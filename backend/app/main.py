"""The FastAPI application (the 'waiter' between browser and Python)."""
import io
import re
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import agent
from .cleaning import apply_edits, fill_missing, fill_options
from .config import LLM_BASE_URL, LLM_MODEL, MAX_TABLE_ROWS, USE_LLM
from .file_handler import FileError, load_dataframe, save_temp_file, validate_upload
from .insights import generate_insights
from .llm_client import check_status
from .tools import ToolError, categorical_columns, date_columns, get_dataset_info, \
    get_summary_statistics, numeric_columns, to_table

app = FastAPI(title="Data Analysis AI Agent", version="1.1.0")

# CORS lets a page served from another origin call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500", "http://localhost:8000",
                   "http://127.0.0.1:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# dataset_id -> {"df", "name", "version", "undo": [DataFrames], "log": [...]}
# In memory = simple; lost when the server restarts.
DATASETS: dict[str, dict] = {}
MAX_DATASETS = 20
MAX_UNDO = 5
PAGE_SIZE_MAX = 100


# ------------------------------------------------------------------ models
class AskRequest(BaseModel):
    dataset_id: str
    question: str = Field(max_length=2000)
    history: list[dict] = []


class InsightsRequest(BaseModel):
    dataset_id: str


class FillRequest(BaseModel):
    column: str | None = None          # None / "__all__" = every column with missing values
    method: str
    value: str | None = None           # only for method "constant"


class CellEdit(BaseModel):
    row: int                           # 0-based row position
    column: str
    value: str | None = None           # text typed by the user; "" means "make it missing"


class EditRequest(BaseModel):
    edits: list[CellEdit] = []
    delete_rows: list[int] = []


# ----------------------------------------------------------------- helpers
def _entry(dataset_id: str) -> dict:
    if dataset_id not in DATASETS:
        raise HTTPException(404, "Dataset not found. Please upload your file again.")
    return DATASETS[dataset_id]


def get_dataset(dataset_id: str):
    return _entry(dataset_id)["df"]


def _commit(entry: dict, new_df, message: str) -> None:
    """Replace the dataset with a new version and remember the old one for undo."""
    entry["undo"].append(entry["df"])
    del entry["undo"][:-MAX_UNDO]
    entry["df"] = new_df
    entry["version"] += 1
    entry["log"].append({"version": entry["version"], "message": message})
    del entry["log"][:-30]


def build_overview(dataset_id: str, entry: dict) -> dict:
    df = entry["df"]
    info = get_dataset_info(df)["data"]
    return {
        "dataset_id": dataset_id,
        "filename": entry["name"],
        "version": entry["version"],
        "can_undo": bool(entry["undo"]),
        "change_log": entry["log"][-10:],
        "rows": info["rows"],
        "columns": info["columns"],
        "numeric_columns": numeric_columns(df),
        "categorical_columns": categorical_columns(df),
        "date_columns": date_columns(df),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_details": info["column_details"],
        "preview": to_table(df, MAX_TABLE_ROWS // 2),
        "statistics": get_summary_statistics(df).get("table") if numeric_columns(df) else None,
        "fill_options": fill_options(df),
    }


async def _read_upload(file: UploadFile):
    content = await file.read()
    try:
        ext = validate_upload(file.filename, len(content))
        _, path = save_temp_file(content, ext)
        try:
            return load_dataframe(path)
        finally:
            path.unlink(missing_ok=True)  # we keep the data in memory only
    except FileError as exc:
        raise HTTPException(400, str(exc))


# ------------------------------------------------------------------ routes
@app.get("/api/health")
def health():
    out = {"status": "ok", "llm_enabled": USE_LLM, "llm_model": LLM_MODEL, "llm_url": LLM_BASE_URL}
    if USE_LLM:
        out["llm"] = check_status()
    return out


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    df = await _read_upload(file)
    if len(DATASETS) >= MAX_DATASETS:
        DATASETS.pop(next(iter(DATASETS)))
    dataset_id = uuid.uuid4().hex
    DATASETS[dataset_id] = {"df": df, "name": file.filename, "version": 1, "undo": [],
                            "log": [{"version": 1, "message": f"Uploaded {file.filename}"}]}
    return build_overview(dataset_id, DATASETS[dataset_id])


@app.post("/api/dataset/{dataset_id}/replace")
async def replace(dataset_id: str, file: UploadFile = File(...)):
    """Upload a newer version of the file; the analysis is re-run by the frontend."""
    entry = _entry(dataset_id)
    df = await _read_upload(file)
    _commit(entry, df, f"Replaced data with {file.filename}")
    entry["name"] = file.filename
    return {**build_overview(dataset_id, entry), "message": f"Loaded {file.filename}."}


@app.get("/api/dataset/{dataset_id}/overview")
def overview(dataset_id: str):
    return build_overview(dataset_id, _entry(dataset_id))


@app.get("/api/dataset/{dataset_id}/data")
def data_page(dataset_id: str, page: int = Query(0, ge=0), page_size: int = Query(25, ge=1, le=PAGE_SIZE_MAX)):
    """A page of raw rows for the editable table."""
    df = _entry(dataset_id)["df"]
    start = page * page_size
    table = to_table(df.iloc[start:start + page_size], page_size)
    kinds = {}
    for c in df.columns:
        kinds[c] = ("number" if c in numeric_columns(df) else "date" if c in date_columns(df) else "text")
    return {**table, "start": start, "page": page, "page_size": page_size,
            "total_rows": len(df), "kinds": kinds}


@app.post("/api/dataset/{dataset_id}/fill-missing")
def fill_missing_route(dataset_id: str, req: FillRequest):
    entry = _entry(dataset_id)
    try:
        new_df, message = fill_missing(entry["df"], req.column, req.method, req.value)
    except ToolError as exc:
        raise HTTPException(400, str(exc))
    _commit(entry, new_df, message)
    return {**build_overview(dataset_id, entry), "message": message}


@app.post("/api/dataset/{dataset_id}/edit")
def edit_route(dataset_id: str, req: EditRequest):
    entry = _entry(dataset_id)
    try:
        new_df, message = apply_edits(entry["df"], [e.model_dump() for e in req.edits], req.delete_rows)
    except ToolError as exc:
        raise HTTPException(400, str(exc))
    _commit(entry, new_df, message)
    return {**build_overview(dataset_id, entry), "message": message}


@app.post("/api/dataset/{dataset_id}/undo")
def undo_route(dataset_id: str):
    entry = _entry(dataset_id)
    if not entry["undo"]:
        raise HTTPException(400, "Nothing to undo.")
    entry["df"] = entry["undo"].pop()
    entry["version"] += 1
    entry["log"].append({"version": entry["version"], "message": "Undid the last change."})
    return {**build_overview(dataset_id, entry), "message": "Undid the last change."}


@app.get("/api/dataset/{dataset_id}/download")
def download(dataset_id: str, format: str = "csv"):
    entry = _entry(dataset_id)
    df = entry["df"]
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(entry["name"]).stem) or "dataset"
    if format == "xlsx":
        buf = io.BytesIO()
        df.to_excel(buf, index=False)
        return Response(buf.getvalue(), headers={"Content-Disposition": f'attachment; filename="{stem}_edited.xlsx"'},
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    return Response(df.to_csv(index=False).encode("utf-8"),
                    headers={"Content-Disposition": f'attachment; filename="{stem}_edited.csv"'},
                    media_type="text/csv")


@app.post("/api/ask")
def ask(req: AskRequest):
    entry = _entry(req.dataset_id)
    try:
        return agent.ask(entry["df"], req.question, req.history)
    except ToolError as exc:
        raise HTTPException(400, str(exc))
    except Exception:  # never leak a stack trace to the browser
        raise HTTPException(500, "Something went wrong while analysing the data.")


@app.post("/api/insights")
def insights(req: InsightsRequest):
    return generate_insights(get_dataset(req.dataset_id))


# Serve the frontend from the same server -> http://localhost:8000
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(FRONTEND_DIR / "index.html")
