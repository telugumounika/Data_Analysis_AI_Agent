"""Stage 4: validate an uploaded file and load it into a Pandas DataFrame."""
import uuid
from pathlib import Path

import pandas as pd

from .config import ALLOWED_EXTENSIONS, MAX_FILE_MB, UPLOAD_DIR


class FileError(Exception):
    """Raised for any problem with the uploaded file (shown to the user)."""


def validate_upload(filename: str, size_bytes: int) -> str:
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise FileError(f"Unsupported file type '{ext}'. Upload a CSV or Excel file.")
    if size_bytes == 0:
        raise FileError("The uploaded file is empty.")
    if size_bytes > MAX_FILE_MB * 1024 * 1024:
        raise FileError(f"File is larger than {MAX_FILE_MB} MB.")
    return ext


def save_temp_file(content: bytes, ext: str) -> tuple[str, Path]:
    """Save under a random name so user-supplied filenames can never escape the folder."""
    dataset_id = uuid.uuid4().hex
    path = UPLOAD_DIR / f"{dataset_id}{ext}"
    path.write_bytes(content)
    return dataset_id, path


def load_dataframe(path: Path) -> pd.DataFrame:
    try:
        if path.suffix.lower() == ".csv":
            try:
                df = pd.read_csv(path)
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding="latin-1")
        else:
            df = pd.read_excel(path)
    except Exception as exc:  # pandas raises many different errors
        raise FileError(f"Could not read the file: {exc}") from exc

    if df.empty or len(df.columns) == 0:
        raise FileError("The dataset has no rows or columns.")

    df.columns = [str(c).strip() for c in df.columns]
    df = _parse_date_columns(df)
    return df


def _parse_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Turn text columns that look like dates into real datetime columns."""
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            name = col.lower()
            if "date" in name or "time" in name or "month" in name:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() > 0.8:
                    df[col] = parsed
    return df
