import io

import pandas as pd
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _upload(name, data, ctype="text/csv"):
    return client.post("/api/upload", files={"file": (name, data, ctype)})


def test_health():
    assert client.get("/api/health").json()["status"] == "ok"


def test_csv_upload(csv_bytes):
    r = _upload("sales.csv", csv_bytes)
    assert r.status_code == 200
    body = r.json()
    assert body["rows"] == 5 and body["missing_values"] == 2 and "Order_Date" in body["date_columns"]


def test_excel_upload(df):
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    assert _upload("sales.xlsx", buf.getvalue()).status_code == 200


def test_unsupported_file_type():
    r = _upload("notes.txt", b"hello")
    assert r.status_code == 400 and "Unsupported" in r.json()["detail"]


def test_empty_file():
    assert _upload("empty.csv", b"").status_code == 400


def test_header_only_csv():
    assert _upload("h.csv", b"a,b,c\n").status_code == 400


def test_ask_flow(csv_bytes):
    ds = _upload("sales.csv", csv_bytes).json()["dataset_id"]
    r = client.post("/api/ask", json={"dataset_id": ds, "question": "total sales"})
    assert r.status_code == 200 and "1650" in r.json()["answer"]
    assert client.post("/api/insights", json={"dataset_id": ds}).status_code == 200


def test_unknown_dataset():
    r = client.post("/api/ask", json={"dataset_id": "nope", "question": "hi"})
    assert r.status_code == 404


def test_frontend_served():
    assert client.get("/").status_code == 200


# ---------------- editing / cleaning / auto-refresh support ----------------
def _ds(csv_bytes):
    return _upload("sales.csv", csv_bytes).json()["dataset_id"]


def test_fill_missing_endpoint_bumps_version(csv_bytes):
    ds = _ds(csv_bytes)
    r = client.post(f"/api/dataset/{ds}/fill-missing", json={"column": "Profit", "method": "mean"})
    body = r.json()
    assert r.status_code == 200 and body["version"] == 2 and body["missing_values"] == 1
    assert body["can_undo"] and "Filled 1" in body["message"]
    # analysis on the updated data sees the change
    ans = client.post("/api/ask", json={"dataset_id": ds, "question": "Are there any missing values?"}).json()
    assert "1 missing" in ans["answer"] or "missing values found" in ans["answer"]


def test_fill_missing_bad_method(csv_bytes):
    ds = _ds(csv_bytes)
    r = client.post(f"/api/dataset/{ds}/fill-missing", json={"column": "Region", "method": "mean"})
    assert r.status_code == 400


def test_edit_endpoint_and_data_page(csv_bytes):
    ds = _ds(csv_bytes)
    page = client.get(f"/api/dataset/{ds}/data?page=0&page_size=2").json()
    assert page["total_rows"] == 5 and len(page["rows"]) == 2 and page["kinds"]["Sales"] == "number"
    r = client.post(f"/api/dataset/{ds}/edit", json={"edits": [{"row": 0, "column": "Sales", "value": "5000"}]})
    assert r.status_code == 200
    ans = client.post("/api/ask", json={"dataset_id": ds, "question": "total sales"}).json()
    assert "6550" in ans["answer"]
    assert client.post(f"/api/dataset/{ds}/edit", json={"edits": [{"row": 0, "column": "Sales", "value": "x"}]}).status_code == 400


def test_delete_rows_and_undo(csv_bytes):
    ds = _ds(csv_bytes)
    r = client.post(f"/api/dataset/{ds}/edit", json={"delete_rows": [0, 1]}).json()
    assert r["rows"] == 3
    r = client.post(f"/api/dataset/{ds}/undo").json()
    assert r["rows"] == 5
    assert client.post(f"/api/dataset/{ds}/undo").status_code == 400  # nothing left to undo


def test_download_csv_and_xlsx(csv_bytes):
    ds = _ds(csv_bytes)
    client.post(f"/api/dataset/{ds}/fill-missing", json={"column": "Profit", "method": "constant", "value": "0"})
    r = client.get(f"/api/dataset/{ds}/download")
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    assert len(pd.read_csv(io.BytesIO(r.content))) == 5
    x = client.get(f"/api/dataset/{ds}/download?format=xlsx")
    assert pd.read_excel(io.BytesIO(x.content))["Profit"].isna().sum() == 0


def test_replace_file(csv_bytes):
    ds = _ds(csv_bytes)
    new = b"Order_Date,Category,Sales\n2025-01-01,A,10\n2025-01-02,B,20\n"
    r = client.post(f"/api/dataset/{ds}/replace", files={"file": ("new.csv", new, "text/csv")})
    assert r.status_code == 200 and r.json()["rows"] == 2 and r.json()["version"] == 2
    assert client.post(f"/api/dataset/{ds}/replace", files={"file": ("x.txt", b"hi", "text/plain")}).status_code == 400


def test_ask_fill_returns_clarification_and_changes_nothing(csv_bytes):
    ds = _ds(csv_bytes)
    r = client.post("/api/ask", json={"dataset_id": ds, "question": "fill missing values"}).json()
    assert r["needs_input"]["type"] == "fill_missing"
    assert client.get(f"/api/dataset/{ds}/overview").json()["version"] == 1
