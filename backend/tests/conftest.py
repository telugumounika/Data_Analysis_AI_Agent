import os
import sys
from pathlib import Path

os.environ["USE_LLM"] = "false"  # tests must never need a running LLM
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest


@pytest.fixture
def df():
    return pd.DataFrame({
        "Order_Date": pd.to_datetime(["2025-01-05", "2025-01-20", "2025-02-10", "2025-03-01", "2025-03-15"]),
        "Category": ["A", "B", "A", "C", "B"],
        "Region": ["N", "S", "N", None, "S"],
        "Sales": [100.0, 200.0, 300.0, 50.0, 1000.0],
        "Profit": [10.0, 20.0, None, 5.0, 100.0],
    })


@pytest.fixture
def csv_bytes(df):
    return df.to_csv(index=False).encode()
