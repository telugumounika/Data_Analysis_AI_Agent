"""Data cleaning and editing.

Everything here returns a NEW DataFrame - the caller decides whether to keep it.
That makes undo easy and keeps the analysis tools read-only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .tools import ToolError, resolve_column

MAX_EDITS = 500

METHODS = [
    {"id": "mean", "label": "Mean (average)", "numeric_only": True, "needs_value": False,
     "description": "Replace with the column average. Numeric columns only."},
    {"id": "median", "label": "Median", "numeric_only": True, "needs_value": False,
     "description": "Replace with the middle value. Safer than mean when there are outliers."},
    {"id": "mode", "label": "Mode (most frequent value)", "numeric_only": False, "needs_value": False,
     "description": "Replace with the most common value. Works for text and numbers."},
    {"id": "ffill", "label": "Forward fill (ffill)", "numeric_only": False, "needs_value": False,
     "description": "Copy the value from the row above. Best for time-ordered data."},
    {"id": "bfill", "label": "Backward fill (bfill)", "numeric_only": False, "needs_value": False,
     "description": "Copy the value from the row below."},
    {"id": "interpolate", "label": "Linear interpolation", "numeric_only": True, "needs_value": False,
     "description": "Estimate from the neighbouring values. Numeric columns only."},
    {"id": "constant", "label": "A value I type", "numeric_only": False, "needs_value": True,
     "description": "Fill with your own value, e.g. 0 or Unknown."},
    {"id": "drop_rows", "label": "Delete rows with missing values", "numeric_only": False, "needs_value": False,
     "description": "Remove every row that has a missing value in the chosen column(s)."},
]
METHOD_IDS = {m["id"] for m in METHODS}


def fill_options(df: pd.DataFrame) -> dict:
    """What the frontend needs to ask the user 'how should I fill?'."""
    cols = [{"name": c, "missing": int(df[c].isna().sum()), "dtype": str(df[c].dtype),
             "numeric": bool(pd.api.types.is_numeric_dtype(df[c]))}
            for c in df.columns if df[c].isna().any()]
    return {"columns": cols, "methods": METHODS}


# ------------------------------------------------------------ fill missing
def _coerce_constant(s: pd.Series, value):
    if value is None or str(value).strip() == "":
        raise ToolError("Please type the value to fill with.")
    if pd.api.types.is_numeric_dtype(s):
        try:
            return float(value)
        except ValueError:
            raise ToolError(f"'{value}' is not a number, but column '{s.name}' is numeric.")
    if pd.api.types.is_datetime64_any_dtype(s):
        try:
            return pd.to_datetime(value)
        except (ValueError, TypeError):
            raise ToolError(f"'{value}' is not a valid date for column '{s.name}'.")
    return str(value)


def _fill_column(s: pd.Series, method: str, value) -> tuple[pd.Series, str]:
    numeric = pd.api.types.is_numeric_dtype(s)
    if method in {"mean", "median", "interpolate"} and not numeric:
        raise ToolError(f"'{method}' only works on numeric columns, but '{s.name}' is {s.dtype}.")
    if method in {"mean", "median"}:
        fill = s.mean() if method == "mean" else s.median()
        if pd.isna(fill):
            raise ToolError(f"Column '{s.name}' has no values to calculate the {method} from.")
        return s.fillna(fill), f"{method} ({round(float(fill), 4)})"
    if method == "mode":
        m = s.mode(dropna=True)
        if m.empty:
            raise ToolError(f"Column '{s.name}' has no values to take a mode from.")
        return s.fillna(m.iloc[0]), f"mode ({m.iloc[0]})"
    if method == "ffill":
        return s.ffill(), "forward fill"
    if method == "bfill":
        return s.bfill(), "backward fill"
    if method == "interpolate":
        return s.interpolate(limit_direction="both"), "linear interpolation"
    if method == "constant":
        fill = _coerce_constant(s, value)
        return s.fillna(fill), f"value ({value})"
    raise ToolError(f"Unknown method '{method}'.")


def fill_missing(df: pd.DataFrame, column: str | None, method: str, value=None) -> tuple[pd.DataFrame, str]:
    if method not in METHOD_IDS:
        raise ToolError(f"Unknown method '{method}'. Choose one of {sorted(METHOD_IDS)}.")
    all_cols = column in (None, "", "__all__")
    if all_cols:
        targets = [c for c in df.columns if df[c].isna().any()]
    else:
        targets = [resolve_column(df, column)]
        if not df[targets[0]].isna().any():
            raise ToolError(f"Column '{targets[0]}' has no missing values.")
    if not targets:
        raise ToolError("There are no missing values to fill.")

    out = df.copy()
    if method == "drop_rows":
        before = len(out)
        out = out.dropna(subset=targets).reset_index(drop=True)
        if out.empty:
            raise ToolError("That would delete every row. Choose another method.")
        return out, f"Deleted {before - len(out)} rows with missing values in {', '.join(targets)}."

    parts, skipped, total = [], [], 0
    for c in targets:
        before = int(out[c].isna().sum())
        try:
            new, desc = _fill_column(out[c], method, value)
        except ToolError:
            if not all_cols:
                raise
            skipped.append(c)
            continue
        out[c] = new
        after = int(new.isna().sum())
        total += before - after
        parts.append(f"{c}: {before - after} filled using {desc}" + (f", {after} still missing" if after else ""))
    if not parts:
        raise ToolError(f"The '{method}' method can't be applied to the columns that have missing values.")
    msg = f"Filled {total} missing values. " + "; ".join(parts) + "."
    if skipped:
        msg += f" Skipped (method not applicable): {', '.join(skipped)}."
    return out, msg


# ------------------------------------------------------------- manual edits
def _convert_cell(s: pd.Series, raw):
    """Convert text typed in the browser into the column's real type."""
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        return pd.NaT if pd.api.types.is_datetime64_any_dtype(s) else np.nan
    if pd.api.types.is_bool_dtype(s):
        if str(raw).strip().lower() in {"true", "1", "yes"}:
            return True
        if str(raw).strip().lower() in {"false", "0", "no"}:
            return False
        raise ToolError(f"'{raw}' is not true/false for column '{s.name}'.")
    if pd.api.types.is_numeric_dtype(s):
        try:
            return float(str(raw).replace(",", ""))
        except ValueError:
            raise ToolError(f"'{raw}' is not a valid number for column '{s.name}'.")
    if pd.api.types.is_datetime64_any_dtype(s):
        try:
            return pd.to_datetime(raw)
        except (ValueError, TypeError):
            raise ToolError(f"'{raw}' is not a valid date for column '{s.name}'.")
    return str(raw)


def apply_edits(df: pd.DataFrame, edits: list[dict], delete_rows: list[int]) -> tuple[pd.DataFrame, str]:
    if not edits and not delete_rows:
        raise ToolError("Nothing to save.")
    if len(edits) > MAX_EDITS:
        raise ToolError(f"Too many edits at once (max {MAX_EDITS}).")
    out = df.copy()

    for e in edits:
        row = int(e["row"])
        if not 0 <= row < len(out):
            raise ToolError(f"Row {row + 1} does not exist.")
        col = resolve_column(out, e["column"])
        s = out[col].copy()
        val = _convert_cell(s, e.get("value"))
        if pd.api.types.is_integer_dtype(s) and (pd.isna(val) or not float(val).is_integer()):
            s = s.astype("float64")
        elif pd.api.types.is_integer_dtype(s):
            val = int(val)
        s.iloc[row] = val
        out[col] = s

    removed = 0
    if delete_rows:
        idx = sorted({int(i) for i in delete_rows})
        if idx[0] < 0 or idx[-1] >= len(out):
            raise ToolError("One of the rows to delete does not exist.")
        if len(idx) >= len(out):
            raise ToolError("You can't delete every row.")
        keep = np.ones(len(out), dtype=bool)
        keep[idx] = False
        out = out[keep].reset_index(drop=True)
        removed = len(idx)

    parts = []
    if edits:
        parts.append(f"Edited {len(edits)} cell(s)")
    if removed:
        parts.append(f"deleted {removed} row(s)")
    return out, ", ".join(parts) + "."
