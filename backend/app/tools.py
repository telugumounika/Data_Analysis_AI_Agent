"""Stages 5 & 8: the analysis tools.

Every tool is a normal Python function that takes the DataFrame plus simple
arguments and returns a JSON-friendly dict. The LLM never runs code itself -
it can only ask us to call one of these functions. That is what makes the
agent safe.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .charts import make_chart
from .config import MAX_TABLE_ROWS


class ToolError(Exception):
    """A problem the agent / user can understand (bad column name, etc.)."""


# ---------------------------------------------------------------- helpers
def _clean(value: Any) -> Any:
    """Convert numpy / pandas values into plain JSON-safe Python values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if (math.isnan(value) or math.isinf(value)) else round(float(value), 4)
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    if value is pd.NaT or value is None:
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def to_table(df: pd.DataFrame, max_rows: int = MAX_TABLE_ROWS) -> dict:
    df = df.head(max_rows)
    rows = [[_clean(v) for v in row] for row in df.itertuples(index=False, name=None)]
    return {"columns": [str(c) for c in df.columns], "rows": rows}


def resolve_column(df: pd.DataFrame, name: str | None) -> str:
    """Match a column name case-insensitively so 'sales' finds 'Sales'."""
    if name is None or str(name).strip() == "":
        raise ToolError("A column name is required.")
    name = str(name).strip()
    if name in df.columns:
        return name
    lowered = {c.lower().replace("_", " "): c for c in df.columns}
    key = name.lower().replace("_", " ")
    if key in lowered:
        return lowered[key]
    raise ToolError(f"Column '{name}' not found. Available columns: {list(df.columns)}")


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()


def date_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="datetime").columns.tolist()


def _require_numeric(df: pd.DataFrame, col: str) -> None:
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise ToolError(f"Column '{col}' is not numeric.")


AGGS = {"sum", "mean", "median", "min", "max", "count"}


def _agg_name(agg: str | None) -> str:
    agg = (agg or "sum").lower()
    if agg == "average":
        agg = "mean"
    if agg not in AGGS:
        raise ToolError(f"Unsupported aggregation '{agg}'. Use one of {sorted(AGGS)}.")
    return agg


# ------------------------------------------------------------------ tools
def get_dataset_info(df: pd.DataFrame, **_) -> dict:
    cols = [
        {"name": c, "type": str(df[c].dtype), "missing": int(df[c].isna().sum()),
         "unique": int(df[c].nunique())}
        for c in df.columns
    ]
    return {
        "summary": f"The dataset has {len(df)} rows and {len(df.columns)} columns.",
        "data": {
            "rows": len(df),
            "columns": len(df.columns),
            "numeric_columns": numeric_columns(df),
            "categorical_columns": categorical_columns(df),
            "date_columns": date_columns(df),
            "column_details": cols,
        },
    }


def get_summary_statistics(df: pd.DataFrame, **_) -> dict:
    nums = numeric_columns(df)
    if not nums:
        raise ToolError("The dataset has no numeric columns.")
    desc = df[nums].describe().T.reset_index().rename(columns={"index": "column"})
    return {"summary": f"Summary statistics for {len(nums)} numeric columns.",
            "table": to_table(desc)}


def find_missing_values(df: pd.DataFrame, **_) -> dict:
    miss = df.isna().sum()
    miss = miss[miss > 0]
    total_cells = df.shape[0] * df.shape[1]
    pct_total = round(float(miss.sum()) / total_cells * 100, 2)
    if miss.empty:
        return {"summary": "There are no missing values in the dataset.",
                "data": {"total_missing": 0}}
    table = pd.DataFrame({"column": miss.index, "missing": miss.values,
                          "percent": (miss.values / len(df) * 100).round(2)})
    return {"summary": f"{int(miss.sum())} missing values found in {len(miss)} columns "
                       f"({pct_total}% of all cells).",
            "table": to_table(table), "data": {"total_missing": int(miss.sum())}}


def find_duplicates(df: pd.DataFrame, **_) -> dict:
    dup = int(df.duplicated().sum())
    return {"summary": f"{dup} duplicate rows found." if dup else "No duplicate rows found.",
            "data": {"duplicate_rows": dup}}


def calculate_statistics(df: pd.DataFrame, column: str = None, stat: str = "sum", **_) -> dict:
    col = resolve_column(df, column)
    stat = _agg_name(stat)
    if stat != "count":
        _require_numeric(df, col)
    value = _clean(getattr(df[col], stat)())
    return {"summary": f"The {stat} of {col} is {value}.", "data": {"column": col, "stat": stat, "value": value}}


def group_data(df: pd.DataFrame, group_by: str = None, value_column: str = None,
               agg: str = "sum", top_n: int = 20, ascending: bool = False, **_) -> dict:
    g = resolve_column(df, group_by)
    agg = _agg_name(agg)
    if agg == "count" and not value_column:
        result = df.groupby(g).size().rename("count")
        label = "count"
    else:
        v = resolve_column(df, value_column)
        if agg != "count":
            _require_numeric(df, v)
        result = df.groupby(g)[v].agg(agg)
        label = f"{agg}_{v}"
    result = result.sort_values(ascending=ascending).head(int(top_n)).reset_index()
    result.columns = [g, label]
    best = result.iloc[0]
    return {"summary": f"Grouped by {g} ({label}). Top entry: {_clean(best[g])} = {_clean(best[label])}.",
            "table": to_table(result, int(top_n)),
            "data": {"group_by": g, "metric": label}}


def find_top_records(df: pd.DataFrame, column: str = None, n: int = 10,
                     group_by: str | None = None, agg: str = "sum", **_) -> dict:
    return _top_bottom(df, column, n, group_by, agg, ascending=False)


def find_bottom_records(df: pd.DataFrame, column: str = None, n: int = 10,
                        group_by: str | None = None, agg: str = "sum", **_) -> dict:
    return _top_bottom(df, column, n, group_by, agg, ascending=True)


def _top_bottom(df, column, n, group_by, agg, ascending) -> dict:
    n = max(1, min(int(n or 10), 100))
    col = resolve_column(df, column)
    _require_numeric(df, col)
    word = "Bottom" if ascending else "Top"
    if group_by:
        out = group_data(df, group_by=group_by, value_column=col, agg=agg, top_n=n, ascending=ascending)
        out["summary"] = f"{word} {n} by {out['data']['metric']}. " + out["summary"].split(". ", 1)[-1]
        return out
    sub = df.sort_values(col, ascending=ascending).head(n)
    return {"summary": f"{word} {len(sub)} records by {col}.", "table": to_table(sub, n)}


def filter_data(df: pd.DataFrame, column: str = None, operator: str = "==",
                value: Any = None, **_) -> dict:
    col = resolve_column(df, column)
    ops = {"==", "!=", ">", "<", ">=", "<=", "contains"}
    if operator not in ops:
        raise ToolError(f"Unsupported operator '{operator}'. Use one of {sorted(ops)}.")
    series = df[col]
    if operator == "contains":
        mask = series.astype(str).str.contains(str(value), case=False, na=False)
    else:
        if pd.api.types.is_numeric_dtype(series):
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ToolError(f"'{value}' is not a number, but {col} is numeric.")
        elif pd.api.types.is_datetime64_any_dtype(series):
            value = pd.to_datetime(value)
        else:
            series = series.astype(str).str.lower()
            value = str(value).lower()
        mask = {"==": series == value, "!=": series != value, ">": series > value,
                "<": series < value, ">=": series >= value, "<=": series <= value}[operator]
    sub = df[mask]
    return {"summary": f"{len(sub)} of {len(df)} rows match {col} {operator} {value}.",
            "table": to_table(sub), "data": {"matching_rows": int(len(sub))}}


def calculate_correlation(df: pd.DataFrame, **_) -> dict:
    nums = numeric_columns(df)
    if len(nums) < 2:
        raise ToolError("Need at least two numeric columns for correlation.")
    corr = df[nums].corr()
    pairs = (corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
             .stack().reset_index())
    pairs.columns = ["column_1", "column_2", "correlation"]
    pairs["abs"] = pairs["correlation"].abs()
    pairs = pairs.sort_values("abs", ascending=False).drop(columns="abs")
    top = pairs.iloc[0]
    return {"summary": f"Strongest correlation: {top.column_1} vs {top.column_2} "
                       f"({round(float(top.correlation), 3)}).",
            "table": to_table(pairs, 10)}


def find_outliers(df: pd.DataFrame, column: str | None = None, **_) -> dict:
    """IQR rule: values beyond 1.5 * IQR from the quartiles are flagged."""
    cols = [resolve_column(df, column)] if column else numeric_columns(df)
    rows = []
    for c in cols:
        _require_numeric(df, c)
        s = df[c].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((s < low) | (s > high)).sum())
        rows.append({"column": c, "outliers": count, "lower_bound": low, "upper_bound": high})
    if not rows:
        raise ToolError("No numeric data available for outlier detection.")
    table = pd.DataFrame(rows)
    total = int(table["outliers"].sum())
    return {"summary": f"{total} potential outliers found using the IQR rule.",
            "table": to_table(table), "data": {"total_outliers": total}}


def time_trend(df: pd.DataFrame, date_column: str = None, value_column: str = None,
               freq: str = "M", agg: str = "sum", **_) -> dict:
    d = resolve_column(df, date_column)
    v = resolve_column(df, value_column)
    _require_numeric(df, v)
    if not pd.api.types.is_datetime64_any_dtype(df[d]):
        raise ToolError(f"Column '{d}' is not a date column.")
    agg = _agg_name(agg)
    freq = {"M": "M", "month": "M", "monthly": "M", "Q": "Q", "quarter": "Q",
            "Y": "Y", "year": "Y", "D": "D", "day": "D"}.get(freq, "M")
    period = df[d].dt.to_period(freq)
    result = df.groupby(period)[v].agg(agg).reset_index()
    result[d] = result[d].astype(str)
    result.columns = ["period", f"{agg}_{v}"]
    peak = result.loc[result.iloc[:, 1].idxmax()]
    trend = "increased" if result.iloc[-1, 1] > result.iloc[0, 1] else "decreased"
    return {"summary": f"{agg} of {v} by period. It {trend} from the first to the last period; "
                       f"peak was {peak['period']} ({_clean(peak.iloc[1])}).",
            "table": to_table(result, 60), "data": {"periods": len(result)}}


def create_visualization(df: pd.DataFrame, chart_type: str = "bar", x: str | None = None,
                         y: str | None = None, agg: str = "sum", title: str | None = None,
                         top_n: int = 15, **_) -> dict:
    chart_type = (chart_type or "bar").lower()
    allowed = {"bar", "line", "pie", "hist", "histogram", "scatter", "heatmap"}
    if chart_type not in allowed:
        raise ToolError(f"Unsupported chart type '{chart_type}'. Use one of {sorted(allowed)}.")
    agg = _agg_name(agg)

    if chart_type == "heatmap":
        nums = numeric_columns(df)
        if len(nums) < 2:
            raise ToolError("Need at least two numeric columns for a heatmap.")
        image = make_chart("heatmap", title or "Correlation heatmap", corr=df[nums].corr())
        return {"summary": "Correlation heatmap of the numeric columns.", "chart": image}

    if chart_type in {"hist", "histogram"}:
        col = resolve_column(df, x or y)
        _require_numeric(df, col)
        image = make_chart("hist", title or f"Distribution of {col}", values=df[col].dropna(), xlabel=col)
        return {"summary": f"Histogram showing the distribution of {col}.", "chart": image}

    if chart_type == "scatter":
        xc, yc = resolve_column(df, x), resolve_column(df, y)
        _require_numeric(df, xc)
        _require_numeric(df, yc)
        sample = df[[xc, yc]].dropna().head(5000)
        image = make_chart("scatter", title or f"{yc} vs {xc}", x=sample[xc], y=sample[yc],
                           xlabel=xc, ylabel=yc)
        return {"summary": f"Scatter plot of {yc} against {xc}.", "chart": image}

    # bar / line / pie need grouped data
    xc = resolve_column(df, x)
    if y:
        yc = resolve_column(df, y)
        _require_numeric(df, yc) if agg != "count" else None
    elif agg == "count":
        yc = None
    else:
        raise ToolError("A numeric column (y) is required for this chart.")

    if pd.api.types.is_datetime64_any_dtype(df[xc]):
        key = df[xc].dt.to_period("M")
        data = (df.groupby(key)[yc].agg(agg) if yc else df.groupby(key).size())
        data.index = data.index.astype(str)
        data = data.sort_index()
        if chart_type in {"bar", "pie"} and chart_type == "pie":
            chart_type = "bar"
    else:
        data = (df.groupby(xc)[yc].agg(agg) if yc else df.groupby(xc).size())
        data = data.sort_values(ascending=False).head(int(top_n))
    label = f"{agg} of {yc}" if yc else "count"
    image = make_chart(chart_type, title or f"{label} by {xc}", series=data, xlabel=xc, ylabel=label)
    table = data.reset_index()
    table.columns = [xc, label]
    return {"summary": f"{chart_type.title()} chart of {label} by {xc}.",
            "chart": image, "table": to_table(table, 30)}


# ------------------------------------------------------------- registry
def _fn(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required or []}}}


_S = {"type": "string"}
_AGG = {"type": "string", "enum": sorted(AGGS), "description": "Aggregation to apply"}

TOOLS: dict[str, Any] = {
    "get_dataset_info": get_dataset_info,
    "get_summary_statistics": get_summary_statistics,
    "find_missing_values": find_missing_values,
    "find_duplicates": find_duplicates,
    "find_outliers": find_outliers,
    "calculate_statistics": calculate_statistics,
    "group_data": group_data,
    "find_top_records": find_top_records,
    "find_bottom_records": find_bottom_records,
    "filter_data": filter_data,
    "calculate_correlation": calculate_correlation,
    "time_trend": time_trend,
    "create_visualization": create_visualization,
}

TOOL_SCHEMAS = [
    _fn("get_dataset_info", "Rows, columns, column types. Use for questions about the dataset structure.", {}),
    _fn("get_summary_statistics", "describe() statistics for all numeric columns.", {}),
    _fn("find_missing_values", "Count missing values per column.", {}),
    _fn("find_duplicates", "Count duplicate rows.", {}),
    _fn("find_outliers", "Detect unusual values with the IQR rule. Optional single column.",
        {"column": _S}),
    _fn("calculate_statistics", "One number for one column: total/sum, average/mean, median, min, max, count.",
        {"column": _S, "stat": _AGG}, ["column", "stat"]),
    _fn("group_data", "Group by a categorical column and aggregate a numeric column. "
        "Use for 'which X has the highest/lowest Y' and 'Y by X'.",
        {"group_by": _S, "value_column": _S, "agg": _AGG, "top_n": {"type": "integer"},
         "ascending": {"type": "boolean"}}, ["group_by", "agg"]),
    _fn("find_top_records", "Top N rows (or top N groups when group_by is given) by a numeric column. "
        "Use for 'top 10 customers'.",
        {"column": _S, "n": {"type": "integer"}, "group_by": _S, "agg": _AGG}, ["column"]),
    _fn("find_bottom_records", "Bottom N rows or groups by a numeric column.",
        {"column": _S, "n": {"type": "integer"}, "group_by": _S, "agg": _AGG}, ["column"]),
    _fn("filter_data", "Filter rows with one condition and return the count and a preview.",
        {"column": _S, "operator": {"type": "string", "enum": ["==", "!=", ">", "<", ">=", "<=", "contains"]},
         "value": _S}, ["column", "operator", "value"]),
    _fn("calculate_correlation", "Correlation between all numeric columns.", {}),
    _fn("time_trend", "Aggregate a numeric column over time (monthly by default). Needs a date column.",
        {"date_column": _S, "value_column": _S, "freq": {"type": "string", "enum": ["M", "Q", "Y", "D"]},
         "agg": _AGG}, ["date_column", "value_column"]),
    _fn("create_visualization", "Create a chart. chart_type: bar, line, pie, histogram, scatter, heatmap. "
        "bar/line/pie: x = category or date column, y = numeric column. histogram: x = numeric column. "
        "scatter: x and y numeric.",
        {"chart_type": {"type": "string", "enum": ["bar", "line", "pie", "histogram", "scatter", "heatmap"]},
         "x": _S, "y": _S, "agg": _AGG, "title": _S}, ["chart_type"]),
]


def run_tool(name: str, df: pd.DataFrame, args: dict | None) -> dict:
    """The ONLY way the agent touches data: look the name up in a whitelist."""
    if name not in TOOLS:
        raise ToolError(f"Unknown tool '{name}'.")
    args = args or {}
    # Drop any argument the model invented that is not a plain value.
    safe_args = {k: v for k, v in args.items() if isinstance(v, (str, int, float, bool, type(None)))}
    return TOOLS[name](df, **safe_args)
