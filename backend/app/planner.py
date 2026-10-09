"""Rule-based planner: a simple 'brain' that maps a question to a tool call.

Why does it exist?
1. The app still works when no LLM is running (great for demos and tests).
2. It is the fallback when the LLM fails.
3. It shows you the idea of the agent in its simplest form: question -> tool.
"""
import re

import pandas as pd

from .tools import categorical_columns, date_columns, numeric_columns

MONEY_WORDS = ["sales", "revenue", "amount", "profit", "price", "total", "cost", "quantity"]
CHART_WORDS = ["chart", "plot", "graph", "visual", "show me", "histogram", "heatmap", "pie", "scatter"]


def _mentioned_columns(df: pd.DataFrame, q: str) -> list[str]:
    found = []
    for c in df.columns:
        name = c.lower().replace("_", " ")
        if re.search(rf"\b{re.escape(name)}s?\b", q) or name in q:
            found.append(c)
    return found


def _default_value_column(df: pd.DataFrame, mentioned: list[str]) -> str | None:
    nums = numeric_columns(df)
    for c in mentioned:
        if c in nums:
            return c
    for word in MONEY_WORDS:
        for c in nums:
            if word in c.lower():
                return c
    return nums[0] if nums else None


def _agg_from_question(q: str) -> str:
    if re.search(r"average|mean|avg", q):
        return "mean"
    if "median" in q:
        return "median"
    if re.search(r"\b(min|minimum|lowest|smallest)\b", q):
        return "min"
    if re.search(r"\b(max|maximum|highest|largest|biggest)\b", q):
        return "max"
    if re.search(r"\bcount|how many|number of\b", q):
        return "count"
    return "sum"


def plan(df: pd.DataFrame, question: str) -> list[dict]:
    """Return a list of {'tool': ..., 'args': {...}} steps."""
    q = question.lower().strip()
    mentioned = _mentioned_columns(df, q)
    cats = [c for c in mentioned if c in categorical_columns(df)]
    dates = date_columns(df)
    value = _default_value_column(df, mentioned)
    wants_chart = any(w in q for w in CHART_WORDS)
    n_match = re.search(r"\b(?:top|bottom|first|best|worst)\s+(\d+)\b", q)
    n = int(n_match.group(1)) if n_match else 10

    if "missing" in q or "null" in q or "empty value" in q:
        return [{"tool": "find_missing_values", "args": {}}]
    if "duplicate" in q:
        return [{"tool": "find_duplicates", "args": {}}]
    if "outlier" in q or "unusual" in q or "anomal" in q:
        return [{"tool": "find_outliers", "args": {}}]
    if "correlat" in q or "relationship" in q:
        if wants_chart or "heatmap" in q:
            return [{"tool": "create_visualization", "args": {"chart_type": "heatmap"}}]
        return [{"tool": "calculate_correlation", "args": {}}]
    if re.search(r"insight|summari[sz]e|overview|key findings", q):
        return [{"tool": "__insights__", "args": {}}]
    if re.search(r"statistic|describe|distribution summary", q) and not wants_chart:
        return [{"tool": "get_summary_statistics", "args": {}}]
    if re.search(r"how many (rows|columns)|shape|columns|what is in|dataset info|structure", q) and not wants_chart:
        return [{"tool": "get_dataset_info", "args": {}}]

    # time trends
    if dates and re.search(r"month|trend|over time|daily|yearly|quarter|time", q):
        freq = "Q" if "quarter" in q else "Y" if re.search(r"year|annual", q) else "M"
        if wants_chart or "trend" in q or "monthly" in q:
            return [{"tool": "create_visualization",
                     "args": {"chart_type": "line", "x": dates[0], "y": value, "agg": "sum"}}] \
                if wants_chart else [{"tool": "time_trend",
                                      "args": {"date_column": dates[0], "value_column": value, "freq": freq}}]

    # charts
    if wants_chart:
        chart = ("pie" if "pie" in q else "histogram" if re.search(r"histogram|distribution", q)
                 else "scatter" if "scatter" in q else "bar")
        if chart == "histogram":
            return [{"tool": "create_visualization", "args": {"chart_type": "histogram", "x": value}}]
        if chart == "scatter":
            nums = [c for c in mentioned if c in numeric_columns(df)] or numeric_columns(df)
            if len(nums) >= 2:
                return [{"tool": "create_visualization",
                         "args": {"chart_type": "scatter", "x": nums[0], "y": nums[1]}}]
        group = cats[0] if cats else (categorical_columns(df) or [None])[0]
        if group:
            return [{"tool": "create_visualization",
                     "args": {"chart_type": chart, "x": group, "y": value, "agg": "sum"}}]

    # top / bottom / highest
    group = cats[0] if cats else None
    if re.search(r"\b(top|best)\b", q) or re.search(r"\b(bottom|worst)\b", q):
        tool = "find_bottom_records" if re.search(r"bottom|worst", q) else "find_top_records"
        if not group:
            # "top 10 customers": the plural noun may not exactly match a column name
            for c in categorical_columns(df):
                if c.lower().split()[0].rstrip("s") in q:
                    group = c
                    break
        return [{"tool": tool, "args": {"column": value, "n": n, "group_by": group, "agg": "sum"}}]

    if group and re.search(r"highest|most|lowest|least|best|worst|which|by|each|per", q):
        asc = bool(re.search(r"lowest|least|worst", q))
        return [{"tool": "group_data",
                 "args": {"group_by": group, "value_column": value, "agg": "sum", "ascending": asc,
                          "top_n": 20}}]
    if not group and re.search(r"which|highest|most|lowest", q):
        for c in categorical_columns(df):
            if c.lower().rstrip("s") in q:
                return [{"tool": "group_data",
                         "args": {"group_by": c, "value_column": value, "agg": "sum",
                                  "ascending": bool(re.search(r"lowest|least", q))}}]

    if value and re.search(r"total|sum|average|mean|median|minimum|maximum|max|min|count|how many", q):
        return [{"tool": "calculate_statistics", "args": {"column": value, "stat": _agg_from_question(q)}}]

    return []
