import pandas as pd
import pytest

from app.tools import ToolError, run_tool


def test_total_sales(df):
    assert run_tool("calculate_statistics", df, {"column": "sales", "stat": "sum"})["data"]["value"] == 1650


def test_missing_values(df):
    r = run_tool("find_missing_values", df, {})
    assert r["data"]["total_missing"] == 2


def test_group_by_highest(df):
    r = run_tool("group_data", df, {"group_by": "Category", "value_column": "Sales", "agg": "sum"})
    assert r["table"]["rows"][0] == ["B", 1200.0]


def test_top_records_grouped(df):
    r = run_tool("find_top_records", df, {"column": "Sales", "n": 2, "group_by": "Category"})
    assert len(r["table"]["rows"]) == 2


def test_filter(df):
    r = run_tool("filter_data", df, {"column": "Sales", "operator": ">", "value": "150"})
    assert r["data"]["matching_rows"] == 3


def test_outliers_found(df):
    assert run_tool("find_outliers", df, {"column": "Sales"})["data"]["total_outliers"] >= 1


def test_time_trend(df):
    r = run_tool("time_trend", df, {"date_column": "Order_Date", "value_column": "Sales"})
    assert len(r["table"]["rows"]) == 3


@pytest.mark.parametrize("chart", ["bar", "line", "pie", "histogram", "scatter", "heatmap"])
def test_charts(df, chart):
    args = {"chart_type": chart, "x": "Category", "y": "Sales"}
    if chart == "line":
        args["x"] = "Order_Date"
    if chart == "histogram":
        args = {"chart_type": chart, "x": "Sales"}
    if chart == "scatter":
        args = {"chart_type": chart, "x": "Sales", "y": "Profit"}
    r = run_tool("create_visualization", df, args)
    assert len(r["chart"]) > 500  # base64 PNG


def test_missing_column_error(df):
    with pytest.raises(ToolError):
        run_tool("calculate_statistics", df, {"column": "Nope", "stat": "sum"})


def test_non_numeric_error(df):
    with pytest.raises(ToolError):
        run_tool("calculate_statistics", df, {"column": "Category", "stat": "sum"})


def test_unknown_tool_blocked(df):
    with pytest.raises(ToolError):
        run_tool("__import__('os').system", df, {})


def test_large_dataset():
    big = pd.DataFrame({"g": ["a", "b"] * 250_000, "v": range(500_000)})
    assert run_tool("group_data", big, {"group_by": "g", "value_column": "v", "agg": "sum"})["table"]
