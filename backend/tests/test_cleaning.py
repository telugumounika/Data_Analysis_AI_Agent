import pandas as pd
import pytest

from app import agent
from app.cleaning import apply_edits, fill_missing, fill_options
from app.tools import ToolError


def test_fill_options_lists_only_columns_with_missing(df):
    names = [c["name"] for c in fill_options(df)["columns"]]
    assert names == ["Region", "Profit"]


@pytest.mark.parametrize("method,expected", [("mean", 33.75), ("median", 15.0)])
def test_fill_numeric_statistics(df, method, expected):
    out, msg = fill_missing(df, "Profit", method)
    assert out["Profit"].isna().sum() == 0 and out.loc[2, "Profit"] == expected
    assert "Filled 1" in msg


def test_fill_mode_text(df):
    out, _ = fill_missing(df, "Region", "mode")
    assert out["Region"].isna().sum() == 0


def test_ffill_and_bfill(df):
    assert fill_missing(df, "Profit", "ffill")[0].loc[2, "Profit"] == 20.0
    assert fill_missing(df, "Profit", "bfill")[0].loc[2, "Profit"] == 5.0


def test_interpolate(df):
    assert fill_missing(df, "Profit", "interpolate")[0].loc[2, "Profit"] == 12.5


def test_constant_and_validation(df):
    assert fill_missing(df, "Region", "constant", "Unknown")[0].loc[3, "Region"] == "Unknown"
    with pytest.raises(ToolError):
        fill_missing(df, "Profit", "constant", "abc")
    with pytest.raises(ToolError):
        fill_missing(df, "Profit", "constant", "")


def test_mean_on_text_column_rejected(df):
    with pytest.raises(ToolError):
        fill_missing(df, "Region", "mean")


def test_all_columns_skips_inapplicable(df):
    out, msg = fill_missing(df, None, "mean")
    assert out["Profit"].isna().sum() == 0 and out["Region"].isna().sum() == 1
    assert "Skipped" in msg


def test_drop_rows(df):
    out, _ = fill_missing(df, None, "drop_rows")
    assert len(out) == 3


def test_original_not_modified(df):
    fill_missing(df, "Profit", "mean")
    assert df["Profit"].isna().sum() == 1


def test_no_missing_error(df):
    with pytest.raises(ToolError):
        fill_missing(df, "Sales", "mean")


def test_edit_cells_and_types(df):
    out, _ = apply_edits(df, [{"row": 0, "column": "Sales", "value": "999.5"},
                              {"row": 1, "column": "Category", "value": "Z"},
                              {"row": 2, "column": "Order_Date", "value": "2025-05-05"}], [])
    assert out.loc[0, "Sales"] == 999.5 and out.loc[1, "Category"] == "Z"
    assert str(out.loc[2, "Order_Date"].date()) == "2025-05-05"


def test_edit_blank_makes_missing(df):
    out, _ = apply_edits(df, [{"row": 0, "column": "Sales", "value": ""}], [])
    assert pd.isna(out.loc[0, "Sales"])


def test_edit_integer_column():
    d = pd.DataFrame({"n": [1, 2, 3]})
    assert apply_edits(d, [{"row": 0, "column": "n", "value": "7"}], [])[0]["n"].tolist() == [7, 2, 3]
    assert pd.isna(apply_edits(d, [{"row": 0, "column": "n", "value": ""}], [])[0].loc[0, "n"])


def test_bad_edits_rejected(df):
    with pytest.raises(ToolError):
        apply_edits(df, [{"row": 0, "column": "Sales", "value": "abc"}], [])
    with pytest.raises(ToolError):
        apply_edits(df, [{"row": 99, "column": "Sales", "value": "1"}], [])
    with pytest.raises(ToolError):
        apply_edits(df, [], [0, 1, 2, 3, 4])


def test_delete_rows(df):
    out, _ = apply_edits(df, [], [0, 2])
    assert len(out) == 3 and out.index.tolist() == [0, 1, 2]


def test_agent_asks_before_filling(df):
    for q in ["fill missing values", "Please impute the nulls", "handle missing data", "remove missing values"]:
        r = agent.ask(df, q)
        assert r["mode"] == "clarify" and r["needs_input"]["type"] == "fill_missing", q
        assert df["Profit"].isna().sum() == 1   # nothing changed


def test_agent_suggests_method_from_question(df):
    r = agent.ask(df, "fill missing values in Profit with the median")
    assert r["needs_input"]["suggested_method"] == "median"
    assert r["needs_input"]["suggested_column"] == "Profit"


def test_checking_missing_is_not_a_fill_request(df):
    assert agent.ask(df, "Are there any missing values?")["mode"] != "clarify"
