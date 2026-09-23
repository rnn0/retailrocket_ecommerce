import numpy as np
import pandas as pd
import pytest

from src.data.sanitization import (
    sanitize_events,
    sanitize_item_properties,
    sanitize_category_tree,
    parse_item_property_value,
    SanitizationReport,
    BOT_EVENT_THRESHOLD,
)


def _base_events():
    return pd.DataFrame(
        {
            "timestamp": [1, 2, 3, 3, 4, 5],
            "visitorid": [1, 1, 2, 2, 3, 3],
            "event": ["view", "addtocart", "view", "view", "transaction", "purchase_invalido"],
            "itemid": [10, 10, 20, 20, 30, 40],
            "transactionid": [np.nan, np.nan, np.nan, np.nan, 999, np.nan],
        }
    )


def test_sanitize_events_removes_invalid_event_types():
    df = sanitize_events(_base_events())
    assert set(df["event"].unique()) <= {"view", "addtocart", "transaction"}
    assert "purchase_invalido" not in df["event"].values


def test_sanitize_events_removes_exact_duplicates():
    df = _base_events()
    # A linha 2 e 3 (index) são idênticas -> duplicata exata
    dup = pd.concat([df, df.iloc[[2]]], ignore_index=True)
    result = sanitize_events(dup)
    assert len(result[(result.visitorid == 2) & (result.itemid == 20)]) == 1


def test_sanitize_events_removes_transactionid_inconsistency():
    df = _base_events().copy()
    # Evento "view" com transactionid preenchido = inconsistência
    df.loc[0, "transactionid"] = 555
    result = sanitize_events(df)
    assert not ((result["event"] != "transaction") & result["transactionid"].notnull()).any()


def test_sanitize_events_removes_bot_users():
    n = BOT_EVENT_THRESHOLD + 50
    bot_events = pd.DataFrame(
        {
            "timestamp": range(n),
            "visitorid": [999] * n,
            "event": ["view"] * n,
            "itemid": range(n),
            "transactionid": [np.nan] * n,
        }
    )
    df = pd.concat([_base_events(), bot_events], ignore_index=True)
    result = sanitize_events(df)
    assert 999 not in result["visitorid"].values


def test_sanitization_report_tracks_every_step():
    report = SanitizationReport()
    sanitize_events(_base_events(), report=report)
    log_df = report.as_dataframe()
    assert "rows_removed" in log_df.columns
    assert (log_df["step"] == "TOTAL").any()


def test_parse_item_property_value_numeric():
    assert parse_item_property_value("n15360.000") == pytest.approx(15.36)


def test_parse_item_property_value_non_numeric_kept_as_is():
    assert parse_item_property_value("1116713") == "1116713"


def test_sanitize_item_properties_keeps_latest_value_per_item():
    props = pd.DataFrame(
        {
            "timestamp": [1, 2, 1],
            "itemid": [1, 1, 1],
            "property": ["790", "790", "categoryid"],
            "value": ["n1000.000", "n2000.000", "5"],
        }
    )
    result = sanitize_item_properties(props)
    row = result[result["itemid"] == 1].iloc[0]
    assert row["price"] == pytest.approx(2.0)  # fica com o valor mais recente (timestamp=2)
    assert row["categoryid"] == 5


def test_sanitize_category_tree_removes_orphan_parents():
    tree = pd.DataFrame({"categoryid": [1, 2, 3], "parentid": [np.nan, 1, 999]})
    result = sanitize_category_tree(tree)
    assert 3 not in result["categoryid"].values
    assert {1, 2} <= set(result["categoryid"].values)
