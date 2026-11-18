import pandas as pd

from src.data.preprocessing import build_interaction_matrix
from src.evaluation.compare import build_relevant_items, popularity_ranking, split_users_by_history


def _train():
    return pd.DataFrame({
        "timestamp": range(8),
        "visitorid": [1, 1, 1, 1, 2, 2, 3, 3],
        "event": ["view"] * 6 + ["transaction", "view"],
        "itemid": [10, 11, 12, 13, 10, 11, 10, 11],
    })


def test_relevant_items_exclude_already_seen_and_unknown_items():
    train = _train()
    im = build_interaction_matrix(train)
    test = pd.DataFrame({"timestamp": [9, 10, 11], "visitorid": [2, 2, 2], "event": ["view"] * 3, "itemid": [10, 12, 999]})
    rel = build_relevant_items(train, test, im)
    assert rel == {2: {12}}


def test_split_users_by_history_threshold():
    train = _train()
    im = build_interaction_matrix(train)
    groups = split_users_by_history({1: {99}, 2: {99}}, im)
    assert groups["com_historico"] == [1]
    assert groups["pouco_historico"] == [2]


def test_popularity_ranking_orders_by_weighted_events():
    assert popularity_ranking(_train(), top_n=2)[0] == 10  # tem a transação
