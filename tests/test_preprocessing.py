import pandas as pd

from src.data.preprocessing import build_interaction_matrix, clean_events, temporal_train_test_split


def _events():
    return pd.DataFrame({
        "timestamp": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "visitorid": [1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
        "event": ["view", "addtocart", "transaction", "view", "view", "view", "view", "view", "view", "view"],
        "itemid": [10, 10, 10, 11, 12, 13, 10, 11, 12, 13],
    })


def test_matrix_weights_follow_event_weights():
    im = build_interaction_matrix(_events())
    u, i = im.user_to_idx[1], im.item_to_idx[10]
    assert im.matrix[u, i] > 1  # view + addtocart + transaction acumulam


def test_mappings_are_consistent():
    im = build_interaction_matrix(_events())
    for user, idx in im.user_to_idx.items():
        assert im.idx_to_user[idx] == user


def test_temporal_split_never_leaks_future():
    train, test = temporal_train_test_split(_events(), test_frac=0.2)
    assert train["timestamp"].max() <= test["timestamp"].min()


def test_clean_events_drops_duplicates_and_nulls():
    df = pd.concat([_events(), _events().iloc[:1]])
    df.loc[df.index[-1], "itemid"] = None
    assert len(clean_events(df)) <= len(_events())
