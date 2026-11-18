import pandas as pd

from src.features.build_features import build_user_features


def test_user_features_rfm_and_conversion():
    ev = pd.DataFrame({
        "timestamp": [0, 86_400_000, 2 * 86_400_000, 2 * 86_400_000],
        "visitorid": [1, 1, 1, 2],
        "event": ["view", "view", "transaction", "view"],
        "itemid": [10, 11, 10, 12],
    })
    f = build_user_features(ev).set_index("visitorid")
    assert f.loc[1, "frequency"] == 3
    assert f.loc[1, "recency_days"] == 0
    assert f.loc[2, "recency_days"] == 0
    assert f.loc[1, "n_transactions"] == 1
    assert f.loc[1, "conversion_rate"] == 0.5
    assert f.loc[2, "conversion_rate"] == 0
