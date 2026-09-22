import pandas as pd

from src.models.clustering import fit_kmeans, cluster_profile


def _fake_user_features(n=30):
    return pd.DataFrame(
        {
            "visitorid": range(1, n + 1),
            "recency_days": [i % 10 for i in range(n)],
            "frequency": [10 + i for i in range(n)],
            "n_unique_items": [5 + i % 7 for i in range(n)],
            "engagement_score": [20.0 + i for i in range(n)],
            "conversion_rate": [0.1 * (i % 5) for i in range(n)],
        }
    )


def test_fit_kmeans_assigns_cluster_to_every_user():
    df = _fake_user_features()
    _, _, clustered = fit_kmeans(df, n_clusters=3)
    assert "cluster" in clustered.columns
    assert clustered["cluster"].notna().all()
    assert clustered["cluster"].nunique() <= 3


def test_cluster_profile_has_one_row_per_cluster():
    df = _fake_user_features()
    _, _, clustered = fit_kmeans(df, n_clusters=3)
    profile = cluster_profile(clustered)
    assert len(profile) == clustered["cluster"].nunique()


def test_fit_kmeans_handles_small_dataset_gracefully():
    # Menos usuários que clusters pedidos: não pode quebrar
    df = _fake_user_features(n=4)
    _, _, clustered = fit_kmeans(df, n_clusters=10)
    assert clustered["cluster"].nunique() <= 4
