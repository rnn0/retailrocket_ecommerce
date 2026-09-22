"""
Fase CRISP-DM: Modeling.
K-Means para segmentação de consumidores a partir das features RFM.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE


FEATURE_COLS = [
    "recency_days",
    "frequency",
    "n_unique_items",
    "engagement_score",
    "conversion_rate",
]


def choose_k_by_silhouette(X_scaled: np.ndarray, k_range=range(2, 9)) -> dict:
    """Testa vários K e retorna o silhouette de cada um (para decidir o K final)."""
    scores = {}
    for k in k_range:
        if k >= len(X_scaled):
            break
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels)
    return scores


def fit_kmeans(user_features: pd.DataFrame, n_clusters: int = 5):
    X = user_features[FEATURE_COLS].fillna(0).to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_clusters = min(n_clusters, max(2, len(X_scaled) - 1))
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)

    result = user_features.copy()
    result["cluster"] = labels
    return km, scaler, result


def cluster_profile(clustered_users: pd.DataFrame) -> pd.DataFrame:
    """Perfil médio de cada cluster -> usado para interpretação dos resultados."""
    return clustered_users.groupby("cluster")[FEATURE_COLS].mean().round(2)


def top_items_per_cluster(clustered_users: pd.DataFrame, events: pd.DataFrame, n: int = 10) -> dict:
    """Itens mais populares por cluster -> fallback de recomendação para cold-start."""
    merged = events.merge(clustered_users[["visitorid", "cluster"]], on="visitorid", how="inner")
    result = {}
    for cluster_id, group in merged.groupby("cluster"):
        result[int(cluster_id)] = group["itemid"].value_counts().head(n).index.tolist()
    return result
