"""
Fase CRISP-DM: Modeling.
K-Means para segmentação de consumidores a partir das features RFM.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.config import RANDOM_STATE

FEATURE_COLS = [
    "recency_days",
    "frequency",
    "n_unique_items",
    "engagement_score",
    "conversion_rate",
]


def make_feature_pipeline() -> Pipeline:
    """log1p + StandardScaler.

    As features RFM têm cauda muito pesada (a maioria dos usuários tem poucos eventos; uns
    poucos têm centenas). Com StandardScaler puro, esses poucos outliers dominam a distância
    euclidiana e o K-Means gasta um cluster inteiro isolando poucos usuários (silhouette
    artificialmente alto, sem valor de negócio). O log1p comprime a cauda antes de
    padronizar, e os segmentos passam a refletir comportamento, não só extremos."""
    return Pipeline([("log1p", FunctionTransformer(np.log1p, validate=False)), ("scale", StandardScaler())])


def choose_k_by_silhouette(X_scaled: np.ndarray, k_range=range(2, 9), sample_size: int = 10_000) -> dict:
    """Testa vários K e retorna o silhouette de cada um (para decidir o K final).

    O silhouette_score é O(n²) em memória/tempo (matriz de distâncias par a par).
    Para datasets grandes (ex: Retailrocket, ~160k usuários), calcular sobre a base
    inteira é inviável (>100 GB de RAM). Por isso, quando o dataset excede
    `sample_size`, o score é estimado sobre uma amostra aleatória — prática padrão
    (é o que o parâmetro `sample_size` do próprio scikit-learn existe para fazer)."""
    scores = {}
    effective_sample = sample_size if len(X_scaled) > sample_size else None
    # n_init menor quando o dataset é grande: mantém o custo computacional viável
    # sem comprometer a estabilidade do resultado (KMeans já converge bem com poucos
    # reinícios em datasets com muitas amostras).
    n_init = 3 if len(X_scaled) > 50_000 else 10
    for k in k_range:
        if k >= len(X_scaled):
            break
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=n_init)
        labels = km.fit_predict(X_scaled)
        scores[k] = silhouette_score(
            X_scaled, labels, sample_size=effective_sample, random_state=RANDOM_STATE
        )
    return scores


def fit_kmeans(user_features: pd.DataFrame, n_clusters: int = 5):
    X = user_features[FEATURE_COLS].fillna(0).to_numpy()
    scaler = make_feature_pipeline()
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


def label_clusters(profile: pd.DataFrame) -> dict:
    """Rótulo de negócio de cada cluster, a partir do ranking de engajamento médio.
    Retorna {cluster_id: rótulo}. É o que torna o K-Means interpretável para o negócio."""
    order = profile["engagement_score"].sort_values(ascending=False).index.tolist()
    labels = {}
    for rank, cluster_id in enumerate(order):
        if rank == 0:
            labels[int(cluster_id)] = "Alto engajamento"
        elif rank == len(order) - 1:
            labels[int(cluster_id)] = "Baixo engajamento"
        else:
            labels[int(cluster_id)] = "Engajamento intermediário"
    return labels
