"""
Fase CRISP-DM: Data Preparation (feature engineering para clustering).
Constrói features agregadas por usuário (estilo RFM + comportamento) para o K-Means.
"""
import pandas as pd
from src.config import EVENT_WEIGHTS


def build_user_features(events: pd.DataFrame) -> pd.DataFrame:
    df = events.copy()
    ref_ts = df["timestamp"].max()

    grouped = df.groupby("visitorid")

    features = grouped.agg(
        recency=("timestamp", lambda x: (ref_ts - x.max()) / (1000 * 60 * 60 * 24)),  # dias desde a última interação
        frequency=("timestamp", "count"),
        n_sessions_proxy=("itemid", "nunique"),
    )

    # Monetário / engajamento proxy: soma dos pesos de evento (view/addtocart/transaction)
    df["weight"] = df["event"].map(EVENT_WEIGHTS)
    monetary = grouped["weight"].sum().rename("engagement_score")

    # Taxa de conversão view -> compra
    event_counts = df.pivot_table(index="visitorid", columns="event", values="weight", aggfunc="count", fill_value=0)
    for col in ("view", "addtocart", "transaction"):
        if col not in event_counts.columns:
            event_counts[col] = 0
    conversion_rate = (event_counts["transaction"] / event_counts["view"].replace(0, pd.NA)).fillna(0).rename(
        "conversion_rate"
    )

    result = (
        features.join(monetary)
        .join(conversion_rate)
        .join(event_counts[["view", "addtocart", "transaction"]])
        .reset_index()
    )
    result.columns = [
        "visitorid",
        "recency_days",
        "frequency",
        "n_unique_items",
        "engagement_score",
        "conversion_rate",
        "n_views",
        "n_addtocart",
        "n_transactions",
    ]
    return result
