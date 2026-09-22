"""
Fase CRISP-DM: Data Preparation.
Limpeza, filtragem de cold-start extremo e construção da matriz esparsa usuário-item.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.config import EVENT_WEIGHTS, MIN_ITEM_INTERACTIONS, MIN_USER_INTERACTIONS


@dataclass
class InteractionMatrix:
    matrix: csr_matrix          # shape (n_users, n_items), pesos de interação
    user_to_idx: dict
    item_to_idx: dict
    idx_to_user: dict
    idx_to_item: dict


def clean_events(events: pd.DataFrame) -> pd.DataFrame:
    df = events.dropna(subset=["visitorid", "itemid", "event"]).copy()
    df = df.drop_duplicates()
    df = df[df["event"].isin(EVENT_WEIGHTS.keys())]

    # Remove usuários e itens com poucas interações (cold-start extremo atrapalha CF)
    user_counts = df["visitorid"].value_counts()
    item_counts = df["itemid"].value_counts()
    valid_users = user_counts[user_counts >= MIN_USER_INTERACTIONS].index
    valid_items = item_counts[item_counts >= MIN_ITEM_INTERACTIONS].index
    df = df[df["visitorid"].isin(valid_users) & df["itemid"].isin(valid_items)]
    return df.reset_index(drop=True)


def build_interaction_matrix(events: pd.DataFrame) -> InteractionMatrix:
    df = events.copy()
    df["weight"] = df["event"].map(EVENT_WEIGHTS)

    # Agrega pesos por par (usuário, item) -> soma (visualizou várias vezes + comprou, etc.)
    agg = df.groupby(["visitorid", "itemid"], as_index=False)["weight"].sum()

    users = sorted(agg["visitorid"].unique())
    items = sorted(agg["itemid"].unique())
    user_to_idx = {u: i for i, u in enumerate(users)}
    item_to_idx = {it: i for i, it in enumerate(items)}

    rows = agg["visitorid"].map(user_to_idx).to_numpy()
    cols = agg["itemid"].map(item_to_idx).to_numpy()
    vals = agg["weight"].to_numpy(dtype=np.float32)

    matrix = csr_matrix((vals, (rows, cols)), shape=(len(users), len(items)))

    return InteractionMatrix(
        matrix=matrix,
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        idx_to_user={v: k for k, v in user_to_idx.items()},
        idx_to_item={v: k for k, v in item_to_idx.items()},
    )


def temporal_train_test_split(events: pd.DataFrame, test_frac: float = 0.2):
    """Split temporal: treino com eventos mais antigos, teste com os mais recentes.
    Mais realista para recsys do que split aleatório (evita vazamento de futuro)."""
    df = events.sort_values("timestamp")
    cutoff = int(len(df) * (1 - test_frac))
    return df.iloc[:cutoff].reset_index(drop=True), df.iloc[cutoff:].reset_index(drop=True)
