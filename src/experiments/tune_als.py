"""
Experimento de hiperparâmetros do ALS (Fase CRISP-DM: Modeling/Evaluation).
Gera reports/als_tuning.csv, que justifica os parâmetros usados no pipeline.

    python -m src.experiments.tune_als
"""
import logging
import time

import numpy as np
import pandas as pd

from src.config import RANDOM_STATE, REPORTS_DIR, TOP_K_DEFAULT
from src.data.ingestion import load_events
from src.data.preprocessing import build_interaction_matrix, clean_events, temporal_train_test_split
from src.data.sanitization import sanitize_events
from src.evaluation.compare import build_relevant_items, split_users_by_history
from src.evaluation.metrics import evaluate_recommender
from src.models.collaborative_filtering import ALSCollaborativeFiltering

logger = logging.getLogger("recsys.tuning")

GRID = [
    {"factors": 32, "regularization": 0.05, "weighting": None},   # configuração inicial
    {"factors": 64, "regularization": 0.01, "weighting": None},
    {"factors": 64, "regularization": 0.10, "weighting": None},
    {"factors": 32, "regularization": 0.05, "weighting": "bm25"},
    {"factors": 64, "regularization": 0.01, "weighting": "bm25"},
    {"factors": 64, "regularization": 0.10, "weighting": "bm25"},
    {"factors": 128, "regularization": 0.05, "weighting": "bm25"},
]


def main(n_users_per_segment: int = 1000, iterations: int = 15):
    events = clean_events(sanitize_events(load_events()))
    train, test = temporal_train_test_split(events, test_frac=0.2)
    matrix = build_interaction_matrix(train)
    relevant_all = build_relevant_items(train, test, matrix)
    segments = split_users_by_history(relevant_all, matrix)

    rng = np.random.default_rng(RANDOM_STATE)
    samples = {}
    for name, users in segments.items():
        users = sorted(users)
        chosen = rng.choice(users, size=min(n_users_per_segment, len(users)), replace=False)
        samples[name] = {int(u): relevant_all[int(u)] for u in chosen}

    rows = []
    for params in GRID:
        start = time.time()
        als = ALSCollaborativeFiltering(iterations=iterations, **params).fit(matrix.matrix)
        train_seconds = time.time() - start
        row = {**params, "weighting": params["weighting"] or "none", "train_seconds": round(train_seconds, 1)}
        for seg, relevant in samples.items():
            def recommend(user_id, als=als):
                return [matrix.idx_to_item[i] for i, _ in als.recommend(matrix.user_to_idx[user_id], n=TOP_K_DEFAULT)]

            m = evaluate_recommender(recommend, relevant, k=TOP_K_DEFAULT)
            row[f"ndcg@5_{seg}"] = m["ndcg@5"]
            row[f"recall@5_{seg}"] = m["recall@5"]
        rows.append(row)
        logger.info("%s", row)

    result = pd.DataFrame(rows)
    result["ndcg@5_medio"] = result[[c for c in result.columns if c.startswith("ndcg@5_")]].mean(axis=1).round(4)
    result = result.sort_values("ndcg@5_medio", ascending=False)
    result.to_csv(REPORTS_DIR / "als_tuning.csv", index=False)
    logger.info("Melhor configuração:\n%s", result.head(1).to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    main()
