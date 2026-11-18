"""
Orquestrador do pipeline ponta a ponta (CRISP-DM: Data Understanding -> Data Preparation
-> Modeling -> Evaluation -> artefatos para Deployment).

Uso:
    python -m src.pipeline                      # pipeline completo
    python -m src.pipeline --eval-users 200     # comparação de estratégias mais rápida
    python -m src.pipeline --skip-comparison    # pula a comparação (só treina e salva)

Saídas:
    models/   -> artefatos serializados + metadata.json (versão, parâmetros, métricas)
    reports/  -> CSVs/JSON de qualidade, clusters e comparação + figuras (reports/figures)
"""
import argparse
import json
import logging
import subprocess
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from src.config import (
    EVAL_USERS_DEFAULT,
    MODELS_DIR,
    N_CLUSTERS_DEFAULT,
    RANDOM_STATE,
    REPORTS_DIR,
    TOP_K_DEFAULT,
)
from src.data.ingestion import data_quality_report, load_category_tree, load_events, load_item_properties
from src.data.preprocessing import build_interaction_matrix, clean_events, temporal_train_test_split
from src.data.sanitization import SanitizationReport, build_item_catalog, sanitize_events
from src.evaluation.compare import compare_strategies
from src.features.build_features import build_user_features
from src.models.clustering import (
    FEATURE_COLS,
    choose_k_by_silhouette,
    cluster_profile,
    fit_kmeans,
    label_clusters,
    make_feature_pipeline,
    top_items_per_cluster,
)
from src.models.collaborative_filtering import ALSCollaborativeFiltering, MemoryBasedCF
from src.reporting import (
    plot_cluster_profile,
    plot_event_distribution,
    plot_sanitization,
    plot_silhouette,
    plot_strategy_comparison,
)

logger = logging.getLogger("recsys.pipeline")

# Escolhidos pelo experimento reports/als_tuning.csv (python -m src.experiments.tune_als):
# BM25 + 128 fatores foi a melhor combinação (nDCG@5 ~2,2x a configuração inicial).
ALS_PARAMS = {"factors": 128, "regularization": 0.05, "iterations": 15, "weighting": "bm25"}
MEMORY_PARAMS = {"k_neighbors": 20}


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def _pick_demo_users(interaction_matrix, n_each: int = 4) -> dict:
    """Usuários de exemplo para a demo (aplicação de e-commerce): com e sem histórico."""
    n_items = np.diff(interaction_matrix.matrix.indptr)
    rng = np.random.default_rng(RANDOM_STATE)

    def pick(idx):
        if len(idx) == 0:
            return []
        chosen = rng.choice(idx, size=min(n_each, len(idx)), replace=False)
        return [int(interaction_matrix.idx_to_user[i]) for i in chosen]

    return {
        "com_historico": pick(np.where(n_items >= 10)[0]),
        "pouco_historico": pick(np.where(n_items < 3)[0]),
    }


def run(eval_users: int = EVAL_USERS_DEFAULT, skip_comparison: bool = False) -> dict:
    t_start = time.time()
    version = datetime.now().strftime("%Y%m%d-%H%M%S")

    # ---------------------------------------------------------------- 1. DATA UNDERSTANDING
    logger.info("1/6 INGESTÃO (Data Understanding)")
    events_raw = load_events()
    quality = data_quality_report(events_raw)
    (REPORTS_DIR / "data_quality.json").write_text(json.dumps(quality, indent=2, ensure_ascii=False))
    logger.info(
        "Qualidade dos dados: %s",
        json.dumps({k: quality[k] for k in ("n_rows", "n_unique_users", "n_unique_items", "n_duplicates", "sparsity_pct")}),
    )
    plot_event_distribution(quality["event_distribution"])

    # ---------------------------------------------------------------- 2. DATA PREPARATION
    logger.info("2/6 SANITIZAÇÃO, LIMPEZA E MATRIZ USUÁRIO-ITEM (Data Preparation)")
    sanitization_report = SanitizationReport()
    events_sanitized = sanitize_events(events_raw, report=sanitization_report)
    report_df = sanitization_report.as_dataframe()
    report_df.to_csv(REPORTS_DIR / "sanitization_report.csv", index=False)
    plot_sanitization(report_df)
    logger.info("Sanitização:\n%s", report_df.to_string(index=False))

    events = clean_events(events_sanitized)
    train_events, test_events = temporal_train_test_split(events, test_frac=0.2)
    interaction_matrix = build_interaction_matrix(train_events)
    n_users, n_items = interaction_matrix.matrix.shape
    density = interaction_matrix.matrix.nnz / (n_users * n_items)
    logger.info(
        "Matriz: %d usuários x %d itens | esparsidade %.4f%% | treino=%d teste=%d eventos",
        n_users, n_items, 100 * (1 - density), len(train_events), len(test_events),
    )

    logger.info("3/6 FEATURE ENGINEERING (RFM para clustering)")
    user_features = build_user_features(train_events)

    logger.info("Catálogo de produtos (item_properties + category_tree)")
    try:
        item_catalog = build_item_catalog(load_item_properties(), load_category_tree())
    except FileNotFoundError:
        logger.warning("item_properties não encontrado: a API responderá sem detalhes de produto.")
        item_catalog = pd.DataFrame(columns=["itemid", "price", "available", "categoryid"])
    logger.info("Catálogo: %d itens com propriedades", len(item_catalog))

    # ---------------------------------------------------------------- 3. MODELING
    logger.info("4/6 MODELAGEM")
    logger.info("-> CF memory-based (cosseno, sob demanda)")
    memory_cf = MemoryBasedCF(**MEMORY_PARAMS).fit(interaction_matrix.matrix)

    logger.info("-> CF model-based (ALS)")
    als_cf = ALSCollaborativeFiltering(**ALS_PARAMS).fit(interaction_matrix.matrix)

    logger.info("-> K-Means (escolha de K via silhouette)")
    X_features = user_features[FEATURE_COLS].fillna(0).to_numpy()
    silhouettes = choose_k_by_silhouette(make_feature_pipeline().fit_transform(X_features))
    best_k = max(silhouettes, key=silhouettes.get) if silhouettes else N_CLUSTERS_DEFAULT
    kmeans_model, scaler, clustered_users = fit_kmeans(user_features, n_clusters=best_k)
    profile = cluster_profile(clustered_users)
    labels = label_clusters(profile)
    sizes = {int(k): int(v) for k, v in clustered_users["cluster"].value_counts().items()}
    profile_out = profile.copy()
    profile_out.insert(0, "label", [labels[int(c)] for c in profile_out.index])
    profile_out.insert(1, "n_users", [sizes[int(c)] for c in profile_out.index])
    profile_out.to_csv(REPORTS_DIR / "cluster_profile.csv")
    pd.Series(silhouettes, name="silhouette").rename_axis("k").to_csv(REPORTS_DIR / "silhouette_by_k.csv")
    plot_silhouette(silhouettes, best_k)
    plot_cluster_profile(profile, labels, sizes)
    logger.info(
        "K=%d | silhouette por K: %s\nPerfil dos clusters:\n%s",
        best_k, {k: round(v, 4) for k, v in silhouettes.items()}, profile_out.to_string(),
    )

    cluster_items = top_items_per_cluster(clustered_users, train_events, n=50)

    # ---------------------------------------------------------------- 4. EVALUATION
    logger.info("5/6 AVALIAÇÃO (Evaluation)")
    X_scaled = scaler.transform(X_features)
    sample = 10_000 if len(X_scaled) > 10_000 else None
    final_silhouette = float(
        silhouette_score(X_scaled, clustered_users["cluster"], sample_size=sample, random_state=RANDOM_STATE)
    )
    logger.info("Silhouette final do K-Means: %.4f", final_silhouette)

    comparison = pd.DataFrame()
    if not skip_comparison:
        comparison = compare_strategies(
            train_events, test_events, interaction_matrix, als_cf, memory_cf,
            clustered_users, cluster_items, k=TOP_K_DEFAULT, n_users_per_segment=eval_users,
        )
        comparison.to_csv(REPORTS_DIR / "strategy_comparison.csv", index=False)
        if not comparison.empty:
            plot_strategy_comparison(comparison, "ndcg@5")

    # ---------------------------------------------------------------- 5. ARTEFATOS (Deployment prep)
    logger.info("6/6 SALVANDO ARTEFATOS (versão %s)", version)
    artifacts = {
        "interaction_matrix": interaction_matrix,
        "als_model": als_cf,
        "memory_cf_model": memory_cf,
        "kmeans_model": kmeans_model,
        "scaler": scaler,
        "clustered_users": clustered_users,
        "cluster_top_items": cluster_items,
        "item_catalog": item_catalog,
    }
    for name, obj in artifacts.items():
        joblib.dump(obj, MODELS_DIR / f"{name}.pkl")

    metadata = {
        "version": version,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "data": {
            "raw_events": int(quality["n_rows"]),
            "events_after_sanitization": int(len(events_sanitized)),
            "train_events": int(len(train_events)),
            "test_events": int(len(test_events)),
            "n_users": int(n_users),
            "n_items": int(n_items),
            "matrix_sparsity_pct": round(100 * (1 - density), 4),
        },
        "params": {"als": ALS_PARAMS, "memory_cf": MEMORY_PARAMS, "k_means_k": int(best_k), "top_k": TOP_K_DEFAULT},
        "clustering": {
            "silhouette_final": round(final_silhouette, 4),
            "silhouette_by_k": {str(k): round(v, 4) for k, v in silhouettes.items()},
            "cluster_labels": {str(k): v for k, v in labels.items()},
            "cluster_sizes": {str(k): v for k, v in sizes.items()},
        },
        "evaluation": comparison.to_dict(orient="records") if not comparison.empty else [],
        "demo_users": _pick_demo_users(interaction_matrix),
        "pipeline_seconds": round(time.time() - t_start, 1),
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    with open(MODELS_DIR / "registry.jsonl", "a") as f:  # histórico de todas as versões treinadas
        summary = {k: metadata[k] for k in ("version", "trained_at", "git_commit", "data", "params")}
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    logger.info("Pipeline concluído em %.1fs. Artefatos em %s", metadata["pipeline_seconds"], MODELS_DIR)
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Pipeline do motor de recomendação")
    parser.add_argument("--eval-users", type=int, default=EVAL_USERS_DEFAULT, help="usuários por segmento na comparação")
    parser.add_argument("--skip-comparison", action="store_true", help="não executa a comparação de estratégias")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    run(eval_users=args.eval_users, skip_comparison=args.skip_comparison)


if __name__ == "__main__":
    main()
