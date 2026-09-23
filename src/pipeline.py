"""
Orquestrador do pipeline ponta a ponta (CRISP-DM: Data Prep -> Modeling -> Evaluation -> Deployment prep).
Rodar com: python -m src.pipeline
"""
import json
import joblib

from src.config import MODELS_DIR, N_CLUSTERS_DEFAULT, TOP_K_DEFAULT, RANDOM_STATE
from src.data.ingestion import load_events, data_quality_report
from src.data.sanitization import sanitize_events, SanitizationReport
from src.data.preprocessing import clean_events, build_interaction_matrix, temporal_train_test_split
from src.features.build_features import build_user_features
from src.models.collaborative_filtering import MemoryBasedCF, ALSCollaborativeFiltering
from src.models.clustering import fit_kmeans, cluster_profile, top_items_per_cluster, choose_k_by_silhouette
from src.evaluation.metrics import evaluate_recommender


def run():
    print("=" * 60)
    print("1/6 - INGESTÃO (Data Understanding)")
    events_raw = load_events()
    report = data_quality_report(events_raw)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    print("=" * 60)
    print("2/6 - SANITIZAÇÃO, LIMPEZA E MATRIZ USUÁRIO-ITEM (Data Preparation)")
    sanitization_report = SanitizationReport()
    events_sanitized = sanitize_events(events_raw, report=sanitization_report)
    print(sanitization_report.as_dataframe().to_string(index=False))

    events = clean_events(events_sanitized)
    train_events, test_events = temporal_train_test_split(events, test_frac=0.2)
    interaction_matrix = build_interaction_matrix(train_events)
    print(f"Usuários: {len(interaction_matrix.user_to_idx)} | Itens: {len(interaction_matrix.item_to_idx)}")
    print(f"Treino: {len(train_events)} eventos | Teste: {len(test_events)} eventos")

    print("=" * 60)
    print("3/6 - FEATURE ENGINEERING (RFM para clustering)")
    user_features = build_user_features(train_events)
    print(user_features.describe().round(2))

    print("=" * 60)
    print("4/6 - MODELAGEM")
    print("-> Memory-based CF (cosseno)")
    memory_cf = MemoryBasedCF(k_neighbors=20).fit(interaction_matrix.matrix)

    print("-> Model-based CF (ALS)")
    als_cf = ALSCollaborativeFiltering(factors=32, iterations=15).fit(interaction_matrix.matrix)

    print("-> K-Means (escolha de K via silhouette)")
    silhouettes = choose_k_by_silhouette(
        user_features[["recency_days", "frequency", "n_unique_items", "engagement_score", "conversion_rate"]]
        .fillna(0)
        .to_numpy()
    )
    print(f"Silhouette por K: {silhouettes}")
    best_k = max(silhouettes, key=silhouettes.get) if silhouettes else N_CLUSTERS_DEFAULT

    kmeans_model, scaler, clustered_users = fit_kmeans(user_features, n_clusters=best_k)
    print(f"K escolhido: {best_k}")
    print("Perfil dos clusters:")
    print(cluster_profile(clustered_users))

    cluster_items = top_items_per_cluster(clustered_users, train_events, n=10)

    print("=" * 60)
    print("5/6 - AVALIAÇÃO (Evaluation)")

    def als_recommend_fn(user_id):
        if user_id not in interaction_matrix.user_to_idx:
            return None
        idx = interaction_matrix.user_to_idx[user_id]
        recs = als_cf.recommend(idx, n=TOP_K_DEFAULT)
        return [interaction_matrix.idx_to_item[i] for i, _ in recs]

    als_metrics = evaluate_recommender(als_recommend_fn, test_events, k=TOP_K_DEFAULT)
    print(f"ALS CF: {als_metrics}")

    from sklearn.metrics import silhouette_score

    feature_cols = ["recency_days", "frequency", "n_unique_items", "engagement_score", "conversion_rate"]
    X_scaled_final = scaler.transform(user_features[feature_cols].fillna(0).to_numpy())
    sample_size = 10_000 if len(X_scaled_final) > 10_000 else None
    sil = silhouette_score(
        X_scaled_final, clustered_users["cluster"], sample_size=sample_size, random_state=RANDOM_STATE
    )
    print(f"K-Means Silhouette Score final: {sil:.4f}")

    print("=" * 60)
    print("6/6 - SALVANDO ARTEFATOS (Deployment prep)")
    from src.config import PROCESSED_DIR

    sanitization_report.as_dataframe().to_csv(PROCESSED_DIR / "sanitization_report.csv", index=False)
    joblib.dump(interaction_matrix, MODELS_DIR / "interaction_matrix.pkl")
    joblib.dump(als_cf, MODELS_DIR / "als_model.pkl")
    joblib.dump(memory_cf, MODELS_DIR / "memory_cf_model.pkl")
    joblib.dump(kmeans_model, MODELS_DIR / "kmeans_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(clustered_users, MODELS_DIR / "clustered_users.pkl")
    joblib.dump(cluster_items, MODELS_DIR / "cluster_top_items.pkl")
    print(f"Artefatos salvos em: {MODELS_DIR}")
    print("Pipeline concluído com sucesso.")


if __name__ == "__main__":
    run()
