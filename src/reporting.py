"""Geração dos gráficos do relatório (salvos em reports/figures)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import FIGURES_DIR  # noqa: E402

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


def _save(fig, name: str):
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=150)
    plt.close(fig)


def plot_event_distribution(event_counts: dict):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    labels = ["view", "addtocart", "transaction"]
    values = [event_counts.get(k, 0) for k in labels]
    bars = ax.bar(labels, values, color=COLORS[:3])
    ax.set_yscale("log")
    ax.set_ylabel("Eventos (escala log)")
    ax.set_title("Distribuição dos tipos de evento (funil)")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.15, f"{v:,}".replace(",", "."), ha="center", fontsize=9)
    _save(fig, "01_event_distribution.png")


def plot_sanitization(report: pd.DataFrame):
    steps = report[report["step"] != "TOTAL"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.barh(steps["step"], steps["rows_removed"], color=COLORS[3])
    ax.set_xlabel("Linhas removidas")
    ax.set_title("Impacto de cada etapa de sanitização")
    for y, v in enumerate(steps["rows_removed"]):
        ax.text(v, y, f" {v:,}".replace(",", "."), va="center", fontsize=9)
    _save(fig, "02_sanitization_impact.png")


def plot_silhouette(silhouettes: dict, best_k: int):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ks, vals = list(silhouettes.keys()), list(silhouettes.values())
    ax.plot(ks, vals, marker="o", color=COLORS[0], linewidth=2)
    ax.scatter([best_k], [silhouettes[best_k]], color=COLORS[2], s=140, zorder=5, label=f"K escolhido = {best_k}")
    ax.set_xlabel("K (número de clusters)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Escolha de K via Silhouette")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, "03_silhouette_by_k.png")


def plot_cluster_profile(profile: pd.DataFrame, labels: dict, sizes: dict):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(profile))
    width = 0.35
    ax.bar(x - width / 2, profile["frequency"], width, label="Frequência média", color=COLORS[0])
    ax.bar(x + width / 2, profile["engagement_score"], width, label="Engagement médio", color=COLORS[2])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c}: {labels[int(c)]}\n(n={sizes[int(c)]:,})".replace(",", ".") for c in profile.index], fontsize=8)
    ax.set_title("Perfil dos clusters (K-Means)")
    ax.legend()
    _save(fig, "04_cluster_profile.png")


def plot_strategy_comparison(comparison: pd.DataFrame, metric: str = "ndcg@5"):
    pivot = comparison.pivot(index="estrategia", columns="segmento", values=metric)
    order = [
        s for s in ["popularidade_global", "popularidade_por_cluster", "cf_memory_based", "cf_als", "hibrido_limiar_3", "hibrido_limiar_1"]
        if s in pivot.index
    ]
    pivot = pivot.loc[order]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(pivot))
    segs = list(pivot.columns)
    width = 0.8 / len(segs)
    for i, seg in enumerate(segs):
        ax.bar(x + i * width - 0.4 + width / 2, pivot[seg], width, label=seg.replace("_", " "), color=COLORS[i])
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in pivot.index], fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(f"Comparação de estratégias ({metric})")
    ax.legend()
    _save(fig, "05_strategy_comparison.png")
