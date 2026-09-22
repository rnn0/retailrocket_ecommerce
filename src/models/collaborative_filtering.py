"""
Fase CRISP-DM: Modeling.
Duas estratégias de Filtragem Colaborativa:
  1. Memory-based (similaridade de cosseno usuário-usuário)
  2. Model-based (Matrix Factorization via ALS, biblioteca `implicit`)
"""
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from src.config import RANDOM_STATE


class MemoryBasedCF:
    """CF user-based: recomenda itens que usuários semelhantes consumiram."""

    def __init__(self, k_neighbors: int = 20):
        self.k_neighbors = k_neighbors
        self.matrix: csr_matrix | None = None
        self.user_similarity: np.ndarray | None = None

    def fit(self, matrix: csr_matrix):
        self.matrix = matrix
        self.user_similarity = cosine_similarity(matrix)
        return self

    def recommend(self, user_idx: int, n: int = 5) -> list[tuple[int, float]]:
        sims = self.user_similarity[user_idx]
        neighbor_idxs = np.argsort(-sims)[1 : self.k_neighbors + 1]  # exclui o próprio usuário
        neighbor_weights = sims[neighbor_idxs]

        # Score do item = soma ponderada das interações dos vizinhos
        neighbor_interactions = self.matrix[neighbor_idxs].toarray()
        scores = neighbor_weights @ neighbor_interactions

        already_seen = self.matrix[user_idx].toarray().flatten() > 0
        scores[already_seen] = -np.inf

        top_items = np.argsort(-scores)[:n]
        return [(int(i), float(scores[i])) for i in top_items if scores[i] > -np.inf]


class ALSCollaborativeFiltering:
    """CF model-based via Alternating Least Squares (dados implícitos)."""

    def __init__(self, factors: int = 32, regularization: float = 0.05, iterations: int = 15):
        from implicit.als import AlternatingLeastSquares

        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=RANDOM_STATE,
        )
        self._fitted_matrix: csr_matrix | None = None

    def fit(self, matrix: csr_matrix):
        # implicit espera matriz (usuários x itens) de confiança/peso
        self._fitted_matrix = matrix.tocsr()
        self.model.fit(self._fitted_matrix)
        return self

    def recommend(self, user_idx: int, n: int = 5) -> list[tuple[int, float]]:
        item_idxs, scores = self.model.recommend(
            user_idx, self._fitted_matrix[user_idx], N=n, filter_already_liked_items=True
        )
        return list(zip(item_idxs.tolist(), scores.tolist()))
