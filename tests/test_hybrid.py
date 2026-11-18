import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.data.preprocessing import InteractionMatrix
from src.models.hybrid import hybrid_recommend


class FakeALS:
    def recommend(self, user_idx, n=5):
        return [(i, 1.0) for i in range(n)]


def _setup():
    # usuário 10 tem 3 itens; usuário 20 tem 1 item
    m = csr_matrix(np.array([[1, 1, 1, 0], [1, 0, 0, 0]], dtype=float))
    im = InteractionMatrix(
        matrix=m,
        user_to_idx={10: 0, 20: 1},
        item_to_idx={100: 0, 101: 1, 102: 2, 103: 3},
        idx_to_user={0: 10, 1: 20},
        idx_to_item={0: 100, 1: 101, 2: 102, 3: 103},
    )
    clustered = pd.DataFrame({"visitorid": [10, 20, 30], "cluster": [0, 1, 1]})
    top = {0: [100, 101], 1: [102, 103]}
    return im, clustered, top


def test_known_user_uses_als_by_default():
    im, clustered, top = _setup()
    items, strategy = hybrid_recommend(20, im, FakeALS(), clustered, top, im.idx_to_item, n=2)
    assert strategy == "collaborative_filtering_als"
    assert items == [100, 101]


def test_sparse_user_goes_to_fallback_with_higher_threshold():
    im, clustered, top = _setup()
    items, strategy = hybrid_recommend(20, im, FakeALS(), clustered, top, im.idx_to_item, n=2, min_history=3)
    assert strategy == "cluster_popularity_fallback"
    assert items == [102, 103]


def test_unknown_user_gets_largest_cluster_popularity():
    im, clustered, top = _setup()
    items, strategy = hybrid_recommend(999, im, FakeALS(), clustered, top, im.idx_to_item, n=2)
    assert strategy == "cluster_popularity_fallback"
    assert items == [102, 103]  # cluster 1 é o maior
