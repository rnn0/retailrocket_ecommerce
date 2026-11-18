import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from api.services import recommendation_service as svc
from src.data.preprocessing import InteractionMatrix


class FakeALS:
    def recommend(self, user_idx, n=5):
        return [(i, 1.0) for i in range(min(n, 6))]


class FakeArtifacts:
    def __init__(self):
        self.interaction_matrix = InteractionMatrix(
            matrix=csr_matrix(np.ones((1, 6))),
            user_to_idx={1: 0},
            item_to_idx={i: i for i in range(6)},
            idx_to_user={0: 1},
            idx_to_item={i: i for i in range(6)},
        )
        self.als_model = FakeALS()
        self.clustered_users = pd.DataFrame({"visitorid": [1], "cluster": [0]})
        self.cluster_top_items = {0: [0, 1, 2]}
        self.cluster_labels = {0: "Alto engajamento"}
        self.cluster_of = {1: 0}
        self.out_of_stock = {0, 1}

    def is_available(self, item_id):
        return item_id not in self.out_of_stock

    def describe_item(self, item_id, rank):
        return {"rank": rank, "item_id": str(item_id), "name": f"Produto {item_id}"}


def test_parse_user_id():
    assert svc.parse_user_id("42") == 42
    assert svc.parse_user_id("abc") == "abc"


def test_prefer_available_keeps_order_and_demotes_out_of_stock():
    art = FakeArtifacts()
    assert svc.prefer_available([0, 1, 2, 3, 4], art, 3) == [2, 3, 4]
    # sem disponíveis suficientes: completa com indisponíveis para respeitar k
    assert svc.prefer_available([0, 1, 2], art, 3) == [2, 0, 1]


def test_service_returns_segment_and_products():
    result = svc.get_recommendations_for_user("1", FakeArtifacts(), k=3)
    assert result.segment == {"cluster_id": 0, "label": "Alto engajamento"}
    assert result.item_ids == ["2", "3", "4"]
    assert result.products[0]["name"] == "Produto 2"
