from src.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k


def test_precision_at_k_perfect_match():
    recommended = ["a", "b", "c"]
    relevant = {"a", "b", "c"}
    assert precision_at_k(recommended, relevant, k=3) == 1.0


def test_precision_at_k_no_match():
    assert precision_at_k(["a", "b"], {"x", "y"}, k=2) == 0.0


def test_recall_at_k_partial_match():
    recommended = ["a", "x", "y"]
    relevant = {"a", "b"}
    assert recall_at_k(recommended, relevant, k=3) == 0.5


def test_ndcg_at_k_order_matters():
    relevant = {"a"}
    ndcg_first = ndcg_at_k(["a", "b", "c"], relevant, k=3)
    ndcg_last = ndcg_at_k(["b", "c", "a"], relevant, k=3)
    assert ndcg_first > ndcg_last


def test_ndcg_at_k_no_relevant_found():
    assert ndcg_at_k(["x", "y"], {"a"}, k=2) == 0.0
