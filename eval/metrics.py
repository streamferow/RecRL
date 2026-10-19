import math
from collections.abc import Sequence


def recall_at_k(recommendations: Sequence[Sequence[int]], targets: Sequence[int], k: int) -> float:
    hits = sum(target in prediction[:k] for prediction, target in zip(recommendations, targets))
    return hits / len(targets) if targets else 0.0


def ndcg_at_k(recommendations: Sequence[Sequence[int]], targets: Sequence[int], k: int) -> float:
    score = 0.0
    for prediction, target in zip(recommendations, targets):
        if target in prediction[:k]:
            rank = prediction.index(target) + 1
            score += 1.0 / math.log2(rank + 1)
    return score / len(targets) if targets else 0.0


def map_at_k(recommendations: Sequence[Sequence[int]], targets: Sequence[int], k: int) -> float:
    score = 0.0
    for prediction, target in zip(recommendations, targets):
        if target in prediction[:k]:
            rank = prediction.index(target) + 1
            score += 1.0 / rank
    return score / len(targets) if targets else 0.0
