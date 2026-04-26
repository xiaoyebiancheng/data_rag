from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence


def _normalize_ids(values: Sequence[object]) -> List[str]:
    return [str(value) for value in values if value is not None and str(value) != ""]


def hit_at_k(retrieved_ids: Sequence[object], golden_ids: Sequence[object], top_k: int) -> Optional[float]:
    normalized_golden = set(_normalize_ids(golden_ids))
    if not normalized_golden:
        return None
    normalized_retrieved = _normalize_ids(retrieved_ids)[:top_k]
    return 1.0 if any(chunk_id in normalized_golden for chunk_id in normalized_retrieved) else 0.0


def recall_at_k(retrieved_ids: Sequence[object], golden_ids: Sequence[object], top_k: int) -> Optional[float]:
    normalized_golden = set(_normalize_ids(golden_ids))
    if not normalized_golden:
        return None
    normalized_retrieved = set(_normalize_ids(retrieved_ids)[:top_k])
    hit_count = len(normalized_golden & normalized_retrieved)
    return hit_count / len(normalized_golden)


def mean_reciprocal_rank(retrieved_ids: Sequence[object], golden_ids: Sequence[object]) -> Optional[float]:
    normalized_golden = set(_normalize_ids(golden_ids))
    if not normalized_golden:
        return None
    for rank, chunk_id in enumerate(_normalize_ids(retrieved_ids), start=1):
        if chunk_id in normalized_golden:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[object], golden_ids: Sequence[object], top_k: int) -> Optional[float]:
    normalized_golden = set(_normalize_ids(golden_ids))
    if not normalized_golden:
        return None

    dcg = 0.0
    for rank, chunk_id in enumerate(_normalize_ids(retrieved_ids)[:top_k], start=1):
        rel = 1.0 if chunk_id in normalized_golden else 0.0
        if rel > 0:
            dcg += rel / math.log2(rank + 1)

    ideal_hits = min(len(normalized_golden), top_k)
    if ideal_hits == 0:
        return None

    idcg = 0.0
    for rank in range(1, ideal_hits + 1):
        idcg += 1.0 / math.log2(rank + 1)
    if idcg == 0:
        return None
    return dcg / idcg


def average(values: Iterable[Optional[float]]) -> Optional[float]:
    valid_values = [value for value in values if value is not None]
    if not valid_values:
        return None
    return sum(valid_values) / len(valid_values)


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    if q <= 0:
        return min(values)
    if q >= 100:
        return max(values)

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * (q / 100.0)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return ordered[lower_index]

    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    weight = position - lower_index
    return lower_value + (upper_value - lower_value) * weight
