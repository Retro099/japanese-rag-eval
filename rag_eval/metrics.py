"""Retrieval metrics. A retrieved chunk is relevant if it comes from the gold (doc, page)."""

from typing import Dict, List


def is_relevant(chunk: Dict, gold: Dict) -> bool:
    return chunk["doc"] == gold["doc"] and chunk["page"] == gold["page"]


def first_hit_rank(retrieved: List[Dict], gold: Dict) -> int:
    """1-based rank of the first relevant chunk, 0 if none."""
    for r, c in enumerate(retrieved, 1):
        if is_relevant(c, gold):
            return r
    return 0


def evidence_rank(retrieved: List[Dict], gold: Dict) -> int:
    """Stricter: 1-based rank of the first chunk that contains the evidence string."""
    for r, c in enumerate(retrieved, 1):
        if gold["evidence"] in c["text"]:
            return r
    return 0


def summarize(ranks: List[int], ks=(1, 3, 5)) -> Dict[str, float]:
    n = len(ranks) or 1
    out = {f"hit@{k}": sum(1 for r in ranks if 0 < r <= k) / n for k in ks}
    out["mrr@5"] = sum(1 / r for r in ranks if 0 < r <= 5) / n
    return {k: round(v, 4) for k, v in out.items()}
