from __future__ import annotations

from typing import Dict, List


RELEVANCE_WEIGHT = {
    "high": 1.0,
    "medium": 0.7,
}


def rerank_evidence(evidence_pool: List[Dict]) -> List[Dict]:
    def _score(item: Dict) -> float:
        rel = RELEVANCE_WEIGHT.get(item.get("relevance", "medium"), 0.5)
        role_bonus = 0.15 if item.get("role") == "direct_evidence" else 0.0
        return rel + role_bonus

    return sorted(evidence_pool, key=_score, reverse=True)
