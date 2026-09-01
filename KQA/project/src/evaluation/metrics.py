from __future__ import annotations

from typing import Dict, Optional


def mcq_accuracy(predicted: str, gold: Optional[str]) -> Dict:
    if not gold:
        return {"has_gold": False, "accuracy": None}
    return {"has_gold": True, "accuracy": 1.0 if predicted == gold else 0.0}
