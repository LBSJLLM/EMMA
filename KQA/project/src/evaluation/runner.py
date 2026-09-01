from __future__ import annotations

from typing import Dict

from src.evaluation.metrics import mcq_accuracy


def evaluate_single(prediction_obj: Dict, question_obj: Dict) -> Dict:
    return mcq_accuracy(prediction_obj.get("predicted_option", ""), question_obj.get("metadata", {}).get("gold_option"))
