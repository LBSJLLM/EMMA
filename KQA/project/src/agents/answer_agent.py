from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

from src.utils.llm_client import APILLMClient


class AnswerAgent:
    def __init__(self, use_llm_for_answer: bool = True, llm_client: APILLMClient | None = None):
        self.use_llm_for_answer = use_llm_for_answer
        self.llm_client = llm_client
        self.prompt_template = self._load_prompt_template()

    @staticmethod
    def _load_prompt_template() -> str:
        p = Path(__file__).resolve().parents[1] / "prompts" / "answer_prompt.txt"
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return ""

    @staticmethod
    def _render_prompt(template: str, mapping: Dict[str, Any]) -> str:
        text = template
        for key, value in mapping.items():
            text = text.replace("{" + key + "}", str(value))
        return text

    def _answer_with_llm(self, question: str, options: Dict[str, str], evidence_pool: List[Dict]) -> Dict | None:
        if not self.llm_client or not self.llm_client.available():
            return None
        if self.prompt_template:
            user = self._render_prompt(
                self.prompt_template,
                {
                    "question": question,
                    "options": json.dumps(options, ensure_ascii=False),
                    "evidence_pool": json.dumps(evidence_pool, ensure_ascii=False),
                },
            )
        else:
            user = (
                "Use only evidence to answer MCQ. Output JSON keys: predicted_option, confidence, reasoning, key_evidence_ids. key_evidence_ids must contain memory_id values.\n"
                f"Question: {question}\nOptions: {options}\nEvidence: {evidence_pool}"
            )
        text = self.llm_client.chat("You are the Answer Agent. Return JSON only.", user, max_tokens=900)
        obj = self.llm_client.extract_json_object(text)
        if isinstance(obj, dict):
            obj.pop("answerable", None)
            return obj
        return None

    def answer(self, question: str, options: Dict[str, str], evidence_pool: List[Dict]) -> Dict:
        if self.use_llm_for_answer:
            out = self._answer_with_llm(question, options, evidence_pool)
            if isinstance(out, dict):
                return out

        option_scores = {k: 0.0 for k in options.keys()}
        for e in evidence_pool:
            text = str(e.get("text", "")).lower()
            rel = 1.0 if e.get("relevance") == "high" else 0.5
            role_weight = 1.0 if e.get("role") == "direct_evidence" else 0.6
            for k, v in options.items():
                token = str(v).strip().lower()
                if token and token in text:
                    option_scores[k] += 2.0 * rel * role_weight
                else:
                    words = re.findall(r"[\w\u4e00-\u9fff]+", token)
                    if words and any(w in text for w in words[:2]):
                        option_scores[k] += 0.7 * rel * role_weight

        predicted = max(option_scores, key=option_scores.get) if option_scores else ""
        best_score = option_scores.get(predicted, 0.0)
        answerable = best_score > 0.5
        confidence = "high" if best_score >= 2.0 else "medium" if best_score >= 1.0 else "low"

        key_ids = [e.get("memory_id", "") for e in evidence_pool if e.get("role") == "direct_evidence"]
        reasoning = (
            f"Selected {predicted} based on strongest direct evidence overlap. "
            f"Top option score={best_score:.2f}."
        )
        return {
            "predicted_option": predicted,
            "confidence": confidence,
            "reasoning": reasoning,
            "key_evidence_ids": key_ids[:5],
        }
