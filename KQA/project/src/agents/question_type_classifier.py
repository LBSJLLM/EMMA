from __future__ import annotations

from typing import Dict

from src.utils.parser import classify_question_type
from src.utils.llm_client import APILLMClient


class QuestionTypeClassifier:
    def __init__(self, use_llm: bool = False, llm_client: APILLMClient | None = None):
        self.use_llm = use_llm
        self.llm_client = llm_client

    def _llm_classify(self, question: str, options: Dict[str, str]) -> str:
        if not self.llm_client or not self.llm_client.available():
            return ""
        system = "You are a classifier. Return JSON only."
        user = (
            "Classify question into one of: counting, causal_why, descriptive_attribute, "
            "temporal_order, identity_or_role, long_term_habit_or_fact, spatial_relation, open_other.\n"
            f"Question: {question}\nOptions: {options}\n"
            "Output: {\"question_type\": \"...\"}"
        )
        text = self.llm_client.chat(system, user, max_tokens=128)
        obj = self.llm_client.extract_json_object(text)
        if isinstance(obj, dict):
            qtype = str(obj.get("question_type") or "").strip()
            if qtype:
                return qtype
        return ""

    def classify(self, question: str, options: Dict[str, str]) -> str:
        if self.use_llm:
            qtype = self._llm_classify(question, options)
            if qtype:
                return qtype
        return classify_question_type(question)
