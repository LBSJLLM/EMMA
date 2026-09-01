from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.utils.llm_client import APILLMClient


_ROLES = {"direct_evidence", "supporting_context", "conflict_evidence"}
_LEVELS = {"STM", "MTM", "LTM", "RAW_VIDEO"}


class SingleAgent:
    def __init__(self, llm_client: APILLMClient | None = None):
        self.llm_client = llm_client
        self.prompt_template = self._load_prompt_template()

    @staticmethod
    def _load_prompt_template() -> str:
        p = Path(__file__).resolve().parents[1] / "prompts" / "single_agent_prompt.txt"
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

    @staticmethod
    def _normalize_time_filter(value: Any) -> List[float] | None:
        if not isinstance(value, list) or len(value) != 2:
            return None
        parsed = []
        for v in value:
            try:
                parsed.append(float(v))
            except Exception:
                return None
        a, b = parsed
        if b < a:
            b = a
        return [a, b]

    @staticmethod
    def _normalize_ranked(items: Any) -> List[Dict]:
        if not isinstance(items, list):
            return []
        out = []
        seen = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            mid = str(item.get("memory_id") or "").strip()
            lvl = str(item.get("memory_level") or "").upper().strip()
            if not mid or lvl not in _LEVELS:
                continue
            key = (mid, lvl)
            if key in seen:
                continue
            seen.add(key)
            role = str(item.get("role") or "supporting_context").strip()
            if role not in _ROLES:
                role = "supporting_context"
            out.append({"memory_id": mid, "memory_level": lvl, "role": role})
        return out

    @staticmethod
    def _normalize_answer(value: Any, options: Dict[str, str]) -> Dict | None:
        if not isinstance(value, dict):
            return None
        pred = str(value.get("predicted_option") or "").strip().upper()
        if pred not in {str(k).upper() for k in options.keys()}:
            return None
        conf = str(value.get("confidence") or "low").strip().lower()
        if conf not in {"high", "medium", "low"}:
            conf = "low"
        key_ids = value.get("key_evidence_ids") if isinstance(value.get("key_evidence_ids"), list) else []
        return {
            "predicted_option": pred,
            "confidence": conf,
            "reasoning": str(value.get("reasoning") or ""),
            "key_evidence_ids": [str(x) for x in key_ids if str(x).strip()][:8],
        }

    def step(
        self,
        *,
        question: str,
        options: Dict[str, str],
        round_idx: int,
        max_rounds: int,
        enabled_memory_levels: List[str],
        use_video_fallback: bool,
        existing_evidence_pool: List[Dict],
        retrieved_candidates: List[Dict],
        previous_search_history: List[Dict],
    ) -> Dict:
        if not self.llm_client or not self.llm_client.available() or not self.prompt_template:
            return self._fallback_step(question, options, round_idx, max_rounds, enabled_memory_levels)

        prompt = self._render_prompt(
            self.prompt_template,
            {
                "question": question,
                "options": json.dumps(options, ensure_ascii=False),
                "round_idx": round_idx,
                "max_rounds": max_rounds,
                "is_final_round": str(round_idx >= max_rounds - 1).lower(),
                "enabled_memory_levels": json.dumps(enabled_memory_levels, ensure_ascii=False),
                "use_video_fallback": str(bool(use_video_fallback)).lower(),
                "existing_evidence_pool": json.dumps(existing_evidence_pool, ensure_ascii=False),
                "retrieved_candidates": json.dumps(retrieved_candidates, ensure_ascii=False),
                "previous_search_history": json.dumps(previous_search_history, ensure_ascii=False),
            },
        )
        text = self.llm_client.chat("You are a single-agent multi-round VideoQA controller. Return JSON only.", prompt, max_tokens=1800)
        obj = self.llm_client.extract_json_object(text)
        if not isinstance(obj, dict):
            return self._fallback_step(question, options, round_idx, max_rounds, enabled_memory_levels)
        return self._normalize_output(obj, question, options, round_idx, max_rounds, enabled_memory_levels, use_video_fallback)

    def _normalize_output(
        self,
        obj: Dict,
        question: str,
        options: Dict[str, str],
        round_idx: int,
        max_rounds: int,
        enabled_memory_levels: List[str],
        use_video_fallback: bool,
    ) -> Dict:
        ranked = self._normalize_ranked(obj.get("ranked_evidence"))
        missing = obj.get("missing_information") if isinstance(obj.get("missing_information"), list) else []
        answer = self._normalize_answer(obj.get("final_answer"), options)

        action = obj.get("next_action") if isinstance(obj.get("next_action"), dict) else {}
        action_type = str(action.get("type") or "").lower().strip()
        is_enough = bool(obj.get("is_enough", False))
        if is_enough or action_type == "answer":
            action_type = "answer"
        else:
            action_type = "retrieve"

        enabled = [str(x).upper() for x in enabled_memory_levels]
        available = list(enabled) + (["RAW_VIDEO"] if use_video_fallback else [])
        target = str(action.get("target_source") or "").upper().strip()
        if target not in available:
            target = "MTM" if "MTM" in available else (available[0] if available else "MTM")

        main_query = str(action.get("main_query") or "").strip() or question
        time_filter = self._normalize_time_filter(action.get("time_filter"))
        if target == "RAW_VIDEO" and time_filter is not None:
            if float(time_filter[1]) - float(time_filter[0]) > 120.0:
                target = "STM" if "STM" in available else "MTM"
                time_filter = None
        if target != "RAW_VIDEO" and time_filter is not None and len(time_filter) != 2:
            time_filter = None

        if action_type == "answer" and answer is None:
            # Force a valid answer on malformed final output.
            pred = next(iter(options.keys()), "A")
            answer = {
                "predicted_option": str(pred).upper(),
                "confidence": "low",
                "reasoning": "Malformed final answer; selected a fallback option.",
                "key_evidence_ids": [x.get("memory_id", "") for x in ranked[:3]],
            }

        return {
            "ranked_evidence": ranked,
            "removed_evidence": [],
            "is_enough": bool(action_type == "answer" and answer is not None),
            "missing_information": [str(x) for x in missing if str(x).strip()],
            "next_action": {
                "type": action_type,
                "target_source": target,
                "main_query": main_query,
                "time_filter": time_filter,
                "reason": str(action.get("reason") or ""),
            },
            "final_answer": answer,
        }

    @staticmethod
    def _fallback_step(
        question: str,
        options: Dict[str, str],
        round_idx: int,
        max_rounds: int,
        enabled_memory_levels: List[str],
    ) -> Dict:
        if round_idx >= max_rounds - 1:
            pred = next(iter(options.keys()), "A")
            return {
                "ranked_evidence": [],
                "removed_evidence": [],
                "is_enough": True,
                "missing_information": [],
                "next_action": {"type": "answer", "target_source": "MTM", "main_query": question, "time_filter": None, "reason": "Fallback final round."},
                "final_answer": {"predicted_option": str(pred).upper(), "confidence": "low", "reasoning": "Fallback answer.", "key_evidence_ids": []},
            }
        target = "MTM" if "MTM" in [str(x).upper() for x in enabled_memory_levels] else str(enabled_memory_levels[0]).upper()
        return {
            "ranked_evidence": [],
            "removed_evidence": [],
            "is_enough": False,
            "missing_information": ["Need relevant evidence."],
            "next_action": {"type": "retrieve", "target_source": target, "main_query": question, "time_filter": None, "reason": "Fallback retrieval."},
            "final_answer": None,
        }
