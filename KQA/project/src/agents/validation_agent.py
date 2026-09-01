from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from src.utils.dedup import jaccard_text
from src.utils.llm_client import APILLMClient


_ALLOWED_ROLES = {"direct_evidence", "supporting_context", "conflict_evidence"}


class ValidationAgent:
    def __init__(
        self,
        use_llm_for_validation: bool = False,
        llm_client: APILLMClient | None = None,
        disable_evidence_removal: bool = False,
    ):
        self.use_llm_for_validation = use_llm_for_validation
        self.llm_client = llm_client
        self.disable_evidence_removal = disable_evidence_removal
        self.prompt_template = self._load_prompt_template()

    @staticmethod
    def _load_prompt_template() -> str:
        p = Path(__file__).resolve().parents[1] / "prompts" / "validation_prompt.txt"
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
    def _normalize_ranked_item(item: Dict) -> Dict | None:
        if not isinstance(item, dict):
            return None
        mid = str(item.get("memory_id") or "").strip()
        lvl = str(item.get("memory_level") or "").upper().strip()
        role = str(item.get("role") or "supporting_context").strip()
        if not mid or lvl not in {"STM", "MTM", "LTM", "RAW_VIDEO"}:
            return None
        if role not in _ALLOWED_ROLES:
            role = "supporting_context"
        return {
            "memory_id": mid,
            "memory_level": lvl,
            "role": role,
        }

    @staticmethod
    def _format_search_history(search_history: List[Dict]) -> str:
        """Render prior retrievals compactly so validation can detect loops."""
        if not search_history:
            return "None (this is the first retrieval round.)"
        lines = []
        for idx, plan in enumerate(search_history):
            targets = plan.get("retrieval_targets") or plan.get("target_source") or []
            target = str(targets[0] if isinstance(targets, list) and targets else targets or "?")
            queries = plan.get("queries") or []
            query = queries[0] if queries else {}
            query_text = str(query.get("query_text") or query.get("main_query") or plan.get("main_query") or "")
            time_filter = query.get("time_filter") or query.get("time_range") or plan.get("time_filter")
            time_note = f" [time: {time_filter}]" if isinstance(time_filter, list) and len(time_filter) == 2 else ""
            lines.append(f"Round {idx}: {target} query {query_text!r}{time_note}")
        return "\n".join(lines)

    def _validate_with_llm(
        self,
        question: str,
        options: Dict[str, str],
        existing_evidence_pool: List[Dict],
        retrieved_candidates: List[Dict],
        round_notice: str = "",
        search_history: List[Dict] | None = None,
    ) -> Dict | None:
        if not self.llm_client or not self.llm_client.available():
            return None
        search_history_text = self._format_search_history(search_history or [])
        if self.prompt_template:
            user = self._render_prompt(
                self.prompt_template,
                {
                    "question": question,
                    "options": json.dumps(options, ensure_ascii=False),
                    "existing_evidence_pool": json.dumps(existing_evidence_pool, ensure_ascii=False),
                    "retrieved_candidates": json.dumps(retrieved_candidates, ensure_ascii=False),
                    "search_history": search_history_text,
                },
            )
        else:
            user = (
                "Return JSON only with keys ranked_evidence, removed_evidence, is_enough, missing_information, reason.\n"
                f"Question: {question}\nOptions: {options}\n"
                f"Existing evidence: {existing_evidence_pool}\n"
                f"Retrieved candidates: {retrieved_candidates}"
            )

        if round_notice:
            user = f"{user}\n\n[Round Notice]\n{round_notice}"
        text = self.llm_client.chat("You are the Validation Agent. Return JSON only.", user, max_tokens=1400)
        obj = self.llm_client.extract_json_object(text)
        if not isinstance(obj, dict):
            return None
        ranked_raw = obj.get("ranked_evidence") if isinstance(obj.get("ranked_evidence"), list) else []
        ranked = []
        for x in ranked_raw:
            nx = self._normalize_ranked_item(x)
            if nx is not None:
                ranked.append(nx)

        removed_raw = obj.get("removed_evidence") if isinstance(obj.get("removed_evidence"), list) else []
        removed = []
        for r in removed_raw:
            if not isinstance(r, dict):
                continue
            mid = str(r.get("memory_id") or "").strip()
            if not mid:
                continue
            removed.append({"memory_id": mid, "reason": str(r.get("reason") or "")})

        is_enough = bool(obj.get("is_enough", False))
        missing = obj.get("missing_information") if isinstance(obj.get("missing_information"), list) else []
        reason = str(obj.get("reason") or "")

        if self.disable_evidence_removal:
            # Move all removed items back into ranked as supporting_context
            ranked_ids = {x.get("memory_id") for x in ranked}
            for r in removed:
                mid = str(r.get("memory_id") or "").strip()
                if mid and mid not in ranked_ids:
                    ranked.append({
                        "memory_id": mid,
                        "memory_level": "",
                        "role": "supporting_context",
                    })
                    ranked_ids.add(mid)
            removed = []

        return {
            "ranked_evidence": ranked,
            "removed_evidence": removed,
            "is_enough": is_enough,
            "missing_information": [str(x) for x in missing if str(x).strip()],
            "reason": reason,
        }

    def _is_useful(self, question: str, candidate_text: str) -> bool:
        return jaccard_text(question, candidate_text) >= 0.05 or any(
            token in candidate_text.lower() for token in ["because", "then", "first", "after", "left", "right"]
        )

    def _assign_role(self, text: str) -> str:
        low = text.lower()
        if any(k in low for k in ["however", "but", "conflict", "not", "instead"]):
            return "conflict_evidence"
        if any(k in low for k in ["because", "so", "therefore", "随后", "于是"]):
            return "direct_evidence"
        return "supporting_context"

    def validate(
        self,
        question: str,
        options: Dict[str, str],
        existing_evidence_pool: List[Dict],
        retrieved_candidates: List[Dict],
        round_notice: str = "",
        search_history: List[Dict] | None = None,
    ) -> Dict:
        if self.use_llm_for_validation:
            out = self._validate_with_llm(
                question,
                options,
                existing_evidence_pool,
                retrieved_candidates,
                round_notice=round_notice,
                search_history=search_history,
            )
            if isinstance(out, dict):
                return out

        useful = []
        removed = []
        for cand in retrieved_candidates:
            text = str(cand.get("text", ""))
            if not self._is_useful(question, text):
                if self.disable_evidence_removal:
                    pass  # fall through to add as supporting_context below
                else:
                    removed.append({"memory_id": cand.get("memory_id", ""), "reason": "irrelevant_to_question"})
                    continue

            role = self._assign_role(text)
            if role not in _ALLOWED_ROLES:
                role = "supporting_context"
            useful.append(
                {
                    "memory_id": cand.get("memory_id", ""),
                    "memory_level": str(cand.get("memory_level", "")).upper(),
                    "role": role,
                }
            )

        joined = "\n".join(str(x.get("text") or "") for x in retrieved_candidates)
        opt_hits = 0
        for value in options.values():
            if value and re.search(re.escape(value[:15]), joined, flags=re.IGNORECASE):
                opt_hits += 1

        has_direct = any(x.get("role") == "direct_evidence" for x in useful + existing_evidence_pool)

        is_enough = has_direct and opt_hits >= 1
        missing = []
        if not has_direct:
            missing.append("Need direct causal/temporal detail tied to candidate options.")
        if opt_hits == 0:
            missing.append("Need evidence that explicitly supports or rejects at least one option.")

        return {
            "ranked_evidence": useful,
            "removed_evidence": removed,
            "is_enough": is_enough,
            "reason": "Evidence sufficient." if is_enough else "Evidence still insufficient.",
            "missing_information": missing,
        }
