from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.agents.question_type_classifier import QuestionTypeClassifier
from src.utils.llm_client import APILLMClient


class QueryAgent:
    def __init__(
        self,
        use_llm_for_question_type: bool = False,
        llm_client: APILLMClient | None = None,
        ego_mode: bool = False,
    ):
        self.llm_client = llm_client
        self.ego_mode = ego_mode
        self.classifier = QuestionTypeClassifier(
            use_llm=use_llm_for_question_type,
            llm_client=llm_client,
        )
        self.prompt_template = self._load_prompt_template()

    @staticmethod
    def _load_prompt_template() -> str:
        p = Path(__file__).resolve().parents[1] / "prompts" / "query_prompt.txt"
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
    def _preferred_target(qtype: str) -> str:
        """Return the ideally preferred target for a question type, ignoring availability."""
        if qtype in ("counting", "causal_why", "temporal_order", "identity_or_role"):
            return "MTM"
        if qtype in ("descriptive_attribute", "spatial_relation"):
            return "STM"
        return "MTM"

    def _strategy_and_targets(self, qtype: str, enabled_levels: List[str], include_raw_video: bool) -> Dict:
        preferred = self._preferred_target(qtype)
        all_available = list(enabled_levels) + (["RAW_VIDEO"] if include_raw_video else [])
        if preferred in all_available:
            return {"query_strategy": "single", "retrieval_target": preferred}
        return {"query_strategy": "single", "retrieval_target": all_available[0] if all_available else "MTM"}

    @staticmethod
    def _build_sources_section(enabled_memory_levels: List[str], include_raw_video: bool) -> tuple[str, str, str]:
        """Build (sources_section, source_names, target_options) for the prompt."""
        descriptions = {
            "MTM": (
                "MTM (mid-term memory):\n"
                "- event-level memory\n"
                "- best for locating relevant events, major happenings, and coarse temporal structure"
            ),
            "STM": (
                "STM (short-term memory):\n"
                "- clip-level fine-grained memory\n"
                "- best for local actions, attributes, short-range order, exact visual details, and short utterances"
            ),
            "RAW_VIDEO": (
                "RAW_VIDEO:\n"
                "- use only when memory evidence is still insufficient and there is already a likely relevant time region\n"
                "- ONLY useful when the missing information is purely visual: color, position, count, action details, object appearance, written text on screen, etc.\n"
                "- Do NOT use RAW_VIDEO for: identifying a person by name (VLM cannot match faces to names without prior context), determining specific timestamps or dates, or questions about habits/relationships/recurring patterns\n"
                "- main_query for RAW_VIDEO is NOT a retrieval keyword — it is a natural-language instruction to the VLM observer; write it differently from MTM/STM queries\n"
                "- include any prerequisite facts already established by memory retrieval that the VLM needs to locate the correct target\n"
                "- then specify the precise visual detail still needed to answer the question\n"
                "- Never inspect the full video by default. If RAW_VIDEO is chosen, provide a short time_filter (end - start ≤ 120s)."
            ),
        }
        ordered_keys = [k for k in ("MTM", "STM") if k in enabled_memory_levels]
        if include_raw_video:
            ordered_keys.append("RAW_VIDEO")

        lines = []
        for i, key in enumerate(ordered_keys, 1):
            lines.append(f"{i}. {descriptions[key]}")
        sources_section = "\n\n".join(lines)
        source_names = ", ".join(ordered_keys)
        target_options = " | ".join(f'"{k}"' for k in ordered_keys)
        return sources_section, source_names, target_options

    @staticmethod
    def _time_filter_format(ego_mode: bool) -> str:
        if ego_mode:
            return 'null or ["DAY1 17:30:00", "DAY1 17:45:00"]  — use ego timestamp format (DAY{n} HH:MM:SS)'
        return "null or [start_seconds, end_seconds]  — use float seconds"

    def _plan_with_llm(
        self,
        question: str,
        options: Dict[str, str],
        evidence_summary: List[Dict],
        missing_information: List[str],
        previous_search_history: List[Dict],
        enabled_memory_levels: List[str],
        include_raw_video: bool,
        round_notice: str = "",
    ) -> Dict | None:
        if not self.llm_client or not self.llm_client.available() or not self.prompt_template:
            return None
        sources_section, source_names, target_options = self._build_sources_section(
            enabled_memory_levels, include_raw_video
        )
        prompt = self._render_prompt(
            self.prompt_template,
            {
                "question": question,
                "options": json.dumps(options, ensure_ascii=False),
                "evidence_summary": json.dumps(evidence_summary, ensure_ascii=False),
                "missing_information": json.dumps(missing_information, ensure_ascii=False),
                "previous_search_history": json.dumps(previous_search_history, ensure_ascii=False),
                "available_sources_section": sources_section,
                "available_source_names": source_names,
                "available_target_options": target_options,
                "time_filter_format": self._time_filter_format(self.ego_mode),
            },
        )
        if round_notice:
            prompt = f"{prompt}\n\n[Round Notice]\n{round_notice}"
        text = self.llm_client.chat("You are Query Agent. Return JSON only.", prompt, max_tokens=900)
        obj = self.llm_client.extract_json_object(text)
        return obj if isinstance(obj, dict) else None

    @staticmethod
    def _normalize_time_filter(value) -> List[float] | None:
        if not isinstance(value, list) or len(value) != 2:
            return None
        from src.utils.ego_time import ego_timestamp_to_seconds
        parsed = []
        for v in value:
            if isinstance(v, (int, float)):
                parsed.append(float(v))
            elif isinstance(v, str):
                sec = ego_timestamp_to_seconds(v)
                if sec is not None:
                    parsed.append(sec)
                else:
                    try:
                        parsed.append(float(v))
                    except Exception:
                        return None
            else:
                return None
        a, b = parsed
        if b < a:
            b = a
        return [a, b]

    def plan(
        self,
        question: str,
        options: Dict[str, str],
        evidence_summary: List[Dict],
        missing_information: List[str],
        enabled_memory_levels: List[str],
        previous_search_history: List[Dict] | None = None,
        round_notice: str = "",
        use_video_fallback: bool = True,
    ) -> Dict:
        prev_history = previous_search_history or []
        include_raw_video = bool(use_video_fallback)
        question_type = self.classifier.classify(question, options)
        llm_plan = self._plan_with_llm(
            question=question,
            options=options,
            evidence_summary=evidence_summary,
            missing_information=missing_information,
            previous_search_history=prev_history,
            enabled_memory_levels=enabled_memory_levels,
            include_raw_video=include_raw_video,
            round_notice=round_notice,
        )

        target = ""
        main_query = ""
        time_filter = None
        reason = ""

        if isinstance(llm_plan, dict):
            target = str(llm_plan.get("target_source") or "").upper().strip()
            main_query = str(llm_plan.get("main_query") or "").strip()
            time_filter = self._normalize_time_filter(llm_plan.get("time_filter"))
            reason = str(llm_plan.get("reason") or "").strip()
            llm_qtype = str(llm_plan.get("question_type") or "").strip()
            if llm_qtype:
                question_type = llm_qtype

        all_available = list(enabled_memory_levels) + (["RAW_VIDEO"] if include_raw_video else [])
        if not target or target not in all_available:
            target = self._strategy_and_targets(question_type, enabled_memory_levels, include_raw_video).get("retrieval_target", all_available[0] if all_available else "MTM")

        if not main_query:
            if missing_information:
                main_query = f"{question} {' '.join(missing_information)}"
            else:
                main_query = question

        return {
            "question_type": question_type,
            "query_strategy": "single",
            "retrieval_targets": [target],
            "queries": [
                {
                    "query_id": "query_1",
                    "query_text": main_query,
                    "target_levels": [target],
                    "time_filter": time_filter,
                    "purpose": reason or "Single best next retrieval step.",
                }
            ],
            "notes": f"evidence_pool_size={len(evidence_summary)} target={target}",
        }
