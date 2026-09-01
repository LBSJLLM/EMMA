from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

from src.agents.answer_agent import AnswerAgent
from src.agents.query_agent import QueryAgent
from src.agents.search_agent import SearchAgent
from src.agents.validation_agent import ValidationAgent
from src.agents.video_fallback_agent import VideoFallbackAgent, QwenVLVideoReader
from src.memory.retriever import ResultsMemoryRetriever
from src.pipeline.state import init_state, state_to_dict
from src.utils.llm_client import APILLMClient
from src.utils.dedup import merge_evidence_with_dedup
from src.utils.ego_time import ego_span
from src.utils.timers import timed


class QAController:
    @staticmethod
    def _source_file_for_level(level: str) -> str:
        lvl = str(level or "").upper()
        if lvl == "STM":
            return "outputs/short_term.json"
        if lvl == "MTM":
            return "outputs/medium_term.json"
        if lvl == "LTM":
            return "outputs/long_term.json"
        return ""

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.ego_mode = bool(cfg.get("ego_mode", False))
        _llm_temp = cfg.get("llm_temperature")
        self.llm_client = APILLMClient(
            model=cfg.get("llm_model", "deepseek-chat"),
            api_key=cfg.get("openai_api_key"),
            base_url=cfg.get("openai_base_url"),
            temperature=float(_llm_temp) if _llm_temp is not None else 0.0,
            rate_limit_qps=float(cfg.get("llm_rate_limit_qps", 0.0) or 0.0),
            max_concurrency=int(cfg.get("llm_max_concurrency", 0) or 0),
            max_retries=int(cfg.get("llm_max_retries", 2) or 2),
            retry_backoff_seconds=float(cfg.get("llm_retry_backoff_seconds", 1.0) or 1.0),
            request_timeout_seconds=float(cfg.get("llm_request_timeout_seconds", 0.0) or 0.0),
        )
        _answer_model = cfg.get("answer_llm_model") or cfg.get("llm_model", "deepseek-chat")
        if _answer_model != cfg.get("llm_model", "deepseek-chat"):
            self.answer_llm_client = APILLMClient(
                model=_answer_model,
                api_key=cfg.get("openai_api_key"),
                base_url=cfg.get("openai_base_url"),
                temperature=float(_llm_temp) if _llm_temp is not None else 0.0,
                rate_limit_qps=float(cfg.get("llm_rate_limit_qps", 0.0) or 0.0),
                max_concurrency=int(cfg.get("llm_max_concurrency", 0) or 0),
                max_retries=int(cfg.get("llm_max_retries", 2) or 2),
                retry_backoff_seconds=float(cfg.get("llm_retry_backoff_seconds", 1.0) or 1.0),
                request_timeout_seconds=float(cfg.get("llm_request_timeout_seconds", 0.0) or 0.0),
            )
        else:
            self.answer_llm_client = self.llm_client
        self.query_agent = QueryAgent(
            use_llm_for_question_type=cfg.get("use_llm_for_question_type", False),
            llm_client=self.llm_client,
            ego_mode=self.ego_mode,
        )
        self.retriever = ResultsMemoryRetriever(
            results_root=cfg["results_root"],
            embed_model=cfg.get("embed_model", "text-embedding-3-large"),
            api_key=cfg.get("embed_api_key") or cfg.get("openai_api_key"),
            base_url=cfg.get("embed_base_url") or cfg.get("openai_base_url"),
            scoring_mix=(0.7, 0.3),
        )
        self.search_agent = SearchAgent(
            retriever=self.retriever,
            top_k_per_query=cfg.get("top_k_per_query", 5),
            top_k_per_level=cfg.get("top_k_per_level", 5),
        )
        self.validation_agent = ValidationAgent(
            use_llm_for_validation=cfg.get("use_llm_for_validation", False),
            llm_client=self.llm_client,
            disable_evidence_removal=bool(cfg.get("disable_evidence_removal", False)),
        )
        self.video_fallback_agent = VideoFallbackAgent(cfg.get("video_fallback_window_expand_seconds", 5))
        self.video_reader = QwenVLVideoReader(
            results_root=cfg["results_root"],
            vlm_checkpoint=cfg.get("vlm_checkpoint", "/root/models/Qwen3-VL-8B-Instruct"),
            raw_video_root=cfg.get("raw_video_root", "/root/dataset/videomme/videos"),
            raw_video_extensions=cfg.get("raw_video_extensions", [".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"]),
        )
        self.answer_agent = AnswerAgent(
            use_llm_for_answer=cfg.get("use_llm_for_answer", True),
            llm_client=self.answer_llm_client,
        )

    @staticmethod
    def _make_memory_ref(memory_level: str, raw_id: str) -> str:
        level = str(memory_level or "").upper()
        rid = str(raw_id or "").strip()
        if not rid:
            return ""
        if "::" in rid:
            return rid
        if level in {"STM", "MTM", "LTM", "RAW_VIDEO"}:
            return f"{level}::{rid}"
        return rid

    @staticmethod
    def _parse_memory_ref(memory_ref: str, fallback_level: str = "") -> Tuple[str, str]:
        ref = str(memory_ref or "").strip()
        if "::" in ref:
            lvl, rid = ref.split("::", 1)
            lvl = str(lvl or "").upper()
            if lvl in {"STM", "MTM", "LTM", "RAW_VIDEO"}:
                return lvl, str(rid or "")
        return str(fallback_level or "").upper(), ref

    def _normalize_memory_item(self, item: Dict) -> Dict:
        out = deepcopy(item)
        level = str(out.get("memory_level") or "").upper()
        mid = str(out.get("memory_id") or "")
        parsed_level, raw_id = self._parse_memory_ref(mid, fallback_level=level)
        norm_level = parsed_level or level
        out["memory_level"] = norm_level
        out["memory_id"] = self._make_memory_ref(norm_level, raw_id)
        return out

    def _normalize_pool_memory_ids(self, items: List[Dict]) -> List[Dict]:
        return [self._normalize_memory_item(x) for x in items]

    def _build_evidence_items(self, evidence: List[Dict]) -> List[Dict]:
        out = []
        for e in evidence:
            out.append(
                {
                    "memory_id": e.get("memory_id", ""),
                    "memory_level": e.get("memory_level", ""),
                    "event_id": e.get("event_id", ""),
                    "time_span": e.get("time_span", [0.0, 0.0]),
                    "text": e.get("text", ""),
                    "relevance": e.get("relevance", "medium"),
                    "role": e.get("role", "supporting_context"),
                    "source_locator": e.get("source_locator", {}),
                }
            )
        return out

    @staticmethod
    def _parse_vlm_is_helpful(text: str) -> Optional[bool]:
        """Extract is_helpful from VLM JSON output text. Returns None if unparseable."""
        try:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                obj = json.loads(m.group())
                val = obj.get("is_helpful")
                if isinstance(val, bool):
                    return val
        except Exception:
            pass
        return None

    @staticmethod
    def _merge_raw_text(vlm_text: str, asr_text: str) -> str:
        vt = (vlm_text or "").strip()
        at = (asr_text or "").strip()
        if vt and at:
            return f"[VLM]\n{vt}\n\n[ASR]\n{at}"
        return vt or at

    def _register_raw_video_results(self, raw_bank: Dict[str, Dict], items: List[Dict]) -> None:
        for item in items:
            if str(item.get("memory_level", "")).upper() != "RAW_VIDEO":
                continue
            rid = str(item.get("memory_id") or "").strip()
            if not rid:
                continue
            mid = self._make_memory_ref("RAW_VIDEO", rid)
            extra = item.get("extra", {}) if isinstance(item.get("extra"), dict) else {}
            vlm_text = str(item.get("text") or "")
            asr_text = str(extra.get("asr_context") or "")
            raw_bank[mid] = {
                "memory_id": mid,
                "raw_memory_id": rid,
                "event_id": str(item.get("event_id") or "raw_window"),
                "memory_level": "RAW_VIDEO",
                "time_span": item.get("time_span", [0.0, 0.0]),
                "vlm_text": vlm_text,
                "asr_context": asr_text,
                "merged_text": self._merge_raw_text(vlm_text, asr_text),
                "query": str(extra.get("query") or ""),
                "goal": str(extra.get("goal") or ""),
                "video_path": str(extra.get("video_path") or ""),
                "source_file": "generated/video_fallback",
            }

    def _hydrate_memory_item(self, item: Dict, raw_bank: Dict[str, Dict] | None = None) -> Dict:
        x = deepcopy(item)
        level = str(x.get("memory_level") or "").upper()
        parsed_level, raw_id = self._parse_memory_ref(str(x.get("memory_id") or ""), fallback_level=level)
        if parsed_level:
            level = parsed_level
        x["memory_level"] = level
        x["memory_id"] = self._make_memory_ref(level, raw_id)
        if level == "RAW_VIDEO":
            if raw_bank is not None:
                mid = str(x.get("memory_id") or "")
                rb = raw_bank.get(mid)
                if rb:
                    x["event_id"] = x.get("event_id") or rb.get("event_id", "raw_window")
                    x["time_span"] = x.get("time_span") or rb.get("time_span", [0.0, 0.0])
                    x["text"] = rb.get("merged_text", x.get("text", ""))
            if self.ego_mode:
                es = ego_span(x.get("time_span"))
                if es:
                    x["ego_time_span"] = es
                    x.pop("time_span", None)
            return x
        resolved = self.retriever.resolve_memory(level, raw_id)
        if not resolved:
            return x
        if not x.get("event_id"):
            x["event_id"] = resolved.get("event_id", "")
        if not x.get("time_span") or x.get("time_span") == [0.0, 0.0]:
            x["time_span"] = resolved.get("time_span", [0.0, 0.0])
        x["text"] = resolved.get("text", x.get("text", ""))
        if self.ego_mode:
            es = ego_span(x.get("time_span"))
            if es:
                x["ego_time_span"] = es
                x.pop("time_span", None)
        return x

    def _hydrate_pool(self, pool: List[Dict], raw_bank: Dict[str, Dict] | None = None) -> List[Dict]:
        return [self._hydrate_memory_item(item, raw_bank=raw_bank) for item in pool]

    def _merge_with_validator_order(self, existing: List[Dict], incoming: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Deduplicate evidence while preserving the validator's priority order."""
        merged, dropped = merge_evidence_with_dedup(
            existing,
            incoming,
            float(self.cfg.get("evidence_dedup_threshold", 0.85)),
        )
        merged_ids = {str(item.get("memory_id") or "") for item in merged}
        ordered: List[Dict] = []
        seen = set()
        for item in incoming + merged:
            memory_id = str(item.get("memory_id") or "")
            if not memory_id or memory_id not in merged_ids or memory_id in seen:
                continue
            ordered.append(item)
            seen.add(memory_id)
        max_items = max(1, int(self.cfg.get("max_evidence_pool_items", 40) or 40))
        return ordered[:max_items], dropped

    def _resolve_ranked_item(
        self,
        ranked: Dict,
        existing_pool: List[Dict],
        retrieved: List[Dict],
        raw_bank: Dict[str, Dict] | None = None,
    ) -> Dict:
        level = str(ranked.get("memory_level") or "").upper()
        mid = str(ranked.get("memory_id") or "")
        parsed_level, raw_id = self._parse_memory_ref(mid, fallback_level=level)
        if parsed_level:
            level = parsed_level
        mid = self._make_memory_ref(level, raw_id)
        role = str(ranked.get("role") or "supporting_context")
        text = ""

        candidate = None
        for src in retrieved + existing_pool:
            if str(src.get("memory_id") or "") == mid and str(src.get("memory_level") or "").upper() == level:
                candidate = src
                break

        event_id = ""
        time_span = [0.0, 0.0]
        relevance = "medium"
        if candidate is not None:
            event_id = str(candidate.get("event_id") or "")
            time_span = candidate.get("time_span", [0.0, 0.0])
            relevance = "high" if float(candidate.get("score", 0.0)) >= 0.25 else str(candidate.get("relevance") or "medium")

        if level != "RAW_VIDEO":
            resolved = self.retriever.resolve_memory(level, raw_id)
            if resolved:
                event_id = event_id or str(resolved.get("event_id") or "")
                time_span = time_span if time_span != [0.0, 0.0] else resolved.get("time_span", [0.0, 0.0])
                text = str(resolved.get("text") or text)
        else:
            if raw_bank is not None and mid in raw_bank:
                rb = raw_bank[mid]
                event_id = event_id or str(rb.get("event_id") or "raw_window")
                time_span = time_span if time_span != [0.0, 0.0] else rb.get("time_span", [0.0, 0.0])
                text = rb.get("merged_text", "")
            elif candidate is not None:
                text = str(candidate.get("text") or "")

        source_locator = {
            "memory_id": mid,
            "memory_level": level,
            "source_file": self._source_file_for_level(level),
            "raw_memory_id": raw_id,
        }
        if level == "RAW_VIDEO" and raw_bank is not None and mid in raw_bank:
            rb = raw_bank[mid]
            source_locator["source_file"] = str(rb.get("source_file") or "generated/video_fallback")
            source_locator["video_path"] = str(rb.get("video_path") or "")
            source_locator["query"] = str(rb.get("query") or "")
            source_locator["goal"] = str(rb.get("goal") or "")

        return {
            "memory_id": mid,
            "memory_level": level,
            "event_id": event_id,
            "time_span": time_span,
            "text": str(text or ""),
            "relevance": relevance,
            "role": role,
            "source_locator": source_locator,
        }

    @staticmethod
    def _prepare_validation_alias_payload(
        existing_evidence_pool: List[Dict],
        retrieved_candidates: List[Dict],
    ) -> Tuple[List[Dict], List[Dict], Dict[str, Dict]]:
        alias_to_real: Dict[str, Dict] = {}
        key_to_alias: Dict[str, str] = {}
        alias_idx = 1

        def _alias_for(item: Dict) -> str:
            nonlocal alias_idx
            level = str(item.get("memory_level") or "").upper()
            mid = str(item.get("memory_id") or "")
            key = mid or f"{level}::"
            alias = key_to_alias.get(key)
            if alias:
                return alias
            alias = f"mem_{alias_idx:03d}"
            alias_idx += 1
            key_to_alias[key] = alias
            alias_to_real[alias] = {
                "memory_id": mid,
                "memory_level": level,
            }
            return alias

        def _transform(items: List[Dict]) -> List[Dict]:
            out = []
            for src in items:
                item = deepcopy(src)
                item["memory_id"] = _alias_for(item)
                out.append(item)
            return out

        existing_view = _transform(existing_evidence_pool)
        retrieved_view = _transform(retrieved_candidates)
        return existing_view, retrieved_view, alias_to_real

    @staticmethod
    def _map_validation_output_ids(
        vout: Dict,
        alias_to_real: Dict[str, Dict],
    ) -> Dict:
        out = deepcopy(vout)

        def _resolve(mid: str) -> Dict | None:
            if mid in alias_to_real:
                return alias_to_real[mid]
            return None

        ranked = out.get("ranked_evidence") if isinstance(out.get("ranked_evidence"), list) else []
        new_ranked = []
        for item in ranked:
            if not isinstance(item, dict):
                continue
            mid = str(item.get("memory_id") or "").strip()
            if not mid:
                continue
            mapped = _resolve(mid)
            if mapped:
                item["memory_id"] = mapped.get("memory_id", "")
                if not item.get("memory_level"):
                    item["memory_level"] = mapped.get("memory_level", "")
            new_ranked.append(item)
        out["ranked_evidence"] = new_ranked

        removed = out.get("removed_evidence") if isinstance(out.get("removed_evidence"), list) else []
        new_removed = []
        for item in removed:
            if not isinstance(item, dict):
                continue
            mid = str(item.get("memory_id") or "").strip()
            if not mid:
                continue
            mapped = _resolve(mid)
            if mapped:
                item["memory_id"] = mapped.get("memory_id", "")
            new_removed.append(item)
        out["removed_evidence"] = new_removed

        return out

    def _build_question_text(self, question_obj: Dict) -> str:
        text = str(question_obj.get("question", ""))
        if not self.ego_mode:
            return text
        from src.utils.ego_time import seconds_to_ego_timestamp
        meta = question_obj.get("metadata", {}) if isinstance(question_obj.get("metadata"), dict) else {}
        qt_secs = meta.get("query_time_secs")
        if qt_secs is not None:
            try:
                ts = seconds_to_ego_timestamp(float(qt_secs))
                text = f"[Query time: {ts}]\n{text}"
            except Exception:
                pass
        return text

    def run(self, question_obj: Dict) -> Tuple[Dict, Dict, List[str]]:
        state = init_state(question_obj)
        self.retriever.set_video(str(question_obj.get("video_id", "")))
        raw_video_memory_bank: Dict[str, Dict] = {}
        rounds = []
        txt_logs: List[str] = []
        timers: Dict[str, float] = {}
        max_rounds = int(self.cfg.get("max_rounds", 5))
        next_missing: List[str] = []
        question_text = self._build_question_text(question_obj)

        # Coarse-to-fine recovery path.  MTM locates an event, STM supplies
        # grounded detail, and RAW_VIDEO is reserved for unresolved visual facts.
        configured_levels = [str(level).upper() for level in self.cfg.get("enabled_memory_levels", ["MTM", "STM"])]
        escalation_levels = [level for level in ("MTM", "STM") if level in configured_levels]
        if self.cfg.get("use_video_fallback", True):
            escalation_levels.append("RAW_VIDEO")
        empty_streak = 0
        insufficient_streak = 0
        escalation_idx = 0
        escalation_threshold = max(1, int(self.cfg.get("stop_if_no_new_evidence_rounds", 2) or 2))
        escalate_on_insufficient = bool(self.cfg.get("escalate_on_insufficient_streak", True))

        for ridx in range(max_rounds):
            state.round_idx = ridx
            is_last_round = ridx == max_rounds - 1
            query_round_notice = (
                "This is the final retrieval round. You must issue your best and final retrieval query now."
                if is_last_round
                else ""
            )
            hydrated_for_query = self._hydrate_pool(state.evidence_pool, raw_bank=raw_video_memory_bank)
            with timed(timers, "query"):
                plan = self.query_agent.plan(
                    question=question_text,
                    options=question_obj["options"],
                    evidence_summary=hydrated_for_query,
                    missing_information=next_missing,
                    enabled_memory_levels=self.cfg.get("enabled_memory_levels", ["STM", "MTM", "LTM"]),
                    previous_search_history=state.query_history,
                    round_notice=query_round_notice,
                    use_video_fallback=self.cfg.get("use_video_fallback", True),
                )

            # Do not leave the source choice entirely to the planner after it
            # has demonstrated that the current level is exhausted.
            if escalation_idx > 0 and escalation_idx < len(escalation_levels):
                forced_level = escalation_levels[escalation_idx]
                plan["retrieval_targets"] = [forced_level]
                for query in plan.get("queries", []) or []:
                    if isinstance(query, dict):
                        query["target_levels"] = [forced_level]

            # A deterministic guard against repeated LLM plans.  This is not
            # an LLM rewrite; it merely prevents an exact request loop.
            if state.query_history:
                previous = state.query_history[-1]
                previous_query = (previous.get("queries") or [{}])[0]
                current_query = (plan.get("queries") or [{}])[0]
                previous_target = str((previous.get("retrieval_targets") or [""])[0]).upper()
                current_target = str((plan.get("retrieval_targets") or [""])[0]).upper()
                previous_text = str(previous_query.get("query_text") or "").strip().lower()
                current_text = str(current_query.get("query_text") or "").strip().lower()
                if current_target == previous_target and current_text == previous_text and current_target != "RAW_VIDEO":
                    suffix = " ".join(str(item) for item in next_missing[:2]).strip() or "details"
                    current_query["query_text"] = f"{current_text} {suffix}".strip()
            state.question_type = plan.get("question_type")
            state.query_history.append(plan)

            with timed(timers, "search"):
                retrieved = self.search_agent.run(plan)
            retrieved = self._normalize_pool_memory_ids(retrieved)
            video_plan = {"need_video_fallback": False, "fallback_requests": [], "notes": ""}
            video_results = []
            query_target = ""
            q0 = {}
            if plan.get("queries"):
                q0 = plan.get("queries", [{}])[0]
                levels = q0.get("target_levels", []) if isinstance(q0, dict) else []
                if levels:
                    query_target = str(levels[0]).upper()

            # ── RAW_VIDEO path ────────────────────────────────────────────
            should_fallback = query_target == "RAW_VIDEO"
            if self.cfg.get("use_video_fallback", True) and should_fallback:
                with timed(timers, "video_fallback"):
                    fallback_reason = ""
                    if isinstance(state.sufficiency, dict):
                        fallback_reason = str(state.sufficiency.get("reason", "")).strip()
                    if not fallback_reason and next_missing:
                        fallback_reason = "; ".join(str(x) for x in next_missing)
                    video_plan = self.video_fallback_agent.plan(
                        question=question_text,
                        options=question_obj["options"],
                        evidence_pool=self._hydrate_pool(state.evidence_pool, raw_bank=raw_video_memory_bank),
                        video_fallback_reason=fallback_reason,
                        fallback_query=q0.get("query_text", "") if plan.get("queries") else "",
                        fallback_time_filter=q0.get("time_filter", None) if plan.get("queries") else None,
                    )
                    if video_plan.get("need_video_fallback"):
                        # Make request_ids unique per round so each round's
                        # RAW_VIDEO result gets a distinct memory_id in the
                        # bank (avoids later rounds overwriting earlier ones).
                        for _req in video_plan.get("fallback_requests", []):
                            base = str(_req.get("request_id") or "vf")
                            if not base.endswith(f"_r{ridx}"):
                                _req["request_id"] = f"{base}_r{ridx}"
                        video_results = self.video_reader.inspect(
                            question_obj["video_id"],
                            video_plan.get("fallback_requests", []),
                            question=question_obj.get("question", ""),
                            options=question_obj.get("options"),
                        )
                        video_results = self._normalize_pool_memory_ids(video_results)
                        # Always register in bank for traceability.
                        self._register_raw_video_results(raw_video_memory_bank, video_results)
                        # Filter out unhelpful VLM results before adding to evidence pool.
                        # is_helpful=False means VLM saw nothing relevant; keeping such
                        # evidence would mislead the Answer Agent.  Annotate the query
                        # history so the next round's Query Agent knows which windows
                        # were already tried and found uninformative.
                        helpful_results = []
                        unhelpful_windows = []
                        for vr in video_results:
                            helpful = self._parse_vlm_is_helpful(vr.get("text", ""))
                            if helpful is False:
                                unhelpful_windows.append(vr.get("time_span", []))
                            else:
                                helpful_results.append(vr)
                        if unhelpful_windows and state.query_history:
                            state.query_history[-1]["notes"] = (
                                str(state.query_history[-1].get("notes") or "")
                                + f"; RAW_VIDEO windows {unhelpful_windows} returned no relevant content"
                                  " — try a different time window for the next RAW_VIDEO call"
                            )
                        video_results = helpful_results

            retrieved_all = list(retrieved)
            if video_results:
                retrieved_all.extend(video_results)
            state.retrieval_history.append({"round": ridx, "results": retrieved_all})

            hydrated_existing = self._hydrate_pool(state.evidence_pool, raw_bank=raw_video_memory_bank)
            hydrated_retrieved = self._hydrate_pool(retrieved_all, raw_bank=raw_video_memory_bank)
            validate_existing_view, validate_retrieved_view, alias_to_real = self._prepare_validation_alias_payload(
                hydrated_existing,
                hydrated_retrieved,
            )
            validation_round_notice_parts = []
            if video_results:
                helpful_count = len(video_results)
                validation_round_notice_parts.append(
                    f"This round used RAW_VIDEO retrieval: {helpful_count} helpful VLM observation(s) "
                    f"were returned after filtering out uninformative windows. "
                    f"Treat these as direct visual evidence; do not discount them solely because they are VLM-generated."
                )
            validation_round_notice = "\n".join(validation_round_notice_parts)
            with timed(timers, "validate"):
                vout = self.validation_agent.validate(
                    question=question_text,
                    options=question_obj["options"],
                    existing_evidence_pool=validate_existing_view,
                    retrieved_candidates=validate_retrieved_view,
                    round_notice=validation_round_notice,
                    search_history=list(state.query_history),
                )
            vout = self._map_validation_output_ids(vout, alias_to_real)

            # A medium-term summary is useful for localization but is not a
            # sufficient final basis on its own.  Require clip-level or direct
            # video evidence before allowing early termination.
            if bool(vout.get("is_enough", False)) and bool(self.cfg.get("validation_require_stm_or_raw", True)):
                has_grounded_evidence = any(
                    str(item.get("memory_level") or "").upper() in {"STM", "RAW_VIDEO"}
                    for item in list(validate_existing_view) + list(validate_retrieved_view)
                )
                if not has_grounded_evidence:
                    vout["is_enough"] = False
                    vout["reason"] = "Enforced: STM or RAW_VIDEO evidence is required before answering."
                    missing = list(vout.get("missing_information") or [])
                    if "STM_or_RAW_required" not in missing:
                        missing.append("STM_or_RAW_required")
                    vout["missing_information"] = missing

            ranked = vout.get("ranked_evidence", []) if isinstance(vout.get("ranked_evidence"), list) else []
            resolved_ranked = [
                self._resolve_ranked_item(x, state.evidence_pool, retrieved_all, raw_bank=raw_video_memory_bank) for x in ranked
            ]
            incoming = self._build_evidence_items(resolved_ranked)
            if incoming:
                empty_streak = 0
            else:
                empty_streak += 1
            if escalate_on_insufficient:
                insufficient_streak = 0 if bool(vout.get("is_enough", False)) else insufficient_streak + 1
            else:
                insufficient_streak = 0
            if (
                (empty_streak >= escalation_threshold or insufficient_streak >= escalation_threshold)
                and escalation_idx < len(escalation_levels) - 1
            ):
                escalation_idx += 1
                empty_streak = 0
                insufficient_streak = 0

            ordered_pool, dropped_dups = self._merge_with_validator_order(state.evidence_pool, incoming)
            state.removed_evidence_pool.extend(vout.get("removed_evidence", []))
            state.removed_evidence_pool.extend(dropped_dups)
            state.evidence_pool = ordered_pool

            state.video_fallback_history.append(video_plan)
            state.sufficiency = {
                "is_enough": bool(vout.get("is_enough", False)),
                "reason": vout.get("reason", ""),
                "missing_information": vout.get("missing_information", []),
            }
            next_missing = list(vout.get("missing_information", []))

            round_entry = {
                "round_idx": ridx,
                "query_plan": plan,
                "retrieval_results": retrieved_all,
                "validator_output": vout,
                "video_fallback_request": video_plan,
                "video_fallback_results": video_results,
            }
            rounds.append(round_entry)

            txt_logs.append(
                f"round={ridx} levels={plan.get('retrieval_targets', [])} events={sorted(set(x.get('event_id','') for x in retrieved_all if x.get('event_id')))}"
            )
            txt_logs.append(
                f"round={ridx} enough={vout.get('is_enough', False)} missing={vout.get('missing_information', [])} fallback={video_plan.get('need_video_fallback', False)}"
            )

            if bool(vout.get("is_enough", False)):
                state.terminated = True
                state.termination_reason = "validator_sufficient"
                break

        if not state.terminated:
            state.terminated = True
            state.termination_reason = "max_rounds_reached"

        # Final fallback: surface the closest MTM and STM candidates even when
        # normal retrieval rejected every score.  This is deliberately a last
        # resort, not a replacement for the regular multi-round path.
        if state.termination_reason == "max_rounds_reached" and len(state.evidence_pool) == 0:
            with timed(timers, "final_mtm_fallback"):
                fallback_mtm_items = max(0, int(self.cfg.get("final_fallback_mtm_items", 15) or 0))
                fallback_stm_items = max(0, int(self.cfg.get("final_fallback_stm_items", 25) or 0))
                fallback_all = []
                if fallback_mtm_items:
                    fallback_all.extend(
                        item.__dict__.copy()
                        for item in self.retriever.retrieve(
                            question_text, "MTM", fallback_mtm_items, min_score=float("-inf")
                        )
                    )
                if fallback_stm_items:
                    fallback_all.extend(
                        item.__dict__.copy()
                        for item in self.retriever.retrieve(
                            question_text, "STM", fallback_stm_items, min_score=float("-inf")
                        )
                    )
                fallback_all = self._normalize_pool_memory_ids(fallback_all)
                fallback_notice = (
                    "Final fallback evidence selection. Keep the most useful MTM or STM evidence item in ranked_evidence."
                )
                fe_existing = self._hydrate_pool(state.evidence_pool, raw_bank=raw_video_memory_bank)
                fe_retrieved = self._hydrate_pool(fallback_all, raw_bank=raw_video_memory_bank)
                fe_existing_view, fe_retrieved_view, alias_to_real3 = self._prepare_validation_alias_payload(
                    fe_existing,
                    fe_retrieved,
                )
                fallback_vout = self.validation_agent.validate(
                    question=question_text,
                    options=question_obj["options"],
                    existing_evidence_pool=fe_existing_view,
                    retrieved_candidates=fe_retrieved_view,
                    round_notice=fallback_notice,
                    search_history=list(state.query_history),
                )
                fallback_vout = self._map_validation_output_ids(fallback_vout, alias_to_real3)

                ranked_fb = (
                    fallback_vout.get("ranked_evidence", [])
                    if isinstance(fallback_vout.get("ranked_evidence"), list)
                    else []
                )

                resolved_fb = [
                    self._resolve_ranked_item(x, state.evidence_pool, fallback_all, raw_bank=raw_video_memory_bank)
                    for x in ranked_fb
                ]
                incoming_fb = self._build_evidence_items(resolved_fb)
                state.evidence_pool, dropped_fb = self._merge_with_validator_order(state.evidence_pool, incoming_fb)
                state.removed_evidence_pool.extend(fallback_vout.get("removed_evidence", []))
                state.removed_evidence_pool.extend(dropped_fb)
                state.sufficiency = {
                    "is_enough": bool(fallback_vout.get("is_enough", False)),
                    "reason": fallback_vout.get("reason", ""),
                    "missing_information": fallback_vout.get("missing_information", []),
                }
                rounds.append(
                    {
                        "round_idx": max_rounds,
                        "query_plan": {
                            "question_type": state.question_type,
                            "query_strategy": "single",
                            "retrieval_targets": ["MTM", "STM"],
                            "queries": [
                                {
                                    "query_id": "query_final_mtm_stm_fallback",
                                    "query_text": "<CLOSEST_MTM_STM_FALLBACK>",
                                    "target_levels": ["MTM", "STM"],
                                    "time_filter": None,
                                    "purpose": "Final fallback: provide closest MTM and STM candidates to validation.",
                                }
                            ],
                            "notes": "final_mtm_stm_fallback",
                        },
                        "retrieval_results": fallback_all,
                        "validator_output": fallback_vout,
                        "video_fallback_request": {"need_video_fallback": False, "fallback_requests": [], "notes": ""},
                        "video_fallback_results": [],
                    }
                )
                txt_logs.append(f"final_mtm_stm_fallback applied candidates={len(fallback_all)} kept={len(incoming_fb)}")

        answer = self.answer_agent.answer(
            question=question_text,
            options=question_obj["options"],
            evidence_pool=self._hydrate_pool(state.evidence_pool, raw_bank=raw_video_memory_bank),
        )
        if state.sufficiency.get("is_enough") is False:
            if answer.get("confidence") == "high":
                answer["confidence"] = "low"
        state.final_answer = answer

        if self.ego_mode:
            from src.utils.ego_time import ego_span as _ego_span
            for rnd in rounds:
                for item in rnd.get("retrieval_results", []):
                    if "ego_time_span" not in item:
                        es = _ego_span(item.get("time_span"))
                        if es:
                            item["ego_time_span"] = es

        trace = {
            "question_id": question_obj.get("question_id", ""),
            "question_type": state.question_type,
            "rounds": rounds,
            "final_evidence_pool": state.evidence_pool,
            "raw_video_memory_bank": raw_video_memory_bank,
            "final_answer": state.final_answer,
            "termination_reason": state.termination_reason,
            "timers": timers,
        }

        txt_logs.append(f"final_answer={answer.get('predicted_option')} confidence={answer.get('confidence')}")
        return state_to_dict(state), trace, txt_logs
