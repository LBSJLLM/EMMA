from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Tuple

from src.agents.single_agent import SingleAgent
from src.pipeline.controller import QAController
from src.pipeline.state import init_state, state_to_dict
from src.utils.timers import timed


class SingleAgentQAController(QAController):
    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.single_agent = SingleAgent(llm_client=self.llm_client)

    @staticmethod
    def _plan_from_action(action: Dict, question_type: str | None = None) -> Dict:
        target = str(action.get("target_source") or "MTM").upper()
        return {
            "question_type": question_type or "unknown",
            "query_strategy": "single",
            "retrieval_targets": [target],
            "queries": [
                {
                    "query_id": "query_1",
                    "query_text": str(action.get("main_query") or ""),
                    "target_levels": [target],
                    "time_filter": action.get("time_filter", None),
                    "purpose": str(action.get("reason") or "Single-agent retrieval step."),
                }
            ],
            "notes": "single_agent_multiround",
        }

    def _apply_ranked_evidence(
        self,
        state,
        ranked: List[Dict],
        retrieved_all: List[Dict],
        raw_video_memory_bank: Dict[str, Dict],
    ) -> Tuple[int, List[Dict]]:
        resolved = [
            self._resolve_ranked_item(x, state.evidence_pool, retrieved_all, raw_bank=raw_video_memory_bank)
            for x in ranked
        ]
        incoming = self._build_evidence_items(resolved)
        state.evidence_pool, dropped_dups = self._merge_with_validator_order(state.evidence_pool, incoming)
        state.removed_evidence_pool.extend(dropped_dups)
        return len(incoming), dropped_dups

    @staticmethod
    def _has_grounded_evidence(evidence_pool: List[Dict]) -> bool:
        return any(
            str(item.get("memory_level") or "").upper() in {"STM", "RAW_VIDEO"}
            for item in evidence_pool
        )

    @staticmethod
    def _force_action(action: Dict, target: str, question_text: str, missing: List[str]) -> Dict:
        suffix = " ".join(str(item) for item in missing[:2]).strip()
        query = str(action.get("main_query") or question_text).strip()
        if suffix and suffix.lower() not in query.lower():
            query = f"{query} {suffix}".strip()
        return {
            "type": "retrieve",
            "target_source": target,
            "main_query": query,
            "time_filter": None,
            "reason": f"Controller-enforced escalation to {target}.",
        }

    def _run_plan(
        self,
        *,
        question_obj: Dict,
        question_text: str,
        state,
        plan: Dict,
        ridx: int,
        raw_video_memory_bank: Dict[str, Dict],
        timers: Dict[str, float],
    ) -> Tuple[List[Dict], Dict, List[Dict]]:
        with timed(timers, "search"):
            retrieved = self.search_agent.run(plan)
        retrieved = self._normalize_pool_memory_ids(retrieved)

        video_plan = {"need_video_fallback": False, "fallback_requests": [], "notes": ""}
        video_results: List[Dict] = []
        q0 = plan.get("queries", [{}])[0] if plan.get("queries") else {}
        levels = q0.get("target_levels", []) if isinstance(q0, dict) else []
        query_target = str(levels[0]).upper() if levels else ""

        if self.cfg.get("use_video_fallback", True) and query_target == "RAW_VIDEO":
            with timed(timers, "video_fallback"):
                reason = str(q0.get("purpose") or "")
                video_plan = self.video_fallback_agent.plan(
                    question=question_text,
                    options=question_obj["options"],
                    evidence_pool=self._hydrate_pool(state.evidence_pool, raw_bank=raw_video_memory_bank),
                    video_fallback_reason=reason,
                    fallback_query=q0.get("query_text", ""),
                    fallback_time_filter=q0.get("time_filter", None),
                )
                if video_plan.get("need_video_fallback"):
                    for req in video_plan.get("fallback_requests", []):
                        base = str(req.get("request_id") or "vf")
                        if not base.endswith(f"_r{ridx}"):
                            req["request_id"] = f"{base}_r{ridx}"
                    video_results = self.video_reader.inspect(
                        question_obj["video_id"],
                        video_plan.get("fallback_requests", []),
                        question=question_obj.get("question", ""),
                        options=question_obj.get("options"),
                    )
                    video_results = self._normalize_pool_memory_ids(video_results)
                    self._register_raw_video_results(raw_video_memory_bank, video_results)
                    helpful = []
                    for vr in video_results:
                        if self._parse_vlm_is_helpful(vr.get("text", "")) is False:
                            continue
                        helpful.append(vr)
                    video_results = helpful

        retrieved_all = list(retrieved)
        if video_results:
            retrieved_all.extend(video_results)
        return retrieved_all, video_plan, video_results

    def run(self, question_obj: Dict) -> Tuple[Dict, Dict, List[str]]:
        state = init_state(question_obj)
        self.retriever.set_video(str(question_obj.get("video_id", "")))
        raw_video_memory_bank: Dict[str, Dict] = {}
        rounds = []
        txt_logs: List[str] = []
        timers: Dict[str, float] = {}
        max_rounds = int(self.cfg.get("max_rounds", 5))
        question_text = self._build_question_text(question_obj)
        enabled_levels = [str(level).upper() for level in self.cfg.get("enabled_memory_levels", ["MTM", "STM"])]
        escalation_levels = [level for level in ("MTM", "STM") if level in enabled_levels]
        if self.cfg.get("use_video_fallback", True):
            escalation_levels.append("RAW_VIDEO")
        escalation_idx = 0
        empty_streak = 0
        insufficient_streak = 0
        escalation_threshold = max(1, int(self.cfg.get("stop_if_no_new_evidence_rounds", 2) or 2))
        escalate_on_insufficient = bool(self.cfg.get("escalate_on_insufficient_streak", True))

        previous_retrieved: List[Dict] = []
        previous_plan: Dict | None = None

        for ridx in range(max_rounds):
            state.round_idx = ridx
            hydrated_existing = self._hydrate_pool(state.evidence_pool, raw_bank=raw_video_memory_bank)
            hydrated_retrieved = self._hydrate_pool(previous_retrieved, raw_bank=raw_video_memory_bank)

            with timed(timers, "single_agent"):
                decision = self.single_agent.step(
                    question=question_text,
                    options=question_obj["options"],
                    round_idx=ridx,
                    max_rounds=max_rounds,
                    enabled_memory_levels=enabled_levels,
                    use_video_fallback=bool(self.cfg.get("use_video_fallback", True)),
                    existing_evidence_pool=hydrated_existing,
                    retrieved_candidates=hydrated_retrieved,
                    previous_search_history=state.query_history,
                )

            ranked = decision.get("ranked_evidence", []) if isinstance(decision.get("ranked_evidence"), list) else []
            kept_count, dropped_dups = self._apply_ranked_evidence(
                state,
                ranked,
                previous_retrieved,
                raw_video_memory_bank,
            )

            action = decision.get("next_action") if isinstance(decision.get("next_action"), dict) else {}
            missing = [str(item) for item in decision.get("missing_information", []) if str(item).strip()]
            wants_answer = action.get("type") == "answer" or bool(decision.get("is_enough", False))
            if (
                wants_answer
                and bool(self.cfg.get("validation_require_stm_or_raw", True))
                and not self._has_grounded_evidence(state.evidence_pool)
            ):
                fallback_target = "STM" if "STM" in enabled_levels else (
                    "RAW_VIDEO" if self.cfg.get("use_video_fallback", True) else "MTM"
                )
                action = self._force_action(action, fallback_target, question_text, missing)
                decision["next_action"] = action
                decision["is_enough"] = False
                if "STM_or_RAW_required" not in missing:
                    missing.append("STM_or_RAW_required")
                decision["missing_information"] = missing

            if kept_count == 0:
                empty_streak += 1
            else:
                empty_streak = 0
            if escalate_on_insufficient:
                insufficient_streak = 0 if bool(decision.get("is_enough", False)) else insufficient_streak + 1
            else:
                insufficient_streak = 0
            if (
                (empty_streak >= escalation_threshold or insufficient_streak >= escalation_threshold)
                and escalation_idx < len(escalation_levels) - 1
            ):
                escalation_idx += 1
                empty_streak = 0
                insufficient_streak = 0
            if escalation_idx > 0 and escalation_idx < len(escalation_levels):
                action = self._force_action(action, escalation_levels[escalation_idx], question_text, missing)
                decision["next_action"] = action
                decision["is_enough"] = False

            state.sufficiency = {
                "is_enough": bool(decision.get("is_enough", False)),
                "reason": str((decision.get("next_action") or {}).get("reason") or ""),
                "missing_information": decision.get("missing_information", []),
            }

            if action.get("type") == "answer" or bool(decision.get("is_enough", False)):
                state.terminated = True
                state.termination_reason = "single_agent_answer"
                rounds.append(
                    {
                        "round_idx": ridx,
                        "query_plan": previous_plan or {},
                        "retrieval_results": previous_retrieved,
                        "single_agent_output": decision,
                        "kept_evidence_count": kept_count,
                        "dropped_duplicates": dropped_dups,
                        "video_fallback_request": {"need_video_fallback": False, "fallback_requests": [], "notes": ""},
                        "video_fallback_results": [],
                    }
                )
                break

            plan = self._plan_from_action(action, state.question_type)
            if previous_plan:
                previous_query = (previous_plan.get("queries") or [{}])[0]
                current_query = (plan.get("queries") or [{}])[0]
                previous_target = str((previous_plan.get("retrieval_targets") or [""])[0]).upper()
                current_target = str((plan.get("retrieval_targets") or [""])[0]).upper()
                previous_text = str(previous_query.get("query_text") or "").strip().lower()
                current_text = str(current_query.get("query_text") or "").strip().lower()
                if current_target == previous_target and current_text == previous_text and current_target != "RAW_VIDEO":
                    suffix = " ".join(missing[:2]).strip() or "details"
                    current_query["query_text"] = f"{current_text} {suffix}".strip()
            state.query_history.append(plan)
            retrieved_all, video_plan, video_results = self._run_plan(
                question_obj=question_obj,
                question_text=question_text,
                state=state,
                plan=plan,
                ridx=ridx,
                raw_video_memory_bank=raw_video_memory_bank,
                timers=timers,
            )
            state.retrieval_history.append({"round": ridx, "results": retrieved_all})
            state.video_fallback_history.append(video_plan)
            rounds.append(
                {
                    "round_idx": ridx,
                    "query_plan": deepcopy(plan),
                    "retrieval_results": retrieved_all,
                    "single_agent_output": decision,
                    "kept_evidence_count": kept_count,
                    "dropped_duplicates": dropped_dups,
                    "video_fallback_request": video_plan,
                    "video_fallback_results": video_results,
                }
            )
            txt_logs.append(
                f"round={ridx} single_agent_action={action.get('type')} target={action.get('target_source')} kept={kept_count} retrieved={len(retrieved_all)}"
            )
            previous_retrieved = retrieved_all
            previous_plan = plan

        if not state.terminated:
            state.terminated = True
            state.termination_reason = "single_agent_max_rounds"

        # Last-resort dual-level fallback.  Normal retrieval may filter every
        # item on score; expose the closest MTM and STM candidates once so the
        # evidence planner can still select grounded context for the R1 answer.
        if not state.evidence_pool:
            fallback_mtm_items = max(0, int(self.cfg.get("final_fallback_mtm_items", 15) or 0))
            fallback_stm_items = max(0, int(self.cfg.get("final_fallback_stm_items", 25) or 0))
            fallback_all: List[Dict] = []
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
            if fallback_all:
                with timed(timers, "single_agent_final_fallback"):
                    decision = self.single_agent.step(
                        question=question_text,
                        options=question_obj["options"],
                        round_idx=max_rounds - 1,
                        max_rounds=max_rounds,
                        enabled_memory_levels=enabled_levels,
                        use_video_fallback=bool(self.cfg.get("use_video_fallback", True)),
                        existing_evidence_pool=[],
                        retrieved_candidates=self._hydrate_pool(fallback_all, raw_bank=raw_video_memory_bank),
                        previous_search_history=state.query_history,
                    )
                ranked = decision.get("ranked_evidence", []) if isinstance(decision.get("ranked_evidence"), list) else []
                kept_count, dropped_dups = self._apply_ranked_evidence(
                    state, ranked, fallback_all, raw_video_memory_bank
                )
                rounds.append(
                    {
                        "round_idx": max_rounds,
                        "query_plan": {
                            "retrieval_targets": ["MTM", "STM"],
                            "queries": [{"query_id": "final_mtm_stm_fallback", "query_text": "<CLOSEST_MTM_STM_FALLBACK>"}],
                            "notes": "final_mtm_stm_fallback",
                        },
                        "retrieval_results": fallback_all,
                        "single_agent_output": decision,
                        "kept_evidence_count": kept_count,
                        "dropped_duplicates": dropped_dups,
                        "video_fallback_request": {"need_video_fallback": False, "fallback_requests": [], "notes": ""},
                        "video_fallback_results": [],
                    }
                )
                txt_logs.append(f"final_mtm_stm_fallback candidates={len(fallback_all)} kept={kept_count}")

        # The single agent (DeepSeek-V4-Flash) plans and validates evidence.
        # Final multiple-choice selection is always delegated to answer_llm_model
        # (DeepSeek R1 in the public reference configuration).
        state.final_answer = self.answer_agent.answer(
            question=question_text,
            options=question_obj["options"],
            evidence_pool=self._hydrate_pool(state.evidence_pool, raw_bank=raw_video_memory_bank),
        )

        trace = {
            "question_id": question_obj.get("question_id", ""),
            "question_type": state.question_type,
            "mode": "single_agent_multiround",
            "rounds": rounds,
            "final_evidence_pool": state.evidence_pool,
            "raw_video_memory_bank": raw_video_memory_bank,
            "final_answer": state.final_answer,
            "termination_reason": state.termination_reason,
            "timers": timers,
        }
        txt_logs.append(f"final_answer={state.final_answer.get('predicted_option')} confidence={state.final_answer.get('confidence')}")
        return state_to_dict(state), trace, txt_logs
