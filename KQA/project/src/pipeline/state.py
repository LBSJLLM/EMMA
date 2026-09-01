from __future__ import annotations

from typing import Any, Dict

from src.memory.schemas import PipelineState


def init_state(question_obj: Dict[str, Any]) -> PipelineState:
    return PipelineState(question_info=question_obj)


def state_to_dict(state: PipelineState) -> Dict[str, Any]:
    return {
        "question_info": state.question_info,
        "question_type": state.question_type,
        "round_idx": state.round_idx,
        "query_history": state.query_history,
        "retrieval_history": state.retrieval_history,
        "evidence_pool": state.evidence_pool,
        "removed_evidence_pool": state.removed_evidence_pool,
        "sufficiency": state.sufficiency,
        "video_fallback_history": state.video_fallback_history,
        "final_answer": state.final_answer,
        "terminated": state.terminated,
        "termination_reason": state.termination_reason,
    }
