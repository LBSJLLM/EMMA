from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class QuestionInput:
    video_id: str
    question_id: str
    question: str
    options: Dict[str, str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    memory_id: str
    memory_level: str
    event_id: str
    time_span: List[float]
    text: str
    score: float
    source_query_id: str
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceItem:
    memory_id: str
    memory_level: str
    event_id: str
    time_span: List[float]
    text: str
    relevance: str
    role: str


@dataclass
class SufficiencyState:
    is_enough: bool = False
    reason: str = ""
    missing_information: List[str] = field(default_factory=list)


@dataclass
class PipelineState:
    question_info: Dict[str, Any]
    question_type: Optional[str] = None
    round_idx: int = 0
    query_history: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_history: List[Dict[str, Any]] = field(default_factory=list)
    evidence_pool: List[Dict[str, Any]] = field(default_factory=list)
    removed_evidence_pool: List[Dict[str, Any]] = field(default_factory=list)
    sufficiency: Dict[str, Any] = field(
        default_factory=lambda: {
            "is_enough": False,
            "reason": "",
            "missing_information": [],
        }
    )
    video_fallback_history: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: Optional[Dict[str, Any]] = None
    terminated: bool = False
    termination_reason: Optional[str] = None
