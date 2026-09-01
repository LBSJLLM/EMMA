from __future__ import annotations

from typing import Dict, List, Protocol

from .schemas import RetrievalResult


class Retriever(Protocol):
    def retrieve(
        self,
        query_text: str,
        target_level: str,
        top_k: int,
        time_filter: List[float] | None = None,
    ) -> List[RetrievalResult]:
        """Retrieve memory candidates from a specific memory level."""


class VideoReader(Protocol):
    def inspect(self, video_id: str, fallback_requests: List[Dict]) -> List[RetrievalResult]:
        """Inspect localized raw video windows and return evidence-like results."""
