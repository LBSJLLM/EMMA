from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.memory.schemas import RetrievalResult


def _tokens(text: str) -> set:
    return set(re.findall(r"[\w\u4e00-\u9fff]+", (text or "").lower()))


def _lexical_score(query: str, text: str) -> float:
    q = _tokens(query)
    t = _tokens(text)
    if not q or not t:
        return 0.0
    return len(q & t) / float(len(q))


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        fx = float(x)
        fy = float(y)
        dot += fx * fy
        na += fx * fx
        nb += fy * fy
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


class OpenAIEmbedder:
    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model = model
        # Keep embedding credentials separate when a deployment uses different
        # OpenAI-compatible providers for retrieval and reasoning.  The generic
        # OPENAI_* variables remain a convenient shared fallback.
        key = (
            api_key
            or os.getenv("EMBED_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("QA_OPENAI_KEY")
            or ""
        )
        base = (
            base_url
            or os.getenv("EMBED_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("QA_OPENAI_BASE_URL")
        )
        self.client = None
        if key:
            from openai import OpenAI

            kwargs: Dict[str, Any] = {"api_key": key}
            if base:
                kwargs["base_url"] = base
            self.client = OpenAI(**kwargs)

    def available(self) -> bool:
        return self.client is not None

    def embed(self, text: str) -> Optional[List[float]]:
        if self.client is None:
            return None
        try:
            resp = self.client.embeddings.create(model=self.model, input=text)
            return list(resp.data[0].embedding)
        except Exception:
            return None


class ResultsMemoryRetriever:
    """Retriever over /root/results/<video_id>/outputs/{short,medium,long}_term.json.

    Scoring is fixed to embedding*0.7 + lexical*0.3 when embeddings are available.
    """

    def __init__(
        self,
        results_root: str,
        embed_model: str = "text-embedding-3-large",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        scoring_mix: Tuple[float, float] = (0.7, 0.3),
    ):
        self.results_root = Path(results_root)
        self.embedder = OpenAIEmbedder(embed_model, api_key=api_key, base_url=base_url)
        self.alpha = float(scoring_mix[0])
        self.beta = float(scoring_mix[1])
        self.current_video_id: Optional[str] = None
        self.by_level: Dict[str, List[Dict[str, Any]]] = {"STM": [], "MTM": [], "LTM": []}
        self._index_by_level: Dict[str, Dict[str, Dict[str, Any]]] = {"STM": {}, "MTM": {}, "LTM": {}}

    def _load_json(self, p: Path) -> List[Dict[str, Any]]:
        if not p.exists():
            return []
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []

    def set_video(self, video_id: str) -> None:
        if video_id == self.current_video_id:
            return
        out_dir = self.results_root / video_id / "outputs"
        self.by_level = {
            "STM": self._load_json(out_dir / "short_term.json"),
            "MTM": self._load_json(out_dir / "medium_term.json"),
            "LTM": self._load_json(out_dir / "long_term.json"),
        }
        self._index_by_level = {"STM": {}, "MTM": {}, "LTM": {}}
        for level, items in self.by_level.items():
            idx: Dict[str, Dict[str, Any]] = {}
            for raw in items:
                common = self._to_common_item(level, raw)
                mid = str(common.get("memory_id") or "")
                if mid:
                    idx[mid] = common
            self._index_by_level[level] = idx
        self.current_video_id = video_id

    def resolve_memory(self, memory_level: str, memory_id: str) -> Optional[Dict[str, Any]]:
        level = str(memory_level or "").upper()
        mid = str(memory_id or "")
        if level not in self._index_by_level or not mid:
            return None
        return self._index_by_level[level].get(mid)

    @staticmethod
    def _to_common_item(level: str, item: Dict[str, Any]) -> Dict[str, Any]:
        if level == "STM":
            text = "\n".join(
                [
                    str(item.get("visual_summary") or ""),
                    str(item.get("inferred_intent") or ""),
                    str(item.get("detailed_caption") or ""),
                    str(item.get("ASR") or ""),
                ]
            ).strip()
            ts = item.get("time_range") or [0.0, 0.0]
            return {
                "memory_id": str(item.get("id") or ""),
                "event_id": str(item.get("id") or ""),
                "time_span": [float(ts[0]), float(ts[1])] if isinstance(ts, list) and len(ts) == 2 else [0.0, 0.0],
                "text": text,
                "embedding": item.get("embedding"),
                "extra": {
                    "video_source_path": str(item.get("video_source_path") or ""),
                    "source_file": "outputs/short_term.json",
                },
            }
        if level == "MTM":
            ts = item.get("time_span") or [0.0, 0.0]
            text = "\n".join(
                [
                    str(item.get("topic") or ""),
                    str(item.get("narrative_summary") or ""),
                    str(item.get("semantic_inference") or ""),
                ]
            ).strip()
            return {
                "memory_id": str(item.get("task_id") or ""),
                "event_id": str(item.get("task_id") or ""),
                "time_span": [float(ts[0]), float(ts[1])] if isinstance(ts, list) and len(ts) == 2 else [0.0, 0.0],
                "text": text,
                "embedding": item.get("embedding"),
                "extra": {
                    "topic": str(item.get("topic") or ""),
                    "source_file": "outputs/medium_term.json",
                },
            }
        text = str(item.get("knowledge_content") or "")
        return {
            "memory_id": str(item.get("concept_key") or ""),
            "event_id": str(item.get("concept_key") or ""),
            "time_span": [0.0, 0.1],
            "text": text,
            "embedding": item.get("embedding"),
            "extra": {
                "frequency": int(item.get("frequency") or 0),
                "source_file": "outputs/long_term.json",
            },
        }

    def _score(self, query: str, query_emb: Optional[List[float]], item: Dict[str, Any]) -> float:
        lex = _lexical_score(query, item.get("text", ""))
        emb = 0.0
        raw_emb = item.get("embedding")
        if query_emb is not None and isinstance(raw_emb, list):
            emb = _cosine(query_emb, raw_emb)
        if query_emb is None:
            return lex
        return self.alpha * emb + self.beta * lex

    @staticmethod
    def _time_spans_overlap(span_a: List[float] | None, span_b: List[float] | None) -> bool:
        if not (isinstance(span_a, list) and len(span_a) == 2 and isinstance(span_b, list) and len(span_b) == 2):
            return False
        try:
            a0 = float(span_a[0])
            a1 = float(span_a[1])
            b0 = float(span_b[0])
            b1 = float(span_b[1])
        except Exception:
            return False
        return a0 <= b1 and b0 <= a1

    def retrieve(
        self,
        query_text: str,
        target_level: str,
        top_k: int,
        time_filter: List[float] | None = None,
        min_score: float = 0.0,
    ) -> List[RetrievalResult]:
        level = str(target_level).upper()
        if level not in self.by_level:
            return []
        query_emb = self.embedder.embed(query_text) if self.embedder.available() else None

        scored = []
        for raw in self.by_level[level]:
            item = self._to_common_item(level, raw)
            if time_filter is not None and not self._time_spans_overlap(item.get("time_span"), time_filter):
                continue
            score = self._score(query_text, query_emb, item)
            if score <= min_score:
                continue
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)

        out: List[RetrievalResult] = []
        for score, item in scored[: max(1, int(top_k))]:
            out.append(
                RetrievalResult(
                    memory_id=item["memory_id"],
                    memory_level=level,
                    event_id=item["event_id"],
                    time_span=item["time_span"],
                    text=item["text"],
                    score=float(score),
                    source_query_id="",
                    extra=item["extra"],
                )
            )
        return out

    def retrieve_all(self, target_level: str) -> List[RetrievalResult]:
        level = str(target_level).upper()
        if level not in self.by_level:
            return []

        out: List[RetrievalResult] = []
        for raw in self.by_level[level]:
            item = self._to_common_item(level, raw)
            out.append(
                RetrievalResult(
                    memory_id=item["memory_id"],
                    memory_level=level,
                    event_id=item["event_id"],
                    time_span=item["time_span"],
                    text=item["text"],
                    score=0.0,
                    source_query_id="final_mtm_fallback",
                    extra=item["extra"],
                )
            )
        return out
