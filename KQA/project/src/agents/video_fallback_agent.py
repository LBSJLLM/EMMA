from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.memory.interfaces import VideoReader
from src.utils.vlm_client import QwenVLMClient


class VideoFallbackAgent:
    def __init__(self, window_expand_seconds: float = 5.0):
        self.window_expand_seconds = float(window_expand_seconds)

    @staticmethod
    def _expand_span(span: List[float], sec: float) -> Optional[Tuple[float, float]]:
        if not isinstance(span, list) or len(span) != 2:
            return None
        start = max(0.0, float(span[0]) - sec)
        end = max(start + 0.1, float(span[1]) + sec)
        return (start, end)

    def plan(
        self,
        question: str,
        options: Dict[str, str],
        evidence_pool: List[Dict],
        video_fallback_reason: str,
        fallback_query: str = "",
        fallback_time_filter: Optional[List[float]] = None,
    ) -> Dict:
        if isinstance(fallback_time_filter, list) and len(fallback_time_filter) == 2:
            try:
                s = float(fallback_time_filter[0])
                e = float(fallback_time_filter[1])
                if e >= s:
                    requests = [
                        {
                            "request_id": "vf_1",
                            "query": fallback_query or question,
                            "time_span": [s, e],
                            "goal": f"Inspect local detail for: {video_fallback_reason}",
                            "expected_evidence_type": "disambiguation",
                        }
                    ]
                    return {
                        "need_video_fallback": True,
                        "fallback_requests": requests,
                        "notes": "Use query-provided time filter directly.",
                    }
            except Exception:
                pass

        anchors = [e for e in evidence_pool if e.get("memory_level") in {"STM", "MTM"} and e.get("time_span")]
        requests = []
        for idx, item in enumerate(anchors[:2]):
            span = self._expand_span(item.get("time_span", []), self.window_expand_seconds)
            if not span:
                continue
            requests.append(
                {
                    "request_id": f"vf_{idx + 1}",
                    "query": fallback_query or question,
                    "time_span": [span[0], span[1]],
                    "goal": f"Inspect local detail for: {video_fallback_reason}",
                    "expected_evidence_type": "disambiguation",
                }
            )
        return {
            "need_video_fallback": len(requests) > 0,
            "fallback_requests": requests,
            "notes": "Anchored by STM/MTM windows only.",
        }


class QwenVLVideoReader(VideoReader):
    def __init__(
        self,
        results_root: str,
        vlm_checkpoint: Optional[str] = None,
        raw_video_root: Optional[str] = None,
        raw_video_extensions: Optional[List[str]] = None,
    ):
        self.results_root = Path(results_root)
        self.raw_video_root = Path(raw_video_root) if raw_video_root else None
        exts = raw_video_extensions or [".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"]
        self.raw_video_extensions = [e if str(e).startswith(".") else f".{e}" for e in exts]
        self.vlm = QwenVLMClient(checkpoint=vlm_checkpoint)
        self.prompt_template = self._load_prompt_template()

    @staticmethod
    def _load_prompt_template() -> str:
        p = Path(__file__).resolve().parents[1] / "prompts" / "video_fallback_prompt.txt"
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

    def _find_video_path_from_cfg(self, video_id: str) -> Optional[str]:
        if not self.raw_video_root:
            return None
        vid = str(video_id or "").strip()
        if not vid:
            return None

        candidate_names: List[str] = []
        if Path(vid).suffix:
            candidate_names.append(vid)
        else:
            for ext in self.raw_video_extensions:
                candidate_names.append(f"{vid}{ext}")

        for name in candidate_names:
            p = self.raw_video_root / name
            if p.exists() and p.is_file():
                return str(p)
        return None

    def _find_video_path_from_stm(self, video_id: str) -> Optional[str]:
        st_path = self.results_root / video_id / "outputs" / "short_term.json"
        if not st_path.exists():
            return None
        try:
            with open(st_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                p = str(data[0].get("video_source_path") or "").strip()
                return p or None
        except Exception:
            return None
        return None

    def _find_video_path(self, video_id: str) -> Optional[str]:
        # Preferred path source: configured dataset root + video_id.
        p = self._find_video_path_from_cfg(video_id)
        if p:
            return p
        # Backward-compatible fallback for legacy outputs.
        return self._find_video_path_from_stm(video_id)

    def _load_short_term(self, video_id: str) -> List[Dict]:
        st_path = self.results_root / video_id / "outputs" / "short_term.json"
        if not st_path.exists():
            return []
        try:
            with open(st_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    @staticmethod
    def _overlap(a: List[float], b: List[float]) -> bool:
        if not (isinstance(a, list) and len(a) == 2 and isinstance(b, list) and len(b) == 2):
            return False
        return float(a[0]) <= float(b[1]) and float(b[0]) <= float(a[1])

    def _collect_asr_for_span(self, video_id: str, span: List[float]) -> str:
        st = self._load_short_term(video_id)
        parts = []
        for item in st:
            tr = item.get("time_range", [0.0, 0.0])
            if not self._overlap(tr, span):
                continue
            asr = str(item.get("ASR") or "").strip()
            if asr:
                parts.append(asr)
        return "\n".join(parts).strip()

    @staticmethod
    def _format_options(options: Optional[Dict[str, str]]) -> str:
        if not options:
            return ""
        return "\n".join(f"({k}) {v}" for k, v in sorted(options.items()))

    def inspect(
        self,
        video_id: str,
        fallback_requests: List[Dict],
        question: str = "",
        options: Optional[Dict[str, str]] = None,
    ) -> List:
        video_path = self._find_video_path(video_id)
        options_str = self._format_options(options)
        out = []
        for req in fallback_requests:
            span = req.get("time_span", [0.0, 1.0])
            query = str(req.get("query") or req.get("goal") or "").strip()
            asr_context = self._collect_asr_for_span(video_id, span)
            if self.prompt_template:
                prompt = self._render_prompt(
                    self.prompt_template,
                    {
                        "question": question or query,
                        "options": options_str,
                        "query": query,
                        "time_range": span,
                        "asr_context": asr_context or "",
                        "video_segment": "<provided_by_video_frames>",
                    },
                )
            else:
                prompt = (
                    "Inspect localized video and return evidence JSON. "
                    f"Query: {query}. Time range: {span}. ASR in-range: {asr_context}"
                )
            vlm_text = ""
            if video_path and self.vlm.available():
                try:
                    vlm_text = self.vlm.ask_video(
                        video_path=video_path,
                        prompt=prompt,
                        time_range=(float(span[0]), float(span[1])),
                        fps=1.5,
                    )
                except Exception:
                    vlm_text = ""
            if not vlm_text:
                vlm_text = "RAW fallback could not run VLM; keeping placeholder localized request evidence."
            out.append(
                {
                    "memory_id": f"raw_{req.get('request_id', 'x')}",
                    "memory_level": "RAW_VIDEO",
                    "event_id": "raw_window",
                    "time_span": span,
                    "text": vlm_text,
                    "score": 0.6,
                    "source_query_id": req.get("request_id", ""),
                    "extra": {
                        "video_id": video_id,
                        "goal": req.get("goal", ""),
                        "query": query,
                        "asr_context": asr_context,
                        "video_path": video_path,
                    },
                }
            )
        return out
