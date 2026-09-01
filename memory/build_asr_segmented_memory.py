#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import UUID, uuid4

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OPENAI_API_KEY, OPENAI_BASE_URL
from memory_unit import MediumTermMemory, ShortTermMemory
from prompts import SESSION_SUMMARY_PROMPT_TEMPLATE


DEFAULT_EMBED_MODEL = "text-embedding-3-large"
DEFAULT_LLM_MODEL = "deepseek-chat"
JSON_MAX_ATTEMPTS = 3


class RateLimiter:
    def __init__(self, rpm: float) -> None:
        self.interval_s = 60.0 / max(1.0, float(rpm))
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            if now < self.next_at:
                time.sleep(self.next_at - now)
                now = time.monotonic()
            self.next_at = now + self.interval_s


@dataclass
class BuildStats:
    video_id: str
    status: str
    chapters: int = 0
    stms: int = 0
    mtms: int = 0
    skipped_empty_chapters: int = 0
    elapsed_s: float = 0.0
    error: str = ""


def _build_openai_client() -> OpenAI:
    kwargs: Dict[str, Any] = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    return OpenAI(**kwargs)


def _parse_json_object(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        first_newline = raw.find("\n")
        if first_newline >= 0:
            raw = raw[first_newline + 1 :]
        if raw.endswith("```"):
            raw = raw[:-3].strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        data = json.loads(raw[start : end + 1])
        if isinstance(data, dict):
            return data
    raise ValueError("LLM response is not a JSON object")


def _call_json_summary(
    client: OpenAI,
    prompt: str,
    model: str,
    max_tokens: int,
    rate_limiter: Optional[RateLimiter],
) -> Dict[str, Any]:
    last_error = ""
    last_raw = ""
    for attempt in range(1, JSON_MAX_ATTEMPTS + 1):
        call_prompt = prompt
        if attempt > 1:
            call_prompt = (
                f"{prompt}\n\nReturn valid JSON only with exactly these keys: "
                "topic_label, full_narrative, semantic_inference."
            )
        try:
            if rate_limiter is not None:
                rate_limiter.wait()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": call_prompt}],
                max_tokens=max_tokens,
                temperature=0,
            )
            last_raw = (resp.choices[0].message.content or "").strip()
            data = _parse_json_object(last_raw)
            for key in ("topic_label", "full_narrative", "semantic_inference"):
                if key not in data:
                    raise ValueError(f"summary JSON missing key: {key}")
            return data
        except Exception as exc:
            last_error = str(exc)
            print(
                f"[warn] summary call failed attempt={attempt}/{JSON_MAX_ATTEMPTS}: "
                f"{last_error}; raw={last_raw[:180]!r}",
                flush=True,
            )
    raise RuntimeError(f"summary call failed after {JSON_MAX_ATTEMPTS} attempts: {last_error}")


def _embed_text(
    client: OpenAI,
    text: str,
    model: str,
    rate_limiter: Optional[RateLimiter],
) -> List[float]:
    content = (text or "").strip()
    if not content:
        return []
    if rate_limiter is not None:
        rate_limiter.wait()
    resp = client.embeddings.create(model=model, input=content)
    return [float(x) for x in resp.data[0].embedding]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _valid_json_list(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        data = _load_json(path)
    except Exception:
        return False
    return isinstance(data, list) and len(data) > 0


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _load_stms(path: Path) -> List[ShortTermMemory]:
    rows = _load_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"short_term.json must contain a list: {path}")
    stms = [ShortTermMemory.from_dict(row) for row in rows]
    stms.sort(key=lambda x: (float(x.time_range[0]), float(x.time_range[1]), str(x.id)))
    return stms


def _hhmmss_to_seconds(value: Any) -> Optional[float]:
    parts = str(value or "").strip().split(":")
    try:
        if len(parts) == 3:
            h, m, s = parts
            return float(h) * 3600 + float(m) * 60 + float(s)
        if len(parts) == 2:
            m, s = parts
            return float(m) * 60 + float(s)
    except Exception:
        return None
    return None


def _load_chapter_spans(path: Path, video_duration_s: float) -> List[Tuple[float, float, Dict[str, Any]]]:
    data = _load_json(path)
    chapters = data.get("chapters") if isinstance(data, dict) else None
    if not isinstance(chapters, list) or not chapters:
        raise ValueError(f"chapter_segmentation.json has no chapters: {path}")

    starts: List[Tuple[float, Dict[str, Any]]] = []
    for ch in chapters:
        if not isinstance(ch, dict):
            continue
        st = _hhmmss_to_seconds(ch.get("start_time"))
        if st is None:
            continue
        starts.append((max(0.0, min(float(st), video_duration_s)), ch))
    if not starts:
        raise ValueError(f"chapter_segmentation.json has no parseable starts: {path}")

    starts.sort(key=lambda x: x[0])
    if starts[0][0] > 1.0:
        starts.insert(0, (0.0, {"title": "Beginning", "summary": ""}))
    else:
        starts[0] = (0.0, starts[0][1])

    spans: List[Tuple[float, float, Dict[str, Any]]] = []
    for idx, (st, ch) in enumerate(starts):
        ed = starts[idx + 1][0] if idx + 1 < len(starts) else video_duration_s
        st = max(0.0, min(st, video_duration_s))
        ed = max(st, min(ed, video_duration_s))
        if ed > st:
            spans.append((st, ed, ch))
    return spans


def _stm_overlap(stm: ShortTermMemory, span: Tuple[float, float]) -> float:
    st, ed = map(float, stm.time_range)
    left = max(st, span[0])
    right = min(ed, span[1])
    return max(0.0, right - left)


def _assign_stms_to_chapters(
    stms: Sequence[ShortTermMemory],
    chapter_spans: Sequence[Tuple[float, float, Dict[str, Any]]],
) -> List[List[ShortTermMemory]]:
    groups: List[List[ShortTermMemory]] = [[] for _ in chapter_spans]
    for stm in stms:
        overlaps = [
            _stm_overlap(stm, (float(ch_start), float(ch_end)))
            for ch_start, ch_end, _ in chapter_spans
        ]
        max_overlap = max(overlaps) if overlaps else 0.0
        if max_overlap <= 0.0:
            continue
        # Tie goes to the later chapter because a boundary marks the start of a new unit.
        best_idx = max(i for i, value in enumerate(overlaps) if abs(value - max_overlap) <= 1e-6)
        groups[best_idx].append(stm)
    return groups


def _detail_from_stm(stm: ShortTermMemory) -> str:
    visual = str(stm.detailed_caption or "").strip()
    asr_text = str(stm.ASR or "").strip()
    environment = str(stm.environment or "").strip()
    env_line = f"\n[ENV] {environment}" if environment else ""
    if asr_text and asr_text != "无可用音频或转录失败":
        return f"[VISUAL] {visual}{env_line}\n[ASR] {asr_text}".strip()
    return f"[VISUAL] {visual}{env_line}".strip()


def _summarize_group(
    client: OpenAI,
    stms: Sequence[ShortTermMemory],
    llm_model: str,
    max_tokens: int,
    rate_limiter: Optional[RateLimiter],
) -> Dict[str, str]:
    details = []
    for idx, stm in enumerate(stms, start=1):
        start, end = stm.time_range
        details.append(f"Segment {idx} [{start:.1f}-{end:.1f}s]\n{_detail_from_stm(stm)}")
    prompt = SESSION_SUMMARY_PROMPT_TEMPLATE.replace("{detail_list}", "\n\n".join(details))
    data = _call_json_summary(
        client,
        prompt,
        model=llm_model,
        max_tokens=max_tokens,
        rate_limiter=rate_limiter,
    )
    return {
        "topic_label": str(data.get("topic_label") or "event").strip() or "event",
        "full_narrative": str(data.get("full_narrative") or "").strip(),
        "semantic_inference": str(data.get("semantic_inference") or "").strip(),
    }


def _fallback_summary(stms: Sequence[ShortTermMemory], chapter: Dict[str, Any]) -> Dict[str, str]:
    title = str(chapter.get("title") or "").strip() or "ASR chapter event"
    chapter_summary = str(chapter.get("summary") or "").strip()
    details = [_detail_from_stm(stm) for stm in stms]
    head = details[:2]
    tail = details[-1:] if len(details) > 2 else []
    condensed = "\n\n".join(head + tail)
    if len(condensed) > 5000:
        condensed = condensed[:5000]
    narrative = chapter_summary
    if narrative and condensed:
        narrative = f"{narrative}\n\nSupporting clip evidence:\n{condensed}"
    elif condensed:
        narrative = condensed
    else:
        narrative = title
    return {
        "topic_label": title,
        "full_narrative": narrative,
        "semantic_inference": (
            "This event memory was consolidated from clips assigned to an ASR-derived chapter boundary. "
            "The summary preserves available visual and speech evidence without adding unsupported inferences."
        ),
    }


def _build_video(
    *,
    video_id: str,
    source_root: Path,
    output_root: Path,
    llm_model: str,
    embed_model: str,
    max_tokens: int,
    resume: bool,
    copy_extra_files: bool,
    rate_limiter: Optional[RateLimiter],
) -> BuildStats:
    start_time = time.time()
    source_outputs = source_root / video_id / "outputs"
    output_outputs = output_root / video_id / "outputs"
    out_medium = output_outputs / "medium_term.json"
    out_short = output_outputs / "short_term.json"

    if resume and _valid_json_list(out_medium) and _valid_json_list(out_short):
        rows = _load_json(out_medium)
        return BuildStats(
            video_id=video_id,
            status="skipped",
            mtms=len(rows) if isinstance(rows, list) else 0,
            elapsed_s=time.time() - start_time,
        )

    st_path = source_outputs / "short_term.json"
    ch_path = source_outputs / "chapter_segmentation.json"
    if not st_path.exists():
        raise FileNotFoundError(f"missing source short_term.json: {st_path}")
    if not ch_path.exists():
        raise FileNotFoundError(f"missing source chapter_segmentation.json: {ch_path}")

    stms = _load_stms(st_path)
    if not stms:
        raise ValueError(f"empty source short_term.json: {st_path}")
    video_duration_s = max(float(stm.time_range[1]) for stm in stms)
    chapter_spans = _load_chapter_spans(ch_path, video_duration_s)
    groups = _assign_stms_to_chapters(stms, chapter_spans)

    output_outputs.mkdir(parents=True, exist_ok=True)
    shutil.copy2(st_path, out_short)
    if (source_outputs / "short_term.stream.jsonl").exists():
        shutil.copy2(source_outputs / "short_term.stream.jsonl", output_outputs / "short_term.stream.jsonl")
    if copy_extra_files:
        for name in ("asr_segments.json", "chapter_segmentation.json"):
            src = source_outputs / name
            if src.exists():
                shutil.copy2(src, output_outputs / name)

    mtms: List[MediumTermMemory] = []
    skipped_empty = 0
    client = _build_openai_client()
    for (ch_start, ch_end, chapter), group in zip(chapter_spans, groups):
        if not group:
            skipped_empty += 1
            continue
        try:
            summary = _summarize_group(
                client,
                group,
                llm_model=llm_model,
                max_tokens=max_tokens,
                rate_limiter=rate_limiter,
            )
        except Exception as exc:
            print(
                f"[warn] {video_id} chapter {ch_start:.1f}-{ch_end:.1f}s "
                f"summary fallback after error: {type(exc).__name__}: {exc}",
                flush=True,
            )
            summary = _fallback_summary(group, chapter)
        title = str(chapter.get("title") or "").strip()
        topic = summary["topic_label"] or title or "event"
        narrative = summary["full_narrative"] or "\n".join(_detail_from_stm(stm) for stm in group)
        semantic = summary["semantic_inference"]
        emb_text = f"{narrative}\n{semantic}".strip()
        mtms.append(
            MediumTermMemory(
                task_id=uuid4(),
                topic=topic,
                time_span=(float(ch_start), float(ch_end)),
                narrative_summary=narrative,
                semantic_inference=semantic,
                child_clip_ids=[UUID(str(stm.id)) for stm in group],
                embedding=_embed_text(
                    client,
                    emb_text if emb_text else topic,
                    model=embed_model,
                    rate_limiter=rate_limiter,
                ),
            )
        )

    if not mtms:
        raise ValueError(f"no ASR chapter received STM assignments: {video_id}")

    mtm_rows = [mtm.to_dict() for mtm in mtms]
    _write_json(out_medium, mtm_rows)
    _write_jsonl(output_outputs / "medium_term.stream.jsonl", mtm_rows)

    return BuildStats(
        video_id=video_id,
        status="success",
        chapters=len(chapter_spans),
        stms=len(stms),
        mtms=len(mtms),
        skipped_empty_chapters=skipped_empty,
        elapsed_s=time.time() - start_time,
    )


def _read_video_ids(path: Optional[Path], source_root: Path) -> List[str]:
    if path:
        ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return ids
    ids = []
    for d in sorted(source_root.iterdir()):
        if not d.is_dir():
            continue
        outputs = d / "outputs"
        if (outputs / "short_term.json").exists() and (outputs / "chapter_segmentation.json").exists():
            ids.append(d.name)
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build ASR-only-segmented MTM memories by regrouping existing STMs with ASR chapter boundaries."
    )
    parser.add_argument("--source-root", type=Path, default=Path("/root/results"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--video-list", type=Path, default=None)
    parser.add_argument("--llm-model", default=os.getenv("ASR_ONLY_LLM_MODEL", DEFAULT_LLM_MODEL))
    parser.add_argument("--embed-model", default=os.getenv("ASR_ONLY_EMBED_MODEL", DEFAULT_EMBED_MODEL))
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("ASR_ONLY_SUMMARY_MAX_TOKENS", "1200")))
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--copy-extra-files", action="store_true", default=True)
    parser.add_argument("--summary-name", default="_asr_only_build_summary.json")
    parser.add_argument("--workers", type=int, default=int(os.getenv("ASR_ONLY_WORKERS", "4")))
    parser.add_argument("--rpm", type=float, default=float(os.getenv("ASR_ONLY_RPM", "50")))
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for summary and embedding calls.")

    video_ids = _read_video_ids(args.video_list, args.source_root)
    if args.limit is not None:
        video_ids = video_ids[: max(0, int(args.limit))]
    if not video_ids:
        print("No videos selected.")
        return 0

    args.output_root.mkdir(parents=True, exist_ok=True)
    resume = not args.no_resume
    rate_limiter = RateLimiter(args.rpm) if args.rpm > 0 else None

    stats: List[BuildStats] = []
    stats_lock = threading.Lock()

    def _save_summary() -> None:
        with stats_lock:
            snapshot = list(stats)
        _write_json(
            args.output_root / args.summary_name,
            {
                "source_root": str(args.source_root),
                "output_root": str(args.output_root),
                "video_list": str(args.video_list) if args.video_list else None,
                "llm_model": args.llm_model,
                "embed_model": args.embed_model,
                "workers": max(1, int(args.workers)),
                "rpm": args.rpm,
                "total": len(video_ids),
                "success": sum(1 for s in snapshot if s.status == "success"),
                "failed": sum(1 for s in snapshot if s.status == "failed"),
                "skipped": sum(1 for s in snapshot if s.status == "skipped"),
                "items": [s.__dict__ for s in snapshot],
            },
        )

    def _run_one(item: Tuple[int, str]) -> BuildStats:
        idx, video_id = item
        print(f"[{idx}/{len(video_ids)}] {video_id} start", flush=True)
        try:
            stat = _build_video(
                video_id=video_id,
                source_root=args.source_root,
                output_root=args.output_root,
                llm_model=args.llm_model,
                embed_model=args.embed_model,
                max_tokens=args.max_tokens,
                resume=resume,
                copy_extra_files=args.copy_extra_files,
                rate_limiter=rate_limiter,
            )
            print(
                f"[{idx}/{len(video_ids)}] {video_id} {stat.status} "
                f"chapters={stat.chapters} stms={stat.stms} mtms={stat.mtms} "
                f"empty_chapters={stat.skipped_empty_chapters} elapsed={stat.elapsed_s:.1f}s",
                flush=True,
            )
        except Exception as exc:
            stat = BuildStats(video_id=video_id, status="failed", error=f"{type(exc).__name__}: {exc}")
            print(f"[{idx}/{len(video_ids)}] {video_id} failed: {stat.error}", flush=True)
        return stat

    max_workers = max(1, int(args.workers))
    if max_workers == 1:
        for item in enumerate(video_ids, start=1):
            stat = _run_one(item)
            with stats_lock:
                stats.append(stat)
            _save_summary()
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run_one, item) for item in enumerate(video_ids, start=1)]
            for future in concurrent.futures.as_completed(futures):
                stat = future.result()
                with stats_lock:
                    stats.append(stat)
                _save_summary()

    failed = [s for s in stats if s.status == "failed"]
    print(
        f"Done. success={sum(1 for s in stats if s.status == 'success')} "
        f"skipped={sum(1 for s in stats if s.status == 'skipped')} failed={len(failed)}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
