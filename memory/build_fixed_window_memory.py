#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OPENAI_API_KEY, OPENAI_BASE_URL
from memory_unit import MediumTermMemory, ShortTermMemory
from prompts import TIME_SUMMARY_PROMPT


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
    stms: int = 0
    windows: int = 0
    elapsed_s: float = 0.0
    error: str = ""


def _client() -> OpenAI:
    kwargs: Dict[str, Any] = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    return OpenAI(**kwargs)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


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
    if raw.startswith("{{"):
        try:
            data = json.loads(raw[1:])
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        try:
            fixed = raw
            while fixed.startswith("{{"):
                fixed = fixed[1:]
            while fixed.endswith("}}"):
                fixed = fixed[:-1]
            data = json.loads(fixed)
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


def _call_summary(
    client: OpenAI,
    prompt: str,
    model: str,
    max_tokens: int,
    limiter: Optional[RateLimiter],
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
            if limiter is not None:
                limiter.wait()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": call_prompt}],
                max_tokens=max_tokens,
                temperature=0,
            )
            last_raw = (resp.choices[0].message.content or "").strip()
            data = _parse_json_object(last_raw)
            for key in ("topic_label", "full_narrative"):
                if key not in data:
                    raise ValueError(f"summary JSON missing key: {key}")
            data.setdefault("semantic_inference", "")
            return data
        except Exception as exc:
            last_error = str(exc)
            print(
                f"[warn] summary failed attempt={attempt}/{JSON_MAX_ATTEMPTS}: "
                f"{last_error}; raw={last_raw[:180]!r}",
                flush=True,
            )
    raise RuntimeError(f"summary call failed after {JSON_MAX_ATTEMPTS} attempts: {last_error}")


def _embed_text(client: OpenAI, text: str, model: str, limiter: Optional[RateLimiter]) -> List[float]:
    content = (text or "").strip()
    if not content:
        return []
    if limiter is not None:
        limiter.wait()
    resp = client.embeddings.create(model=model, input=content)
    return [float(x) for x in resp.data[0].embedding]


def _seconds_to_hhmmss(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _load_stms(path: Path) -> List[ShortTermMemory]:
    rows = _load_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"short_term.json must contain a list: {path}")
    stms = [ShortTermMemory.from_dict(row) for row in rows]
    stms.sort(key=lambda x: (float(x.time_range[0]), float(x.time_range[1]), str(x.id)))
    return stms


def _copy_source_tree(source_video_dir: Path, out_video_dir: Path) -> None:
    out_outputs = out_video_dir / "outputs"
    out_outputs.mkdir(parents=True, exist_ok=True)
    src_outputs = source_video_dir / "outputs"
    skip_files = {"medium_term.json", "medium_term.stream.jsonl"}
    for item in src_outputs.iterdir():
        if item.name in skip_files:
            continue
        dst = out_outputs / item.name
        if item.is_dir():
            if item.name == "tiered":
                continue
            if not dst.exists():
                shutil.copytree(item, dst, symlinks=True)
        elif not dst.exists():
            shutil.copy2(item, dst)


def _strict_fixed_windows(stms: Sequence[ShortTermMemory], window_s: float) -> List[List[ShortTermMemory]]:
    if not stms:
        return []
    video_end = max(float(stm.time_range[1]) for stm in stms)
    count = max(1, int((video_end + window_s - 1e-9) // window_s))
    windows: List[List[ShortTermMemory]] = [[] for _ in range(count)]
    for stm in stms:
        start, end = float(stm.time_range[0]), float(stm.time_range[1])
        mid = max(0.0, (start + end) / 2.0)
        idx = min(count - 1, int(mid // window_s))
        windows[idx].append(stm)
    return [w for w in windows if w]


def _summarize_window(
    client: OpenAI,
    win: Sequence[ShortTermMemory],
    model: str,
    embed_model: str,
    max_tokens: int,
    limiter: Optional[RateLimiter],
) -> MediumTermMemory:
    start_s = min(float(stm.time_range[0]) for stm in win)
    end_s = max(float(stm.time_range[1]) for stm in win)
    details = [
        "\n".join(
            [
                f"[{_seconds_to_hhmmss(stm.time_range[0])}-{_seconds_to_hhmmss(stm.time_range[1])}]",
                f"Visual summary: {stm.visual_summary}",
                f"Detailed caption: {stm.detailed_caption}",
                f"ASR: {stm.ASR}",
            ]
        ).strip()
        for stm in win
    ]
    formatted = "\n".join(f"- {d}" for d in details)
    prompt = TIME_SUMMARY_PROMPT.replace("{detail_list}", formatted)
    # TIME_SUMMARY_PROMPT uses doubled braces for .format-style escaping in
    # other scripts. This script uses direct replacement, so normalize the JSON
    # schema shown to the model back to ordinary braces.
    prompt = prompt.replace("{{", "{").replace("}}", "}")
    data = _call_summary(client, prompt, model, max_tokens, limiter)
    topic = str(data.get("topic_label") or "Fixed time window").strip() or "Fixed time window"
    narrative = str(data.get("full_narrative") or "").strip()
    semantic = str(data.get("semantic_inference") or "").strip()
    embed_text = "\n".join([topic, narrative, semantic]).strip()
    embedding = _embed_text(client, embed_text, embed_model, limiter)
    return MediumTermMemory(
        task_id=uuid4(),
        topic=topic,
        time_span=(start_s, end_s),
        narrative_summary=narrative,
        semantic_inference=semantic,
        child_clip_ids=[stm.id for stm in win],
        embedding=embedding,
    )


def _build_video(
    video_id: str,
    source_root: Path,
    output_root: Path,
    window_s: float,
    model: str,
    embed_model: str,
    max_tokens: int,
    limiter: Optional[RateLimiter],
    resume: bool,
) -> BuildStats:
    start_t = time.monotonic()
    src = source_root / video_id
    dst = output_root / video_id
    out_mtm = dst / "outputs" / "medium_term.json"
    if resume and out_mtm.exists() and out_mtm.stat().st_size > 0:
        return BuildStats(video_id=video_id, status="skipped_existing")
    if not (src / "outputs" / "short_term.json").exists():
        raise FileNotFoundError(f"missing short_term.json for {video_id}")
    _copy_source_tree(src, dst)
    if out_mtm.exists():
        out_mtm.unlink()
    stms = _load_stms(src / "outputs" / "short_term.json")
    windows = _strict_fixed_windows(stms, window_s)
    client = _client()
    mtms = [
        _summarize_window(client, win, model, embed_model, max_tokens, limiter)
        for win in windows
    ]
    _write_json(out_mtm, [m.to_dict() for m in mtms])
    elapsed = time.monotonic() - start_t
    return BuildStats(
        video_id=video_id,
        status="ok",
        stms=len(stms),
        windows=len(windows),
        elapsed_s=elapsed,
    )


def _read_video_ids(path: Path) -> List[str]:
    return [x.strip() for x in path.read_text().splitlines() if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build strict fixed-window MTM memories.")
    parser.add_argument("--source-root", type=Path, default=Path("/root/results"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--video-list", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=180.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--rpm", type=float, default=60.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    video_ids = _read_video_ids(args.video_list)
    if args.limit and args.limit > 0:
        video_ids = video_ids[: args.limit]
    args.output_root.mkdir(parents=True, exist_ok=True)
    limiter = RateLimiter(args.rpm) if args.rpm and args.rpm > 0 else None
    stats: List[BuildStats] = []
    errors: List[BuildStats] = []

    def run_one(vid: str) -> BuildStats:
        try:
            return _build_video(
                video_id=vid,
                source_root=args.source_root,
                output_root=args.output_root,
                window_s=float(args.window_seconds),
                model=str(args.model),
                embed_model=str(args.embed_model),
                max_tokens=int(args.max_tokens),
                limiter=limiter,
                resume=bool(args.resume),
            )
        except Exception as exc:
            return BuildStats(video_id=vid, status="failed", error=repr(exc))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futs = {ex.submit(run_one, vid): vid for vid in video_ids}
        for idx, fut in enumerate(concurrent.futures.as_completed(futs), start=1):
            st = fut.result()
            stats.append(st)
            if st.status == "failed":
                errors.append(st)
            print(
                f"[{idx}/{len(video_ids)}] {st.video_id} {st.status} "
                f"stms={st.stms} windows={st.windows} elapsed={st.elapsed_s:.1f}s "
                f"{st.error}",
                flush=True,
            )

    summary = {
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "video_list": str(args.video_list),
        "window_seconds": float(args.window_seconds),
        "videos": len(video_ids),
        "ok": sum(1 for s in stats if s.status == "ok"),
        "skipped_existing": sum(1 for s in stats if s.status == "skipped_existing"),
        "failed": len(errors),
        "stats": [s.__dict__ for s in sorted(stats, key=lambda x: x.video_id)],
    }
    _write_json(args.output_root / "_fixed_window_build_summary.json", summary)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
