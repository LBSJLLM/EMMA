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
from prompts import (
    EVENT_LINK_WITH_ET_PROMPT_AIO,
    SESSION_SUMMARY_PROMPT_TEMPLATE,
    UPDATE_EVENT_TABLE_PROMPT,
)


DEFAULT_EMBED_MODEL = "text-embedding-3-large"
DEFAULT_LLM_MODEL = "deepseek-chat"
JSON_MAX_ATTEMPTS = 3
NO_ASR_TEXT = "无可用音频或转录失败"
_TOKEN_LOG_LOCK = threading.Lock()


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
    mtms: int = 0
    visual_boundaries: int = 0
    candidate_boundaries: int = 0
    elapsed_s: float = 0.0
    error: str = ""


def _record_token_usage(resp: Any, *, model: str, call_name: str) -> None:
    if "deepseek" not in str(model or "").lower():
        return
    log_path = os.getenv("EMMA_TOKEN_LOG", "").strip()
    if not log_path:
        return
    usage = getattr(resp, "usage", None)
    if usage is None:
        return

    def _usage_value(name: str) -> int:
        if isinstance(usage, dict):
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)
        try:
            return int(value or 0)
        except Exception:
            return 0

    row = {
        "ts": time.time(),
        "phase": os.getenv("EMMA_TOKEN_PHASE", "memory"),
        "component": "memory",
        "call_name": call_name,
        "model": model,
        "input_tokens": _usage_value("prompt_tokens"),
        "output_tokens": _usage_value("completion_tokens"),
        "total_tokens": _usage_value("total_tokens"),
    }
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _TOKEN_LOG_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_openai_client() -> OpenAI:
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


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _valid_json_list(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        data = _load_json(path)
    except Exception:
        return False
    return isinstance(data, list) and len(data) > 0


def _load_stms(path: Path) -> List[ShortTermMemory]:
    rows = _load_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"short_term.json must contain a list: {path}")
    stms = [ShortTermMemory.from_dict(row) for row in rows]
    stms.sort(key=lambda x: (float(x.time_range[0]), float(x.time_range[1]), str(x.id)))
    return stms


def _boundary_stm(stm: ShortTermMemory, mask_asr: bool) -> ShortTermMemory:
    return ShortTermMemory(
        id=stm.id,
        video_source_path=stm.video_source_path,
        time_range=stm.time_range,
        visual_summary=stm.visual_summary,
        detailed_caption=stm.detailed_caption,
        embedding=stm.embedding,
        ASR=NO_ASR_TEXT if mask_asr else stm.ASR,
        environment=stm.environment,
    )


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


def _call_json(
    client: OpenAI,
    prompt: str,
    model: str,
    max_tokens: int,
    rate_limiter: Optional[RateLimiter],
    required_keys: Sequence[str],
    call_name: str = "json_call",
) -> Dict[str, Any]:
    last_error = ""
    last_raw = ""
    for attempt in range(1, JSON_MAX_ATTEMPTS + 1):
        call_prompt = prompt
        if attempt > 1:
            call_prompt = f"{prompt}\n\nReturn valid JSON only."
        try:
            if rate_limiter is not None:
                rate_limiter.wait()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": call_prompt}],
                max_tokens=max_tokens,
                temperature=0,
            )
            _record_token_usage(resp, model=model, call_name=call_name)
            last_raw = (resp.choices[0].message.content or "").strip()
            data = _parse_json_object(last_raw)
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"JSON missing key: {key}")
            return data
        except Exception as exc:
            last_error = str(exc)
            print(
                f"[warn] json call failed attempt={attempt}/{JSON_MAX_ATTEMPTS}: "
                f"{last_error}; raw={last_raw[:180]!r}",
                flush=True,
            )
    raise RuntimeError(f"json call failed after {JSON_MAX_ATTEMPTS} attempts: {last_error}")


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


def _event_payload(stm: ShortTermMemory) -> Dict[str, Any]:
    return {
        "clip_id": str(stm.id),
        "time_range": [stm.time_range[0], stm.time_range[1]],
        "visual_summary": stm.visual_summary,
        "detailed_caption": stm.detailed_caption,
        "ASR": stm.ASR,
        "environment": stm.environment,
    }


def _fallback_event_table(pre_et: Optional[Dict[str, Any]], stm: ShortTermMemory) -> Dict[str, Any]:
    identity = (stm.visual_summary or "ongoing activity").strip()
    summary = stm.detailed_caption.strip() or stm.visual_summary.strip() or ""
    if isinstance(pre_et, dict):
        return {
            "event_identity": pre_et.get("event_identity") or identity,
            "event_summary": pre_et.get("event_summary") or summary,
            "entities": pre_et.get("entities") or [],
            "open_questions": pre_et.get("open_questions") or [],
            "delta": "CONTINUE: parse failure fallback.",
        }
    return {
        "event_identity": identity,
        "event_summary": summary,
        "entities": [],
        "open_questions": [],
        "delta": "SHIFT: NONE -> initial visual event.",
    }


def _coerce_event_table(data: Any, pre_et: Optional[Dict[str, Any]], stm: ShortTermMemory) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return _fallback_event_table(pre_et, stm)
    required = {"event_summary", "entities", "open_questions", "delta"}
    if not required.issubset(set(data.keys())):
        return _fallback_event_table(pre_et, stm)
    identity = str(data.get("event_identity") or data.get("event_intent") or "").strip()
    if not identity or not isinstance(data.get("entities"), list) or not isinstance(data.get("open_questions"), list):
        return _fallback_event_table(pre_et, stm)
    return {
        "event_identity": identity,
        "event_summary": str(data.get("event_summary", "")).strip(),
        "entities": data.get("entities") or [],
        "open_questions": data.get("open_questions") or [],
        "delta": str(data.get("delta", "")).strip(),
    }


def _update_event_table(
    client: OpenAI,
    pre_et: Optional[Dict[str, Any]],
    stm: ShortTermMemory,
    llm_model: str,
    rate_limiter: Optional[RateLimiter],
) -> Dict[str, Any]:
    pre_et_text = "null" if not pre_et else json.dumps(pre_et, ensure_ascii=False, indent=2)
    stm_text = json.dumps(_event_payload(stm), ensure_ascii=False, indent=2)
    prompt = f"{UPDATE_EVENT_TABLE_PROMPT}\n\npre_ET:\n{pre_et_text}\n\nSTM_k:\n{stm_text}"
    try:
        data = _call_json(
            client,
            prompt,
            model=llm_model,
            max_tokens=1024,
            rate_limiter=rate_limiter,
            required_keys=("event_summary", "entities", "open_questions", "delta"),
            call_name="update_event_table",
        )
        return _coerce_event_table(data, pre_et, stm)
    except Exception as exc:
        print(f"[warn] update_event_table fallback: {type(exc).__name__}: {exc}", flush=True)
        return _fallback_event_table(pre_et, stm)


def _is_initialization_shift(delta: str) -> bool:
    raw = str(delta or "").strip()
    if not raw.upper().startswith("SHIFT:") or "->" not in raw:
        return False
    lhs = raw.split(":", 1)[1].split("->", 1)[0].strip()
    token = "".join(ch for ch in lhs.upper() if ch.isalnum())
    return token in {"", "NONE", "NULL", "NIL", "NA", "UNKNOWN", "EMPTY", "INITIAL", "INIT", "NOEVENT"}


def _pre_et_payload(pre_et: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(pre_et, dict):
        return None
    return {
        "event_identity": pre_et.get("event_identity") or pre_et.get("event_intent", ""),
        "event_summary": pre_et.get("event_summary", ""),
        "entities": pre_et.get("entities") or [],
    }


def _event_link_prompt(
    pre_et: Optional[Dict[str, Any]],
    anchor_stm: ShortTermMemory,
    pending_stm: ShortTermMemory,
) -> str:
    pre_et_text = "null" if not _pre_et_payload(pre_et) else json.dumps(_pre_et_payload(pre_et), ensure_ascii=False, indent=2)
    prompt = EVENT_LINK_WITH_ET_PROMPT_AIO
    prompt = prompt.replace("{candidate_source}", "ET shift from event table only; ASR chapter stream disabled.")
    prompt = prompt.replace("{asr_confidence}", "low")
    prompt = prompt.replace("{chapter_context}", "segmentation_confidence: low\n\nchapter_list: []")
    prompt = prompt.replace("{pre_et}", pre_et_text)
    prompt = prompt.replace("{anchor_summary}", anchor_stm.visual_summary)
    prompt = prompt.replace("{anchor_caption}", anchor_stm.detailed_caption)
    prompt = prompt.replace("{anchor_ASR}", anchor_stm.ASR)
    prompt = prompt.replace("{pending_summary}", pending_stm.visual_summary)
    prompt = prompt.replace("{pending_caption}", pending_stm.detailed_caption)
    prompt = prompt.replace("{pending_ASR}", pending_stm.ASR)
    return prompt


def _parse_split_seconds(value: Any) -> Optional[float]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except Exception:
        pass
    parts = raw.split(":")
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


def _parse_split_points(data: Dict[str, Any]) -> List[float]:
    raw = data.get("split_point")
    if raw is None:
        raw = data.get("split_points", [])
    if not isinstance(raw, list):
        return []
    points: List[float] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        t_s = _parse_split_seconds(item.get("t"))
        if t_s is not None:
            points.append(t_s)
    return points


def _judge_boundary(
    client: OpenAI,
    pre_et: Optional[Dict[str, Any]],
    anchor_stm: ShortTermMemory,
    pending_stm: ShortTermMemory,
    llm_model: str,
    rate_limiter: Optional[RateLimiter],
) -> Tuple[bool, List[float]]:
    prompt = _event_link_prompt(pre_et, anchor_stm, pending_stm)
    try:
        data = _call_json(
            client,
            prompt,
            model=llm_model,
            max_tokens=256,
            rate_limiter=rate_limiter,
            required_keys=("is_same_event", "split_point"),
            call_name="judge_boundary",
        )
        return not bool(data.get("is_same_event")), _parse_split_points(data)
    except Exception as exc:
        print(f"[warn] boundary judge fallback accept split: {type(exc).__name__}: {exc}", flush=True)
        return True, []


def _snap_split_position(anchor_stm: ShortTermMemory, split_points: Sequence[float]) -> str:
    if not split_points:
        return "before_anchor"
    start, end = map(float, anchor_stm.time_range)
    midpoint = (start + end) / 2.0
    split_t = min(split_points, key=lambda x: abs(float(x) - midpoint))
    return "after_anchor" if split_t > midpoint else "before_anchor"


def _visual_groups(
    client: OpenAI,
    stms: Sequence[ShortTermMemory],
    llm_model: str,
    rate_limiter: Optional[RateLimiter],
    log_path: Path,
    mask_asr_for_boundary: bool,
) -> Tuple[List[List[ShortTermMemory]], int, int]:
    if not stms:
        return [], 0, 0
    boundary_stms = [_boundary_stm(stm, mask_asr_for_boundary) for stm in stms]
    groups: List[List[ShortTermMemory]] = []
    current_group: List[ShortTermMemory] = []
    current_et: Optional[Dict[str, Any]] = None
    candidates = 0
    accepted = 0

    for idx, (orig_stm, bound_stm) in enumerate(zip(stms, boundary_stms)):
        if current_et is None:
            current_et = _update_event_table(client, None, bound_stm, llm_model, rate_limiter)
            current_group = [orig_stm]
            _append_jsonl(log_path, {
                "clip_id": str(orig_stm.id),
                "time_range": list(orig_stm.time_range),
                "decision": "init",
                "event_table": current_et,
            })
            continue

        proposed = _update_event_table(client, current_et, bound_stm, llm_model, rate_limiter)
        delta = str(proposed.get("delta", "")).strip()
        is_candidate = delta.upper().startswith("SHIFT:") and not _is_initialization_shift(delta)
        split = False
        snap_position = ""
        if is_candidate and idx + 1 < len(boundary_stms):
            candidates += 1
            split, split_points = _judge_boundary(
                client,
                current_et,
                bound_stm,
                boundary_stms[idx + 1],
                llm_model,
                rate_limiter,
            )
            if split:
                snap_position = _snap_split_position(bound_stm, split_points)

        if split and snap_position == "before_anchor":
            if current_group:
                groups.append(current_group)
            accepted += 1
            current_et = _update_event_table(client, None, bound_stm, llm_model, rate_limiter)
            current_group = [orig_stm]
            decision = "split_before_anchor"
        elif split and snap_position == "after_anchor":
            current_et = proposed
            current_group.append(orig_stm)
            groups.append(current_group)
            accepted += 1
            current_et = None
            current_group = []
            decision = "split_after_anchor"
        else:
            # The proposed ET already integrates this clip. Reusing it avoids a
            # second LLM call per continued clip and keeps the run tractable.
            current_et = proposed
            current_group.append(orig_stm)
            decision = "continue_candidate_rejected" if is_candidate else "continue"

        _append_jsonl(log_path, {
            "clip_id": str(orig_stm.id),
            "time_range": list(orig_stm.time_range),
            "delta": delta,
            "candidate": is_candidate,
            "decision": decision,
            "snap_split_to": snap_position,
            "event_table": current_et,
        })

    if current_group:
        groups.append(current_group)
    return groups, candidates, accepted


def _detail_from_stm(stm: ShortTermMemory) -> str:
    visual = str(stm.detailed_caption or "").strip()
    asr_text = str(stm.ASR or "").strip()
    environment = str(stm.environment or "").strip()
    env_line = f"\n[ENV] {environment}" if environment else ""
    if asr_text and asr_text != NO_ASR_TEXT:
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
    try:
        data = _call_json(
            client,
            prompt,
            model=llm_model,
            max_tokens=max_tokens,
            rate_limiter=rate_limiter,
            required_keys=("topic_label", "full_narrative", "semantic_inference"),
            call_name="summarize_group",
        )
        return {
            "topic_label": str(data.get("topic_label") or "visual event").strip() or "visual event",
            "full_narrative": str(data.get("full_narrative") or "").strip(),
            "semantic_inference": str(data.get("semantic_inference") or "").strip(),
        }
    except Exception as exc:
        print(f"[warn] summary fallback: {type(exc).__name__}: {exc}", flush=True)
        condensed = "\n\n".join(_detail_from_stm(stm) for stm in list(stms)[:2])
        return {
            "topic_label": stms[0].visual_summary or "visual event",
            "full_narrative": condensed or stms[0].visual_summary or "visual event",
            "semantic_inference": "This event memory was consolidated from clips grouped by visual-only event-table transitions.",
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
    mask_asr_for_boundary: bool,
) -> BuildStats:
    start_time = time.time()
    source_outputs = source_root / video_id / "outputs"
    output_outputs = output_root / video_id / "outputs"
    out_medium = output_outputs / "medium_term.json"
    out_short = output_outputs / "short_term.json"
    if resume and _valid_json_list(out_medium) and _valid_json_list(out_short):
        rows = _load_json(out_medium)
        return BuildStats(video_id=video_id, status="skipped", mtms=len(rows), elapsed_s=time.time() - start_time)

    st_path = source_outputs / "short_term.json"
    if not st_path.exists():
        raise FileNotFoundError(f"missing source short_term.json: {st_path}")
    stms = _load_stms(st_path)
    if not stms:
        raise ValueError(f"empty source short_term.json: {st_path}")

    output_outputs.mkdir(parents=True, exist_ok=True)
    shutil.copy2(st_path, out_short)
    if (source_outputs / "short_term.stream.jsonl").exists():
        shutil.copy2(source_outputs / "short_term.stream.jsonl", output_outputs / "short_term.stream.jsonl")
    if copy_extra_files:
        for name in ("asr_segments.json", "chapter_segmentation.json"):
            src = source_outputs / name
            if src.exists():
                shutil.copy2(src, output_outputs / name)

    client = _build_openai_client()
    log_path = output_outputs / "visual_event_link_logs.jsonl"
    if log_path.exists() and not resume:
        log_path.unlink()
    groups, candidates, accepted = _visual_groups(
        client,
        stms,
        llm_model=llm_model,
        rate_limiter=rate_limiter,
        log_path=log_path,
        mask_asr_for_boundary=mask_asr_for_boundary,
    )
    if not groups:
        raise ValueError(f"visual-only grouping produced no groups: {video_id}")

    mtms: List[MediumTermMemory] = []
    for group in groups:
        summary = _summarize_group(client, group, llm_model, max_tokens, rate_limiter)
        first, last = group[0], group[-1]
        narrative = summary["full_narrative"] or "\n".join(_detail_from_stm(stm) for stm in group)
        semantic = summary["semantic_inference"]
        emb_text = f"{narrative}\n{semantic}".strip()
        mtms.append(
            MediumTermMemory(
                task_id=uuid4(),
                topic=summary["topic_label"] or first.visual_summary or "visual event",
                time_span=(float(first.time_range[0]), float(last.time_range[1])),
                narrative_summary=narrative,
                semantic_inference=semantic,
                child_clip_ids=[UUID(str(stm.id)) for stm in group],
                embedding=_embed_text(client, emb_text or narrative, embed_model, rate_limiter),
            )
        )

    mtm_rows = [mtm.to_dict() for mtm in mtms]
    _write_json(out_medium, mtm_rows)
    _write_jsonl(output_outputs / "medium_term.stream.jsonl", mtm_rows)
    return BuildStats(
        video_id=video_id,
        status="success",
        stms=len(stms),
        mtms=len(mtms),
        visual_boundaries=accepted,
        candidate_boundaries=candidates,
        elapsed_s=time.time() - start_time,
    )


def _read_video_ids(path: Optional[Path], source_root: Path) -> List[str]:
    if path:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = []
    for d in sorted(source_root.iterdir()):
        if not d.is_dir():
            continue
        if (d / "outputs" / "short_term.json").exists():
            ids.append(d.name)
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build memories with the ASR chapter stream disabled, rerunning ET-only boundary decisions over existing STMs."
    )
    parser.add_argument("--source-root", type=Path, default=Path("/root/results"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--video-list", type=Path, default=None)
    parser.add_argument("--llm-model", default=os.getenv("VISUAL_ONLY_LLM_MODEL", DEFAULT_LLM_MODEL))
    parser.add_argument("--embed-model", default=os.getenv("VISUAL_ONLY_EMBED_MODEL", DEFAULT_EMBED_MODEL))
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("VISUAL_ONLY_SUMMARY_MAX_TOKENS", "1200")))
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--copy-extra-files", action="store_true", default=True)
    parser.add_argument("--summary-name", default="_visual_only_build_summary.json")
    parser.add_argument("--workers", type=int, default=int(os.getenv("VISUAL_ONLY_WORKERS", "2")))
    parser.add_argument("--rpm", type=float, default=float(os.getenv("VISUAL_ONLY_RPM", "50")))
    parser.add_argument(
        "--mask-asr-for-boundary",
        action="store_true",
        help="Strict visual-content mode: hide ASR text during ET boundary decisions. Default keeps STM ASR and only disables the ASR chapter stream.",
    )
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for summary, event-table, and embedding calls.")

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
                "mask_asr_for_boundary": bool(args.mask_asr_for_boundary),
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
                mask_asr_for_boundary=args.mask_asr_for_boundary,
            )
            print(
                f"[{idx}/{len(video_ids)}] {video_id} {stat.status} "
                f"stms={stat.stms} mtms={stat.mtms} candidates={stat.candidate_boundaries} "
                f"accepted={stat.visual_boundaries} elapsed={stat.elapsed_s:.1f}s",
                flush=True,
            )
            return stat
        except Exception as exc:
            stat = BuildStats(video_id=video_id, status="failed", error=f"{type(exc).__name__}: {exc}")
            print(f"[{idx}/{len(video_ids)}] {video_id} failed: {stat.error}", flush=True)
            return stat

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(_run_one, item) for item in enumerate(video_ids, start=1)]
        for fut in concurrent.futures.as_completed(futures):
            stat = fut.result()
            with stats_lock:
                stats.append(stat)
            _save_summary()

    _save_summary()
    failed = [s for s in stats if s.status == "failed"]
    print(
        f"Done: total={len(video_ids)} success={sum(1 for s in stats if s.status == 'success')} "
        f"skipped={sum(1 for s in stats if s.status == 'skipped')} failed={len(failed)}",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
