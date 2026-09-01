import os
import json
import gc
import atexit
import re
import queue
import threading
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from uuid import uuid4, UUID
from datetime import datetime

import torch
import torch.distributed as dist

from prompts import (
    CAPTION_PROMPT,
    SESSION_SUMMARY_PROMPT_TEMPLATE,
    CHAPTER_CHUNK as CHAPTER_SEGMENT_PROMPT_JSON,
    UPDATE_EVENT_TABLE_PROMPT,
    EVENT_LINK_WITH_ET_PROMPT_AIO,
)
from memory_unit import ShortTermMemory, MediumTermMemory

from openai import OpenAI
from config import (
    OPENAI_API_KEY, OPENAI_BASE_URL, VLLM_BASE_URL, VLLM_API_KEY, LLM_MODEL,
    MEMORY_TEXT_MODEL, QWEN_CHECKPOINT, WHISPER_MODEL_DIR, OUTPUT_ROOT,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# New unified allocator env var (PyTorch >= 2.4)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

JSON_API_MAX_ATTEMPTS = 3
_TOKEN_LOG_LOCK = threading.Lock()

UPDATE_EVENT_TABLE_PROMPT_FORCE_CONTINUE = (
    UPDATE_EVENT_TABLE_PROMPT + """
    # FORCE CONTINUE MODE (HARD)
    - You MUST output delta in CONTINUE format.
    - You MUST NOT output SHIFT under any circumstance.
    - You MAY slightly adjust event_title (event_identity) to appropriately encompass the current state of the ongoing event, while keeping it consistent with the core semantic of pre_ET.event_title.
    - The adjusted event_title must remain local, stable, and not expand to a global/session-level scope; it should only refine the original title to fit cumulative information of the same unit.
    """
)


def _cleanup_dist() -> None:
    """Best-effort destroy any torch distributed/NCCL process group at exit."""
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


atexit.register(_cleanup_dist)


def _build_vllm_client(base_url: str = VLLM_BASE_URL) -> OpenAI:
    """OpenAI-compatible client pointing to the local vLLM server."""
    return OpenAI(api_key=VLLM_API_KEY, base_url=base_url)


def _build_openai_client() -> OpenAI:
    """External OpenAI client — used only for embeddings."""
    kwargs: Dict[str, Any] = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    return OpenAI(**kwargs)

def _parse_json_safe(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        return json.loads(text)
    except Exception:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                pass
        raise ValueError("Model did not return valid JSON.")


def _raw_snippet(text: Any, limit: int = 200) -> str:
    snippet = str(text or "").replace("\n", " ").strip()
    if len(snippet) > limit:
        return snippet[:limit] + "..."
    return snippet


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
        "component": "memory_aio",
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


def _call_openai_json_with_retry(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: Optional[float] = None,
    validator: Optional[Any] = None,
    call_name: str = "json_call",
    vllm_base_url: str = VLLM_BASE_URL,
) -> Tuple[Optional[Dict[str, Any]], str, str]:
    client = _build_openai_client()
    # Qwen-VL handles visual understanding; all structured text calls in this
    # pipeline use the explicitly configured DeepSeek-compatible text model.
    text_model = (
        os.getenv("EMMA_DEEPSEEK_MODEL")
        or os.getenv("DEEPSEEK_MODEL")
        or os.getenv("MEMORY_TEXT_MODEL")
        or MEMORY_TEXT_MODEL
    )
    last_error = ""
    last_raw = ""

    for attempt in range(1, JSON_API_MAX_ATTEMPTS + 1):
        retry_prompt = prompt
        if attempt > 1:
            retry_prompt = (
                f"{prompt}\n\n"
                "Your previous response could not be consumed by the pipeline. "
                "Return valid JSON only, matching the requested schema exactly. "
                "Do not include explanations, markdown fences, comments, or extra keys."
            )

        kwargs: Dict[str, Any] = {
            "model": text_model,
            "messages": [{"role": "user", "content": retry_prompt}],
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature

        try:
            resp = client.chat.completions.create(**kwargs)
            _record_token_usage(resp, model=text_model, call_name=call_name)
            raw = (resp.choices[0].message.content or "").strip()
            last_raw = raw
            data = _parse_json_safe(raw)
            if validator is not None:
                validator(data)
            return data, raw, ""
        except Exception as e:
            last_error = str(e)
            print(
                f"[warn] {call_name} failed "
                f"(attempt {attempt}/{JSON_API_MAX_ATTEMPTS}): {last_error}; raw={_raw_snippet(last_raw or e)}"
            )

    return None, last_raw, last_error


def _validate_summary_payload(data: Dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise ValueError("summary payload is not a dict")
    required = ("topic_label", "full_narrative", "semantic_inference")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"summary payload missing keys: {missing}")




def _validate_event_link_payload(data: Dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise ValueError("event link payload is not a dict")
    if "is_same_event" not in data:
        raise ValueError("event link payload missing is_same_event")
    if not isinstance(data.get("is_same_event"), bool):
        raise ValueError("event link payload is_same_event must be bool")
    split_raw = data.get("split_point")
    if split_raw is None:
        split_raw = data.get("split_points", [])
    if split_raw is not None and not isinstance(split_raw, list):
        raise ValueError("event link payload split_point(s) must be list")


def _validate_chapter_payload(data: Dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise ValueError("chapter payload is not a dict")
    chapters = data.get("chapters")
    if not isinstance(chapters, list) or not chapters:
        raise ValueError("chapter payload missing non-empty chapters list")
    seg_conf = str(data.get("segmentation_confidence", "")).strip().lower()
    if seg_conf not in {"high", "medium", "low"}:
        raise ValueError(f"invalid segmentation_confidence: {seg_conf}")
    for idx, ch in enumerate(chapters, start=1):
        if not isinstance(ch, dict):
            raise ValueError(f"chapter {idx} is not a dict")
        missing = [k for k in ("start_time", "title", "summary") if k not in ch]
        if missing:
            raise ValueError(f"chapter {idx} missing keys: {missing}")


def _normalize_chapters_start_only(
    chapters: List[Dict[str, Any]],
    video_duration_s: float,
) -> List[Dict[str, Any]]:
    """Normalize and keep chapter fields in start-time-only format."""
    normalized: List[Dict[str, Any]] = []
    for idx, ch in enumerate(chapters, start=1):
        if not isinstance(ch, dict):
            continue
        st_raw = str(ch.get("start_time", "00:00:00"))
        st_s = _hhmmss_to_seconds(st_raw)
        st_s = max(0.0, min(st_s, max(0.0, float(video_duration_s))))
        title = str(ch.get("title", "")).strip() or f"Chapter {idx}"
        summary = str(ch.get("summary", "")).strip()
        normalized.append({
            "chapter_id": int(ch.get("chapter_id", idx) or idx),
            "start_time": _seconds_to_hhmmss(st_s),
            "title": title,
            "summary": summary,
        })

    if not normalized:
        return [{
            "chapter_id": 1,
            "start_time": _seconds_to_hhmmss(0.0),
            "title": "Full Video",
            "summary": "Auto-generated chapter due to missing ASR/segmentation",
        }]

    normalized.sort(key=lambda c: _hhmmss_to_seconds(str(c.get("start_time", "00:00:00"))))

    deduped: List[Dict[str, Any]] = []
    for ch in normalized:
        st = _hhmmss_to_seconds(str(ch.get("start_time", "00:00:00")))
        if deduped:
            prev_st = _hhmmss_to_seconds(str(deduped[-1].get("start_time", "00:00:00")))
            if abs(st - prev_st) <= 1e-6:
                # Keep the richer chapter content if starts are duplicated.
                prev_summary = str(deduped[-1].get("summary", "")).strip()
                curr_summary = str(ch.get("summary", "")).strip()
                if len(curr_summary) > len(prev_summary):
                    deduped[-1] = ch
                continue
        deduped.append(ch)

    if _hhmmss_to_seconds(str(deduped[0].get("start_time", "00:00:00"))) > 0.0:
        deduped[0]["start_time"] = _seconds_to_hhmmss(0.0)

    for i, ch in enumerate(deduped, start=1):
        ch["chapter_id"] = i

    return deduped


def _validate_event_table_payload(data: Dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise ValueError("event table payload is not a dict")
    required = {"event_identity", "event_summary", "entities", "open_questions", "delta"}
    if not required.issubset(set(data.keys())):
        missing = sorted(required - set(data.keys()))
        raise ValueError(f"event table payload missing keys: {missing}")
    identity = str(data.get("event_identity") or data.get("event_intent") or "").strip()
    if not identity:
        raise ValueError("event table payload missing event_identity")
    if not isinstance(data.get("entities"), list):
        raise ValueError("event table payload entities must be list")
    if not isinstance(data.get("open_questions"), list):
        raise ValueError("event table payload open_questions must be list")


def _fill_prompt_template(template: str, **kwargs: Any) -> str:
    """Fill {name} placeholders without interpreting unrelated JSON braces."""
    out = str(template)
    for key, value in kwargs.items():
        out = out.replace("{" + str(key) + "}", str(value))
    return out


def _resolve_video_path(video_path: Path) -> Path:
    """Resolve migrated dataset paths by searching common roots and name variants."""
    if video_path.exists() and video_path.is_file():
        return video_path

    suffix = video_path.suffix or ".mp4"
    stem = video_path.stem if video_path.suffix else video_path.name
    base_name = video_path.name if video_path.suffix else f"{stem}{suffix}"

    stem_variants = [stem]
    if stem.startswith("_"):
        stem_variants.append(stem.lstrip("_"))
    else:
        stem_variants.append(f"_{stem}")

    name_variants: List[str] = [base_name]
    for s in stem_variants:
        n = f"{s}{suffix}"
        if n not in name_variants:
            name_variants.append(n)

    parent = video_path.parent
    if parent.exists() and parent.is_dir():
        for name in name_variants:
            cand = parent / name
            if cand.exists() and cand.is_file():
                print(f"[path] resolved video path: {video_path} -> {cand}")
                return cand

    raise FileNotFoundError(
        f"Input video not found: {video_path}. Checked common dataset roots and filename variants: {name_variants}"
    )


def _resolve_qwen_checkpoint(qwen_checkpoint: str | Path) -> str:
    """Resolve Qwen checkpoint: explicit arg > QWEN_CHECKPOINT env > error."""
    raw = Path(str(qwen_checkpoint)) if qwen_checkpoint else None
    if raw and raw.exists():
        return str(raw)
    cfg = Path(QWEN_CHECKPOINT) if QWEN_CHECKPOINT else None
    if cfg and cfg.exists():
        return str(cfg)
    checked = [str(p) for p in [raw, cfg] if p]
    raise FileNotFoundError(
        f"Qwen checkpoint not found. Tried: {checked}. "
        "Set QWEN_CHECKPOINT in memory/.env or pass --qwen-checkpoint."
    )


def _resolve_whisper_model_dir(whisper_model_dir: str | Path) -> str:
    """Resolve Whisper model dir: explicit arg > WHISPER_MODEL_DIR env > error."""
    raw = Path(str(whisper_model_dir)) if whisper_model_dir else None
    if raw and raw.exists():
        return str(raw)
    cfg = Path(WHISPER_MODEL_DIR) if WHISPER_MODEL_DIR else None
    if cfg and cfg.exists():
        return str(cfg)
    checked = [str(p) for p in [raw, cfg] if p]
    raise FileNotFoundError(
        f"Whisper model dir not found. Tried: {checked}. "
        "Set WHISPER_MODEL_DIR in memory/.env or pass --whisper-model-dir."
    )


def _embed_text_openai(content: str) -> List[float]:
    client = _build_openai_client()
    for attempt in range(5):
        try:
            resp = client.embeddings.create(model="text-embedding-3-large", input=content)
            return resp.data[0].embedding
        except Exception as e:
            if attempt == 4:
                print(f"[warn] embedding failed after 5 attempts, returning empty: {e}")
                return []
            wait = 2 ** attempt
            print(f"[warn] embedding attempt {attempt+1}/5 failed ({e}), retry in {wait}s")
            time.sleep(wait)


def _append_jsonl(path: Path, item: Dict[str, Any]) -> None:
    def _drop_embeddings(obj: Any) -> Any:
        if isinstance(obj, dict):
            out: Dict[str, Any] = {}
            for k, v in obj.items():
                if k == "embedding":
                    continue
                out[k] = _drop_embeddings(v)
            return out
        if isinstance(obj, list):
            return [_drop_embeddings(x) for x in obj]
        return obj

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _drop_embeddings(item) if path.name.endswith(".stream.jsonl") else item
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


# -------------------- Video helpers --------------------

def _get_clip_duration(clip_path: Path) -> float:
    import shutil
    import subprocess
    if shutil.which("ffprobe") is None:
        raise FileNotFoundError("ffprobe not found in PATH. Please install ffmpeg.")
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(clip_path),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    out = res.stdout.decode().strip()
    try:
        return float(out)
    except Exception as e:
        raise RuntimeError(f"Failed to parse duration for {clip_path}: {out}") from e


def _mmss_to_seconds(ts: str) -> float:
    try:
        parts = ts.split(":")
        if len(parts) != 2:
            return 0.0
        m, s = parts
        return float(m) * 60 + float(s)
    except Exception:
        return 0.0

def _hhmmss_to_seconds(ts: str) -> float:
    try:
        parts = ts.split(":")
        if len(parts) == 2:
            return _mmss_to_seconds(ts)
        if len(parts) != 3:
            return 0.0
        h, m, s = parts
        return float(h) * 3600 + float(m) * 60 + float(s)
    except Exception:
        return 0.0

def _seconds_to_hhmmss(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def _normalize_to_hhmmss(ts: str) -> str:
    return _seconds_to_hhmmss(_hhmmss_to_seconds(ts))


# -------------------- Session / memory helpers --------------------

def _summarize_session(details: List[str]) -> Dict[str, str]:
    detail_list = "\n".join(f"- {d}" for d in details)
    prompt = _fill_prompt_template(SESSION_SUMMARY_PROMPT_TEMPLATE, detail_list=detail_list)
    data, raw, _ = _call_openai_json_with_retry(
        model=LLM_MODEL,
        prompt=prompt,
        max_tokens=2048,
        validator=_validate_summary_payload,
        call_name="summarize_session",
    )
    if data is None:
        return {
            "topic_label": "",
            "full_narrative": raw,
            "semantic_inference": "",
        }
    return {
        "topic_label": str(data.get("topic_label", "")).strip(),
        "full_narrative": str(data.get("full_narrative", "")).strip(),
        "semantic_inference": str(data.get("semantic_inference", "")).strip(),
    }





def _build_event_link_prompt(
    pre_et: Optional[Dict[str, Any]],
    anchor_stm: ShortTermMemory,
    pending_stm: ShortTermMemory,
    segmentation_confidence: str,
    chapters: List[Dict[str, Any]],
    candidate_source: str,
) -> str:
    pre_et_payload = _event_link_pre_et_payload(pre_et)
    pre_et_text = "null" if not pre_et_payload else json.dumps(pre_et_payload, ensure_ascii=False, indent=2)
    seg_conf = str(segmentation_confidence or "").strip().lower() or "unknown"

    chapter_context_lines: List[str] = [f"segmentation_confidence: {seg_conf}"]
    if chapters:
        chapter_context_lines.append(
            "chapter_list:\n" + json.dumps(chapters, ensure_ascii=False, indent=2)
        )
    else:
        chapter_context_lines.append("chapter_list: []")

    chapter_context_text = "\n\n".join(chapter_context_lines)

    base = EVENT_LINK_WITH_ET_PROMPT_AIO
    base = base.replace("{candidate_source}", str(candidate_source or "").strip())
    base = base.replace("{asr_confidence}", seg_conf)
    base = base.replace("{chapter_context}", chapter_context_text)
    base = base.replace("{pre_et}", pre_et_text)
    base = base.replace("{anchor_summary}", anchor_stm.visual_summary)
    base = base.replace("{anchor_caption}", anchor_stm.detailed_caption)
    base = base.replace("{anchor_ASR}", anchor_stm.ASR)
    base = base.replace("{pending_summary}", pending_stm.visual_summary)
    base = base.replace("{pending_caption}", pending_stm.detailed_caption)
    base = base.replace("{pending_ASR}", pending_stm.ASR)
    return base


def _event_link_pre_et_payload(pre_et: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(pre_et, dict):
        return None
    return {
        "event_identity": pre_et.get("event_identity") or pre_et.get("event_intent", ""),
        "event_summary": pre_et.get("event_summary", ""),
        "entities": pre_et.get("entities") or [],
    }


def _event_link_current_segments_payload(
    anchor_stm: ShortTermMemory,
    pending_stm: ShortTermMemory,
) -> Dict[str, Any]:
    return {
        "anchor_segment": {
            "clip_id": str(anchor_stm.id),
            "time_range": [anchor_stm.time_range[0], anchor_stm.time_range[1]],
            "summary": anchor_stm.visual_summary,
            "caption": anchor_stm.detailed_caption,
            "asr": anchor_stm.ASR,
        },
        "pending_segment": {
            "clip_id": str(pending_stm.id),
            "time_range": [pending_stm.time_range[0], pending_stm.time_range[1]],
            "summary": pending_stm.visual_summary,
            "caption": pending_stm.detailed_caption,
            "asr": pending_stm.ASR,
        },
    }


def _parse_split_points(data: Dict[str, Any]) -> List[Dict[str, str]]:
    points: List[Dict[str, str]] = []
    if not isinstance(data, dict):
        return points
    raw = data.get("split_point")
    if raw is None:
        raw = data.get("split_points", [])
    if not isinstance(raw, list):
        return points
    for item in raw:
        if not isinstance(item, dict):
            continue
        t = str(item.get("t", "")).strip()
        reason = str(item.get("reason", "")).strip()
        if t:
            points.append({"t": t, "reason": reason})
    return points


def _format_et_shift_candidate(delta: str, evidence: str) -> str:
    raw = str(delta or "").strip()
    if not raw:
        return f"ET shift:SHIFT: unknown -> unknown | evidence={evidence}."

    upper = raw.upper()
    if upper.startswith("SHIFT:"):
        body = raw
    else:
        body = f"SHIFT: {raw}"

    marker = "| evidence="
    idx = body.lower().find(marker)
    if idx != -1:
        body = body[:idx].rstrip()

    if body.endswith("."):
        body = body[:-1].rstrip()

    return f"ET shift:{body} | evidence={evidence}."


def _is_initialization_shift(delta: str) -> bool:
    """Detect SHIFT deltas that are initialization-like (null/none/empty -> event)."""
    raw = str(delta or "").strip()
    if not raw:
        return False
    upper = raw.upper()
    if not upper.startswith("SHIFT:"):
        return False
    body = raw.split(":", 1)[1].strip() if ":" in raw else ""
    if "->" not in body:
        return False
    lhs = body.split("->", 1)[0].strip().strip(":").strip()
    if not lhs:
        return True
    lhs_token = re.sub(r"[^A-Z0-9]+", "", lhs.upper())
    init_tokens = {
        "NONE", "NULL", "NIL", "NA", "NAN", "UNKNOWN", "UNSET", "EMPTY",
        "INITIAL", "INITIALSTATE", "INIT", "NOEVENT",
    }
    return lhs_token in init_tokens


def _judge_event_with_et(
    pre_et: Optional[Dict[str, Any]],
    anchor_stm: ShortTermMemory,
    pending_stm: ShortTermMemory,
    segmentation_confidence: str,
    chapters: List[Dict[str, Any]],
    candidate_source: str,
    log_path: Optional[Path] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[Dict[str, str]]]:
    pre_et_payload = _event_link_pre_et_payload(pre_et)
    current_segments_payload = _event_link_current_segments_payload(anchor_stm, pending_stm)
    prompt = _build_event_link_prompt(
        pre_et,
        anchor_stm,
        pending_stm,
        segmentation_confidence,
        chapters,
        candidate_source,
    )
    data, content, err = _call_openai_json_with_retry(
        model=LLM_MODEL,
        prompt=prompt,
        max_tokens=256,
        validator=_validate_event_link_payload,
        call_name="judge_event_with_et",
    )
    if data is None:
        if log_path:
            _append_jsonl(log_path, {
                "ts": datetime.utcnow().isoformat() + "Z",
                "meta": meta or {},
                "candidate_source": candidate_source,
                "pre_ET": pre_et_payload,
                "current_segments": current_segments_payload,
                "raw_response": content,
                "error": err or "judge_event_with_et failed after retries",
            })
        print(f"[warn] event_link fallback after retries, raw={content}")
        return False, []
    try:
        if log_path:
            _append_jsonl(log_path, {
                "ts": datetime.utcnow().isoformat() + "Z",
                "meta": meta or {},
                "candidate_source": candidate_source,
                "pre_ET": pre_et_payload,
                "current_segments": current_segments_payload,
                "raw_response": content,
                "parsed": data,
            })
        return bool(data.get("is_same_event")), _parse_split_points(data)
    except Exception as e:
        if log_path:
            _append_jsonl(log_path, {
                "ts": datetime.utcnow().isoformat() + "Z",
                "meta": meta or {},
                "candidate_source": candidate_source,
                "pre_ET": pre_et_payload,
                "current_segments": current_segments_payload,
                "raw_response": content,
                "error": str(e),
            })
        print(f"[warn] event_link parse failed, raw={content}")
        return False, []


def _stm_to_event_payload(stm: ShortTermMemory) -> Dict[str, Any]:
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
    if pre_et and isinstance(pre_et, dict):
        return {
            "event_identity": pre_et.get("event_identity") or pre_et.get("event_intent") or identity,
            "event_summary": pre_et.get("event_summary") or summary,
            "entities": pre_et.get("entities") or [],
            "open_questions": pre_et.get("open_questions") or [],
            "delta": "No update due to parse failure.",
        }
    return {
        "event_identity": identity,
        "event_summary": summary,
        "entities": [],
        "open_questions": [],
        "delta": "Initialize event from current clip.",
    }


def _coerce_event_table(data: Any, pre_et: Optional[Dict[str, Any]], stm: ShortTermMemory) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return _fallback_event_table(pre_et, stm)
    required = {"event_summary", "entities", "open_questions", "delta"}
    if not required.issubset(set(data.keys())):
        return _fallback_event_table(pre_et, stm)
    identity = str(data.get("event_identity") or data.get("event_intent") or "").strip()
    if not identity:
        return _fallback_event_table(pre_et, stm)
    if not isinstance(data.get("entities"), list) or not isinstance(data.get("open_questions"), list):
        return _fallback_event_table(pre_et, stm)
    return {
        "event_identity": identity,
        "event_summary": str(data.get("event_summary", "")).strip(),
        "entities": data.get("entities") or [],
        "open_questions": data.get("open_questions") or [],
        "delta": str(data.get("delta", "")).strip(),
    }


def _update_event_table(
    pre_et: Optional[Dict[str, Any]],
    stm: ShortTermMemory,
    force_continue: bool = False,
) -> Dict[str, Any]:
    pre_et_text = "null" if not pre_et else json.dumps(pre_et, ensure_ascii=False, indent=2)
    stm_text = json.dumps(_stm_to_event_payload(stm), ensure_ascii=False, indent=2)
    base_prompt = UPDATE_EVENT_TABLE_PROMPT_FORCE_CONTINUE if force_continue else UPDATE_EVENT_TABLE_PROMPT
    prompt = f"{base_prompt}\n\npre_ET:\n{pre_et_text}\n\nSTM_k:\n{stm_text}"
    data, content, err = _call_openai_json_with_retry(
        model=LLM_MODEL,
        prompt=prompt,
        max_tokens=1024,
        validator=_validate_event_table_payload,
        call_name="update_event_table",
    )
    if data is None:
        print(f"[warn] update_event_table fallback after retries: {err}; raw={_raw_snippet(content)}")
        return _fallback_event_table(pre_et, stm)
    return _coerce_event_table(data, pre_et, stm)


# -------------------- Chapter segmentation --------------------

def _segment_chapters(cleaned_asr_segments: List[Dict[str, str]], video_duration_s: float) -> Dict[str, Any]:
    chapters_only_schema = (
        "You must return JSON only in the following schema:\n"
        "{\n"
        "  \"segmentation_confidence\": string,  // one of: high, medium, low\n"
        "  \"chapters\": [\n"
        "    {\n"
        "      \"chapter_id\": number,\n"
        "      \"start_time\": \"HH:MM:SS\",\n"
        "      \"title\": string,\n"
        "      \"summary\": string\n"
        "    }\n"
        "  ]\n"
        "}\n"
    )
    transcript_lines = []
    for seg in cleaned_asr_segments:
        st = _normalize_to_hhmmss(seg.get("start", "00:00"))
        text = str(seg.get("text", "")).strip()
        transcript_lines.append(f"[{st}] {text}")

    if not transcript_lines:
        raise RuntimeError("ASR chaptering aborted: no ASR transcript lines available for chapter segmentation.")

    prompt = f"{CHAPTER_SEGMENT_PROMPT_JSON}\n\n{chapters_only_schema}\n\n" + "\n".join(transcript_lines)
    chapter_model = LLM_MODEL
    data, _, err = _call_openai_json_with_retry(
        model=chapter_model,
        prompt=prompt,
        max_tokens=8000,
        temperature=0,
        validator=_validate_chapter_payload,
        call_name="segment_chapters",
    )
    if data is not None:
        seg_conf = str(data.get("segmentation_confidence", "")).strip().lower() or "unknown"
        chapters = _normalize_chapters_start_only(data.get("chapters") or [], video_duration_s)
        return {"chapters": chapters, "segmentation_confidence": seg_conf}
    raise RuntimeError(
        "ASR chaptering failed after "
        f"{JSON_API_MAX_ATTEMPTS} attempts; aborting this task. "
        f"error={err}; asr_segments={len(cleaned_asr_segments)}"
    )


def _should_fallback_to_legacy(
    asr_segments: List[Dict[str, str]],
    cleaned_asr_segments: List[Dict[str, str]],
    chapters: List[Dict[str, Any]],
    video_duration_s: float,
    segmentation_confidence: Optional[str] = None,
) -> Tuple[bool, str]:
    try:
        if segmentation_confidence and segmentation_confidence.lower() in ("low"):
            return True, f"chapter segmentation_confidence {segmentation_confidence.lower()}"

        text_len = sum(len(str(seg.get("text", ""))) for seg in cleaned_asr_segments)
        if not asr_segments or not cleaned_asr_segments or text_len < 200:
            return True, "ASR empty or too short"

        if not chapters:
            return True, "chaptering returned empty"

        durations = []
        titles_norm: List[str] = []
        for idx, ch in enumerate(chapters):
            st = _hhmmss_to_seconds(str(ch.get("start_time", "00:00:00")))
            if idx + 1 < len(chapters):
                ed = _hhmmss_to_seconds(str(chapters[idx + 1].get("start_time", "00:00:00")))
            else:
                ed = video_duration_s
            st = max(0.0, min(st, video_duration_s))
            ed = max(st, min(ed, video_duration_s))
            durations.append(ed - st)
            title = str(ch.get("title", "")).strip().lower()
            titles_norm.append(" ".join(title.split()))

        if len(chapters) == 1 and video_duration_s >= 600:
            return True, "single chapter for long video"

        # 注释掉章节过长的回退判断
        # if durations and max(durations) > 900:
        #     return True, "chapter too long"

        if titles_norm:
            from collections import Counter
            ctr = Counter(titles_norm)
            most_common = ctr.most_common(1)[0][1]
            if most_common / max(len(titles_norm), 1) >= 0.70:
                return True, "chapter titles repetitive"

    except Exception as e:
        return True, f"chapter validation error: {e}"

    return False, "ok"


def _collect_asr_boundaries(cleaned_asr_segments: List[Dict[str, str]]) -> List[float]:
    boundaries: List[float] = []
    for seg in cleaned_asr_segments:
        st = _hhmmss_to_seconds(seg.get("start", "00:00:00"))
        ed = _hhmmss_to_seconds(seg.get("end", "00:00:00"))
        if st >= 0:
            boundaries.append(st)
        if ed >= 0:
            boundaries.append(ed)
    return sorted(set(boundaries))


def _collect_chapter_boundaries(chapters: List[Dict[str, Any]]) -> List[float]:
    boundaries: List[float] = []
    for idx, ch in enumerate(chapters):
        if idx == 0:
            continue
        st = _hhmmss_to_seconds(str(ch.get("start_time", "00:00:00")))
        if st >= 0:
            boundaries.append(st)
    return sorted(set(boundaries))


def _chapter_time_span(
    chapters: List[Dict[str, Any]],
    index: int,
    video_duration_s: float,
) -> Tuple[float, float]:
    """Derive chapter [start, end) from start-only chapter list."""
    start_s = _hhmmss_to_seconds(str(chapters[index].get("start_time", "00:00:00")))
    if index + 1 < len(chapters):
        end_s = _hhmmss_to_seconds(str(chapters[index + 1].get("start_time", "00:00:00")))
    else:
        end_s = max(0.0, float(video_duration_s))
    start_s = max(0.0, min(start_s, max(0.0, float(video_duration_s))))
    end_s = max(start_s, min(end_s, max(0.0, float(video_duration_s))))
    return start_s, end_s


def make_asr_aligned_segments(
    start_s: float,
    end_s: float,
    boundaries: List[float],
    target_len_s: float = 30.0,
    min_len_s: float = 20.0,
    max_len_s: float = 40.0,
    snap_window_s: float = 10.0,
) -> List[Dict[str, float]]:
    segments: List[Dict[str, float]] = []
    cur = start_s
    while cur < end_s - 1e-6:
        remaining = end_s - cur
        if remaining <= 5.0:
            if segments:
                segments[-1]["end_s"] = end_s
            else:
                segments.append({"segment_id": 1, "start_s": start_s, "end_s": end_s})
            break

        proposed = min(end_s, cur + target_len_s)
        candidates: List[Tuple[float, float]] = []  # (distance, boundary)
        for b in boundaries:
            if b <= cur or b >= end_s:
                continue
            seg_len = b - cur
            if seg_len < min_len_s or seg_len > max_len_s:
                continue
            if abs(b - proposed) <= snap_window_s:
                candidates.append((abs(b - proposed), b))

        chosen = proposed
        if candidates:
            candidates.sort(key=lambda x: x[0])
            chosen = candidates[0][1]

        seg_end = min(end_s, chosen)
        if seg_end - cur < min_len_s and remaining > min_len_s:
            seg_end = min(end_s, cur + min_len_s)
        if seg_end - cur > max_len_s:
            seg_end = min(end_s, cur + max_len_s)

        if seg_end - cur < 5.0 and segments:
            segments[-1]["end_s"] = end_s
            break

        segments.append({"segment_id": len(segments) + 1, "start_s": cur, "end_s": seg_end})
        cur = seg_end

    return segments

def _get_video_duration(path: Path) -> float:
    return _get_clip_duration(path)

def _extract_video_subclip(
    video_path: Path,
    out_path: Path,
    start_s: float,
    end_s: float,
    timeout_s: int = 180,
) -> Path:
    import shutil
    import subprocess
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg not found in PATH. Please install ffmpeg.")
    start_s = max(0.0, float(start_s))
    end_s = max(start_s, float(end_s))
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_s:.3f}",
        "-to", f"{end_s:.3f}",
        "-i", str(video_path),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        str(out_path),
    ]
    try:
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=int(timeout_s),
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"ffmpeg subclip timeout after {timeout_s}s: [{start_s:.3f}, {end_s:.3f}] {video_path}"
        ) from e
    return out_path

def _asr_for_range(asr_segments: List[Dict[str, str]], start_s: float, end_s: float) -> List[str]:
    lines: List[str] = []
    for seg in asr_segments:
        seg_start = _hhmmss_to_seconds(seg.get("start", "00:00:00"))
        seg_end = _hhmmss_to_seconds(seg.get("end", "00:00:00"))
        # Use half-open interval [start_s, end_s) to avoid boundary duplication
        if seg_end <= start_s or seg_start >= end_s:
            continue
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        st_norm = _normalize_to_hhmmss(seg.get("start", "00:00:00"))
        ed_norm = _normalize_to_hhmmss(seg.get("end", "00:00:00"))
        lines.append(f"[{st_norm}-{ed_norm}]:{text}")
    return lines

def _merge_short_segments(segments: List[Dict[str, Any]], min_len_s: float, chapter_start_s: float, chapter_end_s: float) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for seg in segments:
        st_s = max(chapter_start_s, float(seg.get("start_s", 0.0)))
        ed_s = min(chapter_end_s, float(seg.get("end_s", st_s)))
        if ed_s < st_s:
            st_s, ed_s = ed_s, ed_s
        dur = ed_s - st_s
        if dur < min_len_s and merged:
            merged[-1]["end_s"] = max(merged[-1]["end_s"], ed_s)
        else:
            merged.append({"segment_id": seg.get("segment_id"), "start_s": st_s, "end_s": ed_s})
    return merged
# -------------------- Offline Qwen-VL inference --------------------

def _offline_vu(
    clip_path: Path,
    prompt: str,
    vllm_client: OpenAI,
) -> Dict[str, Any]:
    """Call vLLM server for vision-language inference on a video clip."""
    import urllib.error
    import urllib.request

    def _vlm_endpoint_candidates() -> List[str]:
        base = str(getattr(vllm_client, "base_url", "") or VLLM_BASE_URL).rstrip("/")
        candidates = []
        if base.endswith("/chat/completions"):
            candidates.append(base)
        else:
            candidates.append(base + "/chat/completions")
            if base.endswith("/v1"):
                candidates.append(base[:-3].rstrip("/") + "/chat/completions")
        out = []
        seen = set()
        for url in candidates:
            if url and url not in seen:
                out.append(url)
                seen.add(url)
        return out

    def _post_vlm_json(retry_prompt: str) -> str:
        body = json.dumps({
            "model": LLM_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": f"file://{clip_path.resolve()}"}},
                    {"type": "text", "text": retry_prompt},
                ],
            }],
            "max_tokens": 512,
            "temperature": 0.4,
        }).encode("utf-8")

        last_error = ""
        for url in _vlm_endpoint_candidates():
            req = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {VLLM_API_KEY}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=180) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return str(payload["choices"][0]["message"]["content"]).strip()
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = f"{url} -> HTTP {exc.code}: {detail}"
                if exc.code == 404:
                    continue
                raise RuntimeError(last_error) from exc
            except Exception as exc:
                last_error = f"{url} -> {type(exc).__name__}: {exc}"
                raise RuntimeError(last_error) from exc
        raise RuntimeError(last_error or "no VLM endpoint candidates")

    max_retries = 4
    generated_text = ""
    for attempt in range(max_retries + 1):
        try:
            retry_prompt = prompt if attempt == 0 else (
                f"{prompt}\n\nPlease strictly output valid JSON only. Do not include any extra text."
            )
            generated_text = _post_vlm_json(retry_prompt)
            return _parse_json_safe(generated_text)
        except Exception as e:
            snippet = (generated_text or str(e)).replace("\n", " ")
            print(f"[warn] VU JSON parse/generation failed (attempt {attempt+1}/{max_retries+1}): {snippet}")
            if attempt < max_retries:
                continue
            return {}


# -------------------- Main pipeline --------------------

def run_video_memory_pipeline_off(
    video_path: str | Path,
    work_root: str | Path,
    chunk_seconds: int = 30,
    prefix: str = "chunk",
    whisper_model_dir: Path | str = WHISPER_MODEL_DIR,
    whisper_device: Optional[str] = None,
    whisper_language: Optional[str] = None,
    whisper_batch_size: int = 1,
    require_precomputed_asr: bool = False,
    vllm_base_url: str = VLLM_BASE_URL,
    resume_from_stream: bool = True,
    cleanup: bool = True,
) -> Dict[str, Path]:
    video_path = _resolve_video_path(Path(video_path))
    if whisper_device is None:
        whisper_device = os.getenv("AIO_WHISPER_DEVICE") or "auto"
    whisper_model_dir = _resolve_whisper_model_dir(whisper_model_dir)
    print(f"[pipeline-off] start video_path={video_path} work_root={work_root} chunk_seconds={chunk_seconds}")
    work_root = Path(work_root)
    segments_dir = work_root / "segments"
    outputs_dir = work_root / "outputs"
    segments_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    event_link_log_path = outputs_dir / "event_link_logs.jsonl"
    st_stream_path = outputs_dir / "short_term.stream.jsonl"
    mt_stream_path = outputs_dir / "medium_term.stream.jsonl"
    et_stream_path = outputs_dir / "event_table.stream.jsonl"
    resume_active = bool(resume_from_stream) and any(
        p.exists() and p.is_file() and p.stat().st_size > 0
        for p in (st_stream_path, mt_stream_path, et_stream_path)
    )

    if resume_active:
        print("[resume] detected existing stream state; attempting clip-level recovery")
    else:
        with open(event_link_log_path, "w", encoding="utf-8") as f:
            f.write("")
        for p in (st_stream_path, mt_stream_path, et_stream_path):
            with open(p, "w", encoding="utf-8") as f:
                f.write("")

    # Ensure files exist for append mode even when partially resumed.
    for p in (st_stream_path, mt_stream_path, et_stream_path, event_link_log_path):
        if not p.exists():
            with open(p, "w", encoding="utf-8") as f:
                f.write("")

    # The public runner writes ASR once via faster-whisper/CTranslate2.
    asr_save_path = outputs_dir / "asr_segments.json"
    asr_tmp_work_dir = work_root / "asr_tmp"
    asr_segments: List[Dict[str, str]] = []
    reused_asr = False
    if asr_save_path.exists() and asr_save_path.is_file():
        try:
            with open(asr_save_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if isinstance(cached, list) and cached:
                asr_segments = cached
                reused_asr = True
                print(f"[pipeline-off] Reusing existing ASR transcript: {asr_save_path} ({len(asr_segments)} segments)")
            elif isinstance(cached, list) and not cached:
                print(f"[warn] Existing ASR transcript is empty, rerunning ASR: {asr_save_path}")
            else:
                print(f"[warn] Existing ASR transcript is not a list, rerunning ASR: {asr_save_path}")
        except Exception as e:
            print(f"[warn] Failed to load existing ASR transcript, rerunning ASR: {e}")

    if require_precomputed_asr and not reused_asr:
        raise RuntimeError(
            f"ASR prerequisite missing or empty: {asr_save_path}. "
            "Run batch_ASR first and retry."
        )

    if not reused_asr:
        raise RuntimeError(
            "ASR transcript is missing. The public pipeline requires a "
            "precomputed outputs/asr_segments.json from ASR.py or "
            "run_memory.py."
        )

    try:
        video_duration_s = _get_video_duration(video_path)
    except Exception as e:
        print(f"[warn] ffprobe duration failed: {e}")
        video_duration_s = 0.0

    chapter_out_path = outputs_dir / "chapter_segmentation.json"
    chapter_json: Dict[str, Any] = {}
    reused_chapters = False
    if chapter_out_path.exists() and chapter_out_path.is_file():
        try:
            with open(chapter_out_path, "r", encoding="utf-8") as f:
                cached_chapters = json.load(f)
            if isinstance(cached_chapters, dict) and isinstance(cached_chapters.get("chapters"), list):
                chapter_json = cached_chapters
                reused_chapters = True
                print(
                    f"[chapters] reusing existing chapter segmentation: {chapter_out_path} "
                    f"({len(cached_chapters.get('chapters', []))} chapters)"
                )
            else:
                print(f"[warn] Existing chapter segmentation invalid, rerunning chaptering: {chapter_out_path}")
        except Exception as e:
            print(f"[warn] Failed to load existing chapter segmentation, rerunning chaptering: {e}")

    if not reused_chapters:
        chapter_json = _segment_chapters(asr_segments, video_duration_s)

    seg_conf = "unknown"
    if isinstance(chapter_json, dict):
        seg_conf = str(chapter_json.get("segmentation_confidence", "")).strip().lower() or "unknown"
    chapters = chapter_json.get("chapters", []) if isinstance(chapter_json, dict) else []
    chapters = _normalize_chapters_start_only(chapters, video_duration_s)
    if chapters:
        _, last_chapter_end_before = _chapter_time_span(chapters, len(chapters) - 1, video_duration_s)
        _, last_chapter_end_after = _chapter_time_span(chapters, len(chapters) - 1, video_duration_s)
    else:
        last_chapter_end_before = 0.0
        last_chapter_end_after = 0.0

    try:
        with open(chapter_out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "segmentation_confidence": seg_conf,
                    "chapters": chapters,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[chapters] saved to {chapter_out_path}")
    except Exception as save_err:
        print(f"[warn] failed to save chapter segmentation: {save_err}")

    boundaries = _collect_asr_boundaries(asr_segments)
    chapter_boundaries = _collect_chapter_boundaries(chapters)
    last_asr_ts = max(boundaries) if boundaries else 0.0
    max_ch_dur = 0.0
    if chapters:
        for i, _ in enumerate(chapters):
            st, ed = _chapter_time_span(chapters, i, video_duration_s)
            max_ch_dur = max(max_ch_dur, ed - st)
    print(f"[duration] video={video_duration_s:.3f}s last_asr={last_asr_ts:.3f}s last_ch_end_before={last_chapter_end_before:.3f}s last_ch_end_after={last_chapter_end_after:.3f}s max_ch_dur={max_ch_dur:.3f}s")
    print(f"[chapters] total={len(chapters)} seg_conf={seg_conf} asr_text_len={sum(len(str(seg.get('text',''))) for seg in asr_segments)}")
    fallback_needed, fallback_reason = _should_fallback_to_legacy(asr_segments, asr_segments, chapters, video_duration_s, seg_conf)
    print(f"[fallback-check] needed={fallback_needed} reason={fallback_reason}")
    if fallback_needed:
        print(f"[fallback] skip streaming; continue with unified ASR-aligned pipeline: {fallback_reason}")

    vllm_client = _build_vllm_client(vllm_base_url)
    print(f"[pipeline-off] vLLM server: {vllm_base_url}")
    short_terms: List[ShortTermMemory] = []
    stms_by_id: Dict[UUID, ShortTermMemory] = {}
    # asr_embedding_cache: Dict[UUID, List[float]] = {}
    medium_terms: List[MediumTermMemory] = []
    event_tables: List[Dict[str, Any]] = []

    session_clip_ids: List[UUID] = []
    session_details: List[str] = []
    prev_stm: Optional[ShortTermMemory] = None
    current_event_table: Optional[Dict[str, Any]] = None
    current_event_id: Optional[UUID] = None
    last_accepted_split_time_s = -1.0
    min_split_gap_s = float(chunk_seconds)
    pending_active = False
    pending_anchor_info: Optional[Dict[str, Any]] = None
    pending_stms: List[ShortTermMemory] = []
    pending_event_table: Optional[Dict[str, Any]] = None
    resume_embedding_cache: Dict[str, List[float]] = {}

    def _iter_jsonl(path: Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if not path.exists() or not path.is_file():
            return rows
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
        return rows

    def _safe_uuid(raw: Any) -> Optional[UUID]:
        try:
            return raw if isinstance(raw, UUID) else UUID(str(raw))
        except Exception:
            return None

    def _embed_with_resume_cache(text: str) -> List[float]:
        key = str(text or "").strip()
        if not key:
            return []
        if key in resume_embedding_cache:
            return resume_embedding_cache[key]
        emb = _embed_text_openai(key)
        resume_embedding_cache[key] = emb
        return emb

    def _stm_from_stream_row(d: Dict[str, Any]) -> Optional[ShortTermMemory]:
        try:
            sid = _safe_uuid(d.get("id"))
            tr = d.get("time_range", [0.0, 0.0])
            if sid is None or not isinstance(tr, (list, tuple)) or len(tr) != 2:
                return None
            st = float(tr[0])
            ed = float(tr[1])
            if ed < st:
                return None
            visual_summary = str(d.get("visual_summary", ""))
            detailed_caption = str(d.get("detailed_caption", ""))
            asr_text = str(d.get("ASR", ""))
            emb_raw = d.get("embedding", [])
            emb = emb_raw if isinstance(emb_raw, list) else []
            emb_clean = [float(x) for x in emb if isinstance(x, (int, float))]
            if not emb_clean:
                embed_text = " ".join([visual_summary, detailed_caption, asr_text]).strip() or visual_summary.strip()
                if embed_text:
                    try:
                        emb_clean = _embed_with_resume_cache(embed_text)
                    except Exception as e:
                        print(f"[warn] resume STM embedding backfill failed for clip_id={sid}: {e}")
            return ShortTermMemory(
                id=sid,
                video_source_path=str(d.get("video_source_path", video_path)),
                time_range=(st, ed),
                visual_summary=visual_summary,
                detailed_caption=detailed_caption,
                embedding=emb_clean,
                ASR=asr_text,
                environment=str(d.get("environment", "")),
                inferred_intent=str(d.get("inferred_intent", "")),
            )
        except Exception:
            return None

    def _mtm_from_stream_row(d: Dict[str, Any]) -> Optional[MediumTermMemory]:
        try:
            tid = _safe_uuid(d.get("task_id"))
            sp = d.get("time_span", [0.0, 0.0])
            if tid is None or not isinstance(sp, (list, tuple)) or len(sp) != 2:
                return None
            st = float(sp[0])
            ed = float(sp[1])
            if ed < st:
                return None
            child_raw = d.get("child_clip_ids", [])
            child_ids: List[UUID] = []
            if isinstance(child_raw, list):
                for x in child_raw:
                    uid = _safe_uuid(x)
                    if uid is not None:
                        child_ids.append(uid)
            topic = str(d.get("topic", "") or "event")
            narrative_summary = str(d.get("narrative_summary", ""))
            semantic_inference = str(d.get("semantic_inference", ""))
            emb_raw = d.get("embedding", [])
            emb = emb_raw if isinstance(emb_raw, list) else []
            emb_clean = [float(x) for x in emb if isinstance(x, (int, float))]
            if not emb_clean:
                mtm_embed_text = f"{narrative_summary}\n{semantic_inference}".strip()
                embed_text = mtm_embed_text if mtm_embed_text else narrative_summary
                if embed_text:
                    try:
                        emb_clean = _embed_with_resume_cache(embed_text)
                    except Exception as e:
                        print(f"[warn] resume MTM embedding backfill failed for task_id={tid}: {e}")
            return MediumTermMemory(
                task_id=tid,
                topic=topic,
                time_span=(st, ed),
                narrative_summary=narrative_summary,
                semantic_inference=semantic_inference,
                child_clip_ids=child_ids,
                embedding=emb_clean,
            )
        except Exception:
            return None

    def _record_stm(stm: ShortTermMemory) -> None:
        short_terms.append(stm)
        stms_by_id[stm.id] = stm
        _append_jsonl(st_stream_path, stm.to_dict())

    def _mtm_detail_from_stm(stm: ShortTermMemory) -> str:
        visual = str(stm.detailed_caption or "").strip()
        asr_text = str(stm.ASR or "").strip()
        environment = str(stm.environment or "").strip()
        env_line = f"\n[ENV] {environment}" if environment else ""
        if asr_text and asr_text != "无可用音频或转录失败":
            return f"[VISUAL] {visual}{env_line}\n[ASR] {asr_text}".strip()
        return f"[VISUAL] {visual}{env_line}".strip()

    def _finalize_session() -> None:
        if not session_clip_ids:
            return
        first_stm = stms_by_id[session_clip_ids[0]]
        last_stm = stms_by_id[session_clip_ids[-1]]
        topic = first_stm.visual_summary or "事件"
        time_span = (first_stm.time_range[0], last_stm.time_range[1])
        summary = _summarize_session(session_details)
        topic = summary.get("topic_label") or topic
        narrative = summary.get("full_narrative") or "\n".join(session_details)
        semantic_inference = summary.get("semantic_inference", "")
        mtm_embed_text = f"{narrative}\n{semantic_inference}".strip()
        mtm = MediumTermMemory(
            task_id=uuid4(),
            topic=topic,
            time_span=time_span,
            narrative_summary=narrative,
            semantic_inference=semantic_inference,
            child_clip_ids=session_clip_ids,
            embedding=_embed_text_openai(mtm_embed_text if mtm_embed_text else narrative),
        )
        medium_terms.append(mtm)
        _append_jsonl(mt_stream_path, mtm.to_dict())

    def _chapters_for_segment(start_s: float, end_s: float) -> List[Dict[str, Any]]:
        if not chapters:
            return []
        overlapping: List[Dict[str, Any]] = []
        for i, ch in enumerate(chapters):
            st, ed = _chapter_time_span(chapters, i, video_duration_s)
            if ed >= start_s and st <= end_s:
                enriched = dict(ch)
                enriched["start_time"] = _seconds_to_hhmmss(st)
                enriched["end_time"] = _seconds_to_hhmmss(ed)
                enriched["time_range"] = [
                    _seconds_to_hhmmss(st),
                    _seconds_to_hhmmss(ed),
                ]
                overlapping.append(enriched)
        return overlapping

    def _chapter_shift_text(boundary_t: float) -> str:
        if not chapters:
            return "ASR shift:unknown -> unknown | evidence=asr."
        eps = 1e-3
        next_idx = -1
        for i in range(1, len(chapters)):
            st = _hhmmss_to_seconds(str(chapters[i].get("start_time", "00:00:00")))
            if abs(st - boundary_t) <= 1.0 + eps:
                next_idx = i
                break
        if next_idx <= 0:
            for i in range(1, len(chapters)):
                st = _hhmmss_to_seconds(str(chapters[i].get("start_time", "00:00:00")))
                if st >= boundary_t - eps:
                    next_idx = i
                    break
        if next_idx <= 0:
            return "ASR shift:unknown -> unknown | evidence=asr."
        prev_ch = chapters[next_idx - 1]
        next_ch = chapters[next_idx]
        prev_summary = str(prev_ch.get("summary") or prev_ch.get("title") or "unknown").strip() or "unknown"
        next_summary = str(next_ch.get("summary") or next_ch.get("title") or "unknown").strip() or "unknown"
        return f"ASR shift:{prev_summary}->{next_summary} | evidence=asr."

    def _fixed_length_segments() -> List[Dict[str, float]]:
        step = max(1.0, float(chunk_seconds))
        segments: List[Dict[str, float]] = []
        start_t = 0.0
        while start_t < video_duration_s - 1e-6:
            end_t = min(video_duration_s, start_t + step)
            segments.append({"start_s": start_t, "end_s": end_t})
            start_t = end_t
        if not segments and video_duration_s > 0.0:
            segments.append({"start_s": 0.0, "end_s": video_duration_s})
        for idx, seg in enumerate(segments, start=1):
            seg["segment_id"] = idx
        return segments

    def _prepare_segments() -> List[Dict[str, float]]:
        asr_text_len = sum(len(str(seg.get("text", "")).strip()) for seg in asr_segments)
        if asr_text_len <= 0:
            print("[segments] no usable ASR text detected; falling back to fixed chunking")
            return _fixed_length_segments()

        segs = make_asr_aligned_segments(
            0.0,
            video_duration_s,
            boundaries,
            target_len_s=float(chunk_seconds),
            min_len_s=max(1.0, float(chunk_seconds) - 5.0),
            max_len_s=float(chunk_seconds) + 5.0,
            snap_window_s=10.0,
        )
        segs = _merge_short_segments(segs, min_len_s=5.0, chapter_start_s=0.0, chapter_end_s=video_duration_s)
        merged: List[Dict[str, float]] = []
        for seg in segs:
            st = float(seg.get("start_s", 0.0))
            ed = float(seg.get("end_s", st))
            asr_lines = _asr_for_range(asr_segments, st, ed)
            if not asr_lines and merged:
                merged[-1]["end_s"] = max(merged[-1]["end_s"], ed)
            else:
                merged.append({"start_s": st, "end_s": ed})
        for idx, seg in enumerate(merged, start=1):
            seg["segment_id"] = idx
        return merged

    def _is_segment_already_covered(start_t: float, end_t: float) -> bool:
        """Check whether existing STMs already cover this base segment interval."""
        if not short_terms:
            return False
        eps = 1e-3
        intervals: List[Tuple[float, float]] = []
        for stm in short_terms:
            st = float(stm.time_range[0])
            ed = float(stm.time_range[1])
            if ed <= start_t + eps or st >= end_t - eps:
                continue
            cs = max(start_t, st)
            ce = min(end_t, ed)
            if ce - cs > eps:
                intervals.append((cs, ce))

        if not intervals:
            return False

        intervals.sort(key=lambda x: x[0])
        merged: List[Tuple[float, float]] = []
        for st, ed in intervals:
            if not merged or st > merged[-1][1] + 0.05:
                merged.append((st, ed))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], ed))

        cursor = start_t
        for st, ed in merged:
            if st > cursor + 0.25:
                return False
            cursor = max(cursor, ed)
            if cursor >= end_t - 0.25:
                return True
        return cursor >= end_t - 0.25

    if resume_active:
        restored_st = 0
        restored_mt = 0
        restored_et = 0

        st_rows = _iter_jsonl(st_stream_path)
        parsed_st: List[ShortTermMemory] = []
        seen_st_ids: set[UUID] = set()
        for item in st_rows:
            stm = _stm_from_stream_row(item)
            if stm is None or stm.id in seen_st_ids:
                continue
            parsed_st.append(stm)
            seen_st_ids.add(stm.id)

        # If a crashed/buggy rerun appended fresh clips from t~0 at tail,
        # keep only the monotonic historical prefix and drop the rollback suffix.
        rollback_idx = -1
        prev_end = -1.0
        for idx, stm in enumerate(parsed_st):
            st_v = float(stm.time_range[0])
            ed_v = float(stm.time_range[1])
            if idx > 0 and st_v + 1e-3 < prev_end - 1.0:
                rollback_idx = idx
                break
            prev_end = max(prev_end, ed_v)

        rollback_trimmed = rollback_idx >= 0
        if rollback_trimmed:
            print(
                f"[resume] detected rollback-appended STM tail at row={rollback_idx + 1}; "
                "keeping historical prefix only"
            )
            parsed_st = parsed_st[:rollback_idx]

        for stm in parsed_st:
            short_terms.append(stm)
            stms_by_id[stm.id] = stm
            restored_st += 1

        short_terms.sort(key=lambda s: (float(s.time_range[0]), float(s.time_range[1]), str(s.id)))

        st_resume_end_s = max((float(s.time_range[1]) for s in short_terms), default=-1.0)
        valid_st_ids = set(stms_by_id.keys())

        seen_mtm_ids: set[UUID] = set()
        for item in _iter_jsonl(mt_stream_path):
            mtm = _mtm_from_stream_row(item)
            if mtm is None or mtm.task_id in seen_mtm_ids:
                continue
            if st_resume_end_s >= 0 and float(mtm.time_span[1]) > st_resume_end_s + 1.0:
                continue
            if mtm.child_clip_ids and any(cid not in valid_st_ids for cid in mtm.child_clip_ids):
                continue
            medium_terms.append(mtm)
            seen_mtm_ids.add(mtm.task_id)
            restored_mt += 1

        valid_mtm_ids = {str(m.task_id) for m in medium_terms}

        for item in _iter_jsonl(et_stream_path):
            cid = _safe_uuid(item.get("clip_id"))
            if cid is None or cid not in valid_st_ids:
                continue
            tr = item.get("time_range", [])
            if isinstance(tr, (list, tuple)) and len(tr) == 2 and st_resume_end_s >= 0:
                try:
                    if float(tr[1]) > st_resume_end_s + 1.0:
                        continue
                except Exception:
                    pass
            event_tables.append(item)
            restored_et += 1

        if rollback_trimmed:
            # Rewrite sanitized streams so future resumes are deterministic.
            with open(st_stream_path, "w", encoding="utf-8") as f:
                for s in short_terms:
                    f.write(json.dumps(s.to_dict(), ensure_ascii=True) + "\n")
            with open(mt_stream_path, "w", encoding="utf-8") as f:
                for m in medium_terms:
                    f.write(json.dumps(m.to_dict(), ensure_ascii=True) + "\n")
            with open(et_stream_path, "w", encoding="utf-8") as f:
                for rec in event_tables:
                    f.write(json.dumps(rec, ensure_ascii=True) + "\n")

        # Rebuild active session state from last committed ET event chain.
        if event_tables:
            valid_chain: List[Tuple[Dict[str, Any], UUID]] = []
            for rec in event_tables:
                cid = _safe_uuid(rec.get("clip_id"))
                if cid is None or cid not in stms_by_id:
                    continue
                valid_chain.append((rec, cid))

            if valid_chain:
                last_rec, _ = valid_chain[-1]
                evt_raw = str(last_rec.get("event_id", "")).strip()
                current_event_id = _safe_uuid(evt_raw)
                committed = last_rec.get("event_table_committed")
                if not isinstance(committed, dict):
                    committed = last_rec.get("event_table")
                current_event_table = committed if isinstance(committed, dict) else None

                if current_event_id is not None:
                    for rec, cid in valid_chain:
                        if str(rec.get("event_id", "")).strip() == str(current_event_id):
                            session_clip_ids.append(cid)

                session_details = [
                    _mtm_detail_from_stm(stms_by_id[cid])
                    for cid in session_clip_ids
                    if cid in stms_by_id
                ]
                prev_stm = stms_by_id[session_clip_ids[-1]] if session_clip_ids else None

        for rec in event_tables:
            decision = rec.get("decision")
            if not isinstance(decision, dict):
                continue
            split_t = decision.get("split_t")
            if split_t is None:
                continue
            try:
                split_v = float(split_t)
            except Exception:
                continue
            if split_v > last_accepted_split_time_s:
                last_accepted_split_time_s = split_v

        print(
            "[resume] restored "
            f"ST={restored_st} MT={restored_mt} ET={restored_et} "
            f"active_session_clips={len(session_clip_ids)}"
        )

    def _append_event_table_record(
        *,
        event_id: str,
        clip_id: str,
        time_range: List[float],
        committed_et: Dict[str, Any],
        proposed_et: Optional[Dict[str, Any]] = None,
        decision: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist both pre-judge proposal and committed ET for traceability."""
        rec = {
            "event_id": event_id,
            "clip_id": clip_id,
            "time_range": time_range,
            # Backward compatible field consumed by existing scripts.
            "event_table": committed_et,
            "event_table_proposed": proposed_et if proposed_et is not None else committed_et,
            "event_table_committed": committed_et,
        }
        if decision is not None:
            rec["decision"] = decision
        event_tables.append(rec)
        _append_jsonl(et_stream_path, rec)

    def _is_et_only_cooldown_active(start_t: float, candidate_sources: List[str]) -> bool:
        """Suppress ET-only re-splitting immediately after a confirmed cut."""
        if candidate_sources != ["ET"]:
            return False
        if last_accepted_split_time_s < 0:
            return False
        gap = start_t - last_accepted_split_time_s
        return 0.0 <= gap < min_split_gap_s

    def _apply_split(split_t: float, start_t: float, end_t: float, seg_id: int) -> None:
        nonlocal session_clip_ids
        nonlocal session_details
        nonlocal prev_stm
        nonlocal current_event_table
        nonlocal current_event_id
        nonlocal last_accepted_split_time_s

        pre_stm, _, _ = _run_stm_for_segment(start_t, split_t, f"{seg_id}_a")
        if pre_stm:
            _record_stm(pre_stm)
            session_clip_ids.append(pre_stm.id)
            session_details.append(_mtm_detail_from_stm(pre_stm))
            prev_stm = pre_stm
            pre_proposed = _update_event_table(current_event_table, pre_stm)
            current_event_table = pre_proposed
            _append_event_table_record(
                event_id=str(current_event_id) if current_event_id else "",
                clip_id=str(pre_stm.id),
                time_range=[start_t, split_t],
                proposed_et=pre_proposed,
                committed_et=current_event_table,
                decision={"mode": "split_pre", "split_t": split_t},
            )

        if session_clip_ids:
            _finalize_session()
        session_clip_ids = []
        session_details = []
        prev_stm = None
        current_event_table = None
        current_event_id = None
        last_accepted_split_time_s = split_t

        post_stm, _, _ = _run_stm_for_segment(split_t, end_t, f"{seg_id}_b")
        if post_stm:
            _record_stm(post_stm)
            session_clip_ids = [post_stm.id]
            session_details = [_mtm_detail_from_stm(post_stm)]
            prev_stm = post_stm
            current_event_id = uuid4()
            post_proposed = _update_event_table(None, post_stm)
            current_event_table = post_proposed
            _append_event_table_record(
                event_id=str(current_event_id),
                clip_id=str(post_stm.id),
                time_range=[split_t, end_t],
                proposed_et=post_proposed,
                committed_et=current_event_table,
                decision={"mode": "split_post", "split_t": split_t},
            )

    def _parse_split_seconds(raw_t: str) -> Optional[float]:
        t = str(raw_t or "").strip()
        if not t:
            return None
        try:
            return float(t)
        except Exception:
            pass
        try:
            return _hhmmss_to_seconds(t)
        except Exception:
            return None

    def _pick_split_t(
        split_points: List[Dict[str, str]],
        start_t: float,
        end_t: float,
        fallback_t: Optional[float] = None,
    ) -> Optional[float]:
        eps = 1e-3
        for sp in split_points:
            t_s = _parse_split_seconds(sp.get("t", ""))
            if t_s is None:
                continue
            # Allow split at the left edge (new clip starts a new event),
            # but keep right edge exclusive to avoid empty post-split segment.
            if start_t - eps <= t_s < end_t - eps:
                return t_s
        if fallback_t is not None and start_t - eps <= fallback_t < end_t - eps:
            return float(fallback_t)
        return None

    def _run_stm_for_segment(start_t: float, end_t: float, seg_tag: str) -> Tuple[Optional[ShortTermMemory], List[float], str]:
        out_clip = segments_dir / f"seg_{seg_tag}.mp4"
        print(f"[segment] seg={seg_tag} start={start_t:.3f} end={end_t:.3f}: extracting subclip")
        try:
            _extract_video_subclip(video_path, out_clip, start_t, end_t)
        except Exception as e:
            print(f"[warn] ffmpeg subclip failed for seg_{seg_tag}: {e}")
            return None, [], ""
        print(f"[segment] seg={seg_tag}: subclip ready -> {out_clip}")

        asr_lines = _asr_for_range(asr_segments, start_t, end_t)
        asr_text = "\n".join(asr_lines) if asr_lines else "无可用音频或转录失败"
        pre_et_text = "null" if not current_event_table else json.dumps(current_event_table, ensure_ascii=False, indent=2)
        prompt = CAPTION_PROMPT.replace("{event_table}", pre_et_text)
        print(f"[segment] seg={seg_tag}: running VLM inference")
        cap = _offline_vu(out_clip, prompt, vllm_client=vllm_client)
        print(f"[segment] seg={seg_tag}: VLM inference done")

        visual_summary = str(cap.get("visual_summary", "")).strip()
        detailed_caption = str(cap.get("detailed_caption", "")).strip()
        environment = str(cap.get("environment", "")).strip()
        asr_corrected = asr_text

        embed_text = " ".join([visual_summary, detailed_caption, asr_corrected]).strip()
        embedding = _embed_text_openai(embed_text if embed_text else visual_summary)
        stm = ShortTermMemory(
            id=uuid4(),
            video_source_path=str(video_path),
            time_range=(start_t, end_t),
            visual_summary=visual_summary,
            detailed_caption=detailed_caption,
            embedding=embedding,
            ASR=asr_corrected,
            environment=environment,
        )
        return stm, [], asr_text

    def _rebuild_event_table_from_old(
        old_et: Optional[Dict[str, Any]],
        pending_items: List[ShortTermMemory],
    ) -> Dict[str, Any]:
        rebuilt = old_et
        for item in pending_items:
            rebuilt = _update_event_table(rebuilt, item, force_continue=True)
        if isinstance(rebuilt, dict):
            return rebuilt
        if pending_items:
            return _update_event_table(old_et, pending_items[-1], force_continue=True)
        return old_et or {}

    try:
        segments = _prepare_segments()
        max_segments_raw = os.getenv("AIO_MAX_SEGMENTS", "").strip()
        if max_segments_raw:
            try:
                max_segments = max(0, int(max_segments_raw))
            except Exception:
                max_segments = 0
            if max_segments > 0:
                segments = segments[:max_segments]
                print(f"[segments] limited by AIO_MAX_SEGMENTS={max_segments}")
        print(f"[segments] total={len(segments)} aligned_to_asr=True")
        for seg in segments:
            start_t = round(float(seg["start_s"]), 3)
            end_t = round(float(seg["end_s"]), 3)
            if resume_active and _is_segment_already_covered(start_t, end_t):
                print(
                    f"[resume] skip seg={int(seg.get('segment_id', 0) or 0)} "
                    f"start={start_t:.3f} end={end_t:.3f} (already covered)"
                )
                continue
            segment_chapter_boundaries = [
                b for b in chapter_boundaries if start_t <= b < end_t
            ]

            stm_full, _, _ = _run_stm_for_segment(start_t, end_t, f"{seg['segment_id']}")
            if stm_full is None:
                continue

            if current_event_table is None:
                if segment_chapter_boundaries:
                    boundary_time = min(segment_chapter_boundaries)
                    _append_jsonl(event_link_log_path, {
                        "ts": datetime.utcnow().isoformat() + "Z",
                        "meta": {
                            "segment_id": int(seg.get("segment_id", 0) or 0),
                            "time_range": [start_t, end_t],
                            "boundary_source": "chapter_segment",
                            "boundary_time": boundary_time,
                            "boundary_time_hhmmss": _seconds_to_hhmmss(boundary_time),
                            "lookahead_clips": 1,
                            "init_pre_et": True,
                            "init_action": "direct_split",
                        },
                        "candidate_source": _chapter_shift_text(boundary_time),
                        "pre_ET": None,
                        "current_segments": _event_link_current_segments_payload(stm_full, stm_full),
                        "note": "Chapter boundary observed during initialization; split applied immediately.",
                    })
                    _apply_split(boundary_time, start_t, end_t, int(seg.get("segment_id", 0) or 0))
                    continue

                _record_stm(stm_full)
                session_clip_ids = [stm_full.id]
                session_details = [_mtm_detail_from_stm(stm_full)]
                prev_stm = stm_full
                current_event_id = uuid4()
                init_proposed = _update_event_table(None, stm_full)
                current_event_table = init_proposed
                _append_event_table_record(
                    event_id=str(current_event_id),
                    clip_id=str(stm_full.id),
                    time_range=[start_t, end_t],
                    proposed_et=init_proposed,
                    committed_et=current_event_table,
                    decision={"mode": "init_from_null_pre_et"},
                )
                continue

            if pending_active and pending_anchor_info is not None:
                pending_stms.append(stm_full)
                pending_event_table = _update_event_table(pending_event_table, stm_full)

                left_event_table = pending_anchor_info.get("left_event_table")
                left_event_id = pending_anchor_info.get("left_event_id")
                left_session_ids = list(pending_anchor_info.get("left_session_clip_ids") or [])
                left_session_detail_list = list(pending_anchor_info.get("left_session_details") or [])

                anchor_stm = pending_stms[0]
                pending_confirm_stm = pending_stms[-1]
                pending_start_t = float(pending_stms[0].time_range[0])
                pending_end_t = float(pending_stms[-1].time_range[1])
                pending_chapters = _chapters_for_segment(pending_start_t, pending_end_t)

                effective_candidate_sources = list(pending_anchor_info.get("candidate_sources") or [])
                anchor_end_t = float(pending_anchor_info.get("anchor_end_t", pending_start_t))
                pending_new_chapter_boundaries = [
                    b for b in chapter_boundaries if anchor_end_t <= b < pending_end_t
                ]
                if pending_new_chapter_boundaries and "chapter_segment" not in effective_candidate_sources:
                    effective_candidate_sources.append("chapter_segment")

                has_asr_source = "chapter_segment" in effective_candidate_sources
                has_et_source = "ET" in effective_candidate_sources
                if has_asr_source and pending_new_chapter_boundaries:
                    pending_boundary_time = min(pending_new_chapter_boundaries)
                else:
                    pending_boundary_time = float(
                        pending_anchor_info.get("boundary_time", pending_start_t)
                    )
                pending_source_tag = (
                    "ET+chapter_segment"
                    if len(effective_candidate_sources) > 1
                    else (effective_candidate_sources[0] if effective_candidate_sources else "unknown")
                )
                parts: List[str] = []
                if has_asr_source:
                    parts.append(_chapter_shift_text(pending_boundary_time))
                if has_et_source:
                    et_evidence = "both" if has_asr_source else "et"
                    parts.append(
                        _format_et_shift_candidate(
                            str(pending_anchor_info.get("trigger_delta", "")),
                            et_evidence,
                        )
                    )
                pending_candidate_source_text = " | ".join(parts) if parts else str(
                    pending_anchor_info.get("candidate_source_text", "unknown")
                )
                pending_anchor_info["candidate_sources"] = effective_candidate_sources
                pending_anchor_info["candidate_source_text"] = pending_candidate_source_text
                pending_anchor_info["boundary_source"] = pending_source_tag
                pending_anchor_info["boundary_time"] = pending_boundary_time

                same_event, split_points = _judge_event_with_et(
                    left_event_table,
                    anchor_stm,
                    pending_confirm_stm,
                    seg_conf,
                    pending_chapters,
                    pending_candidate_source_text,
                    log_path=event_link_log_path,
                    meta={
                        "segment_id": int(seg.get("segment_id", 0) or 0),
                        "time_range": [pending_start_t, pending_end_t],
                        "boundary_source": "pending_delayed_gate",
                        "pending_candidate_source": pending_source_tag,
                        "boundary_time": pending_boundary_time,
                        "boundary_time_hhmmss": _seconds_to_hhmmss(pending_boundary_time),
                        "anchor_segment_id": pending_anchor_info.get("anchor_segment_id"),
                        "anchor_time_range": [
                            pending_anchor_info.get("anchor_start_t", pending_start_t),
                            pending_anchor_info.get("anchor_end_t", pending_end_t),
                        ],
                        "trigger_delta": pending_anchor_info.get("trigger_delta", ""),
                    },
                )
                print(
                    f"[pending] anchor_seg={pending_anchor_info.get('anchor_segment_id')} "
                    f"confirm_seg={seg.get('segment_id')} same_event={same_event}"
                )

                if not same_event:
                    split_t = _pick_split_t(
                        split_points,
                        pending_start_t,
                        pending_end_t,
                        fallback_t=float(pending_anchor_info.get("boundary_time", pending_start_t)),
                    )
                    split_idx = -1
                    split_boundary_idx = -1
                    split_seg_start = 0.0
                    split_seg_end = 0.0
                    if split_t is not None:
                        eps = 1e-3
                        for i, pstm in enumerate(pending_stms):
                            st_i = float(pstm.time_range[0])
                            ed_i = float(pstm.time_range[1])
                            if st_i + eps < split_t < ed_i - eps:
                                split_idx = i
                                split_seg_start = st_i
                                split_seg_end = ed_i
                                break
                            # split point lands exactly on clip boundary: split by clip index
                            if abs(ed_i - split_t) <= eps:
                                split_boundary_idx = i + 1
                                break
                            if abs(st_i - split_t) <= eps:
                                split_boundary_idx = i
                                break

                    session_clip_ids = left_session_ids
                    session_details = left_session_detail_list
                    prev_stm = stms_by_id[session_clip_ids[-1]] if session_clip_ids and session_clip_ids[-1] in stms_by_id else None
                    current_event_table = left_event_table if isinstance(left_event_table, dict) else current_event_table
                    current_event_id = left_event_id if isinstance(left_event_id, UUID) else current_event_id

                    if split_idx >= 0 and split_t is not None:
                        for pstm in pending_stms[:split_idx]:
                            _record_stm(pstm)
                            session_clip_ids.append(pstm.id)
                            session_details.append(_mtm_detail_from_stm(pstm))
                            prev_stm = pstm
                            current_event_table = _update_event_table(current_event_table, pstm, force_continue=True)
                            _append_event_table_record(
                                event_id=str(current_event_id) if current_event_id else "",
                                clip_id=str(pstm.id),
                                time_range=[pstm.time_range[0], pstm.time_range[1]],
                                proposed_et=pending_event_table,
                                committed_et=current_event_table,
                                decision={
                                    "mode": "pending_accept_pre_split",
                                    "split_t": split_t,
                                    "pending_index": split_idx,
                                    "candidate_sources": pending_anchor_info.get("candidate_sources") or [],
                                },
                            )

                        _apply_split(split_t, split_seg_start, split_seg_end, int(seg.get("segment_id", 0) or 0))

                        for pstm in pending_stms[split_idx + 1:]:
                            _record_stm(pstm)
                            session_clip_ids.append(pstm.id)
                            session_details.append(_mtm_detail_from_stm(pstm))
                            prev_stm = pstm
                            proposed_after = _update_event_table(current_event_table, pstm, force_continue=True)
                            current_event_table = proposed_after
                            _append_event_table_record(
                                event_id=str(current_event_id) if current_event_id else "",
                                clip_id=str(pstm.id),
                                time_range=[pstm.time_range[0], pstm.time_range[1]],
                                proposed_et=proposed_after,
                                committed_et=current_event_table,
                                decision={
                                    "mode": "split_post_attach",
                                    "split_t": split_t,
                                    "candidate_sources": pending_anchor_info.get("candidate_sources") or [],
                                },
                            )
                    elif split_boundary_idx >= 0 and split_t is not None:
                        # Boundary split: no need to re-run clip inference; split by existing clip boundaries.
                        for pstm in pending_stms[:split_boundary_idx]:
                            _record_stm(pstm)
                            session_clip_ids.append(pstm.id)
                            session_details.append(_mtm_detail_from_stm(pstm))
                            prev_stm = pstm
                            current_event_table = _update_event_table(current_event_table, pstm, force_continue=True)
                            _append_event_table_record(
                                event_id=str(current_event_id) if current_event_id else "",
                                clip_id=str(pstm.id),
                                time_range=[pstm.time_range[0], pstm.time_range[1]],
                                proposed_et=pending_event_table,
                                committed_et=current_event_table,
                                decision={
                                    "mode": "pending_accept_pre_boundary",
                                    "split_t": split_t,
                                    "split_boundary_idx": split_boundary_idx,
                                    "candidate_sources": pending_anchor_info.get("candidate_sources") or [],
                                },
                            )

                        if session_clip_ids:
                            _finalize_session()

                        current_event_id = uuid4()
                        current_event_table = None
                        session_clip_ids = []
                        session_details = []
                        prev_stm = None

                        for pstm in pending_stms[split_boundary_idx:]:
                            _record_stm(pstm)
                            session_clip_ids.append(pstm.id)
                            session_details.append(_mtm_detail_from_stm(pstm))
                            prev_stm = pstm
                            proposed_after = _update_event_table(current_event_table, pstm)
                            current_event_table = proposed_after
                            _append_event_table_record(
                                event_id=str(current_event_id),
                                clip_id=str(pstm.id),
                                time_range=[pstm.time_range[0], pstm.time_range[1]],
                                proposed_et=proposed_after,
                                committed_et=current_event_table,
                                decision={
                                    "mode": "pending_accept_post_boundary",
                                    "split_t": split_t,
                                    "split_boundary_idx": split_boundary_idx,
                                    "candidate_sources": pending_anchor_info.get("candidate_sources") or [],
                                },
                            )
                    else:
                        if session_clip_ids:
                            _finalize_session()

                        current_event_id = uuid4()
                        current_event_table = pending_event_table if isinstance(pending_event_table, dict) else _update_event_table(None, pending_stms[0])
                        session_clip_ids = []
                        session_details = []
                        for idx, pstm in enumerate(pending_stms, start=1):
                            _record_stm(pstm)
                            session_clip_ids.append(pstm.id)
                            session_details.append(_mtm_detail_from_stm(pstm))
                            _append_event_table_record(
                                event_id=str(current_event_id),
                                clip_id=str(pstm.id),
                                time_range=[pstm.time_range[0], pstm.time_range[1]],
                                proposed_et=current_event_table,
                                committed_et=current_event_table,
                                decision={
                                    "mode": "pending_accept_new_event",
                                    "pending_index": idx,
                                    "pending_size": len(pending_stms),
                                    "candidate_sources": pending_anchor_info.get("candidate_sources") or [],
                                },
                            )
                        prev_stm = pending_stms[-1] if pending_stms else None
                    last_accepted_split_time_s = float(split_t if split_t is not None else pending_anchor_info.get("anchor_start_t", pending_start_t))
                else:
                    running_old_et = left_event_table
                    for idx, pstm in enumerate(pending_stms, start=1):
                        _record_stm(pstm)
                        running_old_et = _update_event_table(running_old_et, pstm, force_continue=True)
                        _append_event_table_record(
                            event_id=str(left_event_id) if left_event_id else "",
                            clip_id=str(pstm.id),
                            time_range=[pstm.time_range[0], pstm.time_range[1]],
                            proposed_et=pending_event_table,
                            committed_et=running_old_et,
                            decision={
                                "mode": "pending_reject_force_continue",
                                "pending_index": idx,
                                "pending_size": len(pending_stms),
                                "candidate_sources": pending_anchor_info.get("candidate_sources") or [],
                            },
                        )
                    current_event_table = running_old_et
                    current_event_id = left_event_id if isinstance(left_event_id, UUID) else current_event_id
                    session_clip_ids = left_session_ids + [s.id for s in pending_stms]
                    session_details = left_session_detail_list + [_mtm_detail_from_stm(s) for s in pending_stms]
                    prev_stm = pending_stms[-1]

                pending_active = False
                pending_anchor_info = None
                pending_stms = []
                pending_event_table = None
                continue

            proposed_event_table = _update_event_table(current_event_table, stm_full)
            delta = str(proposed_event_table.get("delta", "")).strip()
            delta_upper = delta.upper()
            is_candidate = (
                current_event_table is not None
                and delta_upper.startswith("SHIFT:")
                and "NONE ->" not in delta_upper
                and not _is_initialization_shift(delta)
            )
            candidate_sources: List[str] = []
            if is_candidate:
                candidate_sources.append("ET")
            if segment_chapter_boundaries:
                candidate_sources.append("chapter_segment")

            et_only_cooldown_active = _is_et_only_cooldown_active(start_t, candidate_sources)

            if et_only_cooldown_active:
                _record_stm(stm_full)
                session_clip_ids.append(stm_full.id)
                session_details.append(_mtm_detail_from_stm(stm_full))
                prev_stm = stm_full
                current_event_table = _update_event_table(current_event_table, stm_full, force_continue=True)
                _append_event_table_record(
                    event_id=str(current_event_id) if current_event_id else "",
                    clip_id=str(stm_full.id),
                    time_range=[start_t, end_t],
                    proposed_et=proposed_event_table,
                    committed_et=current_event_table,
                    decision={
                        "mode": "et_only_cooldown_suppressed",
                        "candidate_sources": candidate_sources,
                        "last_accepted_split_time_s": last_accepted_split_time_s,
                        "cooldown_gap_s": start_t - last_accepted_split_time_s,
                        "cooldown_window_s": min_split_gap_s,
                    },
                )
                _append_jsonl(event_link_log_path, {
                    "ts": datetime.utcnow().isoformat() + "Z",
                    "meta": {
                        "segment_id": int(seg.get("segment_id", 0) or 0),
                        "time_range": [start_t, end_t],
                        "delta": delta,
                        "boundary_source": "ET",
                        "boundary_time": start_t,
                        "boundary_time_hhmmss": _seconds_to_hhmmss(start_t),
                        "last_accepted_split_time_s": last_accepted_split_time_s,
                        "cooldown_gap_s": start_t - last_accepted_split_time_s,
                        "cooldown_window_s": min_split_gap_s,
                    },
                    "candidate_source": _format_et_shift_candidate(delta, "et"),
                    "pre_ET": _event_link_pre_et_payload(current_event_table),
                    "current_segments": _event_link_current_segments_payload(stm_full, stm_full),
                    "proposed_event_table": proposed_event_table,
                    "committed_event_table": current_event_table,
                    "note": "ET-only candidate suppressed by nearby cooldown after a recent confirmed cut.",
                })
                continue

            if candidate_sources and current_event_table is not None:
                boundary_time = min(segment_chapter_boundaries) if segment_chapter_boundaries else start_t
                source_tag = "ET+chapter_segment" if len(candidate_sources) > 1 else candidate_sources[0]
                parts: List[str] = []
                has_asr_source = "chapter_segment" in candidate_sources
                has_et_source = "ET" in candidate_sources
                if has_asr_source:
                    parts.append(_chapter_shift_text(boundary_time))
                if has_et_source:
                    et_evidence = "both" if has_asr_source else "et"
                    parts.append(_format_et_shift_candidate(delta, et_evidence))
                prompt_candidate_source = " | ".join(parts) if parts else "unknown"
                pending_active = True
                pending_anchor_info = {
                    "left_event_table": current_event_table,
                    "left_event_id": current_event_id,
                    "left_session_clip_ids": list(session_clip_ids),
                    "left_session_details": list(session_details),
                    "candidate_sources": list(candidate_sources),
                    "candidate_source_text": prompt_candidate_source,
                    "trigger_delta": delta,
                    "anchor_segment_id": int(seg.get("segment_id", 0) or 0),
                    "anchor_start_t": start_t,
                    "anchor_end_t": end_t,
                    "boundary_source": source_tag,
                    "boundary_time": boundary_time,
                }
                pending_stms = [stm_full]
                pending_event_table = _update_event_table(None, stm_full)
                _append_jsonl(event_link_log_path, {
                    "ts": datetime.utcnow().isoformat() + "Z",
                    "meta": {
                        "segment_id": int(seg.get("segment_id", 0) or 0),
                        "time_range": [start_t, end_t],
                        "delta": delta,
                        "boundary_source": source_tag,
                        "boundary_time": boundary_time,
                        "boundary_time_hhmmss": _seconds_to_hhmmss(boundary_time),
                        "lookahead_clips": 1,
                    },
                    "candidate_source": prompt_candidate_source,
                    "pre_ET": _event_link_pre_et_payload(current_event_table),
                    "current_segments": _event_link_current_segments_payload(stm_full, stm_full),
                    "proposed_event_table": proposed_event_table,
                    "pending_event_table": pending_event_table,
                    "note": "Pending opened: delayed confirmation will judge after one right clip.",
                })
                continue
            else:
                _record_stm(stm_full)
                session_clip_ids.append(stm_full.id)
                session_details.append(_mtm_detail_from_stm(stm_full))
                prev_stm = stm_full
                if _is_initialization_shift(delta):
                    current_event_table = _update_event_table(current_event_table, stm_full, force_continue=True)
                    decision_mode = "no_judge_force_continue_init_shift"
                else:
                    current_event_table = proposed_event_table
                    decision_mode = "no_judge_direct_commit"
                _append_event_table_record(
                    event_id=str(current_event_id) if current_event_id else "",
                    clip_id=str(stm_full.id),
                    time_range=[start_t, end_t],
                    proposed_et=proposed_event_table,
                    committed_et=current_event_table,
                    decision={"mode": decision_mode},
                )

        if pending_active and pending_anchor_info is not None and pending_stms:
            left_event_table = pending_anchor_info.get("left_event_table")
            left_event_id = pending_anchor_info.get("left_event_id")
            left_session_ids = list(pending_anchor_info.get("left_session_clip_ids") or [])
            left_session_detail_list = list(pending_anchor_info.get("left_session_details") or [])
            current_event_table = _rebuild_event_table_from_old(left_event_table, pending_stms)
            current_event_id = left_event_id if isinstance(left_event_id, UUID) else current_event_id
            for pstm in pending_stms:
                _record_stm(pstm)
            session_clip_ids = left_session_ids + [s.id for s in pending_stms]
            session_details = left_session_detail_list + [_mtm_detail_from_stm(s) for s in pending_stms]
            prev_stm = pending_stms[-1]
            for idx, pstm in enumerate(pending_stms, start=1):
                _append_event_table_record(
                    event_id=str(current_event_id) if current_event_id else "",
                    clip_id=str(pstm.id),
                    time_range=[pstm.time_range[0], pstm.time_range[1]],
                    proposed_et=pending_event_table,
                    committed_et=current_event_table,
                    decision={
                        "mode": "pending_flush_reject_no_lookahead",
                        "pending_index": idx,
                        "pending_size": len(pending_stms),
                        "candidate_sources": pending_anchor_info.get("candidate_sources") or [],
                    },
                )
            pending_active = False
            pending_anchor_info = None
            pending_stms = []
            pending_event_table = None

        if session_clip_ids:
            _finalize_session()
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Ensure any distributed/NCCL groups are torn down to avoid leaks
        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception as e:
            print(f"[warn] destroy_process_group failed: {e}")

    st_path = outputs_dir / "short_term.json"
    mt_path = outputs_dir / "medium_term.json"
    et_path = outputs_dir / "event_table.json"

    with open(st_path, "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in short_terms], f, ensure_ascii=False, indent=2)

    with open(mt_path, "w", encoding="utf-8") as f:
        json.dump([m.to_dict() for m in medium_terms], f, ensure_ascii=False, indent=2)

    with open(et_path, "w", encoding="utf-8") as f:
        json.dump(event_tables, f, ensure_ascii=False, indent=2)

    print(f"[done-off] ST={len(short_terms)} MT={len(medium_terms)}")
    # -------------------- Cleanup temporary artifacts --------------------
    if cleanup:
        import shutil
        # Remove extracted segment mp4s
        try:
            if segments_dir.exists():
                shutil.rmtree(segments_dir, ignore_errors=True)
        except Exception as e:
            print(f"[warn] cleanup segments failed: {e}")

        # Remove any non-JSON files in outputs_dir (keep memory JSONs)
        try:
            for p in outputs_dir.iterdir():
                if p.is_file() and p.suffix.lower() not in (".json", ".jsonl"):
                    p.unlink(missing_ok=True)
                elif p.is_dir() and p.name != "":
                    # we already handled asr_chunks; remove any other temp dirs
                    if p.name != "":
                        shutil.rmtree(p, ignore_errors=True)
        except Exception as e:
            print(f"[warn] cleanup outputs extras failed: {e}")

        # Legacy pipeline may create work_root/chunks — remove if present
        try:
            legacy_chunks = Path(work_root) / "chunks"
            if legacy_chunks.exists():
                shutil.rmtree(legacy_chunks, ignore_errors=True)
        except Exception as e:
            print(f"[warn] cleanup legacy chunks failed: {e}")

    return {"short_term": st_path, "medium_term": mt_path, "event_table": et_path}


if __name__ == '__main__':
    default_video = Path(os.getenv("AIO_VIDEO_PATH", ""))
    default_work_root = Path(os.getenv("AIO_WORK_ROOT", str(OUTPUT_ROOT / "aio_run")))
    default_whisper = WHISPER_MODEL_DIR
    default_whisper_device = os.getenv("AIO_WHISPER_DEVICE")
    default_vllm_base_url = os.getenv("VLLM_BASE_URL", VLLM_BASE_URL)

    try:
        video_for_run = _resolve_video_path(default_video)
    except Exception:
        raise RuntimeError(
            "No video path provided. Set AIO_VIDEO_PATH env var or pass a path directly."
        )

    run_video_memory_pipeline_off(
        video_path=video_for_run,
        work_root=default_work_root,
        chunk_seconds=30,
        whisper_model_dir=default_whisper,
        whisper_device=default_whisper_device,
        vllm_base_url=default_vllm_base_url,
    )
