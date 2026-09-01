import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from decord import VideoReader, cpu
from openai import OpenAI
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from config import OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL, QWEN_CHECKPOINT, OUTPUT_ROOT

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

prompt_boundary_recall = '''
You are given the timestamped ASR transcript of a long video.

Your task is to identify candidate timestamps for visual evidence collection for later video chaptering.

This is a recall-oriented step, not the final chaptering step.
Your goal is to capture plausible chapter-start locations that are worth checking with visual evidence.

A candidate should be proposed only when there is a plausible chapter-scale semantic transition, such as:
- a new topic being introduced
- a new major stage or phase in a process
- a change in the main goal, activity, or task
- a clear shift in discussion context
- explicit discourse markers like "next", "now let's move to", "step two", "another part", etc., when they introduce a substantial new segment

Guidelines:
- Prefer recall over precision, but do NOT mark every local step change.
- Focus on chapter-level or phase-level transitions, not every fine-grained action.
- If several nearby steps belong to the same larger activity, keep only the earliest timestamp that best represents the start of that larger segment.
- Prefer broader semantic units over short procedural substeps.
- Only include a new timestamp if it would add meaningful new visual inspection value beyond nearby candidates.
- If unsure whether a point is a true chapter-scale transition, include it only if visual evidence would meaningfully help resolve the uncertainty.
- Do not split for small rephrasings, examples, brief elaborations, or tightly connected consecutive steps within the same activity.
- When possible, use a short semantic candidate title describing the larger segment starting there.
- Avoid generic labels like "New step" or "Possible topic shift" when a more specific title can be inferred.
- If ASR is too sparse, vague, noisy, repetitive, or uninformative to infer a meaningful title, but visual evidence may help, use exactly: Low-information region
- Include 00:00:00 as the first item.
- Keep the candidate list compact and high-value. For a typical 30-45 minute video, prefer roughly 8-20 candidates unless the structure is unusually complex.

STRICT OUTPUT RULES:
- Output ONLY the candidate timestamps.
- Each item must be on its own line.
- Use the exact format: HH:MM:SS - Brief candidate title
- The first item MUST be: 00:00:00 - Beginning
- Timestamps must be in ascending order.

ASR transcript:
{ASR}
'''

prompt_frame_caption = '''
Describe this short video segment for long-video chaptering.

Focus on information that is useful for identifying event or chapter boundaries:
- main people, objects, and actions
- scene or activity type
- visible on-screen text, logos, subtitles, or signs
- whether this frame suggests a new stage, topic, or activity

Rules:
- Be factual and concise.
- Prefer 1-3 sentences.
- Do not speculate beyond visible evidence.
- If readable text appears, include it explicitly.
- If nothing important changes, summarize the most salient visual content in the segment.

Output format:
Caption: <concise visual description>
OCR: <visible text or "None">

'''

prompt_final_chaptering = '''
You are given a multimodal transcript of a long video.

The transcript may contain:
- timestamped ASR speech
- timestamped frame captions
- optional OCR text

Your task is to generate the final chapter boundaries and chapter titles.

A final chapter should correspond to a coherent semantic event or topic.
Use ASR as the primary signal.
Use visual evidence only to confirm, refine, or reject possible semantic transitions.

Guidelines:
- Start a new chapter only when the following content is better understood as a new coherent topic, event, task, or stage.
- Prefer semantically complete chapters over fragmented short ones.
- Do not split for minor elaboration, repeated wording, brief examples, or local visual variation alone.
- The presence of a frame caption or OCR at a timestamp does NOT by itself mean there is a chapter boundary there.
- Some true chapter boundaries may have little or no nearby visual evidence, and some captioned timestamps may not correspond to any chapter boundary.
- Use visual evidence when it clearly supports a meaningful transition, such as a scene change, task change, new slide, title card, interface, object, or location.
- Keep chapter titles short, specific, and semantic.
- Titles should describe the content of the chapter, not the evidence source.
- Chapters must cover the whole video.
- Include 00:00:00 as the first chapter boundary.

STRICT OUTPUT RULES:
- Output ONLY the final chapters.
- Each chapter must be on its own line.
- Use the exact format: HH:MM:SS - Chapter title
- Timestamps must be in ascending order.
- Do NOT include explanations or commentary.

Example:
00:00:00 - Introduction
00:01:32 - Challenge setup
00:05:10 - First attempt
00:08:45 - New strategy
00:12:20 - Wrap-up

Multimodal transcript:
{MULTIMODAL_EVIDENCE}
'''


TEXT_API_MAX_ATTEMPTS = 3


def _build_openai_client() -> OpenAI:
    kwargs: Dict[str, Any] = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    return OpenAI(**kwargs)


def _looks_like_unsupported_param_error(exc: Exception) -> bool:
    text = str(exc).lower()
    markers = [
        "unexpected keyword argument",
        "unknown parameter",
        "unrecognized request argument",
        "extra fields not permitted",
        "is not allowed",
        "invalid_request_error",
        "extra_body",
        "thinking",
        "reasoning",
    ]
    return any(marker in text for marker in markers)


def _chat_completion_create(client: OpenAI, **kwargs: Any) -> Any:
    thinking_overrides: List[Dict[str, Any]] = [
        {"extra_body": {"thinking": {"type": "disabled"}}},
        {"extra_body": {"thinking": {"enabled": False}}},
        {"extra_body": {"reasoning": {"enabled": False}}},
    ]

    for override in thinking_overrides:
        call_kwargs = dict(kwargs)
        if "extra_body" in call_kwargs and isinstance(call_kwargs["extra_body"], dict):
            merged = dict(call_kwargs["extra_body"])
            merged.update(override.get("extra_body", {}))
            call_kwargs["extra_body"] = merged
        else:
            call_kwargs.update(override)
        try:
            return client.chat.completions.create(**call_kwargs)
        except Exception as exc:
            if _looks_like_unsupported_param_error(exc):
                continue
            break

    return client.chat.completions.create(**kwargs)


def _raw_snippet(text: Any, limit: int = 200) -> str:
    snippet = str(text or "").replace("\n", " ").strip()
    if len(snippet) > limit:
        return snippet[:limit] + "..."
    return snippet


def _write_text_now(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())


def _write_json_now(path: Path, payload: Any) -> None:
    _write_text_now(path, json.dumps(payload, ensure_ascii=False, indent=2))


def _seconds_to_hhmmss(seconds: float) -> str:
    sec = max(0.0, float(seconds))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _hhmmss_to_seconds(ts: str) -> float:
    parts = str(ts).strip().split(":")
    try:
        if len(parts) == 3:
            h, m, s = parts
            return float(h) * 3600 + float(m) * 60 + float(s)
        if len(parts) == 2:
            m, s = parts
            return float(m) * 60 + float(s)
    except Exception:
        return 0.0
    return 0.0


def _iter_result_dirs(root_dir: Path) -> Iterable[Path]:
    if not root_dir.exists() or not root_dir.is_dir():
        return []
    return sorted(p for p in root_dir.iterdir() if p.is_dir() and not p.name.startswith("."))


def _call_text_with_retry(model: str, text_prompt: str, max_tokens: int, call_name: str) -> str:
    client = _build_openai_client()
    last_error = ""
    last_raw = ""

    for attempt in range(1, TEXT_API_MAX_ATTEMPTS + 1):
        retry_prompt = text_prompt
        if attempt > 1:
            retry_prompt = (
                f"{text_prompt}\n\n"
                "Your previous response was empty or unusable. "
                "Follow the required output format strictly and return content only."
            )

        try:
            resp = _chat_completion_create(
                client,
                model=model,
                messages=[{"role": "user", "content": retry_prompt}],
                max_tokens=max_tokens,
                temperature=0,
            )
            raw = str(resp.choices[0].message.content or "").strip()
            last_raw = raw
            if raw:
                return raw
            last_error = "empty_content"
        except Exception as exc:
            last_error = str(exc)

        print(
            f"[warn] {call_name} failed "
            f"(attempt {attempt}/{TEXT_API_MAX_ATTEMPTS}): {last_error}; raw={_raw_snippet(last_raw or last_error)}"
        )

    raise RuntimeError(last_error or f"{call_name} returned empty content")


def _parse_boundary_times(boundary_text: str) -> List[float]:
    pattern = re.compile(r"^\s*(\d{2}:\d{2}:\d{2})\s*-\s*.+$")
    times: List[float] = []
    for line in str(boundary_text).splitlines():
        m = pattern.match(line)
        if not m:
            continue
        times.append(_hhmmss_to_seconds(m.group(1)))
    uniq = sorted(set(max(0.0, t) for t in times))
    if not uniq or uniq[0] > 0.0:
        uniq.insert(0, 0.0)
    return uniq


def _parse_asr_entries(asr_text: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    pat = re.compile(r"^\s*(\d{2}:\d{2}:\d{2})\s+(.+?)\s*$")
    for line in str(asr_text).splitlines():
        m = pat.match(line)
        if not m:
            continue
        ts = m.group(1)
        content = m.group(2).strip()
        if not content:
            continue
        entries.append(
            {
                "time_s": _hhmmss_to_seconds(ts),
                "line": f"[ASR]{ts} {content}",
            }
        )
    return entries


def _parse_caption_and_ocr(text: str) -> Tuple[str, Optional[str]]:
    caption = ""
    ocr: Optional[str] = None
    for line in str(text).splitlines():
        s = line.strip()
        if s.lower().startswith("caption:"):
            caption = s.split(":", 1)[1].strip()
        elif s.lower().startswith("ocr:"):
            ocr_raw = s.split(":", 1)[1].strip()
            if ocr_raw and ocr_raw.lower() != "none":
                ocr = ocr_raw
    if not caption:
        caption = str(text).strip().splitlines()[0].strip() if str(text).strip() else ""
    return caption, ocr


def _resolve_video_path(video_dir: Path, video_root: Path) -> Path:
    vid = video_dir.name
    for ext in (".mp4", ".mkv", ".webm", ".mov"):
        p = video_root / f"{vid}{ext}"
        if p.exists() and p.is_file():
            return p

    if video_root.exists():
        matches = sorted(video_root.glob(f"**/{vid}.*"))
        for m in matches:
            if m.is_file() and m.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}:
                return m

    raise FileNotFoundError(f"video not found for id={vid}")


def _get_video_duration_decord(video_path: Path) -> float:
    vr = VideoReader(str(video_path), ctx=cpu(0))
    fps = float(vr.get_avg_fps() or 0.0)
    if fps <= 0:
        raise RuntimeError(f"invalid fps from decord: {video_path}")
    return float(len(vr)) / fps


def _extract_subclip(video_path: Path, out_path: Path, start_s: float, end_s: float) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg not found in PATH")
    start_s = max(0.0, float(start_s))
    end_s = max(start_s, float(end_s))
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_s:.3f}",
        "-to", f"{end_s:.3f}",
        "-i", str(video_path),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "96k",
        str(out_path),
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return out_path


def _prepare_vlm_inputs(messages: List[Dict[str, Any]], processor: AutoProcessor) -> Dict[str, Any]:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    mm_data: Dict[str, Any] = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    return {
        "prompt": text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }


def _run_vlm_caption(
    clip_path: Path,
    llm: LLM,
    processor: AutoProcessor,
    sampling_params: SamplingParams,
    frame_prompt: str,
) -> Tuple[str, Optional[str]]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(clip_path), "fps": 2.0},
                {"type": "text", "text": frame_prompt},
            ],
        }
    ]
    inputs = [_prepare_vlm_inputs(messages, processor)]
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    text = str(outputs[0].outputs[0].text or "").strip()
    return _parse_caption_and_ocr(text)


def _boundary_window(t_s: float, duration_s: float) -> Optional[Tuple[float, float]]:
    st = max(0.0, min(max(0.0, t_s - 3.0), duration_s))
    ed = max(st, min(min(duration_s, t_s + 3.0), duration_s))
    if ed - st < 0.2:
        return None
    return st, ed


def process_video_dir(
    video_dir: Path,
    video_root: Path,
    qwen_checkpoint: str,
    llm: LLM,
    processor: AutoProcessor,
    sampling_params: SamplingParams,
    boundary_model: str,
    final_model: str,
    max_tokens: int,
) -> None:
    outputs_dir = video_dir / "outputs"
    asr_path = outputs_dir / "chapter_asr.txt"
    if not asr_path.exists():
        print(f"[chapter] skip missing ASR file: {asr_path}")
        return

    asr_text = asr_path.read_text(encoding="utf-8")

    # Step 1: ASR -> candidate boundaries
    recall_prompt = prompt_boundary_recall.replace("{ASR}", asr_text)
    candidate_text = _call_text_with_retry(
        model=boundary_model,
        text_prompt=recall_prompt,
        max_tokens=max_tokens,
        call_name="boundary_recall",
    )
    candidate_out = outputs_dir / "chapter_candidates.txt"
    _write_text_now(candidate_out, candidate_text)
    print(f"[chapter] candidate written: {candidate_out}", flush=True)

    candidate_times = _parse_boundary_times(candidate_text)

    # Step 2: decord + VLM caption/OCR on [t-3, t+3] for each candidate boundary
    video_path = _resolve_video_path(video_dir, video_root)
    duration_s = _get_video_duration_decord(video_path)
    tmp_dir = outputs_dir / "chapter_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    visual_entries: List[Dict[str, Any]] = []

    for t_s in candidate_times:
        window = _boundary_window(t_s, duration_s)
        if window is None:
            continue
        st, ed = window

        clip_name = f"{_seconds_to_hhmmss(t_s).replace(':', '-')}_ctx.mp4"
        clip_path = tmp_dir / clip_name
        try:
            _extract_subclip(video_path, clip_path, st, ed)
            caption, ocr = _run_vlm_caption(
                clip_path=clip_path,
                llm=llm,
                processor=processor,
                sampling_params=sampling_params,
                frame_prompt=prompt_frame_caption,
            )
        except Exception as exc:
            print(f"[warn] vlm caption failed @t={_seconds_to_hhmmss(t_s)}: {exc}")
            continue

        ts = _seconds_to_hhmmss(t_s)
        if caption:
            visual_entries.append(
                {"time_s": t_s, "line": f"[caption]{ts} {caption}"}
            )
        if ocr:
            visual_entries.append(
                {"time_s": t_s, "line": f"[OCR]{ts} {ocr}"}
            )

        # Per-candidate visual evidence is kept only in multimodal text, not in a separate JSON file.

    # Step 3: build multimodal evidence and final chaptering
    asr_entries = _parse_asr_entries(asr_text)
    merged = sorted(asr_entries + visual_entries, key=lambda x: x["time_s"])
    multimodal_evidence = "\n".join(item["line"] for item in merged)

    mm_out = outputs_dir / "chapter_multimodal_evidence.txt"
    _write_text_now(mm_out, multimodal_evidence)
    print(f"[chapter] multimodal evidence written: {mm_out}", flush=True)

    final_prompt = prompt_final_chaptering.replace("{MULTIMODAL_EVIDENCE}", multimodal_evidence)
    final_text = _call_text_with_retry(
        model=final_model,
        text_prompt=final_prompt,
        max_tokens=max_tokens,
        call_name="final_chaptering",
    )

    final_out = outputs_dir / "chapter_final.txt"
    _write_text_now(final_out, final_text)
    print(f"[chapter] final written: {final_out}", flush=True)

    print(
        f"[chapter] done video={video_dir.name} candidates={len(candidate_times)} "
        f"visual_items={len(visual_entries)} final={final_out}"
    )

    # Cleanup temporary clips to save disk space.
    try:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception as exc:
        print(f"[warn] cleanup chapter tmp failed: {exc}")


def process_results_root(
    root_dir: Path,
    video_root: Path,
    qwen_checkpoint: str,
    boundary_model: str,
    final_model: str,
    max_tokens: int,
) -> None:
    result_dirs = list(_iter_result_dirs(root_dir))
    if not result_dirs:
        print(f"[chapter] no result directories found under: {root_dir}")
        return

    processor = AutoProcessor.from_pretrained(qwen_checkpoint)
    llm = LLM(
        model=str(qwen_checkpoint),
        mm_encoder_tp_mode="data",
        enable_expert_parallel=False,
        tensor_parallel_size=1,
        max_model_len=32768,
        seed=123,
    )
    sampling_params = SamplingParams(temperature=0.2, max_tokens=384, top_k=-1)

    ok = 0
    fail = 0
    for video_dir in result_dirs:
        try:
            process_video_dir(
                video_dir=video_dir,
                video_root=video_root,
                qwen_checkpoint=qwen_checkpoint,
                llm=llm,
                processor=processor,
                sampling_params=sampling_params,
                boundary_model=boundary_model,
                final_model=final_model,
                max_tokens=max_tokens,
            )
            ok += 1
        except Exception as exc:
            fail += 1
            print(f"[chapter] failed {video_dir.name}: {exc}")

    print(f"[chapter] summary ok={ok} fail={fail} total={len(result_dirs)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-stage chaptering pipeline")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Root directory containing per-video result folders",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path(""),
        help="Video root used to resolve source video by video id",
    )
    parser.add_argument(
        "--qwen-checkpoint",
        type=str,
        default=QWEN_CHECKPOINT,
        help="Qwen-VL checkpoint path",
    )
    parser.add_argument(
        "--boundary-model",
        type=str,
        default=LLM_MODEL,
        help="Model for candidate boundary recall",
    )
    parser.add_argument(
        "--final-model",
        type=str,
        default=LLM_MODEL,
        help="Model for final chapter generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens for each text model call",
    )
    args = parser.parse_args()

    process_results_root(
        root_dir=args.results_root,
        video_root=args.video_root,
        qwen_checkpoint=args.qwen_checkpoint,
        boundary_model=args.boundary_model,
        final_model=args.final_model,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
