#!/usr/bin/env python3
"""
Batch Whisper ASR using faster-whisper (CTranslate2).

特点:
  - 基于 faster-whisper / CTranslate2
  - 模型只加载一次，所有视频共用
  - 逐视频串行推理（长视频分块后连续送入），实时写出 asr_segments.json

用法:
    # 先转换模型（只需做一次）:
    ct2-transformers-converter \
        --model /root/autodl-tmp/MMM/models/whisper-large-v3-turbo \
        --output_dir /root/autodl-tmp/MMM/models/whisper-large-v3-turbo-ct2 \
        --quantization float16 \
        --copy_files tokenizer.json preprocessor_config.json

    # 运行:
    python ASR.py \
        --input-dir /path/to/videos \
        --output-root /path/to/memory-results \
        --model-dir /path/to/whisper-large-v3-turbo-ct2 \
        --device cuda \
        --beam-size 1
"""
import argparse
import gc
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}


# ── helpers ────────────────────────────────────────────────────────────────────

def _sanitize_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    safe = safe.strip(".")
    return safe or "video"


def _ffprobe_duration(path: Path) -> float:
    cmd = ["ffprobe", "-v", "error",
           "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
    try:
        return float(subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip())
    except Exception:
        return 0.0


def _extract_audio(video: Path, out_dir: Path) -> Path:
    audio = out_dir / f"{video.stem}.wav"
    if not audio.exists():
        cmd = ["ffmpeg", "-y", "-i", str(video),
               "-ac", "1", "-ar", "16000", "-vn", str(audio)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio


def _seconds_to_hhmmss(s: float) -> str:
    s = max(0.0, float(s))
    t = int(round(s))
    return f"{t//3600:02d}:{(t%3600)//60:02d}:{t%60:02d}"


def _fix_monotonic(segs: List[Dict]) -> List[Dict]:
    out, prev_end = [], None
    for s in sorted(segs, key=lambda x: x["start"]):
        st, ed, text = float(s["start"]), float(s["end"]), s.get("text", "").strip()
        if not text:
            continue
        if ed < st:
            ed = st
        if prev_end is not None and st < prev_end:
            shift = prev_end - st
            st, ed = prev_end, max(ed + shift, prev_end)
        out.append({"start": st, "end": ed, "text": text})
        prev_end = ed
    return out


def _select_videos(
    input_dir: Path,
    recursive: bool,
    video_list: Optional[str],
    limit: int,
) -> List[Path]:
    """Return a deterministic subset, optionally selected by a text file.

    A list entry can be an absolute path, a path relative to ``input_dir``, a
    file name, or a video stem.  This makes it safe to run one smoke-test video
    without moving the dataset.
    """
    candidates = sorted(
        p for p in (input_dir.rglob("*") if recursive else input_dir.glob("*"))
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )
    if video_list:
        list_path = Path(video_list)
        if not list_path.is_file():
            raise FileNotFoundError(f"Video list not found: {list_path}")
        wanted = {
            line.strip() for line in list_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        selected: List[Path] = []
        for video in candidates:
            keys = {
                str(video), str(video.resolve()), video.name, video.stem,
                str(video.relative_to(input_dir)),
            }
            if keys & wanted:
                selected.append(video)
        missing = []
        for item in wanted:
            path = Path(item)
            if path.is_file():
                continue
            if not any(item in {v.name, v.stem, str(v.relative_to(input_dir))} for v in selected):
                missing.append(item)
        if missing:
            raise FileNotFoundError(f"Video-list entries not found under {input_dir}: {missing[:5]}")
        candidates = selected
    if limit > 0:
        candidates = candidates[:limit]
    return candidates


# ── per-video ASR ─────────────────────────────────────────────────────────────

def _transcribe_video(
    model,
    video: Path,
    tmp_dir: Path,
    language: Optional[str],
    beam_size: int,
    vad: bool,
) -> List[Dict[str, str]]:
    """Transcribe one video and return cleaned segments."""
    audio = _extract_audio(video, tmp_dir)

    # faster-whisper transcribes the full audio file directly (no manual chunking needed)
    segments_iter, info = model.transcribe(
        str(audio),
        language=language,
        beam_size=beam_size,
        word_timestamps=True,
        vad_filter=vad,
        vad_parameters={"min_silence_duration_ms": 300},
    )

    raw: List[Dict] = []
    for seg in segments_iter:          # generator — iterates as decoding proceeds
        if not seg.words:
            if seg.text.strip():
                raw.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
            continue
        # Accumulate word-level into sentence segments
        cur_words, cur_st, prev_end = [], None, None
        end_punct = (".", "?", "!", "。", "？", "！")

        def flush():
            nonlocal cur_words, cur_st, prev_end
            if cur_words and cur_st is not None:
                text = "".join(w.word for w in cur_words).strip()
                if text:
                    raw.append({"start": cur_st, "end": prev_end, "text": text})
            cur_words, cur_st, prev_end = [], None, None

        for w in seg.words:
            if cur_st is None:
                cur_st, prev_end = w.start, w.end
            else:
                gap = max(0.0, w.start - prev_end)
                seg_len = prev_end - cur_st
                if gap > 0.35 or seg_len >= 6.0:
                    flush()
                    cur_st, prev_end = w.start, w.end
                else:
                    prev_end = max(prev_end, w.end)
            cur_words.append(w)
            if w.word.strip().endswith(end_punct):
                flush()
        flush()

    fixed = _fix_monotonic(raw)
    cleaned = [{"start": _seconds_to_hhmmss(s["start"]),
                "end":   _seconds_to_hhmmss(s["end"]),
                "text":  s["text"]} for s in fixed]

    # cleanup audio
    try:
        audio.unlink(missing_ok=True)
    except Exception:
        pass

    return cleaned


# ── main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    from faster_whisper import WhisperModel

    input_dir   = Path(args.input_dir)
    output_root = Path(args.output_root)
    model_dir   = Path(args.model_dir)

    videos = _select_videos(input_dir, args.recursive, args.video_list, args.limit)
    if not videos:
        print(f"No videos found in {input_dir}")
        return

    # filter already done
    todo: List[Path] = []
    for v in videos:
        asr_path = output_root / _sanitize_name(v.stem) / "outputs" / "asr_segments.json"
        if not args.no_skip_existing and asr_path.exists():
            try:
                data = json.loads(asr_path.read_text())
                if isinstance(data, list) and data and data[0].get("text", "").strip():
                    print(f"[skip] {v.name}")
                    continue
            except Exception:
                pass
        todo.append(v)

    if not todo:
        print("All videos already have ASR. Nothing to do.")
        return

    print(f"\nFound {len(todo)} video(s) to transcribe.")
    print(f"Model : {model_dir}")
    print(f"Device: {args.device}  |  compute: float16  |  beam_size: {args.beam_size}  |  VAD: {not args.no_vad}\n")

    # load model once
    print("[model] Loading faster-whisper model ...")
    t0 = time.time()
    model = WhisperModel(
        str(model_dir),
        device=args.device,
        compute_type="float16",
        num_workers=2,               # parallel audio loading
        cpu_threads=4,
    )
    print(f"[model] Loaded in {time.time()-t0:.1f}s\n")

    success = fail = 0
    t_total = time.time()

    for i, video in enumerate(todo, 1):
        key = _sanitize_name(video.stem)
        out_dir = output_root / key / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        asr_path = out_dir / "asr_segments.json"
        tmp_dir = output_root / key / "_asr_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        t_vid = time.time()
        print(f"[{i}/{len(todo)}] {video.name} ...", end="", flush=True)
        try:
            segs = _transcribe_video(
                model, video, tmp_dir,
                language=args.language or None,
                beam_size=int(args.beam_size),
                vad=not args.no_vad,
            )
            asr_path.write_text(json.dumps(segs, ensure_ascii=False, indent=2))
            elapsed = time.time() - t_vid
            print(f" {len(segs)} segs  {elapsed:.0f}s")
            success += 1
        except Exception as e:
            print(f" FAILED: {e}")
            fail += 1
        finally:
            try:
                import shutil
                if not args.no_cleanup:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

    del model
    gc.collect()

    total = time.time() - t_total
    print(f"\n{'='*55}")
    print(f"Done: {success} ok, {fail} failed  |  total {total:.0f}s  ({total/max(1,success):.0f}s/video)")
    print("\nASR is complete. Run memory/run_memory.py for memory building.")


def main() -> None:
    p = argparse.ArgumentParser(description="Batch Whisper ASR via faster-whisper (CTranslate2)")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--model-dir",
                   default="/root/autodl-tmp/MMM/models/whisper-large-v3-turbo-ct2",
                   help="Path to CTranslate2-format model directory")
    p.add_argument("--device", default="cuda",
                   help="cuda or cpu")
    p.add_argument("--beam-size", type=int, default=1,
                   help="1=greedy (fastest), 5=more accurate")
    p.add_argument("--language", default=None,
                   help="Force language code (e.g. zh, en). None=auto-detect.")
    p.add_argument("--no-vad", action="store_true",
                   help="Disable VAD filter (VAD skips silent segments, saves ~20%% time)")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--video-list", default="", help="Optional text file selecting videos")
    p.add_argument("--limit", type=int, default=0, help="Process at most N selected videos; 0 means all")
    p.add_argument("--no-skip-existing", action="store_true")
    p.add_argument("--no-cleanup", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
