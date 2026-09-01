#!/usr/bin/env python3
"""Build EMMA memory with reproducible stage timings.

The pipeline uses two phases:
1. faster-whisper/CTranslate2 transcribes the selected videos and exits;
2. Qwen3-VL-8B is started once as a local vLLM server and stays resident while
   all videos are converted to STM, MTM, and event memories.

DeepSeek-V4-Flash and text embeddings remain OpenAI-compatible API calls, as
specified by memory/.env.  The JSON report separates ASR, model start-up, and
steady-state memory-building latency.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _sanitize_name(name: str) -> str:
    import re

    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip(".")
    return safe or "video"


def _select_videos(input_dir: Path, video_list: str, recursive: bool, limit: int) -> List[Path]:
    videos = sorted(
        p for p in (input_dir.rglob("*") if recursive else input_dir.glob("*"))
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )
    if video_list:
        list_path = Path(video_list)
        if not list_path.is_file():
            raise FileNotFoundError(f"Video list not found: {list_path}")
        wanted = {
            x.strip() for x in list_path.read_text(encoding="utf-8").splitlines()
            if x.strip() and not x.lstrip().startswith("#")
        }
        chosen: List[Path] = []
        for video in videos:
            if wanted & {
                str(video), str(video.resolve()), video.name, video.stem,
                str(video.relative_to(input_dir)),
            }:
                chosen.append(video)
        if len(chosen) != len(wanted):
            found = set()
            for video in chosen:
                found.update({str(video), str(video.resolve()), video.name, video.stem, str(video.relative_to(input_dir))})
            missing = sorted(wanted - found)
            if missing:
                raise FileNotFoundError(f"Video-list entries not found under {input_dir}: {missing[:5]}")
        videos = chosen
    if limit > 0:
        videos = videos[:limit]
    return videos


def _http_ok(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
            return 200 <= resp.status < 300
    except (OSError, urllib.error.URLError, urllib.error.HTTPError):
        return False


def _stop_process(proc: Optional[subprocess.Popen]) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
        proc.wait(timeout=10)


def _start_qwen(args: argparse.Namespace, run_dir: Path) -> Tuple[Optional[subprocess.Popen], Dict[str, Any]]:
    if _http_ok(args.qwen_port):
        return None, {"reused": True, "port": args.qwen_port, "seconds": 0.0}

    log_path = run_dir / "qwen_server.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    command = [
        args.vllm_python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.qwen_model,
        "--served-model-name", "qwen-vl",
        "--host", "127.0.0.1", "--port", str(args.qwen_port),
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", str(args.qwen_memory_utilization),
        "--max-model-len", str(args.qwen_max_model_len),
        "--max-num-seqs", str(args.qwen_max_num_seqs),
        "--trust-remote-code", "--disable-log-requests", "--allowed-local-media-path", "/",
    ]
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            command, stdout=log_file, stderr=subprocess.STDOUT, env=env,
            start_new_session=True,
        )
    deadline = time.monotonic() + args.server_startup_timeout
    while time.monotonic() < deadline:
        if _http_ok(args.qwen_port):
            return proc, {
                "reused": False, "port": args.qwen_port,
                "seconds": round(time.perf_counter() - started, 3), "log": str(log_path),
            }
        code = proc.poll()
        if code is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"Qwen vLLM exited during startup (code {code}).\n{tail}")
        time.sleep(2)
    _stop_process(proc)
    tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    raise TimeoutError(f"Qwen vLLM was not ready within {args.server_startup_timeout}s.\n{tail}")


def _valid_asr(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return isinstance(data, list) and bool(data) and bool(str(data[0].get("text") or "").strip())
    except Exception:
        return False


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build EMMA memory")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--video-list", default="", help="Optional text file listing selected videos")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device-id", type=int, default=0, help="Visible device index")
    p.add_argument("--asr-model-dir", required=True, help="CTranslate2 Whisper model directory")
    p.add_argument("--qwen-model", required=True, help="Local Qwen3-VL checkpoint directory")
    p.add_argument("--asr-python", "--python", dest="asr_python", default=sys.executable, help="Python used for ASR")
    p.add_argument("--vllm-python", default=sys.executable, help="Python with vLLM installed")
    p.add_argument("--asr-beam-size", type=int, default=1)
    p.add_argument("--rerun-asr", action="store_true", help="Overwrite existing ASR transcripts")
    p.add_argument("--qwen-port", type=int, default=8100)
    p.add_argument("--qwen-memory-utilization", type=float, default=0.82)
    p.add_argument("--qwen-max-model-len", type=int, default=32768)
    p.add_argument("--qwen-max-num-seqs", type=int, default=2)
    p.add_argument("--server-startup-timeout", type=int, default=600)
    p.add_argument("--memory-text-model", default="deepseek-v4-flash")
    p.add_argument("--chunk-seconds", type=int, default=30)
    p.add_argument("--resume", action="store_true", help="Reuse partial per-video stream outputs")
    p.add_argument("--keep-clips", action="store_true", help="Keep extracted video clips after success")
    p.add_argument("--reuse-qwen-server", action="store_true", help="Allow an already running server on --qwen-port")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    input_dir, output_root = Path(args.input_dir).resolve(), Path(args.output_root).resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    if not Path(args.asr_model_dir).is_dir():
        raise FileNotFoundError(f"ASR CTranslate2 model not found: {args.asr_model_dir}")
    if not Path(args.qwen_model).is_dir():
        raise FileNotFoundError(f"Qwen checkpoint not found: {args.qwen_model}")
    videos = _select_videos(input_dir, args.video_list, args.recursive, args.limit)
    if not videos:
        raise ValueError("No selected videos found.")

    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / "_runs" / _tag()
    run_dir.mkdir(parents=True, exist_ok=False)
    summary: Dict[str, Any] = {
        "started_at": _utc_now(), "input_dir": str(input_dir), "output_root": str(output_root),
        "device_id": args.device_id, "selected_videos": [str(v) for v in videos], "video_count": len(videos),
        "settings": {
            "asr_beam_size": args.asr_beam_size, "chunk_seconds": args.chunk_seconds,
            "qwen_memory_utilization": args.qwen_memory_utilization,
            "qwen_max_model_len": args.qwen_max_model_len, "memory_text_model": args.memory_text_model,
        },
        "run_dir": str(run_dir), "stages": {}, "videos": [],
    }
    qwen_proc: Optional[subprocess.Popen] = None
    overall_start = time.perf_counter()
    try:
        print(f"[selection] {len(videos)} video(s)")
        for video in videos:
            print(f"  {video}")
        if args.dry_run:
            return

        asr_command = [
            args.asr_python, str(SCRIPT_DIR / "ASR.py"), "--input-dir", str(input_dir),
            "--output-root", str(output_root), "--model-dir", args.asr_model_dir, "--device", "cuda",
            "--beam-size", str(args.asr_beam_size), "--video-list", args.video_list or "",
            "--limit", str(args.limit),
        ]
        if args.recursive:
            asr_command.append("--recursive")
        if args.rerun_asr:
            asr_command.append("--no-skip-existing")
        # The ASR entry point treats an empty --video-list as no filter; drop the empty pair for clarity.
        if not args.video_list:
            idx = asr_command.index("--video-list")
            del asr_command[idx:idx + 2]
        asr_log = run_dir / "asr.log"
        print("[command]", " ".join(asr_command))
        asr_start = time.perf_counter()
        with asr_log.open("w", encoding="utf-8") as log_file:
            result = subprocess.run(asr_command, stdout=log_file, stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(args.device_id)})
        summary["stages"]["asr"] = {"seconds": round(time.perf_counter() - asr_start, 3), "return_code": result.returncode, "log": str(asr_log)}
        if result.returncode != 0:
            raise RuntimeError(f"ASR failed; see {asr_log}")
        missing_asr = [str(v) for v in videos if not _valid_asr(output_root / _sanitize_name(v.stem) / "outputs" / "asr_segments.json")]
        if missing_asr:
            raise RuntimeError("Missing or empty ASR after ASR stage:\n" + "\n".join(missing_asr))

        os.environ["VLLM_BASE_URL"] = f"http://127.0.0.1:{args.qwen_port}/v1"
        os.environ["VLLM_API_KEY"] = "dummy"
        os.environ["LLM_MODEL"] = "qwen-vl"
        os.environ["MEMORY_TEXT_MODEL"] = args.memory_text_model
        if str(SCRIPT_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPT_DIR))
        from config import OPENAI_API_KEY, OPENAI_BASE_URL  # imports memory/.env safely
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is empty. Copy memory/.env.example to memory/.env and configure the DeepSeek/embedding endpoint.")
        if not OPENAI_BASE_URL:
            print("[warn] OPENAI_BASE_URL is empty; the OpenAI SDK default endpoint will be used.")

        start_mark = time.perf_counter()
        qwen_proc, qwen_info = _start_qwen(args, run_dir)
        if qwen_info["reused"] and not args.reuse_qwen_server:
            raise RuntimeError(f"Port {args.qwen_port} already has a healthy server. Use --reuse-qwen-server to use it deliberately.")
        summary["stages"]["qwen_server_start"] = qwen_info
        print(f"[qwen] {'reusing' if qwen_info['reused'] else 'ready'} on device {args.device_id}, port {args.qwen_port} in {qwen_info['seconds']:.3f}s")

        from AIO import run_video_memory_pipeline_off
        memory_start = time.perf_counter()
        for video in videos:
            row: Dict[str, Any] = {"video": str(video), "started_at": _utc_now(), "work_root": str(output_root / _sanitize_name(video.stem))}
            video_start = time.perf_counter()
            try:
                outputs = run_video_memory_pipeline_off(
                    video_path=video,
                    work_root=row["work_root"],
                    chunk_seconds=args.chunk_seconds,
                    whisper_model_dir=args.asr_model_dir,
                    whisper_device="cpu",
                    require_precomputed_asr=True,
                    vllm_base_url=os.environ["VLLM_BASE_URL"],
                    resume_from_stream=args.resume,
                    cleanup=not args.keep_clips,
                )
                row.update({"status": "success", "outputs": {k: str(v) for k, v in outputs.items()}})
                print(f"[video] success {video.name} {time.perf_counter() - video_start:.3f}s")
            except Exception as exc:
                row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
                print(f"[video] failed {video.name}: {row['error']}")
                if args.fail_fast:
                    raise
            finally:
                row["seconds"] = round(time.perf_counter() - video_start, 3)
                row["finished_at"] = _utc_now()
                summary["videos"].append(row)
        summary["stages"]["memory_build"] = {"seconds": round(time.perf_counter() - memory_start, 3)}
        summary["elapsed_before_teardown_seconds"] = round(time.perf_counter() - overall_start, 3)
    finally:
        shutdown_start = time.perf_counter()
        _stop_process(qwen_proc)
        summary["stages"]["server_shutdown"] = {"seconds": round(time.perf_counter() - shutdown_start, 3), "owned_server": qwen_proc is not None}
        summary["finished_at"] = _utc_now()
        summary["total_wall_seconds"] = round(time.perf_counter() - overall_start, 3)
        summary["successful_videos"] = sum(v.get("status") == "success" for v in summary["videos"])
        summary["failed_videos"] = sum(v.get("status") == "failed" for v in summary["videos"])
        steady = [float(v["seconds"]) for v in summary["videos"] if v.get("status") == "success"]
        if steady:
            summary["steady_state_video_seconds"] = {"count": len(steady), "mean": round(sum(steady) / len(steady), 3), "min": min(steady), "max": max(steady)}
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_dir / "per_video_timings.json").write_text(json.dumps(summary["videos"], ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[summary] {run_dir / 'summary.json'}")
        print(f"[summary] total wall time: {summary['total_wall_seconds']:.3f}s")


if __name__ == "__main__":
    main()
