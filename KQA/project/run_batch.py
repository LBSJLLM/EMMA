#!/usr/bin/env python3
"""Run EMMA KQA sequentially.

Qwen3-VL is served once through vLLM for RAW_VIDEO fallback.  Retrieval,
planning, validation, embeddings, and R1 answer generation use the configured
OpenAI-compatible APIs.  A single worker is intentional: it prevents multiple
Qwen engines from competing for resources and keeps API rate limits predictable.
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


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.main import load_cfg
from src.pipeline.qa_pipeline import QAPipeline
from src.utils.dataset_loader import list_video_ids_from_results_dirs, load_videomme_questions, load_videomme_questions_by_video_ids
from src.utils.json_utils import write_json
from src.utils.logging import TraceLogger


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_env(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() and key.strip() not in os.environ:
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


def _healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
            return 200 <= resp.status < 300
    except (OSError, urllib.error.URLError, urllib.error.HTTPError):
        return False


def _stop(proc: Optional[subprocess.Popen]) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
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


def _start_qwen(args: argparse.Namespace, run_root: Path) -> Tuple[Optional[subprocess.Popen], Dict[str, Any]]:
    if _healthy(args.qwen_port):
        if not args.reuse_qwen_server:
            raise RuntimeError(f"Port {args.qwen_port} already has a healthy server. Pass --reuse-qwen-server to use it.")
        return None, {"reused": True, "seconds": 0.0, "port": args.qwen_port}
    log_path = run_root / "qwen_server.log"
    cmd = [
        args.vllm_python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.qwen_model, "--served-model-name", "qwen-vl",
        "--host", "127.0.0.1", "--port", str(args.qwen_port),
        "--tensor-parallel-size", "1", "--gpu-memory-utilization", str(args.qwen_memory_utilization),
        "--max-model-len", str(args.qwen_max_model_len), "--max-num-seqs", "2",
        "--trust-remote-code", "--disable-log-requests", "--allowed-local-media-path", "/",
    ]
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(args.device_id)}, start_new_session=True)
    deadline = time.monotonic() + args.server_startup_timeout
    while time.monotonic() < deadline:
        if _healthy(args.qwen_port):
            return proc, {"reused": False, "seconds": round(time.perf_counter() - started, 3), "port": args.qwen_port, "log": str(log_path)}
        if proc.poll() is not None:
            raise RuntimeError(f"Qwen vLLM exited during startup (code {proc.returncode}).\n" + log_path.read_text(encoding="utf-8", errors="replace")[-4000:])
        time.sleep(2)
    _stop(proc)
    raise TimeoutError(f"Qwen vLLM was not ready within {args.server_startup_timeout}s. See {log_path}")


def _prediction_done(path: Path) -> bool:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return bool(str((obj.get("final_answer") or {}).get("predicted_option") or "").strip())
    except Exception:
        return False


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run EMMA KQA")
    p.add_argument("--config", default=str(PROJECT_DIR / "configs" / "deepseek_v4_flash_r1.yaml"))
    p.add_argument("--videomme-json", required=True)
    p.add_argument("--results-root", required=True)
    p.add_argument("--raw-video-root", default="")
    p.add_argument("--question-id", default="")
    p.add_argument("--top-n-videos", type=int, default=0)
    p.add_argument("--max-questions", type=int, default=0)
    p.add_argument("--out-root", default="")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--qwen-model", required=True)
    p.add_argument("--vllm-python", default=sys.executable)
    p.add_argument("--qwen-port", type=int, default=8100)
    p.add_argument("--qwen-memory-utilization", type=float, default=0.82)
    p.add_argument("--qwen-max-model-len", type=int, default=32768)
    p.add_argument("--server-startup-timeout", type=int, default=600)
    p.add_argument("--reuse-qwen-server", action="store_true")
    p.add_argument("--no-video-fallback", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    _load_env(PROJECT_DIR / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is empty. Copy KQA/project/.env.example to KQA/project/.env and configure it.")
    if not Path(args.qwen_model).is_dir():
        raise FileNotFoundError(f"Qwen checkpoint not found: {args.qwen_model}")
    cfg = load_cfg(args.config)
    cfg["results_root"] = args.results_root
    if args.raw_video_root:
        cfg["raw_video_root"] = args.raw_video_root
    if args.no_video_fallback:
        cfg["use_video_fallback"] = False

    top_n = None if args.top_n_videos <= 0 else args.top_n_videos
    max_q = None if args.max_questions <= 0 else args.max_questions
    if top_n is None and not args.question_id:
        ids = list_video_ids_from_results_dirs(args.results_root)
        questions = load_videomme_questions_by_video_ids(args.videomme_json, ids, max_q)
    else:
        questions = load_videomme_questions(args.videomme_json, args.results_root, top_n, max_q, args.question_id or None)
    if not questions:
        raise ValueError("No valid questions loaded. Check paths and memory outputs.")

    out_root = Path(args.out_root) if args.out_root else PROJECT_DIR / "outputs" / f"batch_{_tag()}"
    pred_dir, log_dir, err_dir = out_root / "predictions", out_root / "logs", out_root / "errors"
    for d in (pred_dir, log_dir, err_dir):
        d.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {"started_at": _now(), "config": args.config, "results_root": args.results_root, "out_root": str(out_root), "device_id": args.device_id, "total_questions": len(questions), "stages": {}}
    qwen_proc: Optional[subprocess.Popen] = None
    started = time.perf_counter()
    try:
        if bool(cfg.get("use_video_fallback", True)):
            qwen_proc, qwen_info = _start_qwen(args, out_root)
            summary["stages"]["qwen_server_start"] = qwen_info
            os.environ["VLM_HTTP_URL"] = f"http://127.0.0.1:{args.qwen_port}"
            os.environ["VLM_MODEL_NAME"] = "qwen-vl"
            os.environ["VLM_CHECKPOINT"] = args.qwen_model
            print(f"[qwen] {'reusing' if qwen_info['reused'] else 'ready'} on device {args.device_id} in {qwen_info['seconds']:.3f}s")

        logger, pipeline = TraceLogger(str(log_dir)), QAPipeline(cfg)
        done = skipped = failed = 0
        work_started = time.perf_counter()
        for question in questions:
            qid = str(question.get("question_id") or "unknown")
            pred_path = pred_dir / f"{qid}_result.json"
            if args.resume and _prediction_done(pred_path):
                skipped += 1
                continue
            try:
                state, trace, txt_lines = pipeline.run_one(question)
                write_json(str(pred_path), state)
                logger.save_trace(qid, trace)
                logger.save_text_log(qid, txt_lines)
                done += 1
                print(f"[question] success {qid}")
            except Exception as exc:
                failed += 1
                write_json(str(err_dir / f"{qid}_error.json"), {"question_id": qid, "error": str(exc), "traceback": traceback.format_exc()})
                print(f"[question] failed {qid}: {exc}")
        summary["stages"]["qa"] = {"seconds": round(time.perf_counter() - work_started, 3)}
        summary.update({"done": done, "skipped": skipped, "failed": failed})
    finally:
        teardown = time.perf_counter()
        _stop(qwen_proc)
        summary["stages"]["server_shutdown"] = {"seconds": round(time.perf_counter() - teardown, 3), "owned_server": qwen_proc is not None}
        summary["finished_at"] = _now()
        summary["total_wall_seconds"] = round(time.perf_counter() - started, 3)
        write_json(str(out_root / "summary.json"), summary)
        print(f"[summary] {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
