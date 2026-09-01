from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.qa_pipeline import QAPipeline
from src.utils.json_utils import read_json, write_json
from src.utils.logging import TraceLogger
from src.utils.dataset_loader import load_videomme_questions


def _parse_scalar(text: str):
    t = text.strip()
    if t.lower() in {"true", "false"}:
        return t.lower() == "true"
    try:
        if "." in t:
            return float(t)
        return int(t)
    except Exception:
        pass
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        return t[1:-1]
    return t


def _simple_yaml_load(path: str) -> dict:
    data = {}
    current_list_key = None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or line.strip().startswith("#"):
                continue
            if line.lstrip().startswith("-") and current_list_key:
                item = line.split("-", 1)[1].strip()
                data[current_list_key].append(_parse_scalar(item))
                continue
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            if val == "":
                data[key] = []
                current_list_key = key
            else:
                data[key] = _parse_scalar(val)
                current_list_key = None
    return data


def load_cfg(path: str) -> dict:
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception:
        cfg = _simple_yaml_load(path)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping.")
    return cfg


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="KQA multi-agent video MCQ pipeline")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--question", default="")
    p.add_argument("--videomme-json", default="/root/dataset/videomme/videomme/videomme.json")
    p.add_argument("--results-root", default="/root/results")
    p.add_argument("--question-id", default="")
    p.add_argument("--top-n-videos", type=int, default=1)
    p.add_argument("--max-questions", type=int, default=1)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = load_cfg(args.config)
    cfg["results_root"] = args.results_root or cfg.get("results_root")

    if args.question:
        questions = [read_json(args.question)]
    else:
        questions = load_videomme_questions(
            videomme_json=args.videomme_json,
            results_root=cfg["results_root"],
            top_n_videos=args.top_n_videos,
            max_questions=args.max_questions,
            question_id=args.question_id or None,
        )
    if not questions:
        raise ValueError("No valid questions loaded. Check videomme path/results root/question-id filters.")

    pipeline = QAPipeline(cfg)

    out_predictions = Path("outputs/predictions")
    out_logs = Path("outputs/logs")
    out_predictions.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)
    logger = TraceLogger(str(out_logs))
    for question_obj in questions:
        state_dict, trace, txt_lines = pipeline.run_one(question_obj)
        qid = question_obj.get("question_id", "unknown")
        write_json(str(out_predictions / f"{qid}_result.json"), state_dict)
        trace_path = logger.save_trace(qid, trace)
        text_log_path = logger.save_text_log(qid, txt_lines)
        print("Pipeline finished")
        print(f"question_id={qid}")
        print(f"trace={trace_path}")
        print(f"text_log={text_log_path}")


if __name__ == "__main__":
    main()
