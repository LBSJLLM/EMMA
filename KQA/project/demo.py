from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.main import load_cfg
from src.pipeline.controller import QAController
from src.utils.dataset_loader import load_videomme_questions
from src.utils.json_utils import read_json, write_json


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except Exception:
        return str(value)


def _truncate_text(value: Any, max_len: int = 320) -> Any:
    if isinstance(value, str):
        if len(value) <= max_len:
            return value
        return value[:max_len] + f" ...<truncated {len(value) - max_len} chars>"
    if isinstance(value, list):
        return [_truncate_text(x, max_len=max_len) for x in value]
    if isinstance(value, dict):
        return {k: _truncate_text(v, max_len=max_len) for k, v in value.items()}
    return value


class FlowLogger:
    def __init__(self, pretty_print_full: bool = False):
        self.pretty_print_full = pretty_print_full
        self.events: List[Dict[str, Any]] = []
        self.counter = 0

    def log(self, stage: str, direction: str, payload: Dict[str, Any]) -> None:
        self.counter += 1
        event = {
            "step": self.counter,
            "timestamp": _now(),
            "stage": stage,
            "direction": direction,
            "payload": _safe(copy.deepcopy(payload)),
        }
        self.events.append(event)

        display = event["payload"] if self.pretty_print_full else _truncate_text(event["payload"])
        print(f"\n[{event['step']:03d}] {event['timestamp']} {stage} {direction}")
        print(json.dumps(display, ensure_ascii=False, indent=2))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Verbose KQA demo runner with full intermediate I/O")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--question", default="", help="Single question json path")
    p.add_argument("--videomme-json", default="/root/dataset/videomme/videomme/videomme.json")
    p.add_argument("--results-root", default="/root/results")
    p.add_argument("--question-id", default="")
    p.add_argument("--top-n-videos", type=int, default=1)
    p.add_argument("--max-questions", type=int, default=1)
    p.add_argument("--full-print", action="store_true", help="Print non-truncated payloads to stdout")
    p.add_argument("--out-dir", default="outputs/demo")
    return p


def load_questions(args: argparse.Namespace, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    if args.question:
        return [read_json(args.question)]
    return load_videomme_questions(
        videomme_json=args.videomme_json,
        results_root=cfg["results_root"],
        top_n_videos=args.top_n_videos,
        max_questions=args.max_questions,
        question_id=args.question_id or None,
    )


def attach_debug_hooks(controller: QAController, logger: FlowLogger) -> None:
    orig_query_plan = controller.query_agent.plan
    orig_search_run = controller.search_agent.run
    orig_retrieve = controller.retriever.retrieve
    orig_validate = controller.validation_agent.validate
    orig_vf_plan = controller.video_fallback_agent.plan
    orig_video_inspect = controller.video_reader.inspect
    orig_answer = controller.answer_agent.answer

    def wrapped_query_plan(*args, **kwargs):
        logger.log("query_agent.plan", "input", {"args": list(args), "kwargs": kwargs})
        out = orig_query_plan(*args, **kwargs)
        logger.log("query_agent.plan", "output", {"plan": out})
        return out

    def wrapped_search_run(*args, **kwargs):
        logger.log("search_agent.run", "input", {"args": list(args), "kwargs": kwargs})
        out = orig_search_run(*args, **kwargs)
        logger.log(
            "search_agent.run",
            "output",
            {
                "retrieved_count": len(out),
                "memory_ids": [str(x.get("memory_id") or "") for x in out],
                "levels": sorted({str(x.get("memory_level") or "") for x in out}),
            },
        )
        return out

    def wrapped_retrieve(*args, **kwargs):
        logger.log("retriever.retrieve", "input", {"args": list(args), "kwargs": kwargs})
        out = orig_retrieve(*args, **kwargs)
        logger.log(
            "retriever.retrieve",
            "output",
            {
                "count": len(out),
                "items": [x.__dict__ if hasattr(x, "__dict__") else str(x) for x in out],
            },
        )
        return out

    def wrapped_validate(*args, **kwargs):
        logger.log("validation_agent.validate", "input", {"args": list(args), "kwargs": kwargs})
        out = orig_validate(*args, **kwargs)
        logger.log("validation_agent.validate", "output", {"validation": out})
        return out

    def wrapped_vf_plan(*args, **kwargs):
        logger.log("video_fallback_agent.plan", "input", {"args": list(args), "kwargs": kwargs})
        out = orig_vf_plan(*args, **kwargs)
        logger.log("video_fallback_agent.plan", "output", {"video_plan": out})
        return out

    def wrapped_video_inspect(*args, **kwargs):
        logger.log("video_reader.inspect", "input", {"args": list(args), "kwargs": kwargs})
        out = orig_video_inspect(*args, **kwargs)
        logger.log("video_reader.inspect", "output", {"video_results": out})
        return out

    def wrapped_answer(*args, **kwargs):
        logger.log("answer_agent.answer", "input", {"args": list(args), "kwargs": kwargs})
        out = orig_answer(*args, **kwargs)
        logger.log("answer_agent.answer", "output", {"answer": out})
        return out

    controller.query_agent.plan = wrapped_query_plan
    controller.search_agent.run = wrapped_search_run
    controller.retriever.retrieve = wrapped_retrieve
    controller.validation_agent.validate = wrapped_validate
    controller.video_fallback_agent.plan = wrapped_vf_plan
    controller.video_reader.inspect = wrapped_video_inspect
    controller.answer_agent.answer = wrapped_answer


def main() -> None:
    args = build_argparser().parse_args()
    cfg = load_cfg(args.config)
    cfg["results_root"] = args.results_root or cfg.get("results_root")

    questions = load_questions(args, cfg)
    if not questions:
        raise ValueError("No valid questions loaded. Check your input arguments.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for q in questions:
        qid = str(q.get("question_id") or "unknown")
        print("\n" + "=" * 88)
        print(f"KQA DEMO START question_id={qid}")
        print("=" * 88)
        print(json.dumps(q, ensure_ascii=False, indent=2))

        logger = FlowLogger(pretty_print_full=bool(args.full_print))
        controller = QAController(cfg)
        attach_debug_hooks(controller, logger)

        logger.log("controller.run", "input", {"question_obj": q})
        state_dict, trace, txt_logs = controller.run(q)
        logger.log(
            "controller.run",
            "output",
            {
                "state_dict": state_dict,
                "trace": trace,
                "txt_logs": txt_logs,
            },
        )

        flow_path = out_dir / f"{qid}_flow.json"
        state_path = out_dir / f"{qid}_state.json"
        trace_path = out_dir / f"{qid}_trace.json"
        text_log_path = out_dir / f"{qid}.log"

        write_json(str(flow_path), {"question": q, "events": logger.events})
        write_json(str(state_path), state_dict)
        write_json(str(trace_path), trace)
        text_log_path.write_text("\n".join(txt_logs) + "\n", encoding="utf-8")

        print("\n" + "-" * 88)
        print("DEMO OUTPUT FILES")
        print(f"flow : {flow_path}")
        print(f"state: {state_path}")
        print(f"trace: {trace_path}")
        print(f"log  : {text_log_path}")
        print("-" * 88)


if __name__ == "__main__":
    main()
