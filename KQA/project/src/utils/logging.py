from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .json_utils import write_json


class TraceLogger:
    def __init__(self, out_dir: str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def save_trace(self, question_id: str, trace_obj: Dict) -> str:
        path = self.out_dir / f"{question_id}_trace.json"
        write_json(str(path), trace_obj)
        return str(path)

    def save_text_log(self, question_id: str, lines: List[str]) -> str:
        path = self.out_dir / f"{question_id}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return str(path)
