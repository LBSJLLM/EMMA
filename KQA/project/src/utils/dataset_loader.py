from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set


_HOUSEKEEPING_DIR_NAMES = {
    "__pycache__",
    "_batch_logs",
    "_logs",
    "_tmp",
    "_cache",
}


def _is_housekeeping_dir(name: str) -> bool:
    n = str(name or "").strip()
    if not n:
        return True
    # Hidden/system folders should be ignored, but leading "_" can be a valid video_id.
    if n.startswith("."):
        return True
    return n in _HOUSEKEEPING_DIR_NAMES


def _opts_list_to_dict(options: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    letters = ["A", "B", "C", "D", "E", "F"]
    for idx, item in enumerate(options):
        if idx >= len(letters):
            break
        text = str(item or "").strip()
        if not text:
            continue
        prefix = f"{letters[idx]}."
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
        out[letters[idx]] = text
    return out


def _has_memory(results_root: Path, video_id: str) -> bool:
    out_dir = results_root / video_id / "outputs"
    return all((out_dir / name).exists() for name in ["short_term.json", "medium_term.json", "long_term.json"])


def list_video_ids_from_results_dirs(results_root: str) -> List[str]:
    rr = Path(results_root)
    if not rr.exists():
        return []
    out: List[str] = []
    for p in sorted(rr.iterdir()):
        if not p.is_dir():
            continue
        if _is_housekeeping_dir(p.name):
            continue
        out.append(p.name)
    return out


def load_videomme_questions_by_video_ids(
    videomme_json: str,
    allowed_video_ids: List[str],
    max_questions: Optional[int] = None,
    question_id: Optional[str] = None,
) -> List[Dict]:
    vm_path = Path(videomme_json)
    with open(vm_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    if not isinstance(items, list):
        raise ValueError("videomme json must be a list")

    allow: Set[str] = set(str(x).strip() for x in allowed_video_ids if str(x).strip())
    out: List[Dict] = []
    for obj in items:
        if not isinstance(obj, dict):
            continue
        qid = str(obj.get("question_id") or "").strip()
        if question_id and qid != question_id:
            continue

        vid = str(obj.get("videoID") or "").strip()
        if vid not in allow:
            continue

        options = obj.get("options") or []
        option_dict = _opts_list_to_dict(options if isinstance(options, list) else [])
        if not option_dict:
            continue

        out.append(
            {
                "video_id": vid,
                "question_id": qid,
                "question": str(obj.get("question") or "").strip(),
                "options": option_dict,
                "metadata": {
                    "dataset": "videomme",
                    "gold_option": str(obj.get("answer") or "").strip() or None,
                    "task_type": str(obj.get("task_type") or "").strip(),
                    "domain": str(obj.get("domain") or "").strip(),
                    "url": str(obj.get("url") or "").strip(),
                },
            }
        )

        if max_questions is not None and len(out) >= int(max_questions):
            break
    return out


def load_videomme_questions(
    videomme_json: str,
    results_root: str,
    top_n_videos: Optional[int] = None,
    max_questions: Optional[int] = None,
    question_id: Optional[str] = None,
) -> List[Dict]:
    rr = Path(results_root)
    available_videos = [
        p.name
        for p in sorted(rr.iterdir())
        if p.is_dir() and (not _is_housekeeping_dir(p.name)) and _has_memory(rr, p.name)
    ]
    if top_n_videos is not None:
        available_videos = available_videos[: max(1, int(top_n_videos))]
    return load_videomme_questions_by_video_ids(
        videomme_json=videomme_json,
        allowed_video_ids=available_videos,
        max_questions=max_questions,
        question_id=question_id,
    )
