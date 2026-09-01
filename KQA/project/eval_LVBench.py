from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_lvbench_records(lvbench_file: Path):
    suffix = lvbench_file.suffix.lower()

    if suffix == ".jsonl":
        with open(lvbench_file, "r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj
        return

    if suffix == ".json":
        data = _load_json(lvbench_file)
        if isinstance(data, list):
            for obj in data:
                if isinstance(obj, dict):
                    yield obj
            return
        raise ValueError(f"LVBench json must be a list: {lvbench_file}")

    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore

            table = pq.read_table(str(lvbench_file))
            for obj in table.to_pylist():
                if isinstance(obj, dict):
                    yield obj
            return
        except Exception:
            pass

        try:
            import pandas as pd  # type: ignore

            df = pd.read_parquet(str(lvbench_file))
            for obj in df.to_dict(orient="records"):
                if isinstance(obj, dict):
                    yield obj
            return
        except Exception as exc:
            raise RuntimeError(
                "Failed to read parquet file. Install pyarrow or pandas with parquet support."
            ) from exc

    raise ValueError(f"Unsupported LVBench file format: {lvbench_file.suffix}")


def _build_lvbench_gold_maps(lvbench_file: Optional[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Return two maps:
    1) qid_gold:   "<video_id>_<uid>" -> "A/B/C/D"
    2) uid_gold:   "<uid>" -> "A/B/C/D" (fallback when qid cannot be reconstructed)
    """
    if not lvbench_file:
        return {}, {}

    p = Path(lvbench_file)
    if not p.exists():
        return {}, {}

    qid_gold: Dict[str, str] = {}
    uid_gold: Dict[str, str] = {}

    for item in _iter_lvbench_records(p):
        video_key = str(item.get("key") or "").strip()
        video_path = str(item.get("video_path") or "").strip()
        video_id = video_key or Path(video_path).stem

        uid = str(item.get("uid") or "").strip()
        gold = str(item.get("answer") or "").strip().upper()
        if gold:
            uid_gold[uid] = gold
            if video_id and uid:
                qid_gold[f"{video_id}_{uid}"] = gold

    return qid_gold, uid_gold


def _iter_prediction_files(batch_dir: Path) -> List[Path]:
    # New layout: <batch_dir>/predictions/*_result.json
    flat_files = list(batch_dir.glob("predictions/*_result.json"))
    shard_files = list(batch_dir.glob("shard_*/predictions/*_result.json"))
    return sorted(set(flat_files + shard_files))


def _extract_pred_gold(
    obj: Dict, qid_gold_map: Dict[str, str], uid_gold_map: Dict[str, str]
) -> Tuple[str, str, str, str]:
    qinfo = obj.get("question_info") if isinstance(obj.get("question_info"), dict) else {}
    qid = str(qinfo.get("question_id") or "").strip()
    vid = str(qinfo.get("video_id") or qinfo.get("videoID") or "").strip()

    final_answer = obj.get("final_answer") if isinstance(obj.get("final_answer"), dict) else {}
    pred = str(final_answer.get("predicted_option") or "").strip().upper()

    meta = qinfo.get("metadata") if isinstance(qinfo.get("metadata"), dict) else {}
    gold = str(meta.get("gold_option") or "").strip().upper()

    if not gold and qid and qid in qid_gold_map:
        gold = qid_gold_map[qid]

    if not gold:
        uid = str(meta.get("uid") or "").strip()
        if uid and uid in uid_gold_map:
            gold = uid_gold_map[uid]

    return qid, vid, pred, gold


def evaluate(batch_dir: Path, lvbench_file: Optional[str]) -> Dict:
    qid_gold_map, uid_gold_map = _build_lvbench_gold_maps(lvbench_file)
    pred_files = _iter_prediction_files(batch_dir)
    legal_options = {"A", "B", "C", "D"}

    total = 0
    valid = 0
    correct = 0
    missing_gold = 0
    missing_pred = 0
    illegal_pred = 0
    wrong_items: List[Dict] = []
    invalid_items: List[Dict] = []

    per_video_total: Dict[str, int] = {}
    per_video_correct: Dict[str, int] = {}

    for fp in pred_files:
        total += 1
        obj = _load_json(fp)
        if not isinstance(obj, dict):
            invalid_items.append({"file": str(fp), "reason": "prediction_json_not_object"})
            continue

        qid, vid, pred, gold = _extract_pred_gold(obj, qid_gold_map, uid_gold_map)
        if not qid:
            invalid_items.append({"file": str(fp), "reason": "missing_question_id"})
            continue
        if not pred:
            missing_pred += 1
            invalid_items.append({"question_id": qid, "video_id": vid, "reason": "missing_predicted_option"})
            continue
        if pred not in legal_options:
            illegal_pred += 1
            invalid_items.append(
                {
                    "question_id": qid,
                    "video_id": vid,
                    "reason": "illegal_predicted_option",
                    "predicted": pred,
                }
            )
            continue
        if not gold:
            missing_gold += 1
            invalid_items.append({"question_id": qid, "video_id": vid, "reason": "missing_gold_option"})
            continue

        valid += 1
        per_video_total[vid] = per_video_total.get(vid, 0) + 1
        is_correct = pred == gold
        if is_correct:
            correct += 1
            per_video_correct[vid] = per_video_correct.get(vid, 0) + 1
        else:
            wrong_items.append(
                {
                    "question_id": qid,
                    "video_id": vid,
                    "predicted": pred,
                    "gold": gold,
                    "prediction_file": str(fp),
                }
            )

    accuracy = (correct / valid) if valid > 0 else 0.0
    per_video = []
    for vid in sorted(per_video_total.keys()):
        vt = per_video_total[vid]
        vc = per_video_correct.get(vid, 0)
        per_video.append(
            {
                "video_id": vid,
                "total": vt,
                "correct": vc,
                "accuracy": (vc / vt) if vt > 0 else 0.0,
            }
        )

    return {
        "batch_dir": str(batch_dir),
        "lvbench_file": str(lvbench_file) if lvbench_file else "",
        "prediction_file_count": len(pred_files),
        "total_seen": total,
        "valid_scored": valid,
        "correct": correct,
        "wrong": valid - correct,
        "accuracy": accuracy,
        "missing_gold": missing_gold,
        "missing_pred": missing_pred,
        "illegal_pred": illegal_pred,
        "invalid_items": invalid_items,
        "wrong_items": wrong_items,
        "wrong_question_ids": [x["question_id"] for x in wrong_items],
        "per_video": per_video,
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate LVBench batch outputs against gold answers")
    p.add_argument("--batch-dir", required=True)
    p.add_argument("--lvbench-file", default="")
    p.add_argument("--out-json", default="")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    batch_dir = Path(args.batch_dir)
    if not batch_dir.exists():
        raise ValueError(f"Batch dir not found: {batch_dir}")

    summary = evaluate(batch_dir, args.lvbench_file)
    out_json = Path(args.out_json) if args.out_json else (batch_dir / "eval_summary_lvbench.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"batch_dir={summary['batch_dir']}")
    print(f"lvbench_file={summary['lvbench_file']}")
    print(f"prediction_file_count={summary['prediction_file_count']}")
    print(f"valid_scored={summary['valid_scored']}")
    print(f"correct={summary['correct']}")
    print(f"wrong={summary['wrong']}")
    print(f"accuracy={summary['accuracy']:.4f}")
    print(f"missing_gold={summary['missing_gold']} missing_pred={summary['missing_pred']}")
    print(f"illegal_pred={summary['illegal_pred']}")
    print(f"eval_summary={out_json}")


if __name__ == "__main__":
    main()
