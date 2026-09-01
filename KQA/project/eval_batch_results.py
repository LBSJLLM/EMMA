from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_gold_map(videomme_json: Optional[str]) -> Dict[str, str]:
    if not videomme_json:
        return {}
    p = Path(videomme_json)
    if not p.exists():
        return {}

    data = _load_json(p)
    if not isinstance(data, list):
        return {}

    out: Dict[str, str] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        qid = str(item.get("question_id") or "").strip()
        ans = str(item.get("answer") or "").strip().upper()
        if qid and ans:
            out[qid] = ans
    return out


def _iter_prediction_files(batch_dir: Path) -> List[Path]:
    # New layout: <batch_dir>/predictions/*_result.json
    flat_files = list(batch_dir.glob("predictions/*_result.json"))
    shard_files = list(batch_dir.glob("shard_*/predictions/*_result.json"))
    return sorted(set(flat_files + shard_files))


def _extract_pred_gold(obj: Dict, gold_map: Dict[str, str]) -> Tuple[str, str, str, str]:
    qinfo = obj.get("question_info") if isinstance(obj.get("question_info"), dict) else {}
    qid = str(qinfo.get("question_id") or "").strip()
    vid = str(qinfo.get("video_id") or qinfo.get("videoID") or "").strip()

    final_answer = obj.get("final_answer") if isinstance(obj.get("final_answer"), dict) else {}
    pred = str(final_answer.get("predicted_option") or "").strip().upper()

    meta = qinfo.get("metadata") if isinstance(qinfo.get("metadata"), dict) else {}
    gold = str(meta.get("gold_option") or "").strip().upper()
    if not gold and qid in gold_map:
        gold = gold_map[qid]

    return qid, vid, pred, gold


def evaluate(batch_dir: Path, videomme_json: Optional[str]) -> Dict:
    gold_map = _load_gold_map(videomme_json)
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

        qid, vid, pred, gold = _extract_pred_gold(obj, gold_map)
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
    p = argparse.ArgumentParser(description="Evaluate a batch output folder against gold answers")
    p.add_argument("--batch-dir", required=True)
    p.add_argument("--videomme-json", default="/root/dataset/videomme/videomme/videomme.json")
    p.add_argument("--out-json", default="")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    batch_dir = Path(args.batch_dir)
    if not batch_dir.exists():
        raise ValueError(f"Batch dir not found: {batch_dir}")

    summary = evaluate(batch_dir, args.videomme_json)
    out_json = Path(args.out_json) if args.out_json else (batch_dir / "eval_summary.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"batch_dir={summary['batch_dir']}")
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
