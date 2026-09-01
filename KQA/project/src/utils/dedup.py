from __future__ import annotations

import re
from typing import Dict, List, Tuple


def _tokens(text: str) -> set:
    return set(re.findall(r"[\w\u4e00-\u9fff]+", (text or "").lower()))


def jaccard_text(a: str, b: str) -> float:
    sa = _tokens(a)
    sb = _tokens(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


def merge_evidence_with_dedup(
    existing: List[Dict],
    incoming: List[Dict],
    threshold: float,
) -> Tuple[List[Dict], List[Dict]]:
    merged = list(existing)
    dropped = []
    for item in incoming:
        dup_found = False
        for old in merged:
            if old.get("memory_id") == item.get("memory_id"):
                dup_found = True
                break
            sim = jaccard_text(old.get("text", ""), item.get("text", ""))
            if sim >= threshold:
                dup_found = True
                break
        if dup_found:
            dropped.append({"memory_id": item.get("memory_id", ""), "reason": "duplicate"})
            continue
        merged.append(item)
    return merged, dropped
