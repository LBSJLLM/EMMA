from __future__ import annotations

import re
from typing import Dict


COUNTING_PAT = re.compile(r"how many|how much|total number|几次|多少", re.IGNORECASE)
CAUSAL_PAT = re.compile(r"why|原因|为什么|because|cause", re.IGNORECASE)
TEMPORAL_PAT = re.compile(r"before|after|first|last|then|顺序|先|后", re.IGNORECASE)
SPATIAL_PAT = re.compile(r"left|right|behind|front|位置|左边|右边", re.IGNORECASE)
IDENTITY_PAT = re.compile(r"who|identity|role|职业|是谁|身份", re.IGNORECASE)
HABIT_PAT = re.compile(r"usually|habit|always|tend to|经常|习惯", re.IGNORECASE)
ATTRIBUTE_PAT = re.compile(r"color|wearing|looks like|属性|特征|颜色", re.IGNORECASE)


def normalize_options(options: Dict[str, str]) -> Dict[str, str]:
    return {str(k).strip().upper(): str(v).strip() for k, v in options.items() if str(v).strip()}


def classify_question_type(question: str) -> str:
    q = (question or "").strip()
    if COUNTING_PAT.search(q):
        return "counting"
    if CAUSAL_PAT.search(q):
        return "causal_why"
    if TEMPORAL_PAT.search(q):
        return "temporal_order"
    if IDENTITY_PAT.search(q):
        return "identity_or_role"
    if HABIT_PAT.search(q):
        return "long_term_habit_or_fact"
    if SPATIAL_PAT.search(q):
        return "spatial_relation"
    if ATTRIBUTE_PAT.search(q):
        return "descriptive_attribute"
    return "open_other"
