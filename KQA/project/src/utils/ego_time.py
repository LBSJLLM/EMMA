from __future__ import annotations

import re

SECONDS_PER_DAY = 86400.0


def seconds_to_ego_timestamp(total_s: float) -> str:
    """Convert global seconds to 'DAY{d} HH:MM:SS' format."""
    total_s = max(0.0, float(total_s))
    day = int(total_s // SECONDS_PER_DAY) + 1
    rem = total_s % SECONDS_PER_DAY
    h = int(rem // 3600)
    m = int((rem % 3600) // 60)
    s = int(rem % 60)
    return f"DAY{day} {h:02d}:{m:02d}:{s:02d}"


def ego_timestamp_to_seconds(ts: str) -> float | None:
    """Parse 'DAY{d} HH:MM:SS' (or plain 'HH:MM:SS') back to global seconds. Returns None on failure."""
    ts = str(ts or "").strip()
    m = re.match(r"DAY(\d+)\s+(\d{1,2}):(\d{2}):(\d{2})", ts)
    if m:
        day = int(m.group(1))
        h, mn, s = int(m.group(2)), int(m.group(3)), int(m.group(4))
        return (day - 1) * SECONDS_PER_DAY + h * 3600 + mn * 60 + float(s)
    m2 = re.match(r"(\d{1,2}):(\d{2}):(\d{2})", ts)
    if m2:
        h, mn, s = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
        return float(h * 3600 + mn * 60 + s)
    return None


def ego_span(time_span: list) -> list[str] | None:
    """Convert [start_s, end_s] to ['DAY{d} HH:MM:SS', 'DAY{d} HH:MM:SS']."""
    if not (isinstance(time_span, list) and len(time_span) == 2):
        return None
    try:
        return [seconds_to_ego_timestamp(float(time_span[0])),
                seconds_to_ego_timestamp(float(time_span[1]))]
    except Exception:
        return None
