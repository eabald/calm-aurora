from __future__ import annotations

import json
from typing import Literal

Mode = Literal["qa", "mapping"]


def detect_mode(query: str) -> Mode:
    lowered = query.lower()
    if any(token in lowered for token in ["table", "fields", "{", "}"]):
        return "mapping"

    stripped = query.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return "mapping"

    try:
        payload = json.loads(query)
        if isinstance(payload, (dict, list)):
            return "mapping"
    except json.JSONDecodeError:
        pass

    return "qa"
