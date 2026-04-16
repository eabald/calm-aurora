from __future__ import annotations

import json
from typing import Literal

Mode = Literal["qa", "mapping"]
TurnIntent = Literal["domain", "followup", "smalltalk", "other"]


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


def detect_turn_intent(query: str) -> TurnIntent:
    lowered = query.strip().lower()
    if lowered in {"hi", "hello", "hey", "thanks", "thank you"}:
        return "smalltalk"
    if any(token in lowered for token in ["this", "that", "it", "which one", "same one", "more", "also"]):
        return "followup"
    if any(token in lowered for token in ["cidoc", "crm", "rdf", "class", "property", "mapping", "e", "p"]):
        return "domain"
    return "other"
