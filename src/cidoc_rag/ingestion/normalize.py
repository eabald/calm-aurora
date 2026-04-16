from __future__ import annotations

import re
from typing import Any, Dict, List

from cidoc_rag.utils import clean_text


def _guess_type(raw_entry: Dict[str, Any], entry_id: str) -> str:
    explicit_type = clean_text(raw_entry.get("type")).lower()
    if explicit_type in {"class", "property"}:
        return explicit_type

    if entry_id.upper().startswith("P"):
        return "property"
    if entry_id.upper().startswith("E"):
        return "class"

    raw_text = clean_text(raw_entry.get("raw_text")).lower()
    keys = {key.lower() for key in raw_entry.keys()}

    if {"domain", "range"}.intersection(keys) or "property" in raw_text:
        return "property"
    return "class"


def _extract_id(raw_entry: Dict[str, Any]) -> str:
    candidates = [
        raw_entry.get("id"),
        raw_entry.get("identifier"),
        raw_entry.get("cidoc_id"),
        raw_entry.get("code"),
    ]
    for candidate in candidates:
        value = clean_text(candidate)
        if value:
            match = re.search(r"\b([EP]\d+[A-Za-z0-9\.]*)\b", value)
            return match.group(1) if match else value

    text_blob = " ".join(clean_text(raw_entry.get(key)) for key in ["title", "label", "name", "raw_text"])
    match = re.search(r"\b([EP]\d+[A-Za-z0-9\.]*)\b", text_blob)
    return match.group(1) if match else ""


def _extract_first(raw_entry: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        if key in raw_entry:
            value = clean_text(raw_entry.get(key))
            if value:
                return value
    return ""


def normalize_entry(raw_entry: Dict[str, Any]) -> Dict[str, Any]:
    entry_id = _extract_id(raw_entry)
    entry_type = _guess_type(raw_entry, entry_id)

    label = _extract_first(raw_entry, ["label", "name", "title"])
    definition = _extract_first(raw_entry, ["definition", "scope_note", "description", "raw_text"])

    if entry_type == "class":
        examples = _extract_first(raw_entry, ["examples", "example"])
        related_props_raw: Any = raw_entry.get("related_properties", raw_entry.get("related_props", []))

        if isinstance(related_props_raw, list):
            related_properties = [clean_text(item) for item in related_props_raw if clean_text(item)]
        else:
            related_properties = [
                clean_text(item)
                for item in re.split(r"[,;]", clean_text(related_props_raw))
                if clean_text(item)
            ]

        return {
            "id": clean_text(entry_id),
            "type": "class",
            "label": label,
            "definition": definition,
            "examples": examples,
            "related_properties": related_properties,
        }

    domain = _extract_first(raw_entry, ["domain", "domain_class"])
    range_value = _extract_first(raw_entry, ["range", "range_class"])

    return {
        "id": clean_text(entry_id),
        "type": "property",
        "label": label,
        "domain": domain,
        "range": range_value,
        "definition": definition,
    }
