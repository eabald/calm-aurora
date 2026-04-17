from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from cidoc_rag.embeddings.service import embed_text
from cidoc_rag.utils import clean_text
from cidoc_rag.vectorstore.faiss_store import build_index_flat_l2


def build_document(entry: Dict[str, Any]) -> str:
    entry_id = clean_text(entry.get("id"))
    label = clean_text(entry.get("label"))
    entry_type = clean_text(entry.get("type")).lower()

    if entry_type == "documentation":
        definition = clean_text(entry.get("definition"))
        source_file = clean_text(entry.get("source_file"))
        parts = [f"{entry_id} {label}".strip(" .")]
        if definition:
            parts.append(f"Content: {definition}")
        if source_file:
            parts.append(f"Source: {source_file}")
        return ". ".join(part for part in parts if part).strip()

    if entry_type == "class":
        definition = clean_text(entry.get("definition"))
        examples = clean_text(entry.get("examples"))
        related = entry.get("related_properties", [])
        related_text = clean_text(related if isinstance(related, list) else str(related))

        parts = [f"{entry_id} {label}".strip(" .")]
        if definition:
            parts.append(f"Definition: {definition}")
        if examples:
            parts.append(f"Examples: {examples}")
        if related_text:
            parts.append(f"Related properties: {related_text}")
        return ". ".join(part for part in parts if part).strip()

    definition = clean_text(entry.get("definition"))
    domain = clean_text(entry.get("domain"))
    range_value = clean_text(entry.get("range"))

    parts = [f"{entry_id} {label}".strip(" .")]
    if domain:
        parts.append(f"Domain: {domain}")
    if range_value:
        parts.append(f"Range: {range_value}")
    if definition:
        parts.append(f"Definition: {definition}")
    return ". ".join(part for part in parts if part).strip()


def build_index(documents: List[Dict[str, Any]]):
    if not documents:
        raise ValueError("No documents available to index.")
    vectors = np.ascontiguousarray(np.array([doc["embedding"] for doc in documents], dtype=np.float32))
    return build_index_flat_l2(vectors)


def embed_documents(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    total = len(entries)
    for idx, entry in enumerate(entries, start=1):
        text = build_document(entry)
        vector = embed_text(text)
        payload.append({"entry": entry, "text": text, "embedding": vector})
        print(f"[progress] Embedded {idx}/{total}")
    return payload
