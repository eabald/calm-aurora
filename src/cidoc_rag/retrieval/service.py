from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from cidoc_rag.config import DEFAULT_INDEX_PATH, DEFAULT_METADATA_PATH
from cidoc_rag.embeddings.service import embed_text
from cidoc_rag.utils import clean_text
from cidoc_rag.vectorstore.faiss_store import load_index, load_metadata, search_index


def retrieve(
    query: str,
    k: int = 5,
    index_path: str = DEFAULT_INDEX_PATH,
    metadata_path: str = DEFAULT_METADATA_PATH,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    if k <= 0:
        raise ValueError("k must be greater than 0")

    index = load_index(index_path)
    metadata = load_metadata(metadata_path)

    if index.ntotal == 0 or not metadata:
        return []

    if index.ntotal != len(metadata):
        print(
            "[warn] Index and metadata counts differ: "
            f"{index.ntotal} vs {len(metadata)}. Retrieval may skip some hits."
        )

    top_k = min(k, index.ntotal)
    query_vector = np.ascontiguousarray(np.array([embed_text(query)], dtype=np.float32))
    _, indices = search_index(index=index, query_vector=query_vector, k=top_k)

    entries: List[Dict[str, Any]] = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(metadata):
            continue
        entry = metadata[int(idx)]
        if isinstance(entry, dict):
            entries.append(entry)

    if debug:
        debug_ids = [clean_text(entry.get("id")) for entry in entries if clean_text(entry.get("id"))]
        print(f"[debug] Retrieved IDs: {', '.join(debug_ids) if debug_ids else 'none'}")
    return entries
