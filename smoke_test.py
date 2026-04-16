#!/usr/bin/env python3
"""End-to-end smoke test for CIDOC ingestion + retrieval pipeline.

The script:
1. Creates a small sample CIDOC dataset.
2. Runs ingestion (normalize -> embed -> index -> save).
3. Reloads artifacts and runs one semantic query.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from hashlib import sha256
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cidoc_rag.embeddings.service import embed_text
from cidoc_rag.config import get_embedding_model, get_ollama_base_url
from cidoc_rag.ingestion.loader import load_raw_data
from cidoc_rag.ingestion.normalize import normalize_entry
from cidoc_rag.ingestion.pipeline import build_document, build_index
from cidoc_rag.retrieval.service import retrieve
from cidoc_rag.vectorstore.faiss_store import (
    load_index,
    load_metadata,
    save_index,
    save_metadata,
    search_index,
)
from cidoc_rag.cli.retrieve_cli import print_results


def write_sample_data(sample_path: Path) -> None:
    sample_entries: List[Dict[str, object]] = [
        {
            "id": "E21",
            "type": "class",
            "label": "Person",
            "definition": "This class comprises real persons who live or are assumed to have lived.",
            "examples": "Leonardo da Vinci",
            "related_properties": ["P14 carried out by"],
        },
        {
            "id": "P14",
            "type": "property",
            "label": "carried out by",
            "domain": "E7 Activity",
            "range": "E39 Actor",
            "definition": "This property describes the active participation of an actor in an activity.",
        },
    ]

    sample_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sample_path, "w", encoding="utf-8") as handle:
        json.dump(sample_entries, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end CIDOC smoke test.")
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Run smoke test without Ollama API calls using deterministic local embeddings.",
    )
    parser.add_argument(
        "--offline-dim",
        type=int,
        default=64,
        help="Embedding dimension used only in --no-api mode.",
    )
    return parser.parse_args()


def offline_embed_text(text: str, dim: int) -> List[float]:
    """Create a deterministic local embedding vector from text for offline testing."""
    if dim <= 0:
        raise ValueError("offline embedding dimension must be greater than 0")

    vector = np.zeros(dim, dtype=np.float32)
    for token in text.lower().split():
        digest = sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], byteorder="big") % dim
        vector[bucket] += 1.0

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector.tolist()


def offline_search(
    query: str,
    index: faiss.Index,
    metadata: List[Dict[str, object]],
    top_k: int,
    dim: int,
) -> List[Dict[str, object]]:
    query_vector = np.ascontiguousarray(np.array([offline_embed_text(query, dim=dim)], dtype=np.float32))
    _, indices = search_index(index=index, query_vector=query_vector, k=top_k)

    results: List[Dict[str, object]] = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(metadata):
            continue
        results.append(metadata[idx])
    return results


def main() -> int:
    load_dotenv()
    args = parse_args()

    if not args.no_api:
        base_url = get_ollama_base_url().rstrip("/")
        embedding_model = get_embedding_model()
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            response.raise_for_status()
            model_info = response.json().get("models", [])
            model_names = {item.get("name") for item in model_info if isinstance(item, dict)}
        except requests.RequestException:
            print(f"[error] Ollama is not reachable at {base_url}. Start Ollama and retry.")
            return 1

        if embedding_model not in model_names:
            print(
                f"[error] Ollama embedding model '{embedding_model}' is not installed. "
                f"Run: ollama pull {embedding_model}"
            )
            return 1

    smoke_root = Path("data/smoke")
    sample_path = smoke_root / "cidoc_sample.json"
    out_dir = smoke_root / "vectorstore"
    index_path = out_dir / "cidoc.index"
    metadata_path = out_dir / "cidoc_metadata.json"

    print("[info] Writing sample CIDOC data...")
    write_sample_data(sample_path)

    print("[info] Loading and normalizing sample data...")
    raw_entries = load_raw_data(str(sample_path))
    normalized_entries = [normalize_entry(entry) for entry in raw_entries]

    print(f"[info] Normalized entries: {len(normalized_entries)}")
    if not normalized_entries:
        print("[error] Smoke test failed: no normalized entries.")
        return 1

    print("[info] Building documents and embeddings...")
    embedded_documents = []
    total = len(normalized_entries)
    for idx, entry in enumerate(normalized_entries, start=1):
        text = build_document(entry)
        if args.no_api:
            vector = offline_embed_text(text, dim=args.offline_dim)
        else:
            vector = embed_text(text)
        embedded_documents.append({"entry": entry, "text": text, "embedding": vector})
        print(f"[progress] Embedded {idx}/{total}")

    print("[info] Building and saving FAISS artifacts...")
    index = build_index(embedded_documents)
    metadata = [item["entry"] for item in embedded_documents]

    out_dir.mkdir(parents=True, exist_ok=True)
    save_index(index, str(index_path))
    save_metadata(metadata, str(metadata_path))

    print(f"[info] Saved index: {index_path}")
    print(f"[info] Saved metadata: {metadata_path}")

    print("[info] Running retrieval smoke query...")
    query = "Who carried out an activity?"
    loaded_index = load_index(str(index_path))
    loaded_metadata = load_metadata(str(metadata_path))
    if args.no_api:
        results = offline_search(
            query=query,
            index=loaded_index,
            metadata=loaded_metadata,
            top_k=2,
            dim=args.offline_dim,
        )
    else:
        results = retrieve(
            query=query,
            k=2,
            index_path=str(index_path),
            metadata_path=str(metadata_path),
        )

    print(f"[info] Query: {query}")
    print_results(results)

    print("[done] Smoke test completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
