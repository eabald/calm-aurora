from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from cidoc_rag.config import DEFAULT_OUT_DIR
from cidoc_rag.ingestion.loader import load_raw_data
from cidoc_rag.ingestion.normalize import normalize_entry
from cidoc_rag.ingestion.pipeline import build_document, build_index
from cidoc_rag.embeddings.service import embed_text
from cidoc_rag.utils import clean_text
from cidoc_rag.vectorstore.faiss_store import save_index, save_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calm Aurora CLI: ingest data and build FAISS index.")
    parser.add_argument("input_path", help="Path to CIDOC file or directory containing CIDOC files.")
    parser.add_argument("--text-delimiter", default=None, help="Optional delimiter for text file splits.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory for artifacts.")
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    print("[info] Loading raw CIDOC data...")
    raw_entries = load_raw_data(args.input_path, text_delimiter=args.text_delimiter)
    print(f"[info] Loaded raw entries: {len(raw_entries)}")

    if not raw_entries:
        print("[error] No entries found. Provide JSON/JSONL/TXT/MD/RDF input files (.ttl, .rdf, .owl, .nt, .n3, .xml).")
        return 1

    print("[info] Normalizing entries...")
    normalized_entries: List[Dict[str, object]] = []
    skipped_normalization = 0
    for raw in raw_entries:
        try:
            normalized_entries.append(normalize_entry(raw))
        except Exception as exc:
            skipped_normalization += 1
            print(f"[warn] Failed to normalize an entry: {exc}")

    print(f"[info] Normalized entries: {len(normalized_entries)} (skipped: {skipped_normalization})")
    if not normalized_entries:
        print("[error] No valid normalized entries to process.")
        return 1

    print("[info] Building embedding documents...")
    documents_payload = [{"entry": entry, "text": build_document(entry)} for entry in normalized_entries]

    print("[info] Generating embeddings...")
    embedded_documents: List[Dict[str, Any]] = []
    skipped_embeddings = 0
    total = len(documents_payload)

    for idx, payload in enumerate(documents_payload, start=1):
        text = str(payload["text"])
        entry = payload["entry"]
        try:
            vector = embed_text(text)
            embedded_documents.append({"entry": entry, "text": text, "embedding": vector})
            print(f"[progress] Embedded {idx}/{total}")
        except Exception as exc:
            skipped_embeddings += 1
            entry_id = clean_text(entry.get("id") if isinstance(entry, dict) else "")
            print(f"[warn] Embedding failed for entry '{entry_id}': {exc}")

    print(f"[info] Embedded entries: {len(embedded_documents)} (skipped: {skipped_embeddings})")
    if not embedded_documents:
        print("[error] No embeddings generated. Check API key, model, and input quality.")
        return 1

    print("[info] Building FAISS index...")
    index = build_index(embedded_documents)
    metadata: List[Dict[str, Any]] = [
        doc["entry"]
        for doc in embedded_documents
        if isinstance(doc.get("entry"), dict)
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "cidoc.index"
    metadata_path = out_dir / "cidoc_metadata.json"

    print("[info] Saving artifacts...")
    save_index(index, str(index_path))
    save_metadata(metadata, str(metadata_path))

    print("[done] Ingestion complete.")
    print(f"[done] Index path: {index_path}")
    print(f"[done] Metadata path: {metadata_path}")
    print(f"[done] Vector count: {index.ntotal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
