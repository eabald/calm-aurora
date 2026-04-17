from __future__ import annotations

import argparse

from dotenv import load_dotenv

from cidoc_rag.config import DEFAULT_INDEX_PATH, DEFAULT_METADATA_PATH
from cidoc_rag.retrieval.service import retrieve
from cidoc_rag.utils import clean_text


def print_results(results):
    if not results:
        print("[info] No results found.")
        return

    for rank, entry in enumerate(results, start=1):
        entry_id = clean_text(entry.get("id"))
        entry_type = clean_text(entry.get("type"))
        label = clean_text(entry.get("label"))
        definition = clean_text(entry.get("definition"))

        print(f"\n[{rank}] {entry_id} ({entry_type}) - {label}")
        if definition:
            print(f"Definition: {definition}")

        if entry_type.lower() == "property":
            domain = clean_text(entry.get("domain"))
            range_value = clean_text(entry.get("range"))
            if domain:
                print(f"Domain: {domain}")
            if range_value:
                print(f"Range: {range_value}")
        elif entry_type.lower() == "class":
            examples = clean_text(entry.get("examples"))
            related = entry.get("related_properties", [])
            if examples:
                print(f"Examples: {examples}")
            if isinstance(related, list) and related:
                related_text = ", ".join(clean_text(item) for item in related if clean_text(item))
                if related_text:
                    print(f"Related properties: {related_text}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calm Aurora CLI: run semantic search over FAISS index.")
    parser.add_argument("query", help="Natural language query to search Calm Aurora entries.")
    parser.add_argument("--index-path", default=DEFAULT_INDEX_PATH, help="Path to FAISS index file.")
    parser.add_argument("--metadata-path", default=DEFAULT_METADATA_PATH, help="Path to metadata JSON file.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of nearest entries to return.")
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    if args.top_k <= 0:
        print("[error] --top-k must be greater than 0.")
        return 1

    print("[info] Running retrieval...")
    results = retrieve(
        query=args.query,
        k=args.top_k,
        index_path=args.index_path,
        metadata_path=args.metadata_path,
    )
    print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
