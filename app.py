#!/usr/bin/env python3
"""Interactive CLI chat for CIDOC CRM RAG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cidoc_rag.config import DEFAULT_INDEX_PATH, DEFAULT_METADATA_PATH
from cidoc_rag.cli.chat_runtime import (
    apply_runtime_command,
    call_llm,
    chat_loop,
    format_answer_for_mode,
    format_retrieved_ids,
    truncate_history,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIDOC CRM interactive RAG chat")
    parser.add_argument("--k", type=int, default=5, help="Initial top-k retrieval count")
    parser.add_argument("--index-path", default=DEFAULT_INDEX_PATH, help="Path to FAISS index file")
    parser.add_argument("--metadata-path", default=DEFAULT_METADATA_PATH, help="Path to metadata JSON file")
    parser.add_argument("--history-turns", type=int, default=4, help="Number of recent turns to keep in memory")
    parser.add_argument("--debug", action="store_true", help="Print retrieved context for each turn")
    return parser.parse_args()


def _handle_missing_dependency(exc: ModuleNotFoundError) -> int:
    message = str(exc).lower()
    if "faiss" in message:
        print("[error] Retrieval backend is unavailable: faiss is not installed.")
        print("[hint] Install dependencies with: python -m pip install -r requirements.txt")
        return 1
    raise exc


def main() -> int:
    load_dotenv()
    args = parse_args()

    if args.k <= 0:
        print("[error] --k must be greater than 0")
        return 1
    if args.history_turns <= 0:
        print("[error] --history-turns must be greater than 0")
        return 1

    try:
        chat_loop(
            k=args.k,
            index_path=args.index_path,
            metadata_path=args.metadata_path,
            history_turns=args.history_turns,
            debug=args.debug,
        )
    except ModuleNotFoundError as exc:
        return _handle_missing_dependency(exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
