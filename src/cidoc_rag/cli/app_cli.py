from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from cidoc_rag.cli.chat_runtime import chat_loop
from cidoc_rag.clients.ollama_client import get_ollama_client
from cidoc_rag.config import DEFAULT_INDEX_PATH, DEFAULT_METADATA_PATH
from cidoc_rag.prompting.builders import build_context, build_prompt
from cidoc_rag.prompting.modes import detect_mode
from cidoc_rag.retrieval.service import retrieve
from cidoc_rag.config import get_llm_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIDOC retrieval + prompt app")
    parser.add_argument("query", nargs="?", help="User question or schema JSON")
    parser.add_argument("--k", type=int, default=5, help="Top-k retrieval count")
    parser.add_argument("--index-path", default=DEFAULT_INDEX_PATH, help="Path to FAISS index file")
    parser.add_argument("--metadata-path", default=DEFAULT_METADATA_PATH, help="Path to metadata JSON file")
    parser.add_argument("--print-context", action="store_true", help="Print context before answer")
    parser.add_argument("--chat", action="store_true", help="Run interactive chat mode")
    parser.add_argument("--history-turns", type=int, default=4, help="Number of recent turns to keep in chat")
    parser.add_argument("--debug", action="store_true", help="Print debug context while in chat mode")
    return parser.parse_args()


def call_llm(prompt: str) -> str:
    model = get_llm_model()
    client = get_ollama_client()
    content = client.chat_completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return content.strip() if content else ""


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

    if args.chat:
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

    if not args.query:
        print("[error] query is required unless --chat is provided")
        return 1

    mode = detect_mode(args.query)
    print(f"[info] Mode: {mode}")

    if args.print_context:
        preview = retrieve(
            query=args.query,
            k=args.k,
            index_path=args.index_path,
            metadata_path=args.metadata_path,
        )
        print("[info] Retrieved context preview:")
        print(build_context(preview))

    try:
        results = retrieve(
            query=args.query,
            k=args.k,
            index_path=args.index_path,
            metadata_path=args.metadata_path,
        )
    except ModuleNotFoundError as exc:
        return _handle_missing_dependency(exc)
    context = build_context(results)
    prompt = build_prompt(query=args.query, context=context, mode=mode, history=None)

    answer = call_llm(prompt)
    if mode == "mapping":
        try:
            answer = json.dumps(json.loads(answer), ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            pass

    print("\n[answer]")
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
