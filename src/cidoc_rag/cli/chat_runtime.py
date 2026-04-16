from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from cidoc_rag.clients.ollama_client import get_ollama_client
from cidoc_rag.config import get_llm_model
from cidoc_rag.prompting.builders import build_context, build_prompt
from cidoc_rag.prompting.modes import Mode, detect_mode
from cidoc_rag.retrieval.service import retrieve
from cidoc_rag.utils import clean_text


RULE = "=" * 72
SUBRULE = "-" * 72


def call_llm(prompt: str) -> str:
    model = get_llm_model()
    client = get_ollama_client()
    content = client.chat_completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return content.strip() if content else ""


def truncate_history(history: List[Dict[str, str]], max_turns: int) -> List[Dict[str, str]]:
    max_messages = max(0, max_turns) * 2
    if max_messages == 0:
        return []
    if len(history) <= max_messages:
        return history
    return history[-max_messages:]


def format_retrieved_ids(results: List[Dict[str, object]]) -> str:
    ids = [clean_text(entry.get("id")) for entry in results if clean_text(entry.get("id"))]
    return ", ".join(ids) if ids else "none"


def format_answer_for_mode(mode: Mode, response_text: str) -> str:
    if mode != "mapping":
        return response_text

    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        return response_text

    return json.dumps(parsed, ensure_ascii=False, indent=2)


def _print_welcome(k: int, history_turns: int, debug: bool) -> None:
    print(RULE)
    print("CIDOC CRM CHAT")
    print(RULE)
    print("Interactive retrieval + grounded generation")
    print(f"Session: k={k} | history_turns={history_turns} | debug={'on' if debug else 'off'}")
    print("Commands: debug on | debug off | k=<int> | exit | quit")
    print("Tip: mapping-style questions can return structured JSON.")
    print(SUBRULE)


def _print_response_block(retrieved_ids: str, answer: str, debug: bool, context: str) -> None:
    print()
    print(SUBRULE)
    print("Retrieved")
    print(SUBRULE)
    print(retrieved_ids)
    if debug:
        print()
        print("Debug Context")
        print(SUBRULE)
        print(context)
    print()
    print("Answer")
    print(SUBRULE)
    print(answer)
    print(SUBRULE)


def apply_runtime_command(command: str, debug: bool, k: int) -> Tuple[bool, bool, int, Optional[str]]:
    lowered = command.strip().lower()
    if lowered == "debug on":
        return True, True, k, "[info] Debug mode enabled"
    if lowered == "debug off":
        return True, False, k, "[info] Debug mode disabled"

    if lowered.startswith("k="):
        value = command.split("=", 1)[1].strip()
        try:
            new_k = int(value)
        except ValueError:
            return True, debug, k, "[warn] Invalid k value. Use k=<positive integer>."
        if new_k <= 0:
            return True, debug, k, "[warn] k must be greater than 0."
        return True, debug, new_k, f"[info] Retrieval depth set to k={new_k}"

    return False, debug, k, None


def chat_loop(
    k: int,
    index_path: str,
    metadata_path: str,
    history_turns: int,
    debug: bool,
) -> List[Dict[str, str]]:
    history: List[Dict[str, str]] = []
    _print_welcome(k=k, history_turns=history_turns, debug=debug)

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[info] Exiting chat.")
            break

        if not user_input:
            continue

        lowered = user_input.lower()
        if lowered in {"exit", "quit"}:
            print("[info] Bye")
            break

        handled, debug, k, message = apply_runtime_command(user_input, debug=debug, k=k)
        if handled:
            if message:
                print(message)
            continue

        mode = detect_mode(user_input)
        print("[info] Thinking...")

        # Retrieval always uses the current user query only.
        results = retrieve(
            query=user_input,
            k=k,
            index_path=index_path,
            metadata_path=metadata_path,
            debug=debug,
        )
        context = build_context(results)
        prompt = build_prompt(query=user_input, context=context, mode=mode, history=history)
        raw_response = call_llm(prompt)
        formatted_response = format_answer_for_mode(mode, raw_response)

        _print_response_block(
            retrieved_ids=format_retrieved_ids(results),
            answer=formatted_response,
            debug=debug,
            context=context,
        )

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": raw_response})
        history = truncate_history(history, max_turns=history_turns)

    return history
