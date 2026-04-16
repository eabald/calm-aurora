from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cidoc_rag.agent.policy import decide_next_action
from cidoc_rag.clients.ollama_client import get_ollama_client
from cidoc_rag.config import get_llm_model
from cidoc_rag.exporters.service import export_session
from cidoc_rag.prompting.builders import build_context, build_prompt
from cidoc_rag.prompting.modes import Mode, detect_mode
from cidoc_rag.retrieval.service import retrieve
from cidoc_rag.utils import clean_text


RULE = "=" * 72
SUBRULE = "-" * 72


@dataclass
class TurnRecord:
    turn: int
    mode: Mode
    decision: str
    reason: str
    user: str
    assistant: str
    retrieved_ids: List[str]
    timestamp: str


@dataclass
class ChatSessionState:
    last_retrieval_results: List[Dict[str, Any]]
    pending_clarification_for: str = ""
    imported_rdf_path: str = ""
    imported_rdf_text: str = ""
    last_assistant_output: str = ""


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
    print("Commands: debug on | debug off | k=<int> | export <json|markdown|rdf|ttl> <path> | exit | quit")
    print("RDF edit: import-rdf <path> | show-rdf | apply-rdf | save-rdf [path] | clear-rdf")
    print("Tip: mapping-style questions can return structured JSON.")
    print(SUBRULE)


def _parse_import_rdf_command(command: str) -> Optional[str]:
    parts = command.strip().split(maxsplit=1)
    if len(parts) != 2:
        return None
    if parts[0].lower() != "import-rdf":
        return None
    return parts[1].strip()


def _parse_save_rdf_command(command: str) -> Optional[Optional[str]]:
    parts = command.strip().split(maxsplit=1)
    if not parts or parts[0].lower() != "save-rdf":
        return None
    if len(parts) == 1:
        return ""
    return parts[1].strip()


def _extract_rdf_payload(text: str) -> str:
    if not text.strip():
        return ""

    code_blocks = re.findall(r"```([A-Za-z0-9_-]*)\n([\s\S]*?)```", text)
    for language, block in code_blocks:
        normalized = language.strip().lower()
        if normalized in {"ttl", "turtle", "rdf", "xml", "owl", "n3", "nt"}:
            candidate = block.strip()
            if candidate:
                return candidate

    for _, block in code_blocks:
        candidate = block.strip()
        if candidate.startswith("@prefix") or "<rdf:RDF" in candidate:
            return candidate

    plain = text.strip()
    if plain.startswith("@prefix") or "<rdf:RDF" in plain:
        return plain
    return ""


def _load_importable_rdf_file(path_text: str) -> Tuple[str, str]:
    supported = {".ttl", ".rdf", ".owl", ".nt", ".n3", ".xml"}
    path = Path(path_text).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"RDF file does not exist: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"RDF path must be a file: {path}")
    if path.suffix.lower() not in supported:
        raise ValueError("Unsupported RDF file extension. Use .ttl, .rdf, .owl, .nt, .n3, or .xml")

    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("RDF file is empty")
    return str(path), text


def _build_effective_context(retrieval_context: str, state: ChatSessionState) -> str:
    if not state.imported_rdf_text:
        return retrieval_context

    # Limit imported RDF payload size to keep prompt size bounded.
    max_chars = 12000
    imported = state.imported_rdf_text.strip()
    truncated = imported[:max_chars]
    if len(imported) > max_chars:
        truncated += "\n... [truncated]"

    imported_header = f"Imported RDF file: {state.imported_rdf_path or '(in-memory)'}"
    imported_block = (
        f"{imported_header}\n"
        "RDF content to edit:\n"
        f"{truncated}"
    )

    if retrieval_context.strip() == "No CIDOC context retrieved.":
        return imported_block

    return f"{retrieval_context}\n\n{imported_block}"


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


def _parse_export_command(command: str) -> Optional[Tuple[str, str]]:
    parts = command.strip().split(maxsplit=2)
    if len(parts) != 3:
        return None
    if parts[0].lower() != "export":
        return None
    return parts[1], parts[2]


def _record_turn(
    records: List[TurnRecord],
    mode: Mode,
    decision: str,
    reason: str,
    user_input: str,
    assistant_output: str,
    retrieved_ids: List[str],
) -> None:
    records.append(
        TurnRecord(
            turn=len(records) + 1,
            mode=mode,
            decision=decision,
            reason=reason,
            user=user_input,
            assistant=assistant_output,
            retrieved_ids=retrieved_ids,
            timestamp=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        )
    )


def _records_to_dict(records: List[TurnRecord]) -> List[Dict[str, Any]]:
    return [
        {
            "turn": record.turn,
            "mode": record.mode,
            "decision": record.decision,
            "reason": record.reason,
            "user": record.user,
            "assistant": record.assistant,
            "retrieved_ids": record.retrieved_ids,
            "timestamp": record.timestamp,
        }
        for record in records
    ]


def chat_loop(
    k: int,
    index_path: str,
    metadata_path: str,
    history_turns: int,
    debug: bool,
) -> List[Dict[str, str]]:
    history: List[Dict[str, str]] = []
    records: List[TurnRecord] = []
    state = ChatSessionState(last_retrieval_results=[])
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

        import_path = _parse_import_rdf_command(user_input)
        if import_path is not None:
            try:
                resolved_path, imported_text = _load_importable_rdf_file(import_path)
                state.imported_rdf_path = resolved_path
                state.imported_rdf_text = imported_text
                print(f"[info] Imported RDF file: {resolved_path} ({len(imported_text)} chars)")
            except (OSError, ValueError) as exc:
                print(f"[warn] {exc}")
            continue

        if lowered == "show-rdf":
            if not state.imported_rdf_text:
                print("[warn] No RDF file imported. Use: import-rdf <path>")
                continue
            preview = state.imported_rdf_text[:1200]
            suffix = "\n... [truncated]" if len(state.imported_rdf_text) > 1200 else ""
            print(SUBRULE)
            print(f"Imported RDF: {state.imported_rdf_path or '(in-memory)'}")
            print(SUBRULE)
            print(preview + suffix)
            print(SUBRULE)
            continue

        if lowered == "clear-rdf":
            state.imported_rdf_path = ""
            state.imported_rdf_text = ""
            print("[info] Cleared imported RDF context")
            continue

        if lowered == "apply-rdf":
            if not state.last_assistant_output:
                print("[warn] No assistant response available to apply.")
                continue
            payload = _extract_rdf_payload(state.last_assistant_output)
            if not payload:
                print("[warn] Last assistant response does not contain RDF/Turtle content.")
                continue
            state.imported_rdf_text = payload
            print(f"[info] Applied RDF update from last assistant response ({len(payload)} chars)")
            continue

        save_path = _parse_save_rdf_command(user_input)
        if save_path is not None:
            if not state.imported_rdf_text:
                print("[warn] No RDF content to save. Import a file first or run apply-rdf.")
                continue

            target = save_path or state.imported_rdf_path
            if not target:
                print("[warn] No target path available. Use: save-rdf <path>")
                continue

            try:
                target_path = Path(target).expanduser()
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(state.imported_rdf_text.rstrip() + "\n", encoding="utf-8")
                state.imported_rdf_path = str(target_path)
                print(f"[info] Saved RDF file: {target_path}")
            except OSError as exc:
                print(f"[error] Failed to save RDF file: {exc}")
            continue

        export_cmd = _parse_export_command(user_input)
        if export_cmd is not None:
            export_format, export_path = export_cmd
            try:
                written = export_session(
                    export_format=export_format,
                    output_path=export_path,
                    turns=_records_to_dict(records),
                    session={
                        "k": k,
                        "history_turns": history_turns,
                        "debug": debug,
                    },
                )
                print(f"[info] Export complete: {written}")
            except ValueError as exc:
                print(f"[warn] {exc}")
            except OSError as exc:
                print(f"[error] Failed to export session: {exc}")
            continue

        handled, debug, k, message = apply_runtime_command(user_input, debug=debug, k=k)
        if handled:
            if message:
                print(message)
            continue

        effective_query = user_input
        if state.pending_clarification_for:
            effective_query = f"{state.pending_clarification_for}\nClarification: {user_input}"
            state.pending_clarification_for = ""

        mode = detect_mode(effective_query)
        decision = decide_next_action(
            query=effective_query,
            history=history,
            last_retrieval=state.last_retrieval_results,
            mode=mode,
        )

        if decision.action == "ask_clarifying":
            clarifying_question = decision.clarification_question or "Could you clarify your request?"
            print("[info] I need one clarification before answering.")
            print(clarifying_question)

            _record_turn(
                records=records,
                mode=mode,
                decision=decision.action,
                reason=decision.reason,
                user_input=user_input,
                assistant_output=clarifying_question,
                retrieved_ids=[],
            )
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": clarifying_question})
            history = truncate_history(history, max_turns=history_turns)
            state.pending_clarification_for = user_input
            continue

        print("[info] Thinking...")

        used_retrieval = decision.action == "retrieve_and_answer"
        results: List[Dict[str, Any]] = []
        if used_retrieval:
            results = retrieve(
                query=effective_query,
                k=decision.retrieval_k_override or k,
                index_path=index_path,
                metadata_path=metadata_path,
                debug=debug,
            )
            state.last_retrieval_results = results

        retrieval_context = build_context(results)
        context = _build_effective_context(retrieval_context, state)
        prompt = build_prompt(
            query=effective_query,
            context=context,
            mode=mode,
            history=history,
            retrieval_used=used_retrieval,
        )
        raw_response = call_llm(prompt)
        state.last_assistant_output = raw_response
        formatted_response = format_answer_for_mode(mode, raw_response)
        if not used_retrieval:
            print(f"[info] Retrieval skipped: {decision.reason}")

        retrieved_ids = [clean_text(entry.get("id")) for entry in results if clean_text(entry.get("id"))]

        _print_response_block(
            retrieved_ids=format_retrieved_ids(results),
            answer=formatted_response,
            debug=debug,
            context=context,
        )

        _record_turn(
            records=records,
            mode=mode,
            decision=decision.action,
            reason=decision.reason,
            user_input=user_input,
            assistant_output=raw_response,
            retrieved_ids=retrieved_ids,
        )

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": raw_response})
        history = truncate_history(history, max_turns=history_turns)

    return history
