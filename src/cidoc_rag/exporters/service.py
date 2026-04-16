from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _serialize_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _serialize_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# CIDOC Chat Export")
    lines.append("")
    lines.append(f"Generated: {payload.get('generated_at', '')}")
    lines.append("")
    lines.append("## Session")
    lines.append("")

    session = payload.get("session", {})
    for key in ["k", "history_turns", "debug"]:
        lines.append(f"- {key}: {session.get(key)}")
    lines.append("")

    turns = payload.get("turns", [])
    lines.append("## Turns")
    lines.append("")
    for turn in turns:
        lines.append(f"### Turn {turn.get('turn')} ({turn.get('mode')})")
        lines.append(f"- Decision: {turn.get('decision')}")
        lines.append(f"- Reason: {turn.get('reason')}")
        if turn.get("retrieved_ids"):
            lines.append(f"- Retrieved IDs: {', '.join(turn['retrieved_ids'])}")
        lines.append("")
        lines.append("User:")
        lines.append("")
        lines.append(turn.get("user", ""))
        lines.append("")
        lines.append("Assistant:")
        lines.append("")
        lines.append(turn.get("assistant", ""))
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _cidoc_uri(identifier: str) -> str:
    safe = "".join(ch for ch in identifier.strip() if ch.isalnum() or ch in {"_", "-"})
    return f"http://www.cidoc-crm.org/cidoc-crm/{safe}" if safe else ""


def _escape_literal(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _serialize_rdf(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "@prefix chat: <urn:cidoc-chat:> .",
        "@prefix chatp: <urn:cidoc-chat:predicate:> .",
        "",
    ]

    turns = payload.get("turns", [])
    for turn in turns:
        turn_id = int(turn.get("turn", 0))
        subject = f"chat:turn-{turn_id}"
        lines.append(f"{subject} chatp:userQuery \"{_escape_literal(str(turn.get('user', '')))}\" .")
        lines.append(f"{subject} chatp:assistantAnswer \"{_escape_literal(str(turn.get('assistant', '')))}\" .")
        for cidoc_id in turn.get("retrieved_ids", []):
            uri = _cidoc_uri(str(cidoc_id))
            if uri:
                lines.append(f"{subject} chatp:usesCitation <{uri}> .")
            else:
                lines.append(f"{subject} chatp:usesCitation \"{_escape_literal(str(cidoc_id))}\" .")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def export_session(
    export_format: str,
    output_path: str,
    turns: List[Dict[str, Any]],
    session: Dict[str, Any],
) -> str:
    normalized = export_format.strip().lower()
    if normalized in {"md", "markdown"}:
        normalized = "markdown"
    elif normalized in {"json"}:
        normalized = "json"
    elif normalized in {"rdf", "ttl"}:
        normalized = "rdf"
    else:
        raise ValueError("Unsupported export format. Use json, markdown, or rdf.")

    payload = {
        "generated_at": _now_iso(),
        "session": session,
        "turns": turns,
    }

    path = Path(output_path)
    if normalized == "json":
        _serialize_json(path, payload)
    elif normalized == "markdown":
        _serialize_markdown(path, payload)
    else:
        _serialize_rdf(path, payload)

    return str(path)
