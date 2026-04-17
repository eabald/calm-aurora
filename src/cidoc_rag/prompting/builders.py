from __future__ import annotations

from typing import Any, Dict, List, Optional

from cidoc_rag.prompting.modes import Mode
from cidoc_rag.utils import clean_text


def build_context(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "No CIDOC context retrieved."

    blocks: List[str] = []
    for entry in results:
        entry_id = clean_text(entry.get("id"))
        entry_type = clean_text(entry.get("type")).lower()
        label = clean_text(entry.get("label"))
        definition = clean_text(entry.get("definition"))

        if entry_type == "documentation":
            title = f"{entry_id} {label}".strip() or "Documentation chunk"
            lines = [f"{title} (Documentation)"]
            if definition:
                lines.append(f"Content: {definition}")
            source_file = clean_text(entry.get("source_file"))
            if source_file:
                lines.append(f"Source: {source_file}")
            blocks.append("\n".join(lines))
            continue

        if entry_type == "class":
            lines = [f"{entry_id} {label} (Class)".strip()]
            if definition:
                lines.append(f"Definition: {definition}")

            examples = clean_text(entry.get("examples"))
            if examples:
                lines.append(f"Examples: {examples}")

            related = entry.get("related_properties", [])
            if isinstance(related, list):
                related_text = ", ".join(clean_text(item) for item in related if clean_text(item))
            else:
                related_text = clean_text(related)
            if related_text:
                lines.append(f"Related properties: {related_text}")

            blocks.append("\n".join(lines))
            continue

        lines = [f"{entry_id} {label} (Property)".strip()]
        domain = clean_text(entry.get("domain"))
        range_value = clean_text(entry.get("range"))

        if domain:
            lines.append(f"Domain: {domain}")
        if range_value:
            lines.append(f"Range: {range_value}")
        if definition:
            lines.append(f"Definition: {definition}")

        blocks.append("\n".join(lines))

    return "\n\n".join(block for block in blocks if block)


def _format_history(history: Optional[List[Dict[str, str]]]) -> str:
    if not history:
        return ""

    lines: List[str] = ["Conversation history:"]
    for message in history:
        role = clean_text(message.get("role")).lower()
        content = clean_text(message.get("content"))
        if not content:
            continue

        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")

    return "\n".join(lines)


def build_prompt(
    query: str,
    context: str,
    mode: Mode,
    history: Optional[List[Dict[str, str]]] = None,
    retrieval_used: bool = True,
) -> str:
    history_text = _format_history(history)
    history_block = f"{history_text}\n\n" if history_text else ""

    if mode == "mapping":
        intro = "Use ONLY the provided context."
        if not retrieval_used:
            intro = (
                "No new retrieval context was used for this turn. "
                "Use conversation history and state assumptions explicitly."
            )
        return (
            "You are a CIDOC CRM expert.\n\n"
            f"{intro}\n\n"
            "Task:\n"
            "Map the following question or schema to CIDOC CRM.\n\n"
            "Context:\n"
            f"{context}\n\n"
            f"{history_block}"
            "Current question:\n"
            f"{query}\n\n"
            "Return JSON in this format:\n"
            "{\n"
            '  "class": "...",\n'
            '  "properties": [\n'
            "    {\n"
            '      "property": "...",\n'
            '      "field": "...",\n'
            '      "reason": "..."\n'
            "    }\n"
            "  ],\n"
            '  "notes": "...",\n'
            '  "confidence": "low | medium | high"\n'
            "}\n\n"
            "Rules:\n"
            "- Only use valid CIDOC CRM IDs (E*, P*)\n"
            "- If unsure, include that in notes\n"
            "- Prefer multiple options if ambiguous"
        )

    qa_intro = "Use ONLY the provided context. If the answer is not in the context, say \"I don't know\"."
    if not retrieval_used:
        qa_intro = (
            "No new retrieval context was used for this turn. "
            "Use conversation history and be explicit about assumptions."
        )

    return (
        "You are a CIDOC CRM expert.\n\n"
        f"{qa_intro}\n\n"
        "Context:\n"
        f"{context}\n\n"
        f"{history_block}"
        "Current question:\n"
        f"{query}\n\n"
        "Answer clearly and concisely."
    )
