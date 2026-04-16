from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

from cidoc_rag.prompting.modes import Mode
from cidoc_rag.utils import clean_text

TurnAction = Literal["retrieve_and_answer", "answer_without_retrieval", "ask_clarifying"]


@dataclass(frozen=True)
class Decision:
    action: TurnAction
    reason: str
    clarification_question: str = ""
    retrieval_k_override: Optional[int] = None


_GREETINGS = {
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank you",
    "good morning",
    "good afternoon",
    "good evening",
}

_DOMAIN_PATTERN = re.compile(
    r"\b(cidoc|crm|ontology|rdf|class|property|mapping|domain|range|triple|e\d+|p\d+)\b",
    flags=re.IGNORECASE,
)

_CONTEXT_DEPENDENT_PATTERN = re.compile(
    r"\b(this|that|it|those|these|which one|same one|more|expand|elaborate)\b",
    flags=re.IGNORECASE,
)


def _is_smalltalk(query: str) -> bool:
    lowered = clean_text(query).lower()
    return lowered in _GREETINGS


def _has_domain_cues(query: str, mode: Mode) -> bool:
    if mode == "mapping":
        return True
    return bool(_DOMAIN_PATTERN.search(query))


def _is_context_dependent_followup(query: str) -> bool:
    lowered = clean_text(query).lower()
    if not lowered:
        return False
    if lowered.startswith(("and ", "also ", "what about", "how about")):
        return True
    return bool(_CONTEXT_DEPENDENT_PATTERN.search(lowered))


def _needs_clarification(query: str) -> bool:
    lowered = clean_text(query).lower()
    tokens = lowered.split()
    if not tokens:
        return True
    if len(tokens) <= 3 and _is_context_dependent_followup(lowered):
        return True
    if lowered in {"which one", "what do you mean", "can you clarify"}:
        return True
    return False


def _build_clarification_question(mode: Mode) -> str:
    if mode == "mapping":
        return "Could you share the exact table/fields you want mapped to CIDOC CRM?"
    return "Could you clarify which CIDOC class, property, or concept you mean?"


def decide_next_action(
    query: str,
    history: Optional[List[Dict[str, str]]],
    last_retrieval: Optional[List[Dict[str, object]]],
    mode: Mode,
) -> Decision:
    if _is_smalltalk(query):
        return Decision(action="answer_without_retrieval", reason="smalltalk-or-meta")

    has_domain_cues = _has_domain_cues(query, mode=mode)
    is_followup = _is_context_dependent_followup(query)
    has_recent_retrieval = bool(last_retrieval)

    if is_followup and not has_recent_retrieval and _needs_clarification(query):
        return Decision(
            action="ask_clarifying",
            reason="context-dependent-query-without-grounding",
            clarification_question=_build_clarification_question(mode),
        )

    if is_followup and has_recent_retrieval and not has_domain_cues:
        return Decision(action="answer_without_retrieval", reason="followup-uses-recent-context")

    if has_domain_cues:
        return Decision(action="retrieve_and_answer", reason="domain-cues-detected")

    if history and len(history) >= 2 and is_followup:
        return Decision(action="answer_without_retrieval", reason="followup-with-conversation-history")

    return Decision(action="retrieve_and_answer", reason="default-grounded-answer")
