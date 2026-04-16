from __future__ import annotations

from typing import Dict, List, Optional

from cidoc_rag.clients.ollama_client import get_ollama_client
from cidoc_rag.config import DEFAULT_INDEX_PATH, DEFAULT_METADATA_PATH, get_llm_model
from cidoc_rag.prompting.builders import build_context, build_prompt
from cidoc_rag.prompting.modes import detect_mode
from cidoc_rag.retrieval.service import retrieve


def generate_answer(
    query: str,
    k: int = 5,
    index_path: str = DEFAULT_INDEX_PATH,
    metadata_path: str = DEFAULT_METADATA_PATH,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    mode = detect_mode(query)
    results = retrieve(query=query, k=k, index_path=index_path, metadata_path=metadata_path)
    context = build_context(results)
    prompt = build_prompt(query=query, context=context, mode=mode, history=history)

    model = get_llm_model()
    client = get_ollama_client()
    content = client.chat_completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return content.strip() if content else ""
