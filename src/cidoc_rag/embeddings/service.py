from __future__ import annotations

from typing import List

from cidoc_rag.clients.ollama_client import get_ollama_client
from cidoc_rag.config import get_embedding_model


def embed_text(text: str) -> List[float]:
    client = get_ollama_client()
    model = get_embedding_model()
    return client.embeddings(model=model, input_text=text)
