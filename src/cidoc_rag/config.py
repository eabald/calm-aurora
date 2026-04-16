from __future__ import annotations

import os

DEFAULT_INDEX_PATH = "data/vectorstore/cidoc.index"
DEFAULT_METADATA_PATH = "data/vectorstore/cidoc_metadata.json"
DEFAULT_OUT_DIR = "data/vectorstore"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_LLM_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"


def get_embedding_model() -> str:
    return os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def get_llm_model() -> str:
    return os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)


def get_ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
