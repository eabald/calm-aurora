from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests


_ollama_client: Optional[Any] = None


class OllamaClient:
    def __init__(self, base_url: str, timeout_s: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            response = requests.post(url, json=payload, timeout=self.timeout_s)
            response.raise_for_status()
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else "unknown"
            response_excerpt = ""
            if exc.response is not None:
                body = exc.response.text.strip()
                if body:
                    response_excerpt = f" Response: {body[:240]}"
            raise RuntimeError(
                f"Ollama request failed ({status_code}) for endpoint {path} at {self.base_url}."
                f"{response_excerpt}"
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(
                "Failed to communicate with Ollama. Ensure Ollama is running and reachable at "
                f"{self.base_url}."
            ) from exc

        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError(f"Ollama returned a non-JSON response for endpoint {path}.") from exc

    def chat_completion(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        data = self._post_json("/api/chat", payload)
        message = data.get("message")
        if not isinstance(message, dict):
            raise RuntimeError(
                f"Ollama did not return a valid chat message. Ensure model '{model}' is available locally."
            )

        content = message.get("content")
        if isinstance(content, str):
            return content
        return ""

    def embeddings(self, model: str, input_text: str) -> List[float]:
        errors: List[str] = []

        try:
            # Newer Ollama API style.
            data = self._post_json("/api/embed", {"model": model, "input": input_text})
            embeddings = data.get("embeddings")
            if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
                return [float(value) for value in embeddings[0]]
            errors.append("/api/embed returned no embeddings in expected format")
        except RuntimeError as exc:
            errors.append(str(exc))

        try:
            # Legacy Ollama API style.
            data = self._post_json("/api/embeddings", {"model": model, "prompt": input_text})
            embedding = data.get("embedding")
            if isinstance(embedding, list):
                return [float(value) for value in embedding]
            errors.append("/api/embeddings returned no embedding in expected format")
        except RuntimeError as exc:
            errors.append(str(exc))

        joined_errors = " | ".join(errors)
        raise RuntimeError(
            "Failed to get embeddings from Ollama. "
            f"Ensure embedding model '{model}' is installed (for example: ollama pull {model}). "
            f"Details: {joined_errors}"
        )


def get_ollama_client() -> Any:
    global _ollama_client
    if _ollama_client is not None:
        return _ollama_client

    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()
    if not base_url:
        raise ValueError("OLLAMA_BASE_URL is empty. Set it to your Ollama host URL.")

    timeout_raw = os.getenv("OLLAMA_TIMEOUT_S", "180").strip()
    try:
        timeout_s = float(timeout_raw)
    except ValueError as exc:
        raise ValueError("OLLAMA_TIMEOUT_S must be a number.") from exc
    if timeout_s <= 0:
        raise ValueError("OLLAMA_TIMEOUT_S must be greater than 0.")

    _ollama_client = OllamaClient(base_url=base_url, timeout_s=timeout_s)
    return _ollama_client
