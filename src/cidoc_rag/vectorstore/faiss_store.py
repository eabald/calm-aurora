from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import faiss
except ModuleNotFoundError:  # pragma: no cover - exercised only in environments without faiss
    faiss = None


def _require_faiss() -> Any:
    if faiss is None:
        raise ModuleNotFoundError("faiss is not installed. Install faiss-cpu to use vector store operations.")
    return faiss


def load_index(path: str) -> faiss.Index:
    faiss_mod = _require_faiss()
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Index file not found: {path}")
    return faiss_mod.read_index(str(file_path))


def save_index(index: faiss.Index, path: str) -> None:
    faiss_mod = _require_faiss()
    faiss_mod.write_index(index, path)


def load_metadata(path: str) -> List[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    with open(file_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError("Metadata JSON must be a list of entry dictionaries.")

    return [item for item in data if isinstance(item, dict)]


def save_metadata(metadata: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def build_index_flat_l2(vectors: np.ndarray) -> faiss.IndexFlatL2:
    faiss_mod = _require_faiss()
    arr = np.ascontiguousarray(vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("Embeddings must be a 2D float32 matrix.")

    index = faiss_mod.IndexFlatL2(arr.shape[1])
    # Use the high-level Python wrapper API for broad FAISS wheel compatibility.
    index_any: Any = index
    index_any.add(arr)
    return index


def search_index(index: faiss.Index, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    q = np.ascontiguousarray(query_vector, dtype=np.float32)
    if q.ndim != 2:
        raise ValueError("Query vector must be a 2D float32 matrix.")
    if k <= 0:
        raise ValueError("k must be greater than 0")

    index_any: Any = index
    distances, indices = index_any.search(q, k)
    return distances, indices
