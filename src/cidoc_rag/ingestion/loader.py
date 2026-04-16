from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote


def _extract_cidoc_id_label(value: str) -> tuple[str, str]:
    text = " ".join(value.split()).strip()
    # Support CIDOC and extension local-name patterns such as:
    # E85_Joining, P65i_is_shown_by, P62.1_mode_of_depiction,
    # AP32i_was_discarded_by, TXP9_is_encoded_using.
    match = re.search(r"([A-Z]{1,4}\d+(?:\.\d+)?i?)(?=_|$|[^A-Za-z0-9])", text)
    if not match:
        return "", text

    cidoc_id = match.group(1)
    label = text.replace(cidoc_id, "", 1).strip(" .:-")
    return cidoc_id, label


def _local_name(term: Any) -> str:
    text = str(term)
    if "#" in text:
        return unquote(text.rsplit("#", 1)[1])
    if "/" in text:
        return unquote(text.rsplit("/", 1)[1])
    return unquote(text)


def _first_non_empty(values: List[str]) -> str:
    for value in values:
        stripped = " ".join(str(value).split()).strip()
        if stripped:
            return stripped
    return ""


def _label_for_term(graph: Any, term: Any) -> str:
    try:
        from rdflib.namespace import RDFS, SKOS
    except Exception:
        return _local_name(term)

    label_candidates = [str(obj) for obj in graph.objects(term, RDFS.label)]
    label_candidates.extend(str(obj) for obj in graph.objects(term, SKOS.prefLabel))
    label = _first_non_empty(label_candidates)
    return label if label else _local_name(term)


def _load_rdf_file(file_path: Path) -> List[Dict[str, Any]]:
    try:
        from rdflib import Graph
        from rdflib.namespace import OWL, RDF, RDFS, SKOS
    except Exception as exc:
        raise RuntimeError(
            "RDF ingestion requires rdflib. Install dependencies from requirements.txt."
        ) from exc

    format_by_suffix = {
        ".ttl": "turtle",
        ".nt": "nt",
        ".n3": "n3",
        ".owl": "xml",
        ".rdf": "xml",
        ".xml": "xml",
    }

    graph = Graph()
    rdf_format = format_by_suffix.get(file_path.suffix.lower())
    try:
        if rdf_format:
            graph.parse(str(file_path), format=rdf_format)
        else:
            graph.parse(str(file_path))
    except Exception:
        graph.parse(str(file_path))

    class_types = {RDFS.Class, OWL.Class}
    property_types = {RDF.Property, OWL.ObjectProperty, OWL.DatatypeProperty, OWL.AnnotationProperty}

    subjects = set()
    for type_iri in class_types.union(property_types):
        for subject in graph.subjects(RDF.type, type_iri):
            subjects.add(subject)

    entries: List[Dict[str, Any]] = []
    for subject in subjects:
        subject_types = {obj for obj in graph.objects(subject, RDF.type)}
        uri_local = _local_name(subject)

        label_candidates = [str(obj) for obj in graph.objects(subject, RDFS.label)]
        label_candidates.extend(str(obj) for obj in graph.objects(subject, SKOS.prefLabel))
        label_text = _first_non_empty(label_candidates) or uri_local

        cidoc_id, parsed_label = _extract_cidoc_id_label(label_text)
        if not cidoc_id:
            cidoc_id, parsed_label = _extract_cidoc_id_label(uri_local)
        if not cidoc_id:
            continue

        is_property = bool(subject_types.intersection(property_types)) or cidoc_id.startswith("P")
        entry_type = "property" if is_property else "class"

        definition_candidates = [str(obj) for obj in graph.objects(subject, RDFS.comment)]
        definition_candidates.extend(str(obj) for obj in graph.objects(subject, SKOS.definition))
        definition = _first_non_empty(definition_candidates)

        if entry_type == "property":
            domain = _first_non_empty([_label_for_term(graph, obj) for obj in graph.objects(subject, RDFS.domain)])
            range_value = _first_non_empty([_label_for_term(graph, obj) for obj in graph.objects(subject, RDFS.range)])
            entries.append(
                {
                    "id": cidoc_id,
                    "type": "property",
                    "label": parsed_label,
                    "domain": domain,
                    "range": range_value,
                    "definition": definition,
                }
            )
            continue

        entries.append(
            {
                "id": cidoc_id,
                "type": "class",
                "label": parsed_label,
                "definition": definition,
                "examples": "",
                "related_properties": [],
            }
        )

    return entries


def parse_text_chunk(chunk: str) -> Dict[str, Any]:
    text = chunk.strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    parsed: Dict[str, Any] = {"raw_text": text}

    for line in lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip().lower().replace(" ", "_")] = value.strip()

    if lines:
        first_line = lines[0]
        if first_line[:1] in {"E", "P"}:
            parts = first_line.split(maxsplit=1)
            parsed.setdefault("id", parts[0])
            if len(parts) > 1:
                parsed.setdefault("label", parts[1].strip(" .:-"))

    return parsed


def _load_json_or_jsonl_file(file_path: Path) -> List[Dict[str, Any]]:
    content = file_path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    try:
        payload = json.loads(content)
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            return [payload]
    except json.JSONDecodeError:
        pass

    entries: List[Dict[str, Any]] = []
    for line_no, line in enumerate(content.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                entries.append(obj)
        except json.JSONDecodeError as exc:
            print(f"[warn] Skipping malformed JSONL line {line_no} in {file_path}: {exc}")
    return entries


def _load_text_file(file_path: Path, text_delimiter: Optional[str]) -> List[Dict[str, Any]]:
    content = file_path.read_text(encoding="utf-8")
    if not content.strip():
        return []

    if text_delimiter:
        chunks = [chunk for chunk in content.split(text_delimiter) if chunk.strip()]
        return [parse_text_chunk(chunk) for chunk in chunks]

    return [parse_text_chunk(content)]


def load_raw_data(path: str, text_delimiter: Optional[str] = None) -> List[Dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    supported_text_suffixes = {".txt", ".md"}
    supported_rdf_suffixes = {".ttl", ".rdf", ".owl", ".nt", ".n3", ".xml"}
    raw_entries: List[Dict[str, Any]] = []

    if input_path.is_file():
        file_paths = [input_path]
    else:
        file_paths = sorted(p for p in input_path.rglob("*") if p.is_file())

    for file_path in file_paths:
        suffix = file_path.suffix.lower()
        if suffix in {".json", ".jsonl"}:
            entries = _load_json_or_jsonl_file(file_path)
        elif suffix in supported_text_suffixes:
            entries = _load_text_file(file_path, text_delimiter=text_delimiter)
        elif suffix in supported_rdf_suffixes:
            entries = _load_rdf_file(file_path)
        else:
            continue

        print(f"[info] Processed file: {file_path} (entries: {len(entries)})")

        for entry in entries:
            entry["_source_file"] = str(file_path)
            raw_entries.append(entry)

    return raw_entries
