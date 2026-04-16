# Calm Aurora

[![Build Status](https://github.com/eabald/calm-aurora/actions/workflows/ci.yml/badge.svg)](https://github.com/eabald/calm-aurora/actions/workflows/ci.yml)

Calm Aurora is an open-source, research-oriented RAG assistant for CIDOC CRM and related RDF vocabularies.

> MVP status: Calm Aurora is an early-stage prototype. Features, CLI behavior, prompts, and output formats may change between releases.

It combines:

- FAISS-based semantic retrieval
- CIDOC-aware prompt construction
- Agentic chat turn policy (retrieve, reuse context, or clarify)
- RDF/Turtle import-edit-save workflow directly in chat

Project status: MVP (actively evolving).

## MVP Scope and Expectations

- This project is currently focused on fast iteration and research utility, not production hardening.
- No formal stable release has been published yet.
- Backward compatibility is not guaranteed yet.
- Generated mappings and RDF edits require expert review before use in authoritative datasets.
- Edge cases and performance regressions may still be present.
- Feedback and issue reports are highly encouraged at this stage.

## Why Calm Aurora

Calm Aurora is designed for cultural heritage and ontology-heavy workflows where answers must be grounded in schema definitions and provenance-aware context. The repository is structured to support:

- Experimentation with retrieval and prompting strategies
- Reproducible local runs (including an offline smoke mode)
- Practical annotation and ontology editing loops with LLM assistance

## Core Capabilities

- Ingestion from JSON, JSONL, text, markdown, and RDF-family formats
- Normalization into CIDOC-like entries (class/property oriented)
- Embedding generation + FAISS index build
- Retrieval CLI and grounded QA/mapping CLI
- Interactive chat with runtime controls and export
- Session export to JSON, Markdown, and RDF citation triples
- In-chat RDF editing via `import-rdf`, `apply-rdf`, and `save-rdf`

## Supported Input Formats

- `.json`
- `.jsonl`
- `.txt`
- `.md`
- `.ttl`
- `.rdf`
- `.owl`
- `.nt`
- `.n3`
- `.xml`

## Repository Layout

```text
.
|- src/cidoc_rag/
|  |- agent/           # turn policy
|  |- cli/             # ingest/retrieve/app/chat entrypoints
|  |- embeddings/      # embedding client/service
|  |- exporters/       # session export serializers
|  |- ingestion/       # loading, normalization, indexing pipeline helpers
|  |- prompting/       # context + prompt builders
|  \- retrieval/       # retrieval service
|- tests/              # unit tests
|- data/               # sample/raw/vector artifacts
|- smoke_test.py       # end-to-end smoke runner
\- Makefile            # developer shortcuts
```

## System Overview

```text
Input files -> Loader/Normalizer -> Embeddings -> FAISS index + metadata
                           |
                           v
                       Retrieval service
                           |
User query -> Mode detection + turn policy -> Prompt builder -> LLM -> Answer
                  |
                  \-> Chat exports + RDF edit workflow
```

## Requirements

- Python 3.10+
- Ollama (for online embedding/generation paths)
- Local disk space for FAISS index and metadata artifacts

## Installation

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

1. Optional editable install:

```bash
python -m pip install -e .
```

1. Configure environment:

```bash
cp .env.example .env
```

1. Start Ollama and pull models:

```bash
ollama serve
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

If model cold-start is slow, increase timeout in `.env`:

```bash
OLLAMA_TIMEOUT_S=180
```

## Quick Start

1. Ingest data:

```bash
PYTHONPATH=src python -m cidoc_rag.cli.ingest_cli data/raw --out-dir data/vectorstore
```

1. Retrieve:

```bash
PYTHONPATH=src python -m cidoc_rag.cli.retrieve_cli "What is E21?" --top-k 5
```

1. Ask grounded question:

```bash
PYTHONPATH=src python -m cidoc_rag.cli.app_cli "What is E21?" --k 5
```

1. Start chat:

```bash
PYTHONPATH=src python -m cidoc_rag.cli.app_cli --chat --k 5 --history-turns 4
```

## Chat Runtime Commands

```text
debug on
debug off
k=10
export json ./exports/session.json
export markdown ./exports/session.md
export rdf ./exports/session.ttl
import-rdf ./data/raw/my_ontology.ttl
show-rdf
apply-rdf
save-rdf ./data/raw/my_ontology.edited.ttl
clear-rdf
exit
quit
```

### Chat Behavior

- Retrieval is policy-driven per turn.
- Small-talk/meta turns can skip retrieval.
- Ambiguous follow-ups can trigger one clarifying question.
- Recent conversation is truncated by `--history-turns`.
- Mapping responses are pretty-printed JSON when valid.

## RDF/Turtle Edit Workflow

```text
you> import-rdf ./data/raw/my_ontology.ttl
you> Add rdfs:comment to E21 and return full updated TTL in a ttl code block.
you> apply-rdf
you> save-rdf ./data/raw/my_ontology.edited.ttl
```

Notes:

- `apply-rdf` extracts RDF/Turtle from the last assistant response.
- `save-rdf` without a path overwrites the currently imported RDF path.

## Make Targets

```bash
make help
make install
make install-editable
make ingest
make ingest-batch BATCH_PATH=data/raw
make ingest-rdf-sample
make retrieve QUERY="What is E21?" TOP_K=5
make app QUERY="What is E21?"
make app-chat
make app-mapping SCHEMA='{"table":"authors","fields":["name"]}'
make smoke
make smoke-no-api
make test
```

## Reproducibility and Testing

### Unit Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py"
```

### End-to-End Smoke Test

Online mode (requires Ollama):

```bash
PYTHONPATH=src python smoke_test.py
```

Offline deterministic mode (no API calls):

```bash
PYTHONPATH=src python smoke_test.py --no-api --offline-dim 64
```

## Research Usage Notes

- This tool assists ontology interpretation and editing, but does not replace expert modeling decisions.
- LLM-generated ontology edits should be validated before publication.
- For experiments, export sessions and keep generated artifacts under version control where possible.

## Contributing

Contributions are welcome.

Recommended workflow:

1. Fork and create a feature branch.
1. Add or update tests with your change.
1. Run `make test` (and smoke checks if relevant).
1. Open a pull request with rationale and reproducibility notes.

See also:

- `CONTRIBUTING.md`
- `CODE_OF_CONDUCT.md`
- `SECURITY.md`

## Citation

If you use Calm Aurora in research, cite the repository (update author/version as needed):

```bibtex
@software{calm_aurora_2026,
 title = {Calm Aurora},
 author = {Krawczyk, Maciej},
 year = {2026},
 url = {https://github.com/eabald/calm-aurora}
}
```

## License

MIT License. See `LICENSE` for full text.

## Acknowledgments

- CIDOC CRM community and ontology maintainers
- FAISS, RDFlib, and the open-source LLM ecosystem
