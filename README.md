# CIDOC RAG Assistant

Modular FAISS-based RAG assistant for CIDOC CRM.

## Supported Input Formats

1. JSON (`.json`)
1. JSONL (`.jsonl`)
1. Text/Markdown (`.txt`, `.md`)
1. RDF (`.ttl`, `.rdf`, `.owl`, `.nt`, `.n3`, `.xml`)

## Quick Start

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

1. Optional editable install:

```bash
python -m pip install -e .
```

1. Start Ollama and pull required models:

```bash
ollama serve
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

1. Configure environment (copy and adjust as needed):

```bash
cp .env.example .env
```

If Ollama model cold-start is slow on your machine, increase request timeout in `.env`:

```bash
OLLAMA_TIMEOUT_S=180
```

1. Ingest CIDOC data:

```bash
PYTHONPATH=src python -m cidoc_rag.cli.ingest_cli data/raw --out-dir data/vectorstore
```

1. Retrieve:

```bash
PYTHONPATH=src python -m cidoc_rag.cli.retrieve_cli "What is E21?" --top-k 5
```

1. Generate grounded answer:

```bash
PYTHONPATH=src python -m cidoc_rag.cli.app_cli "What is E21?" --k 5
```

1. Start interactive chat:

```bash
PYTHONPATH=src python -m cidoc_rag.cli.app_cli --chat --k 5 --history-turns 4
```

## Interactive Chat

The chat interface continuously accepts questions and runs retrieval + prompt generation for each turn.

1. Start chat mode:

```bash
PYTHONPATH=src python -m cidoc_rag.cli.app_cli --chat --k 5 --history-turns 4 --debug
```

1. Runtime commands while chatting:

```text
debug on
debug off
k=10
exit
quit
```

1. Behavior notes:

- Retrieval always uses the current user query only.
- Conversation memory is short and keeps only recent turns.
- Mapping answers are pretty-printed as JSON when valid JSON is returned.

## RDF Demo

1. Ingest bundled RDF sample:

```bash
PYTHONPATH=src python -m cidoc_rag.cli.ingest_cli data/raw/cidoc_sample.ttl --out-dir data/vectorstore
```

1. Or with Make target:

```bash
make ingest-rdf-sample
```

## Batch Ingestion

1. Ingest all supported files under a directory (recursive):

```bash
make ingest-batch BATCH_PATH=data/raw
```

1. Ingest all supported files under a custom path:

```bash
make ingest-batch BATCH_PATH=/path/to/cidoc/files
```
