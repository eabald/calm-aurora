PYTHON ?= $(if $(wildcard env/bin/python),env/bin/python,python)
PYTHONPATH ?= src
INPUT ?= data/raw
OUT_DIR ?= data/vectorstore
TEXT_DELIMITER ?=
QUERY ?= Who carried out an activity?
SCHEMA ?= {"table":"authors","fields":["name"]}
TOP_K ?= 5
OFFLINE_DIM ?= 64
RDF_SAMPLE ?= data/raw/CIDOC_CRM_v7.1.3_PC.rdf
RDF_FILE ?= $(RDF_SAMPLE)
BATCH_PATH ?= data/raw
LLM_MODEL ?= llama3.1:8b

.PHONY: help install install-editable ingest ingest-batch ingest-rdf-sample retrieve app app-chat app-mapping smoke smoke-no-api test clean-smoke

help:
	@echo "CIDOC RAG Assistant - Make targets"
	@echo ""
	@echo "Usage: make <target> [VAR=value]"
	@echo ""
	@echo "Targets:"
	@echo "  install         Install Python dependencies"
	@echo "  install-editable Install project in editable mode"
	@echo "  ingest          Run ingestion pipeline"
	@echo "  ingest-batch    Run ingestion over all supported files under BATCH_PATH"
	@echo "  ingest-rdf-sample Run ingestion on sample RDF Turtle file"
	@echo "  retrieve        Query FAISS index"
	@echo "  app             Generate grounded answer (QA or mapping auto-detected)"
	@echo "  app-chat        Start interactive chat mode"
	@echo "  app-mapping     Run mapping-mode example with SCHEMA variable"
	@echo "  smoke           Run end-to-end smoke test (Ollama API)"
	@echo "  smoke-no-api    Run end-to-end smoke test (offline local embeddings)"
	@echo "  test            Run unit tests"
	@echo "  clean-smoke     Remove smoke test artifacts"
	@echo ""
	@echo "Variables:"
	@echo "  PYTHON          Python executable (default: python)"
	@echo "  PYTHONPATH      Import path for package modules (default: src)"
	@echo "  INPUT           Input file/folder for ingestion (default: data/raw)"
	@echo "  OUT_DIR         Output directory for index/metadata (default: data/vectorstore)"
	@echo "  TEXT_DELIMITER  Optional text delimiter for ingestion"
	@echo "  QUERY           Retrieval query string"
	@echo "  SCHEMA          Mapping schema JSON string for app-mapping"
	@echo "  RDF_SAMPLE      Default RDF path for ingest-rdf-sample"
	@echo "  RDF_FILE        Override RDF file path for ingest-rdf-sample"
	@echo "  BATCH_PATH      Directory or file path for ingest-batch"
	@echo "  TOP_K           Number of retrieval results"
	@echo "  OFFLINE_DIM     Offline embedding dimension for smoke-no-api"
	@echo "  LLM_MODEL       Chat model for app/app-chat/app-mapping (default: llama3.1:8b)"

install:
	$(PYTHON) -m pip install -r requirements.txt

install-editable:
	$(PYTHON) -m pip install -e .

ingest:
ifeq ($(strip $(TEXT_DELIMITER)),)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m cidoc_rag.cli.ingest_cli $(INPUT) --out-dir $(OUT_DIR)
else
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m cidoc_rag.cli.ingest_cli $(INPUT) --text-delimiter "$(TEXT_DELIMITER)" --out-dir $(OUT_DIR)
endif

ingest-batch:
ifeq ($(strip $(TEXT_DELIMITER)),)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m cidoc_rag.cli.ingest_cli $(BATCH_PATH) --out-dir $(OUT_DIR)
else
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m cidoc_rag.cli.ingest_cli $(BATCH_PATH) --text-delimiter "$(TEXT_DELIMITER)" --out-dir $(OUT_DIR)
endif

ingest-rdf-sample:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m cidoc_rag.cli.ingest_cli $(RDF_FILE) --out-dir $(OUT_DIR)

retrieve:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m cidoc_rag.cli.retrieve_cli "$(QUERY)" --index-path $(OUT_DIR)/cidoc.index --metadata-path $(OUT_DIR)/cidoc_metadata.json --top-k $(TOP_K)

app:
	PYTHONPATH=$(PYTHONPATH) LLM_MODEL=$(LLM_MODEL) $(PYTHON) -m cidoc_rag.cli.app_cli "$(QUERY)" --k $(TOP_K) --index-path $(OUT_DIR)/cidoc.index --metadata-path $(OUT_DIR)/cidoc_metadata.json

app-chat:
	PYTHONPATH=$(PYTHONPATH) LLM_MODEL=$(LLM_MODEL) $(PYTHON) -m cidoc_rag.cli.app_cli --chat --k $(TOP_K) --index-path $(OUT_DIR)/cidoc.index --metadata-path $(OUT_DIR)/cidoc_metadata.json

app-mapping:
	PYTHONPATH=$(PYTHONPATH) LLM_MODEL=$(LLM_MODEL) $(PYTHON) -m cidoc_rag.cli.app_cli '$(SCHEMA)' --k $(TOP_K) --index-path $(OUT_DIR)/cidoc.index --metadata-path $(OUT_DIR)/cidoc_metadata.json

smoke:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) smoke_test.py

smoke-no-api:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) smoke_test.py --no-api --offline-dim $(OFFLINE_DIM)

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -p "test_*.py"

clean-smoke:
	rm -rf data/smoke
