# Contributing

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[train,rag,agent,test]"
python -m pytest -q
```

The local test suite uses the deterministic hash embedding and in-memory vector store, so it does not require model downloads or external services.

## Adding data

Training examples use the JSONL schema in `data/raw/demo.jsonl`. Retrieval documents live under `data/rag/documents`; the ingestion script generates chunks and graph records from that directory. Evaluation questions use `data/rag/eval.jsonl` and identify relevant documents or chunks.

## Code conventions

- Keep state transitions and tool contracts typed.
- Preserve source metadata and citation identifiers when adding retrieval components.
- Add a focused test for new chunking, retrieval, verification, memory, or workflow behavior.
- Keep generated indexes, model checkpoints, memory files, and caches out of commits.
