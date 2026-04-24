# CPT Model

CPT retrieval pipeline used by the medical coding API.

## What It Does

Given free-text clinical notes, this model:

1. Extracts CPT-focused entities from the note.
2. Runs hybrid retrieval (dense + sparse) in Qdrant.
3. Fuses and filters candidate codes.
4. Applies LLM reranking.
5. Returns final CPT predictions with confidence and explanation.

## Main Files

- app/adaptive_retrieval_cpt.py: Main orchestration entrypoint.
- app/preprocessing.py: CPT entity extraction.
- app/embedding.py: Dense and sparse embeddings.
- app/qdrant_rest.py: Qdrant retrieval and fusion helpers.
- app/reranking.py: Final LLM reranking.
- scripts/query_test.py: Interactive local query script.
- scripts/ingest.py: Data ingestion script.

## Expected Environment Variables

Configured via src/.env:

- QDRANT_URL
- QDRANT_API_KEY
- JINA_API_KEY
- LLM_API_KEY
- LLM_MODEL
- QDRANT_TOP_K (optional)
- FINAL_TOP_N (optional)
- MIN_SCORE (optional)

## Run Local Query Test

From repository root:

```bash
cd src/model-cpt/scripts
python query_test.py
```

## Ingest / Rebuild Index

From repository root:

```bash
cd src/model-cpt/scripts
python ingest.py
```

## Evaluation

Evaluation scripts and result files are under:

- scripts/Testing

Run these after prompt updates, retrieval tuning, or collection rebuilds.

## API Integration

This model is exposed through:

- POST /predict
- POST /predict/cpt

See complete API docs at:

- ../../api-documentation.md
