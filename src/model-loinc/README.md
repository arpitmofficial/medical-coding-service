# LOINC Model

LOINC retrieval pipeline used by the medical coding API.

## What It Does

Given free-text lab or observation text, this model:

1. Extracts LOINC-oriented entities (component, system, context).
2. Runs hybrid retrieval in Qdrant (dense + sparse).
3. Fuses scores and applies metadata-aware ranking.
4. Returns a capped candidate pool.

Current behavior:

- The reranking module exists, but reranking is currently disabled in the active orchestration path.
- The active pipeline returns the capped pre-rerank pool (default cap is 50).

## Main Files

- app/adaptive_retrieval_loinc.py: Main orchestration entrypoint.
- app/preprocessing.py: LOINC entity extraction.
- app/reranking.py: LOINC reranking implementation (currently not invoked in active path).
- scripts/query_test.py: Interactive local query script.
- scripts/ingest.py: Ingestion script for LOINC data.

## Expected Environment Variables

Configured via src/.env:

- QDRANT_URL
- QDRANT_API_KEY
- LLM_API_KEY
- LLM_MODEL
- QDRANT_TOP_K (optional)
- LOINC_PRE_RERANK_TOP_N (optional)
- FINAL_TOP_N (reserved for rerank path)
- MIN_SCORE (optional)

## Run Local Query Test

From repository root:

```bash
cd src/model-loinc/scripts
python query_test.py
```

## Ingest / Rebuild Index

From repository root:

```bash
cd src/model-loinc/scripts
python ingest.py
```

## Evaluation

Evaluation scripts and outputs are under:

- scripts/Testing

Common scripts:

- scripts/Testing/eval.py
- scripts/Testing/eval_detailed.py
- scripts/Testing/sweep_thresholds.py

## API Integration

This model is exposed through:

- POST /predict
- POST /predict/loinc

See complete API docs at:

- [API Documentation](../../api-documentation.md)

## Embedding Notes

- Dense embeddings use local SapBERT.
- Sparse embeddings use BM25-style FastEmbed retrieval.
