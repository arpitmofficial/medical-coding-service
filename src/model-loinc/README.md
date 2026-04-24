# LOINC Medical Coding Pipeline

This module retrieves LOINC lab and observation codes from clinical text using adaptive retrieval, metadata-aware scoring, and LLM re-ranking.

## What it does

- Detects whether the Qdrant collection supports hybrid search, named dense vectors, or legacy dense vectors
- Expands common lab abbreviations and related terms
- Retrieves candidate LOINC codes from the `loinc_hybrid` collection
- Re-ranks results and returns the top codes with confidence scores and explanations

## Environment variables

Set these in the shared `src/.env` file or in your shell before running:

- `QDRANT_URL`
- `QDRANT_API_KEY`
- `LLM_API_KEY`
- `LLM_MODEL` (default: `gpt-4o-mini`)
- `QDRANT_TOP_K` (default: `50`)
- `FINAL_TOP_N` (default: `5`)
- `MIN_SCORE` (default: `0.0`)

## Run from terminal

From the project `src/` directory:

```bash
python model-loinc/scripts/query_test.py
```

Then enter a lab or observation query at the prompt, for example:

```text
Enter lab/observation query: HbA1c test
```

Type `quit`, `exit`, or `q` to stop.

## Shared API usage

When the FastAPI gateway is running, LOINC can also be reached through the shared `/predict` endpoint if the model registers successfully at startup.
