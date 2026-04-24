# CPT Medical Coding Pipeline

This module retrieves CPT procedure codes from clinical or procedure descriptions using adaptive Qdrant search, embedding generation, and LLM re-ranking.

## What it does

- Detects whether the Qdrant collection supports hybrid search, named dense vectors, or legacy dense vectors
- Retrieves candidate CPT codes from the `cpt_hybrid` collection
- Re-ranks results with the LLM and returns the top codes with confidence scores and explanations

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
python model-cpt/scripts/query_test.py
```

Then enter a procedure description at the prompt, for example:

```text
Enter procedure description: laparoscopic appendectomy
```

Type `quit`, `exit`, or `q` to stop.

## Shared API usage

When the FastAPI gateway is running, CPT can also be reached through the shared `/predict` endpoint if the model registers successfully at startup.
