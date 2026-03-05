# ICD Retrieval System

This project retrieves ICD codes using vector search.

## Setup

Clone repo:
git clone repo_link
cd medical-coding-service


Install dependencies:
pip install -r requirements.txt


Add `.env` file with:

```
JINA_API_KEY=
QDRANT_URL=
QDRANT_API_KEY=
LLM_API_KEY=
LLM_MODEL=
QDRANT_TOP_K=(50)
FINAL_TOP_N=(5)
MIN_SCORE=(0.70)
```

## Architecture

Query → LLM Entity Parsing → Jina Embedding → Qdrant Vector Search → LLM Re-ranking → Results
