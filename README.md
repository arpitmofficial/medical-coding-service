# ICD Retrieval System

This project retrieves ICD codes using vector search.

## Setup

Clone repo:


git clone repo
cd DASS_Project


Install dependencies:


pip install -r requirements.txt


Add `.env` file with:


JINA_API_KEY=

QDRANT_URL=

QDRANT_API_KEY=


Run test:


python scripts/query_test.py


## Architecture

Query → Jina Embedding → Qdrant Cloud Vector Search → Results
