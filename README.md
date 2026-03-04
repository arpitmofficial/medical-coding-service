git clone repo
pip install -r requirements.txt
# medical-coding-service


Add `.env` file with:


JINA_API_KEY=
QDRANT_URL=
QDRANT_API_KEY=


Run test:


python scripts/query_test.py


## Architecture

Query → Jina Embedding → Qdrant Cloud Vector Search → Results
>>>>>>> master
