import requests
from app.config import JINA_API_KEY

URL = "https://api.jina.ai/v1/embeddings"

HEADERS = {
    "Authorization": f"Bearer {JINA_API_KEY}",
    "Content-Type": "application/json"
}

def get_embeddings_batch(texts):

    payload = {
        "input": texts,
        "model": "jina-embeddings-v2-base-en"
    }

    response = requests.post(URL, headers=HEADERS, json=payload)

    data = response.json()

    return [item["embedding"] for item in data["data"]]