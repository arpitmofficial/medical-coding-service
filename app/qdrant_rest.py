import requests
from app.config import QDRANT_URL, HEADERS

COLLECTION = "icd10"


def search_vectors(vector, limit=20):

    payload = {
        "vector": vector,
        "limit": limit,
        "with_payload": True
    }

    response = requests.post(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
        headers=HEADERS,
        json=payload
    )

    return response.json()["result"]