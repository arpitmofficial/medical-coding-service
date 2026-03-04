from app.embedding import get_embeddings_batch
from app.qdrant_rest import search_vectors


def retrieve_icd_candidates(query, top_k=20):

    vector = get_embeddings_batch([query])[0]

    results = search_vectors(vector, top_k)

    output = []

    for r in results:

        output.append({
            "code": r["payload"]["code"],
            "description": r["payload"]["description"],
            "score": r["score"]
        })

    return output