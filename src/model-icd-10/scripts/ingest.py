import asyncio
import httpx
import logging
import os
import sys
import pandas as pd
import uuid

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.config import QDRANT_URL, QDRANT_HEADERS
from app.embedding import get_embeddings_batch, get_sparse_embeddings_batch

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

COLLECTION_NAME = "icd10_hybrid"
# Path relative to this script's parent directory (model-icd-10/)
EXCEL_FILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "icd10cm_codes_2026.xlsx"
)


async def setup_collection(client: httpx.AsyncClient):
    logger.info(f"Recreating hybrid collection: {COLLECTION_NAME}...")

    # Ignore 404 if it doesn't exist yet
    await client.delete(
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}", headers=QDRANT_HEADERS
    )

    # SapBERT produces 768-dimensional embeddings
    schema_payload = {
        "vectors": {"dense": {"size": 768, "distance": "Cosine"}},
        "sparse_vectors": {"sparse": {}},
    }

    res = await client.put(
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}",
        headers=QDRANT_HEADERS,
        json=schema_payload,
    )
    res.raise_for_status()
    logger.info(" Hybrid collection ready.")


async def ingest_from_excel():
    # 1. Read the Excel file
    logger.info(f"Reading data from {EXCEL_FILE_PATH}...")
    df = pd.read_excel(EXCEL_FILE_PATH)

    # UPDATE THESE to match your actual Excel column headers
    code_column = "ICD10"
    description_column = "Diagnosis Description"

    # Drop empty rows
    df = df.dropna(subset=[code_column, description_column])
    total_records = len(df)
    logger.info(f"Found {total_records} valid ICD-10 records.")

    async with httpx.AsyncClient(timeout=60.0) as client:
        await setup_collection(client)

        batch_size = 100
        total_uploaded = 0

        # 2. Process in batches
        for start_idx in range(0, total_records, batch_size):
            batch_df = df.iloc[start_idx : start_idx + batch_size]

            codes = batch_df[code_column].astype(str).tolist()
            descriptions = batch_df[description_column].astype(str).tolist()

            # 3. Generate BOTH Dense (SapBERT) and Sparse (BM25) vectors locally
            logger.info(
                f"Generating vectors for batch {start_idx} to {start_idx + len(codes)}..."
            )

            # We embed the descriptions to capture the medical meaning
            dense_vectors = await get_embeddings_batch(descriptions)
            sparse_vectors = await get_sparse_embeddings_batch(descriptions)

            # 4. Format payload for Qdrant
            points = []
            for i in range(len(codes)):
                # Qdrant requires a unique ID (UUID or integer).
                # We generate a stable UUID based on the ICD code string.
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, codes[i]))

                points.append(
                    {
                        "id": point_id,
                        "payload": {"code": codes[i], "description": descriptions[i]},
                        "vector": {
                            "dense": dense_vectors[i],
                            "sparse": sparse_vectors[i],
                        },
                    }
                )

            # 5. Upload to Qdrant
            res = await client.put(
                f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points",
                headers=QDRANT_HEADERS,
                json={"points": points},
            )
            res.raise_for_status()

            total_uploaded += len(points)
            logger.info(f" Uploaded {total_uploaded}/{total_records} points.")

        logger.info(" Full Excel ingestion complete!")


if __name__ == "__main__":
    asyncio.run(ingest_from_excel())
