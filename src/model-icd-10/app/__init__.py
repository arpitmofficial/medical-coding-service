"""
ICD-10 Medical Coding Retrieval System with Hybrid Search
=========================================================

A multi-stage AI pipeline for converting unstructured clinical text 
into proper ICD-10 diagnostic codes using hybrid search combining:

Stage 0: LLM-powered entity extraction from clinical text
Stage 1: Dual embedding generation (dense + sparse vectors)
Stage 2: Qdrant hybrid search with Reciprocal Rank Fusion (RRF)
Stage 3: Merge & deduplicate results
Stage 4: LLM re-ranking with clinical reasoning (returns top 5)

The hybrid approach ensures both semantic understanding and exact 
keyword matching for optimal medical code retrieval accuracy.

Usage:
    from app import adaptive_retrieve_icd_candidates
    
    results = await adaptive_retrieve_icd_candidates("patient has fever and cold")
"""

__version__ = "2.0.0"
__author__ = "Medical Coding Team"

# Main pipeline function
from app.adaptive_retrieval import adaptive_retrieve_icd_candidates

# Key components for advanced usage
from app.preprocessing import parse_entities
from app.embedding import get_embeddings_batch, get_sparse_embeddings_batch
from app.reranking import rerank_codes
from app.qdrant_rest import search_vectors

__all__ = [
    "adaptive_retrieve_icd_candidates",  # Main pipeline function
    "parse_entities", 
    "get_embeddings_batch",
    "get_sparse_embeddings_batch",
    "rerank_codes",
    "search_vectors",
]