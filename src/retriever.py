from typing import List
from . import config
from .embeddings import embed_query
from .vector_store import get_client, search

def retrieve(question: str, top_k: int = None):
    top_k = top_k or config.TOP_K
    client = get_client()
    query_vector = embed_query(question)
    results = search(client, query_vector, top_k)
    return results

def has_sufficient_evidence(results) -> bool:
    if not results:
        return False
    return results[0].score >= config.MIN_SCORE
