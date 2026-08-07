"""
Thin wrapper around the Qdrant client: collection setup, upsert, search.
"""
import uuid
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from . import config
from .pdf_utils import Chunk


def get_client() -> QdrantClient:
    # Generous timeout: large batches / free-tier cloud instances can be slow.
    return QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        timeout=180,
    )


def ensure_collection(client: QdrantClient, dim: int, recreate: bool = False) -> None:
    exists = client.collection_exists(config.COLLECTION_NAME)
    if exists and recreate:
        client.delete_collection(config.COLLECTION_NAME)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
        )


def upsert_chunks(client: QdrantClient, chunks: List[Chunk], vectors: List[List[float]]) -> None:
    points = []
    for chunk, vector in zip(chunks, vectors):
        points.append(
            qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "doc_name": chunk.doc_name,
                    "page_number": chunk.page_number,
                    "text": chunk.text,
                },
            )
        )
    if points:
        client.upsert(collection_name=config.COLLECTION_NAME, points=points, wait=True)


def search(client: QdrantClient, query_vector: List[float], top_k: int):
    return client.query_points(
        collection_name=config.COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points
