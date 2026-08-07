import os
from functools import lru_cache
from typing import List

os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")

from sentence_transformers import SentenceTransformer

from . import config

@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    try:
        return SentenceTransformer(config.EMBEDDING_MODEL)
    except Exception:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        return SentenceTransformer(config.EMBEDDING_MODEL)

def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()

def embed_query(text: str) -> List[float]:
    return embed_texts([text])[0]

def get_embedding_dim() -> int:
    return get_model().get_sentence_embedding_dimension()
