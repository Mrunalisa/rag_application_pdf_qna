import os
from dotenv import load_dotenv

load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes")


# OpenRouter (LLM)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_docs")

# Data / chunking
DATA_DIR = os.getenv("DATA_DIR", "data")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Retrieval
TOP_K = int(os.getenv("TOP_K", "4"))
# Cosine similarity threshold below which we consider "not found in documents"
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.30"))

NOT_FOUND_MESSAGE = "The information is not available in the supplied documents."
