"""Token budgets shared by ingestion and embedding retrieval."""

from functools import lru_cache


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MAX_TOKENS = 256
CHUNK_TOKENS = 220
CHUNK_OVERLAP_TOKENS = 30


@lru_cache(maxsize=1)
def get_embedding_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
