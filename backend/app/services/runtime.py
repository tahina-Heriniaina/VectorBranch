from functools import lru_cache

from app.embeddings.model import EmbeddingModel


@lru_cache(maxsize=1)
def get_embedding_model():
    return EmbeddingModel()