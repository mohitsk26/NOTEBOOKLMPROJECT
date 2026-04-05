from sentence_transformers import SentenceTransformer
from typing import List
from config import EMBEDDING_MODEL


class TextEmbedder:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def embed_texts(self, texts: List[str]):
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings

    def embed_query(self, query: str):
        if not query:
            raise ValueError("Query cannot be empty")

        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding
