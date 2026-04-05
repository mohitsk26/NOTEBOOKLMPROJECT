import faiss
import numpy as np
import pickle
from typing import List


class FAISSVectorStore:
    def __init__(self, embedding_dim: int):
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.text_chunks = []

    def add_embeddings(self, embeddings: np.ndarray, chunks: List[str]):
        if len(embeddings) != len(chunks):
            raise ValueError("Embeddings and chunks size mismatch")

        if embeddings.shape[1] != self.index.d:
            raise ValueError("Embedding dimension mismatch")

        self.index.add(embeddings)
        self.text_chunks.extend(chunks)

    def similarity_search(self, query_embedding: np.ndarray, top_k: int = 5):
        if query_embedding.ndim == 1:
            query_embedding = np.array([query_embedding])

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.text_chunks):
                results.append(self.text_chunks[idx])

        return results

    # -----------------------------
    # SAVE BOTH INDEX + CHUNKS
    # -----------------------------
    def save_data(self, index_path: str, chunks_path: str):
        faiss.write_index(self.index, index_path)
        with open(chunks_path, "wb") as f:
            pickle.dump(self.text_chunks, f)

    # -----------------------------
    # LOAD BOTH INDEX + CHUNKS
    # -----------------------------
    def load_data(self, index_path: str, chunks_path: str):
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            self.text_chunks = pickle.load(f)
