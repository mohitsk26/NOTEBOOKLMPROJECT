from embeddings.embedder import TextEmbedder
from vectorstore.faiss_store import FAISSVectorStore
from config import TOP_K


class Retriever:
    """
    Handles retrieval of relevant chunks using:
    1. Semantic search (FAISS)
    2. Keyword search (basic hybrid search)
    """

    def __init__(self, vector_store: FAISSVectorStore):
        # Store FAISS instance
        self.vector_store = vector_store

        # Initialize embedder (same model as used for chunks)
        self.embedder = TextEmbedder()

    # -----------------------------
    # Keyword Search (Hybrid Part)
    # -----------------------------
    def keyword_search(self, query: str, chunks):
        """
        Simple keyword matching (case-insensitive)
        """
        query = query.lower()
        return [chunk for chunk in chunks if query in chunk.lower()]

    # -----------------------------
    # Main Retrieve Function
    # -----------------------------
    def retrieve(self, query: str, top_k: int = TOP_K):
        """
        Retrieve top-k relevant chunks using hybrid approach
        """

        if not query:
            raise ValueError("Query cannot be empty")

        # STEP 1: Query → embedding
        query_embedding = self.embedder.embed_query(query)

        # STEP 2: Semantic search (FAISS)
        semantic_results = self.vector_store.similarity_search(
            query_embedding,
            top_k=top_k
        )

        # STEP 3: Keyword search (basic hybrid)
        keyword_results = self.keyword_search(
            query,
            self.vector_store.text_chunks
        )

        # STEP 4: Combine results (remove duplicates)
        combined_results = list(set(semantic_results + keyword_results))

        # STEP 5: Limit to top_k
        return combined_results[:top_k]
