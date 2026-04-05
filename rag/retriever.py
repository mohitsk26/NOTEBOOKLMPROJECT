from embeddings.embedder import TextEmbedder
from vectorstore.faiss_store import FAISSVectorStore
from config import TOP_K


class Retriever:
    """
    Handles retrieval of relevant chunks using embeddings + FAISS
    """

    def __init__(self, vector_store: FAISSVectorStore):
        # Store reference to FAISS vector store
        self.vector_store = vector_store

        # Initialize embedder (same model used for chunks)
        self.embedder = TextEmbedder()

    def retrieve(self, query: str, top_k: int = TOP_K):
        """
        Retrieve top-k relevant chunks for a query
        """

        if not query:
            raise ValueError("Query cannot be empty")

        # STEP 1: Convert query → embedding
        query_embedding = self.embedder.embed_query(query)

        # STEP 2: Search in FAISS
        results = self.vector_store.similarity_search(
            query_embedding,
            top_k=top_k
        )

        return results
