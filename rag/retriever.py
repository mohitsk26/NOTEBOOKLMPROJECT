from embeddings.embedder import TextEmbedder
from vectorstore.faiss_store import FAISSVectorStore
from config import TOP_K


class Retriever:
    """
    Handles retrieval of relevant chunks using embeddings + FAISS
    """

    def __init__(self, vector_store: FAISSVectorStore):
        # Store FAISS instance
        self.vector_store = vector_store

        # Initialize embedder (same model as used for chunks)
        self.embedder = TextEmbedder()

    def retrieve(self, query: str, top_k: int = TOP_K):
        """
        Convert query → embedding → retrieve top-k chunks
        """

        if not query:
            raise ValueError("Query cannot be empty")

        # STEP 1: Query → embedding
        query_embedding = self.embedder.embed_query(query)

        # STEP 2: FAISS search
        results = self.vector_store.similarity_search(
            query_embedding,
            top_k=top_k
        )

        return results
