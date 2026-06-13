"""
embeddings.py — Text Embedding Model

WHY THIS EXISTS:
    Converts text into numerical vectors (embeddings) so we can do
    similarity search. Separated from vector_store.py because:
    1. The embedding model and the database are independent concerns
    2. We might swap the model (e.g., OpenAI embeddings) without changing storage
    3. Easier to test — embed text without needing a database

INTERVIEW POINT:
    "Embeddings map text to a high-dimensional vector space where similar
    meanings are close together. 'all-MiniLM-L6-v2' produces 384-dim vectors,
    is only 80MB, and runs on CPU — good enough for most RAG use cases."
"""

from sentence_transformers import SentenceTransformer
from app.config import settings


class EmbeddingModel:
    """Wrapper around SentenceTransformer for text → vector conversion."""

    def __init__(self):
        # This is the heaviest init in the app — downloads/loads the ML model
        self.model = SentenceTransformer(settings.embedding_model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Convert a list of text strings into embedding vectors.

        Args:
            texts: List of strings to embed (e.g., document chunks).

        Returns:
            List of vectors, one per input text.
            Each vector is a list of floats (384 dimensions for MiniLM).
        """
        # .encode() returns a numpy array, .tolist() converts to plain Python lists
        return self.model.encode(texts).tolist()
