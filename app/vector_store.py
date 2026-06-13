"""
vector_store.py — Vector Database (ChromaDB)

WHY THIS EXISTS:
    Stores document chunks as vectors and retrieves the most similar ones
    when the user asks a question. This is the "Retrieval" part of RAG.
    Separated from embeddings.py because:
    - The database (ChromaDB) and the embedding model are independent
    - We could swap to Pinecone or Weaviate without touching the embedding code

INTERVIEW POINT:
    "ChromaDB uses HNSW (Hierarchical Navigable Small World) indexing for
    approximate nearest neighbor search. Cosine similarity measures how
    close two vectors are — 1.0 = identical meaning, 0.0 = unrelated."
"""

import chromadb
from app.embeddings import EmbeddingModel
from app.config import settings


class VectorStore:
    """Manages document storage and retrieval using ChromaDB."""

    def __init__(self, embedding_model: EmbeddingModel):
        # PersistentClient saves data to disk so it survives restarts
        self.client = chromadb.PersistentClient(path=settings.vector_store_path)
        self.embedding_model = embedding_model

    def add_documents(
        self,
        collection_name: str,
        texts: list[str],
        ids: list[str] | None = None,
    ) -> None:
        """
        Embed document chunks and store them in a ChromaDB collection.

        Args:
            collection_name: Name of the collection (like a "table" in SQL).
            texts:           List of text chunks to store.
            ids:             Optional custom IDs. Auto-generated if not provided.
        """
        # get_or_create avoids errors if the collection already exists
        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

        # Generate IDs if not provided
        if ids is None:
            # Count existing docs to avoid ID collisions on repeated uploads
            existing_count = collection.count()
            ids = [f"doc_{existing_count + i}" for i in range(len(texts))]

        # Convert text → vectors
        embeddings = self.embedding_model.embed(texts)

        # Store everything together
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
        )

    def search(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 5,
    ) -> list[str]:
        """
        Find the most similar document chunks to a query.

        Args:
            collection_name: Which collection to search.
            query_text:      The user's question.
            n_results:       How many chunks to return.

        Returns:
            List of the most relevant text chunks, ranked by similarity.
        """
        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Embed the query using the same model used for documents
        query_embedding = self.embedding_model.embed([query_text])

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
        )

        # results['documents'] is a list of lists — [0] gets the first (only) query's results
        return results["documents"][0]
