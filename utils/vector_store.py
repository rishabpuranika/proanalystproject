"""
Vector Store Utility Module
============================

Manages ChromaDB vector store operations including creation, loading,
and semantic similarity search for the RAG pipeline.

Uses sentence-transformers/all-MiniLM-L6-v2 for embedding generation,
which produces 384-dimensional dense vectors optimized for semantic
similarity tasks.

ChromaDB Configuration:
    - chunk_size=500: Balances between having enough context for meaningful
      retrieval and keeping chunks focused on specific topics. Larger chunks
      provide more context but may dilute relevance scores.
    - chunk_overlap=50: ~10% overlap ensures that information at chunk
      boundaries is preserved in at least one chunk, preventing loss of
      critical API details that span chunk edges.
"""

import logging
from typing import Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_PERSIST_DIR: str = "chroma_db"
COLLECTION_NAME: str = "upwork_api_docs"


def _get_embedding_function() -> HuggingFaceEmbeddings:
    """
    Initialise and return the HuggingFace embedding model.

    Uses all-MiniLM-L6-v2 — a lightweight, fast model that produces
    384-dimensional vectors well-suited for semantic search tasks.

    Returns:
        HuggingFaceEmbeddings: Configured embedding model instance.
    """
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # L2-normalise for cosine sim
    )
    logger.info("Embedding model loaded successfully.")
    return embeddings


def create_vector_store(
    chunks: list[Document],
    persist_directory: str = DEFAULT_PERSIST_DIR,
) -> Chroma:
    """
    Create a ChromaDB vector store from document chunks and persist it.

    Each chunk is embedded using all-MiniLM-L6-v2 and stored alongside
    its metadata (source file, page number, chunk ID) so that retrieval
    results can be traced back to the original documentation.

    Args:
        chunks: List of LangChain Document objects with page_content and metadata.
        persist_directory: Filesystem path where ChromaDB will persist data.

    Returns:
        Chroma: The populated and persisted vector store instance.

    Raises:
        ValueError: If the chunks list is empty.
        RuntimeError: If vector store creation fails.
    """
    if not chunks:
        raise ValueError("Cannot create vector store from an empty chunk list.")

    import shutil
    import os

    # Clear existing vector store database to prevent duplicate entries
    if os.path.exists(persist_directory):
        logger.info("Removing existing vector store at '%s' to avoid duplicates...", persist_directory)
        shutil.rmtree(persist_directory, ignore_errors=True)

    logger.info(
        "Creating vector store with %d chunks in '%s'...",
        len(chunks),
        persist_directory,
    )

    try:
        # Ensure each chunk carries the required metadata fields
        for idx, chunk in enumerate(chunks):
            if "chunk_id" not in chunk.metadata:
                chunk.metadata["chunk_id"] = idx
            if "source" not in chunk.metadata:
                chunk.metadata["source"] = "upwork_api_reference.pdf"

        embeddings = _get_embedding_function()

        # Create the Chroma vector store — this embeds all chunks and
        # persists the database to the specified directory automatically.
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=COLLECTION_NAME,
        )

        logger.info(
            "Vector store created and persisted. Total vectors: %d",
            len(chunks),
        )
        return vector_store

    except Exception as exc:
        logger.error("Failed to create vector store: %s", exc)
        raise RuntimeError(f"Vector store creation failed: {exc}") from exc


def load_vector_store(
    persist_directory: str = DEFAULT_PERSIST_DIR,
) -> Chroma:
    """
    Load an existing ChromaDB vector store from disk.

    Args:
        persist_directory: Filesystem path where ChromaDB data is stored.

    Returns:
        Chroma: The loaded vector store instance ready for queries.

    Raises:
        FileNotFoundError: If the persist directory does not exist.
        RuntimeError: If loading fails.
    """
    import os

    if not os.path.isdir(persist_directory):
        raise FileNotFoundError(
            f"Vector store directory not found: '{persist_directory}'. "
            "Run ingest.py first to create the database."
        )

    logger.info("Loading vector store from '%s'...", persist_directory)

    try:
        embeddings = _get_embedding_function()

        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )

        logger.info("Vector store loaded successfully.")
        return vector_store

    except Exception as exc:
        logger.error("Failed to load vector store: %s", exc)
        raise RuntimeError(f"Vector store loading failed: {exc}") from exc


def retrieve_context(
    query: str,
    vector_store: Chroma,
    k: int = 3,
) -> list[dict]:
    """
    Perform semantic similarity search and return the top-k most relevant chunks.

    ChromaDB returns results as (Document, distance) tuples where distance
    is the L2 (Euclidean) distance between the query embedding and each
    stored embedding.  Lower distance = higher similarity.

    Because we normalise embeddings during creation, L2 distance is
    directly related to cosine similarity:
        cosine_sim = 1 - (distance² / 2)

    Args:
        query: The user's natural-language question.
        vector_store: A loaded Chroma vector store instance.
        k: Number of top results to return (default 3).

    Returns:
        A list of dictionaries, each containing:
            - content (str):  The chunk text.
            - page (int):     The source PDF page number.
            - score (float):  The similarity distance (lower is better).

    Raises:
        ValueError: If query is empty.
        RuntimeError: If retrieval fails.
    """
    if not query or not query.strip():
        raise ValueError("Search query must not be empty.")

    logger.info("Retrieving top %d chunks for query: '%s'", k, query[:80])

    try:
        # similarity_search_with_score returns List[Tuple[Document, float]]
        # where float is the distance score (lower = more similar).
        results = vector_store.similarity_search_with_score(query, k=k)

        context_list: list[dict] = []
        for doc, distance in results:
            context_list.append(
                {
                    "content": doc.page_content,
                    "page": doc.metadata.get("page", -1),
                    "score": round(float(distance), 4),
                }
            )

        logger.info(
            "Retrieved %d chunks. Best distance: %.4f",
            len(context_list),
            context_list[0]["score"] if context_list else float("inf"),
        )

        return context_list

    except Exception as exc:
        logger.error("Context retrieval failed: %s", exc)
        raise RuntimeError(f"Context retrieval failed: {exc}") from exc
