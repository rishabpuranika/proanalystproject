"""
Data Ingestion Script
======================

Processes the Upwork API documentation PDF, splits it into semantically
meaningful chunks, generates embeddings, and persists them in a ChromaDB
vector store for later retrieval.

Usage:
    python ingest.py

This script must be run once before starting the Streamlit application.
It reads the PDF from data/upwork_api_reference.pdf and creates the
vector database in the chroma_db/ directory.
"""

import os
# Force pure-Python implementation for protobuf to bypass descriptor conflicts on Streamlit Cloud
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import logging
import sys

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from utils.pdf_loader import load_pdf, print_diagnostics
from utils.vector_store import create_vector_store

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PDF_PATH: str = os.path.join("data", "upwork_api_reference.pdf")
CHROMA_PERSIST_DIR: str = "chroma_db"

# Chunking Parameters
# --------------------
# chunk_size = 500 characters
#   - Large enough to capture meaningful API details (endpoint descriptions,
#     parameter lists, code examples) in a single chunk.
#   - Small enough to maintain topical focus so that similarity search
#     returns highly relevant results rather than diluted context.
#
# chunk_overlap = 50 characters (~10% of chunk_size)
#   - Prevents information loss across chunk boundaries: when a sentence
#     or API parameter description straddles two chunks, the overlap
#     ensures it appears in full in at least one chunk.
#   - Preserves technical context: API docs often have tightly coupled
#     sentences (e.g. "The access token is valid for 24 hours. Use the
#     refresh token to obtain a new one.") that should not be separated.
#   - Maintains code snippets and examples: API request/response examples
#     that span multiple lines are less likely to be cut mid-example.

CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50


def create_chunks(documents: list[Document]) -> list[Document]:
    """
    Split loaded PDF documents into smaller, overlapping text chunks.

    Uses LangChain's RecursiveCharacterTextSplitter which tries multiple
    separators (paragraphs → sentences → words) to find natural split
    points, preserving semantic coherence within each chunk.

    Args:
        documents: List of Document objects from the PDF loader (one per page).

    Returns:
        List of Document objects, each representing a single chunk with
        updated metadata including a unique chunk_id.
    """
    logger.info(
        "Splitting documents into chunks (size=%d, overlap=%d)...",
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )

    # RecursiveCharacterTextSplitter attempts to split by these separators
    # in order: double newline → single newline → space → character.
    # This preserves paragraph and sentence boundaries where possible.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = text_splitter.split_documents(documents)

    # Assign sequential chunk IDs for traceability and metadata enrichment.
    # Each chunk retains its original page number from the PDF loader
    # and gains a unique chunk_id for precise reference.
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
        chunk.metadata["source"] = PDF_PATH

    logger.info("Created %d chunks from %d pages.", len(chunks), len(documents))
    return chunks


def main() -> None:
    """
    Main ingestion pipeline.

    Steps:
        1. Load the PDF document.
        2. Print diagnostic information (pages, characters, sample text).
        3. Split into overlapping chunks.
        4. Create embeddings and persist to ChromaDB.
    """
    logger.info("=" * 60)
    logger.info("STARTING DATA INGESTION PIPELINE")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load PDF
    # ------------------------------------------------------------------
    if not os.path.isfile(PDF_PATH):
        logger.error("PDF file not found at '%s'.", PDF_PATH)
        logger.error(
            "Please place the Upwork API documentation PDF in the data/ directory "
            "as 'upwork_api_reference.pdf'."
        )
        sys.exit(1)

    logger.info("Loading PDF from: %s", PDF_PATH)
    documents: list[Document] = load_pdf(PDF_PATH)

    # ------------------------------------------------------------------
    # Step 2: Print diagnostics
    # ------------------------------------------------------------------
    print_diagnostics(documents)

    # ------------------------------------------------------------------
    # Step 3: Chunk documents
    # ------------------------------------------------------------------
    chunks: list[Document] = create_chunks(documents)

    # Display sample chunk for verification
    if chunks:
        print("\n" + "=" * 60)
        print("SAMPLE CHUNK (Chunk 0)")
        print("=" * 60)
        print(f"Content: {chunks[0].page_content[:300]}...")
        print(f"Metadata: {chunks[0].metadata}")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Step 4: Create vector store
    # ------------------------------------------------------------------
    logger.info("Creating vector store and generating embeddings...")
    vector_store = create_vector_store(chunks, persist_directory=CHROMA_PERSIST_DIR)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"Total Pages Processed : {len(documents)}")
    print(f"Total Chunks Created  : {len(chunks)}")
    print(f"Chunk Size            : {CHUNK_SIZE} characters")
    print(f"Chunk Overlap         : {CHUNK_OVERLAP} characters")
    print(f"Vector DB Location    : {CHROMA_PERSIST_DIR}/")
    print("=" * 60)

    logger.info("Ingestion pipeline completed successfully.")


if __name__ == "__main__":
    main()
