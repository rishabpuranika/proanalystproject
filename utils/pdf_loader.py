"""
pdf_loader.py — PDF Document Loading Utility
=============================================

Provides functions to load PDF files into LangChain Document objects
using PyPDFLoader, along with diagnostics for inspecting loaded content.

This module is the first step in the RAG pipeline: ingesting raw PDF
documentation so it can later be chunked, embedded, and stored in a
vector database.

Dependencies:
    - langchain_community (PyPDFLoader)

Usage:
    >>> from utils.pdf_loader import load_pdf, print_diagnostics
    >>> docs = load_pdf("path/to/manual.pdf")
    >>> print_diagnostics(docs)
"""

import os
import logging
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Logger setup — uses the module name so log output is easy to filter,
# e.g. "utils.pdf_loader - INFO - …"
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ===========================================================================
# Public API
# ===========================================================================

def load_pdf(file_path: str) -> List[Document]:
    """Load a PDF file and return a list of LangChain Document objects.

    Each returned Document corresponds to one page of the PDF and carries
    ``page_content`` (the extracted text) plus ``metadata`` (at minimum
    the source file path and the page number).

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the PDF file to load.

    Returns
    -------
    List[Document]
        One Document per page.  The list may be empty if the PDF has no
        extractable text (e.g. scanned images without OCR).

    Raises
    ------
    FileNotFoundError
        If ``file_path`` does not point to an existing file.
    ValueError
        If ``file_path`` does not end with ``.pdf`` (case-insensitive).
    RuntimeError
        If PyPDFLoader fails for any other reason (corrupt file, etc.).
    """

    # --- 1. Validate that the file actually exists on disk ----------------
    if not os.path.isfile(file_path):
        logger.error("File not found: %s", file_path)
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    # --- 2. Quick sanity check on the extension ---------------------------
    if not file_path.lower().endswith(".pdf"):
        logger.error("Invalid file extension for: %s", file_path)
        raise ValueError(
            f"Expected a .pdf file but received '{file_path}'. "
            "Please provide a valid PDF file path."
        )

    # --- 3. Load the PDF using PyPDFLoader --------------------------------
    try:
        logger.info("Loading PDF: %s", file_path)
        loader = PyPDFLoader(file_path)
        documents: List[Document] = loader.load()
        logger.info(
            "Successfully loaded %d page(s) from '%s'.",
            len(documents),
            file_path,
        )
        return documents

    except Exception as exc:
        # Wrap unexpected errors so callers get a clear message.
        logger.exception("Failed to load PDF '%s'.", file_path)
        raise RuntimeError(
            f"An error occurred while loading the PDF '{file_path}': {exc}"
        ) from exc


def print_diagnostics(documents: List[Document]) -> None:
    """Print a quick diagnostic summary of loaded PDF documents.

    Useful for sanity-checking that the PDF was parsed correctly before
    feeding the documents into the chunking / embedding pipeline.

    Prints:
        - Total number of pages (documents).
        - Total character count across all pages.
        - A sample of the first 500 characters from the combined text.

    Parameters
    ----------
    documents : List[Document]
        The list of Document objects returned by :func:`load_pdf`.
    """

    if not documents:
        logger.warning("No documents to diagnose — the list is empty.")
        print("Warning: No documents loaded. Nothing to display.")
        return

    # Concatenate all page texts for aggregate stats.
    full_text: str = "".join(doc.page_content for doc in documents)
    total_pages: int = len(documents)
    total_chars: int = len(full_text)
    sample_text: str = full_text[:500]

    # Display the diagnostics.
    print("=" * 60)
    print("PDF Diagnostics")
    print("=" * 60)
    print(f"  Total Pages     : {total_pages}")
    print(f"  Total Characters: {total_chars:,}")
    print("-" * 60)
    print("  Sample Text (first 500 chars):")
    print("-" * 60)
    print(sample_text)
    print("=" * 60)

    logger.info(
        "Diagnostics — pages: %d, characters: %s",
        total_pages,
        f"{total_chars:,}",
    )
