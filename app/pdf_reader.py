"""
pdf_reader.py — PDF Text Extraction & Chunking

WHY THIS EXISTS:
    LLMs have token limits, so we can't feed an entire PDF at once.
    This module handles two steps:
    1. Extract raw text from a PDF file
    2. Split that text into overlapping chunks

INTERVIEW POINT:
    "Overlap (e.g., 50 chars) ensures we don't lose context at chunk
    boundaries. If a sentence spans two chunks, the overlap captures
    it in both, so the retriever can still find it."
"""

import PyPDF2


def extract_text(pdf_path: str) -> str:
    """
    Read a PDF file and return all its text as a single string.

    Args:
        pdf_path: Path to the PDF file on disk.

    Raises:
        ValueError: If the file can't be read or parsed.
    """
    try:
        text = ""
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        return text
    except Exception as e:
        raise ValueError(f"Failed to read PDF: {e}")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks using a sliding window.

    Example with chunk_size=10, overlap=3:
        "Hello World, how are you?"
        → ["Hello Worl", "orld, how ", "how are yo", "are you?"]

    Args:
        text:       The full text to split.
        chunk_size: Maximum characters per chunk.
        overlap:    Number of characters shared between consecutive chunks.

    Returns:
        List of text chunks.
    """
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks
