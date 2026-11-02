"""Load a folder of TXT/PDF files into searchable chunks."""

from __future__ import annotations

from pathlib import Path

from hr_copilot.chunking import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, chunk_documents
from hr_copilot.models import Chunk, DocumentParseError, ParsedDocument
from hr_copilot.parsing import parse_file

SUPPORTED_SUFFIXES = {".txt", ".pdf"}


def ingest_directory(
    directory: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> tuple[list[Chunk], list[str]]:
    """Parse every supported file in directory (non-recursive).

    Returns (chunks, warnings). Parse failures become warnings so one bad PDF
    does not block the rest of the corpus.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Document directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    documents: list[ParsedDocument] = []
    warnings: list[str] = []
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        warnings.append(f"No .txt or .pdf files in {directory}")
        return [], warnings

    for path in files:
        try:
            parsed = parse_file(path)
        except DocumentParseError as exc:
            warnings.append(str(exc))
            continue
        warnings.extend(parsed.warnings)
        documents.append(parsed)

    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        warnings.append("Ingest produced 0 chunks (empty files or no extractable text).")
    return chunks, warnings
