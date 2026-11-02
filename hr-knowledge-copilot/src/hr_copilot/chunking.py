"""Split page text into overlapping chunks that keep source metadata.

Design (character-based, good for mixed Chinese/English policy text):

- chunk_size=400: about one clause (title + rule + exception) in Chinese HR
  documents. Small enough for a citation snippet, large enough for context.
- overlap=80 (20%): if a sentence is cut at a window boundary, keywords near
  the cut still appear in both chunks so retrieval does not miss them.
- Chunk inside a single page only: page_number on a hit stays true. We do not
  merge page 1 and page 2 into one chunk.
- Prefer breaking at blank lines or sentence punctuation, then fall back to a
  hard character window so tests stay deterministic.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from hr_copilot.models import Chunk, ParsedDocument, ParsedPage

DEFAULT_CHUNK_SIZE = 400
DEFAULT_OVERLAP = 80

_BREAK_CHARS = "。！？；;!?\n"


def chunk_documents(
    documents: Sequence[ParsedDocument],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size=chunk_size, overlap=overlap))
    return chunks


def chunk_document(
    document: ParsedDocument,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in document.pages:
        chunks.extend(
            chunk_page(
                document,
                page,
                chunk_size=chunk_size,
                overlap=overlap,
                id_prefix=f"{document.source_name}-p{page.page_number}",
            )
        )
    return chunks


def chunk_page(
    document: ParsedDocument,
    page: ParsedPage,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    id_prefix: str = "chunk",
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0:
        raise ValueError("overlap must be >= 0.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    text = page.text
    if not text.strip():
        return []

    windows = _windows(text, chunk_size=chunk_size, overlap=overlap)
    chunks: list[Chunk] = []
    for index, (start, end) in enumerate(windows, start=1):
        piece = text[start:end]
        if not piece.strip():
            continue
        paragraph_index = _paragraph_index_at(text, start)
        chunks.append(
            Chunk(
                chunk_id=f"{id_prefix}-c{index}",
                source_path=str(document.source_path),
                source_name=document.source_name,
                source_kind=document.kind,
                page_number=page.page_number,
                paragraph_index=paragraph_index,
                text=piece.strip(),
                char_start=start,
                char_end=end,
            )
        )
    return chunks


def _windows(text: str, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    length = len(text)
    if length <= chunk_size:
        return [(0, length)]

    step = chunk_size - overlap
    starts: list[int] = []
    start = 0
    while start < length:
        starts.append(start)
        if start + chunk_size >= length:
            break
        start += step

    windows: list[tuple[int, int]] = []
    for i, raw_start in enumerate(starts):
        end = min(raw_start + chunk_size, length)
        snapped_end = _snap_end(text, raw_start, end) if i < len(starts) - 1 else end
        windows.append((raw_start, max(snapped_end, raw_start + 1)))
        if snapped_end >= length:
            break
    return windows


def _snap_end(text: str, start: int, end: int) -> int:
    if end >= len(text):
        return len(text)
    window_start = max(start + 1, end - 40)
    region = text[window_start:end]
    for offset, char in enumerate(reversed(region)):
        if char in _BREAK_CHARS:
            return end - offset
    return end


def _paragraph_index_at(text: str, char_start: int) -> int:
    prefix = text[: char_start + 1]
    parts = re.split(r"\n\s*\n", prefix)
    return max(1, len([part for part in parts if part.strip()]))
