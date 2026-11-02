"""Shared data structures for parsing, chunking, and retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


class DocumentParseError(RuntimeError):
    """Raised when a file exists but cannot be parsed into text."""


class QueryError(ValueError):
    """Raised when a search query is empty or otherwise unusable."""


@dataclass(frozen=True)
class ParsedPage:
    """One page of extracted text.

    TXT files use page_number=1 and fill paragraph_count for citation.
    PDF files use the real PDF page index (1-based).
    """

    page_number: int
    text: str
    paragraph_count: int = 0


@dataclass(frozen=True)
class ParsedDocument:
    source_path: Path
    source_name: str
    kind: str
    pages: tuple[ParsedPage, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_path: str
    source_name: str
    source_kind: str
    page_number: int
    paragraph_index: int | None
    text: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float
    matched_terms: tuple[str, ...]
    explanation: str


@dataclass(frozen=True)
class SearchResult:
    query: str
    hits: tuple[SearchHit, ...]
    message: str
    warnings: tuple[str, ...] = field(default_factory=tuple)
