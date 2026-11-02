"""Turn TXT and PDF files into page-level text plus source metadata."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from hr_copilot.models import DocumentParseError, ParsedDocument, ParsedPage


def parse_file(path: Path) -> ParsedDocument:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return parse_txt(path)
    if suffix == ".pdf":
        return parse_pdf(path)
    raise DocumentParseError(f"Unsupported file type '{path.suffix}' for {path.name}.")


def parse_txt(path: Path) -> ParsedDocument:
    raw = path.read_bytes()
    if not raw:
        return ParsedDocument(
            source_path=path,
            source_name=path.name,
            kind="txt",
            pages=(),
            warnings=(f"Empty file: {path.name}",),
        )

    text = _normalize_newlines(_decode_bytes(raw, path.name))
    paragraphs = _paragraphs(text)
    page = ParsedPage(page_number=1, text=text, paragraph_count=len(paragraphs))
    warnings: tuple[str, ...] = ()
    if not text.strip():
        warnings = (f"File has no visible text: {path.name}",)
        return ParsedDocument(
            source_path=path,
            source_name=path.name,
            kind="txt",
            pages=(),
            warnings=warnings,
        )
    return ParsedDocument(
        source_path=path,
        source_name=path.name,
        kind="txt",
        pages=(page,),
        warnings=warnings,
    )


def parse_pdf(path: Path) -> ParsedDocument:
    if path.stat().st_size == 0:
        return ParsedDocument(
            source_path=path,
            source_name=path.name,
            kind="pdf",
            pages=(),
            warnings=(f"Empty file: {path.name}",),
        )

    try:
        reader = PdfReader(str(path))
    except (PdfReadError, OSError, ValueError) as exc:
        raise DocumentParseError(
            f"Unable to parse PDF '{path.name}'. The file may be damaged, encrypted, or not a PDF."
        ) from exc

    if getattr(reader, "is_encrypted", False):
        try:
            unlocked = reader.decrypt("")
        except Exception as exc:  # noqa: BLE001 - pypdf decrypt types vary by version
            raise DocumentParseError(
                f"Unable to parse PDF '{path.name}'. The file is encrypted."
            ) from exc
        if unlocked == 0:
            raise DocumentParseError(
                f"Unable to parse PDF '{path.name}'. The file is encrypted."
            )

    pages: list[ParsedPage] = []
    warnings: list[str] = []
    try:
        total = len(reader.pages)
    except Exception as exc:  # noqa: BLE001
        raise DocumentParseError(
            f"Unable to parse PDF '{path.name}'. The file may be damaged, encrypted, or not a PDF."
        ) from exc

    if total == 0:
        warnings.append(f"PDF has no pages: {path.name}")

    for index, page in enumerate(reader.pages, start=1):
        try:
            extracted = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            raise DocumentParseError(
                f"Unable to extract text from PDF '{path.name}' page {index}."
            ) from exc
        extracted = _normalize_newlines(extracted)
        if not extracted.strip():
            warnings.append(f"No extractable text on page {index} of {path.name}")
            continue
        paragraphs = _paragraphs(extracted)
        pages.append(
            ParsedPage(page_number=index, text=extracted, paragraph_count=len(paragraphs))
        )

    if total > 0 and not pages:
        warnings.append(f"PDF has no extractable text: {path.name}")

    return ParsedDocument(
        source_path=path,
        source_name=path.name,
        kind="pdf",
        pages=tuple(pages),
        warnings=tuple(warnings),
    )


def _decode_bytes(raw: bytes, name: str) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DocumentParseError(f"Unable to decode text file '{name}' as UTF-8 or GB18030.")


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _paragraphs(text: str) -> list[str]:
    parts = [part.strip() for part in text.split("\n\n")]
    return [part for part in parts if part]
