"""Build a tiny ASCII-only PDF for tests and demo fixtures.

This is not a general PDF writer. Chinese cannot be embedded with the
built-in Helvetica font, so demo PDFs that must be searched in Chinese
stay as .txt files. Parsing of real PDFs is handled by pypdf in parsing.py.
"""

from __future__ import annotations

from collections.abc import Sequence


def _escape_pdf_string(text: str) -> str:
    ascii_text = text.encode("ascii", errors="replace").decode("ascii")
    return ascii_text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_ascii_pdf(pages: Sequence[str]) -> bytes:
    """Return PDF bytes. Each item in pages becomes one PDF page."""
    if not pages:
        raise ValueError("build_ascii_pdf requires at least one page.")

    objects: list[str] = []
    page_ids: list[int] = []

    objects.append("<< /Type /Catalog /Pages 2 0 R >>")
    # object 2 (Pages) is filled after we know page ids
    objects.append("")  # placeholder

    font_id = None
    content_ids: list[int] = []

    for page_text in pages:
        content = _page_content_stream(page_text)
        objects.append(f"<< /Length {len(content.encode('ascii'))} >>\nstream\n{content}\nendstream")
        content_ids.append(len(objects))

    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    font_id = len(objects)

    for content_id in content_ids:
        objects.append(
            "<< /Type /Page /Parent 2 0 R "
            "/MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        )
        page_ids.append(len(objects))

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>"

    body_parts: list[bytes] = [b"%PDF-1.4\n"]
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(sum(len(part) for part in body_parts))
        body_parts.append(f"{index} 0 obj\n{obj}\nendobj\n".encode("ascii"))

    xref_pos = sum(len(part) for part in body_parts)
    xref = [f"xref\n0 {len(objects) + 1}\n", "0000000000 65535 f \n"]
    for offset in offsets[1:]:
        xref.append(f"{offset:010d} 00000 n \n")
    trailer = (
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    )
    body_parts.append("".join(xref).encode("ascii"))
    body_parts.append(trailer.encode("ascii"))
    return b"".join(body_parts)


def _page_content_stream(text: str) -> str:
    lines = text.splitlines() or [""]
    commands = ["BT", "/F1 12 Tf", "50 740 Td"]
    first = True
    for line in lines[:48]:
        escaped = _escape_pdf_string(line[:110])
        if first:
            commands.append(f"({escaped}) Tj")
            first = False
        else:
            commands.append("0 -16 Td")
            commands.append(f"({escaped}) Tj")
    commands.append("ET")
    return "\n".join(commands)
