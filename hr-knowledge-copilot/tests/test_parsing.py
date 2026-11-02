from pathlib import Path

import pytest

from hr_copilot.ascii_pdf import build_ascii_pdf
from hr_copilot.models import DocumentParseError
from hr_copilot.parsing import parse_pdf, parse_txt


def test_parse_txt_keeps_filename_and_paragraphs(tmp_path: Path) -> None:
    path = tmp_path / "DEMO_leave.txt"
    path.write_text(
        "【演示数据】\n\n年假每年 5 天。\n\n事假须提前申请。",
        encoding="utf-8",
    )
    parsed = parse_txt(path)
    assert parsed.source_name == "DEMO_leave.txt"
    assert parsed.kind == "txt"
    assert len(parsed.pages) == 1
    assert parsed.pages[0].page_number == 1
    assert parsed.pages[0].paragraph_count == 3
    assert "年假每年 5 天" in parsed.pages[0].text


def test_parse_empty_txt(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_bytes(b"")
    parsed = parse_txt(path)
    assert parsed.pages == ()
    assert any("Empty file" in item for item in parsed.warnings)


def test_parse_whitespace_only_txt(tmp_path: Path) -> None:
    path = tmp_path / "blank.txt"
    path.write_text("   \n\n  ", encoding="utf-8")
    parsed = parse_txt(path)
    assert parsed.pages == ()
    assert any("no visible text" in item for item in parsed.warnings)


def test_parse_pdf_keeps_page_numbers(tmp_path: Path) -> None:
    path = tmp_path / "DEMO_travel.pdf"
    path.write_bytes(
        build_ascii_pdf(
            [
                "DEMO DATA not a real company policy. Travel needs approval.",
                "DEMO DATA page two. Hotel cap is 450 per night.",
            ]
        )
    )
    parsed = parse_pdf(path)
    assert parsed.source_name == "DEMO_travel.pdf"
    assert parsed.kind == "pdf"
    assert [page.page_number for page in parsed.pages] == [1, 2]
    assert "Travel needs approval" in parsed.pages[0].text
    assert "Hotel cap" in parsed.pages[1].text


def test_parse_empty_pdf_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.pdf"
    path.write_bytes(b"")
    parsed = parse_pdf(path)
    assert parsed.pages == ()
    assert any("Empty file" in item for item in parsed.warnings)


def test_parse_corrupt_pdf(tmp_path: Path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"this is not a pdf %PDF")
    with pytest.raises(DocumentParseError) as exc:
        parse_pdf(path)
    assert "broken.pdf" in str(exc.value)
    assert "Unable to parse PDF" in str(exc.value)


def test_parse_pdf_with_no_text(tmp_path: Path) -> None:
    from pypdf import PdfWriter

    path = tmp_path / "blank_pages.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.write(path)
    parsed = parse_pdf(path)
    assert parsed.pages == ()
    assert any("no extractable text" in item.lower() for item in parsed.warnings)
