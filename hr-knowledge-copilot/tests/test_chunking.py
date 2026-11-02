from pathlib import Path

from hr_copilot.chunking import chunk_document
from hr_copilot.models import ParsedDocument, ParsedPage


def _doc(text: str, name: str = "demo.txt", kind: str = "txt", page: int = 1) -> ParsedDocument:
    return ParsedDocument(
        source_path=Path(name),
        source_name=name,
        kind=kind,
        pages=(ParsedPage(page_number=page, text=text, paragraph_count=1),),
    )


def test_short_text_is_one_chunk() -> None:
    chunks = chunk_document(_doc("年假每年 5 天。"), chunk_size=400, overlap=80)
    assert len(chunks) == 1
    assert chunks[0].source_name == "demo.txt"
    assert chunks[0].page_number == 1
    assert chunks[0].paragraph_index == 1
    assert "年假" in chunks[0].text


def test_empty_page_yields_no_chunks() -> None:
    chunks = chunk_document(_doc("   \n"), chunk_size=40, overlap=8)
    assert chunks == []


def test_overlap_and_length() -> None:
    text = "A" * 250
    chunks = chunk_document(_doc(text), chunk_size=100, overlap=20)
    assert len(chunks) >= 3
    for chunk in chunks:
        assert len(chunk.text) <= 100
    first, second = chunks[0], chunks[1]
    overlap = first.text[-20:]
    assert second.text.startswith(overlap) or overlap[-10:] in second.text


def test_chunks_do_not_merge_pages() -> None:
    document = ParsedDocument(
        source_path=Path("two.pdf"),
        source_name="two.pdf",
        kind="pdf",
        pages=(
            ParsedPage(page_number=1, text="PAGEONE " + ("甲" * 40), paragraph_count=1),
            ParsedPage(page_number=2, text="PAGETWO " + ("乙" * 40), paragraph_count=1),
        ),
    )
    chunks = chunk_document(document, chunk_size=30, overlap=5)
    pages = {chunk.page_number for chunk in chunks}
    assert pages == {1, 2}
    assert all("PAGEONE" not in chunk.text or chunk.page_number == 1 for chunk in chunks)
    assert all("PAGETWO" not in chunk.text or chunk.page_number == 2 for chunk in chunks)


def test_paragraph_index_follows_blank_lines() -> None:
    text = "第一段年假规则。\n\n第二段事假规则。"
    chunks = chunk_document(_doc(text), chunk_size=400, overlap=80)
    assert chunks[0].paragraph_index == 1
    long = ("甲" * 30 + "。\n\n") + ("乙" * 80)
    split_chunks = chunk_document(_doc(long), chunk_size=50, overlap=10)
    assert split_chunks[-1].paragraph_index >= 2
