from pathlib import Path
from shutil import copy2

from hr_copilot.ascii_pdf import build_ascii_pdf
from hr_copilot.ingest import ingest_directory
from hr_copilot.retrieve import search_chunks

DEMO_SRC = Path(__file__).resolve().parents[1] / "data" / "demo"


def test_ingest_skips_corrupt_pdf_and_keeps_txt(tmp_path: Path) -> None:
    (tmp_path / "DEMO_ok.txt").write_text(
        "【演示数据】年假每年 5 天。",
        encoding="utf-8",
    )
    (tmp_path / "bad.pdf").write_bytes(b"not-a-pdf")
    (tmp_path / "notes.md").write_text("ignore me", encoding="utf-8")
    chunks, warnings = ingest_directory(tmp_path)
    assert any(chunk.source_name == "DEMO_ok.txt" for chunk in chunks)
    assert any("bad.pdf" in item for item in warnings)
    assert all(chunk.source_name != "notes.md" for chunk in chunks)


def test_ingest_empty_folder(tmp_path: Path) -> None:
    chunks, warnings = ingest_directory(tmp_path)
    assert chunks == []
    assert any("No .txt or .pdf" in item for item in warnings)


def test_demo_corpus_leave_query_offline(tmp_path: Path) -> None:
    for path in DEMO_SRC.glob("DEMO_*.txt"):
        copy2(path, tmp_path / path.name)
    (tmp_path / "DEMO_attendance_policy.pdf").write_bytes(
        build_ascii_pdf(
            [
                "DEMO DATA. Fictitious company. Not a real employer policy.",
                "Clock-in is required before 09:30. Late arrival over 30 minutes is recorded.",
            ]
        )
    )
    chunks, warnings = ingest_directory(tmp_path)
    assert chunks
    names = {chunk.source_name for chunk in chunks}
    assert "DEMO_请假管理规定.txt" in names
    assert "DEMO_报销管理规定.txt" in names
    assert "DEMO_出差管理规定.txt" in names
    assert "DEMO_attendance_policy.pdf" in names
    pdf_hits = [chunk for chunk in chunks if chunk.source_name.endswith(".pdf")]
    assert {chunk.page_number for chunk in pdf_hits} == {1, 2}

    result = search_chunks(chunks, "年假需要提前几天申请")
    assert result.hits
    assert result.hits[0].chunk.source_name == "DEMO_请假管理规定.txt"
    assert result.hits[0].chunk.page_number == 1
    assert "年假" in result.hits[0].chunk.text

    clock = search_chunks(chunks, "clock-in late arrival")
    assert clock.hits
    assert clock.hits[0].chunk.source_name == "DEMO_attendance_policy.pdf"
    assert clock.hits[0].chunk.page_number in {1, 2}
    assert all("Empty file" not in item for item in warnings)


def test_shipped_demo_txt_files_are_marked_as_demo() -> None:
    files = list(DEMO_SRC.glob("DEMO_*.txt"))
    assert len(files) >= 3
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "演示数据" in text
        assert "不代表任何真实企业制度" in text
