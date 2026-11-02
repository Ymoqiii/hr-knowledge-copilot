from pathlib import Path

import pytest

from hr_copilot.models import Chunk, QueryError
from hr_copilot.retrieve import search_chunks, tokenize


def _chunk(chunk_id: str, name: str, text: str, page: int = 1) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        source_path=str(Path("data") / name),
        source_name=name,
        source_kind="txt" if name.endswith(".txt") else "pdf",
        page_number=page,
        paragraph_index=1,
        text=text,
        char_start=0,
        char_end=len(text),
    )


LEAVE = _chunk(
    "leave-1",
    "DEMO_请假管理规定.txt",
    "【演示数据】年假每年 5 个工作日，须提前 3 个工作日申请。事假期间不发放当日基本工资。",
)
EXPENSE = _chunk(
    "exp-1",
    "DEMO_报销管理规定.txt",
    "【演示数据】报销必须附合规发票。无发票的财务有权拒绝。逾期未提交视为放弃报销。",
)
TRAVEL = _chunk(
    "travel-1",
    "DEMO_出差管理规定.txt",
    "【演示数据】出差须提前 2 个工作日提交出差申请。住宿标准为每晚不超过 450 元。",
)
TRAVEL_P2 = _chunk(
    "travel-2",
    "DEMO_出差管理规定.txt",
    "【演示数据】出差当天给予餐食补助 80 元/天。",
    page=2,
)


def test_leave_query_ranks_leave_chunk_first() -> None:
    result = search_chunks([LEAVE, EXPENSE, TRAVEL], "年假如何申请", top_k=3)
    assert result.hits
    assert result.hits[0].chunk.source_name == "DEMO_请假管理规定.txt"
    assert "年假" in result.hits[0].chunk.text
    assert result.hits[0].score > 0
    assert any("年假" in term for term in result.hits[0].matched_terms)


def test_expense_query_ranks_expense_chunk_first() -> None:
    result = search_chunks([LEAVE, EXPENSE, TRAVEL], "报销需要发票吗", top_k=3)
    assert result.hits[0].chunk.source_name == "DEMO_报销管理规定.txt"
    assert result.hits[0].score >= result.hits[-1].score


def test_top_k_limit_and_sources() -> None:
    extra = _chunk("leave-2", "DEMO_请假管理规定.txt", "【演示数据】病假须补交医院开具的病假证明。")
    result = search_chunks([LEAVE, extra, EXPENSE, TRAVEL, TRAVEL_P2], "演示数据 出差 年假 报销", top_k=3)
    assert len(result.hits) == 3
    names = {hit.chunk.source_name for hit in result.hits}
    assert names <= {
        "DEMO_请假管理规定.txt",
        "DEMO_报销管理规定.txt",
        "DEMO_出差管理规定.txt",
    }
    for hit in result.hits:
        assert hit.chunk.source_name
        assert hit.chunk.page_number >= 1
        assert hit.explanation


def test_empty_query_raises() -> None:
    with pytest.raises(QueryError) as exc:
        search_chunks([LEAVE], "   ")
    assert "Empty question" in str(exc.value)


def test_no_match_message() -> None:
    result = search_chunks([LEAVE, EXPENSE], "股权激励归属日历")
    assert result.hits == ()
    assert "No matching" in result.message


def test_no_chunks_message() -> None:
    result = search_chunks([], "年假")
    assert result.hits == ()
    assert "No indexed chunks" in result.message


def test_metadata_not_stripped() -> None:
    result = search_chunks([TRAVEL, TRAVEL_P2], "餐食补助", top_k=1)
    hit = result.hits[0]
    assert hit.chunk.page_number == 2
    assert hit.chunk.source_name == "DEMO_出差管理规定.txt"
    assert "80" in hit.chunk.text


def test_tokenize_chinese_bigrams() -> None:
    tokens = tokenize("年假申请")
    assert "年假" in tokens
    assert "申请" in tokens
