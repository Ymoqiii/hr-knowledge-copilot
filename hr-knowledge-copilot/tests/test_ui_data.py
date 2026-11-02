import json
from pathlib import Path

from hr_copilot.ingest import ingest_directory
from hr_copilot.retrieve import search_chunks


ROOT = Path(__file__).resolve().parents[1]


def test_demo_knowledge_base_loads_and_retrieves_sources():
    chunks, warnings = ingest_directory(ROOT / "data" / "demo")
    assert chunks
    assert not warnings
    result = search_chunks(chunks, "年假有几天？")
    assert result.hits
    assert any("请假管理规定" in hit.chunk.source_name for hit in result.hits)


def test_evaluation_snapshot_has_ui_metrics():
    results = json.loads((ROOT / "rag_eval_results.json").read_text(encoding="utf-8"))
    evaluated = [case for case in results if case.get("status") == "evaluated"]
    assert evaluated
    assert all({"answer_ok", "source_ok", "refusal_ok"} <= case.keys() for case in evaluated)
