from hr_copilot.qa import NO_ANSWER
from hr_copilot.llm import LLMError
from scripts.evaluate_rag import evaluate, summarize


class FakeSettings:
    pass


def test_evaluate_tracks_answer_source_and_refusal(monkeypatch) -> None:
    answers = iter(["年假 5 个工作日\n来源：leave.txt，第 1 页", NO_ANSWER])
    monkeypatch.setattr("scripts.evaluate_rag.answer_question", lambda *args: next(answers))
    dataset = [
        {"id": "ok", "question": "年假", "expected_terms": ["5 个工作日"],
         "expected_source": "leave.txt", "answerable": True},
        {"id": "none", "question": "未知", "expected_terms": [],
         "expected_source": None, "answerable": False},
    ]
    results = evaluate(dataset, [], FakeSettings())
    assert all(item["answer_ok"] and item["source_ok"] and item["refusal_ok"] for item in results)


def test_eval_dataset_covers_at_least_ten_cases() -> None:
    import json
    from scripts.evaluate_rag import DATASET

    cases = json.loads(DATASET.read_text(encoding="utf-8"))
    assert len(cases) >= 10
    assert any(not case["answerable"] for case in cases)
    assert any(case["id"].startswith("confuse-") for case in cases)


def test_pdf_and_unsupported_detail_cases_expect_refusal() -> None:
    import json
    from scripts.evaluate_rag import DATASET

    cases = {case["id"]: case for case in json.loads(DATASET.read_text(encoding="utf-8"))}
    for case_id in ("attendance-late", "confuse-hotel-vs-meal"):
        assert cases[case_id]["answerable"] is False
        assert cases[case_id]["expected_source"] is None


def test_api_error_is_recorded_and_later_cases_continue(monkeypatch) -> None:
    calls = iter([LLMError("Connection error"), "年假 5 个工作日 来源：leave.txt"])

    def fake_answer(*args):
        value = next(calls)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr("scripts.evaluate_rag.answer_question", fake_answer)
    dataset = [
        {"id": "network", "question": "q1", "expected_terms": [], "expected_source": None, "answerable": False},
        {"id": "ok", "question": "q2", "expected_terms": ["5 个工作日"], "expected_source": "leave.txt", "answerable": True},
    ]
    results = evaluate(dataset, [], FakeSettings())
    assert results[0]["status"] == "error"
    assert results[0]["error"] == "Connection error"
    assert results[1]["status"] == "evaluated"
    assert summarize(results) == {
        "evaluated": 1, "errors": 1, "answer_hits": 1,
        "source_correct": 1, "refusal_correct": 1,
    }


def test_summary_handles_no_valid_samples() -> None:
    summary = summarize([{"status": "error", "error": "network"}])
    assert summary["evaluated"] == 0
    assert summary["errors"] == 1
