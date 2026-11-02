"""Run the small labeled RAG evaluation set against the configured model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hr_copilot.config import ConfigError, load_settings
from hr_copilot.ingest import ingest_directory
from hr_copilot.llm import LLMError
from hr_copilot.models import QueryError
from hr_copilot.qa import NO_ANSWER, answer_question

DATASET = ROOT / "data" / "eval" / "rag_eval.json"


def evaluate(dataset: list[dict], chunks, settings) -> list[dict]:
    results = []
    for case in dataset:
        try:
            answer = answer_question(chunks, case["question"], settings)
        except LLMError as exc:
            results.append({**case, "status": "error", "error": str(exc)})
            continue
        refused = NO_ANSWER in answer
        answer_ok = refused if not case["answerable"] else (
            not refused and all(term in answer for term in case["expected_terms"])
        )
        expected_source = case["expected_source"]
        source_ok = (expected_source is None) if expected_source is None else expected_source in answer
        results.append({
            **case,
            "answer": answer,
            "answer_ok": answer_ok,
            "source_ok": source_ok,
            "refusal_ok": refused == (not case["answerable"]),
            "status": "evaluated",
        })
    return results


def summarize(results: list[dict]) -> dict:
    evaluated = [item for item in results if item["status"] == "evaluated"]
    errors = [item for item in results if item["status"] == "error"]
    count = len(evaluated)
    return {
        "evaluated": count,
        "errors": len(errors),
        "answer_hits": sum(item["answer_ok"] for item in evaluated),
        "source_correct": sum(item["source_ok"] for item in evaluated),
        "refusal_correct": sum(item["refusal_ok"] for item in evaluated),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAG answers on the labeled demo set.")
    parser.add_argument("--output", type=Path, help="Optional JSON file for full case details")
    args = parser.parse_args(argv)
    try:
        dataset = json.loads(DATASET.read_text(encoding="utf-8"))
        chunks, warnings = ingest_directory(ROOT / "data" / "demo")
        settings = load_settings()
        for warning in warnings:
            print(f"[warn] {warning}")
        results = evaluate(dataset, chunks, settings)
    except (ConfigError, QueryError, FileNotFoundError, NotADirectoryError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    for item in results:
        if item["status"] == "error":
            print(f"[ERROR] {item['id']} | {item['question']}\n  reason={item['error']}")
            continue
        status = "PASS" if item["answer_ok"] and item["source_ok"] and item["refusal_ok"] else "FAIL"
        print(f"[{status}] {item['id']} | {item['question']}")
        print(f"  answer={item['answer_ok']} source={item['source_ok']} refusal={item['refusal_ok']}")
        if status == "FAIL":
            print(f"  expected_terms={item['expected_terms']} expected_source={item['expected_source']}")
            print(f"  actual={item['answer']}")
    summary = summarize(results)
    print(f"\nSummary: evaluated={summary['evaluated']} errors={summary['errors']}")
    if not summary["evaluated"]:
        print("No valid samples; quality metrics are unavailable.")
    else:
        total = summary["evaluated"]
        print(f"Answer hits: {summary['answer_hits']}/{total}; "
              f"source correct: {summary['source_correct']}/{total}; "
              f"refusal correct: {summary['refusal_correct']}/{total}.")
    if args.output:
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Details written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
