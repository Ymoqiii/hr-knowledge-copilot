"""Ask a policy question using the local demo corpus and configured LLM."""

from __future__ import annotations

import argparse
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
from hr_copilot.qa import answer_question

DEMO_DIR = ROOT / "data" / "demo"


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Ask a question about the demo HR policies.")
    parser.add_argument("question", nargs="+", help="Question about a policy")
    args = parser.parse_args(argv)
    question = " ".join(args.question)

    try:
        settings = load_settings()
        chunks, warnings = ingest_directory(DEMO_DIR)
        for warning in warnings:
            print(f"[warn] {warning}")
        print(answer_question(chunks, question, settings))
    except (ConfigError, LLMError, QueryError, FileNotFoundError, NotADirectoryError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
