"""Offline entry: ingest demo files and print top-3 keyword hits. No LLM call."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hr_copilot.ingest import ingest_directory
from hr_copilot.models import QueryError
from hr_copilot.retrieve import search_chunks

DEMO_DIR = ROOT / "data" / "demo"


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Offline keyword search over demo HR policies.")
    parser.add_argument("query", nargs="?", default="", help="Natural-language question")
    parser.add_argument(
        "--data-dir",
        default=str(DEMO_DIR),
        help="Folder of .txt/.pdf files (default: data/demo)",
    )
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args(argv)

    print("HR Knowledge Copilot — phase 2 offline search")
    print("Demo data only. Not real company policy.")
    print(f"data_dir = {args.data_dir}")

    try:
        chunks, warnings = ingest_directory(Path(args.data_dir))
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"[FAIL] ingest: {exc}")
        return 1

    for warning in warnings:
        print(f"[warn] {warning}")
    print(f"indexed_chunks = {len(chunks)}")

    try:
        result = search_chunks(chunks, args.query, top_k=args.top_k)
    except QueryError as exc:
        print(f"[FAIL] query: {exc}")
        return 2

    print(f"query = {result.query}")
    print(result.message)
    if not result.hits:
        return 0

    for rank, hit in enumerate(result.hits, start=1):
        chunk = hit.chunk
        para = chunk.paragraph_index if chunk.paragraph_index is not None else "-"
        print("-" * 60)
        print(f"#{rank} score={hit.score:.4f}")
        print(f"source = {chunk.source_name} ({chunk.source_kind})")
        print(f"page = {chunk.page_number}  paragraph = {para}")
        print(f"chunk_id = {chunk.chunk_id}")
        print(f"matched_terms = {', '.join(hit.matched_terms) if hit.matched_terms else '(phrase only)'}")
        print(f"why = {hit.explanation}")
        print("text:")
        print(chunk.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
