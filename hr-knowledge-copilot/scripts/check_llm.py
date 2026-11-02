"""Live smoke test: load .env and call the configured model API."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hr_copilot.config import ConfigError, load_settings
from hr_copilot.llm import LLMError, ping_llm


def main() -> int:
    print("HR Knowledge Copilot — phase 1 LLM check")
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"[FAIL] config: {exc}")
        return 1

    print(f"base_url = {settings.base_url}")
    print(f"model    = {settings.model}")
    print(f"api_key  = {settings.masked_api_key()}")

    try:
        reply = ping_llm(settings)
    except LLMError as exc:
        print(f"[FAIL] llm: {exc}")
        return 2

    print("[OK] LLM call succeeded")
    print(f"reply: {reply}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
