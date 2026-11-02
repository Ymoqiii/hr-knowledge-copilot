"""Load runtime settings from environment variables.

Phase 1 only needs LLM connection settings. RAG paths will be added later.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str

    def masked_api_key(self) -> str:
        if len(self.api_key) <= 8:
            return "****"
        return f"{self.api_key[:4]}...{self.api_key[-4:]}"


def load_settings(env_file: Path | None = None) -> Settings:
    """Read `.env` then process environment variables.

    The API key is never printed by this function.
    """
    dotenv_path = env_file or (PROJECT_ROOT / ".env")
    if dotenv_path.exists():
        load_dotenv(dotenv_path, override=False)

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    base_url = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").strip()
    model = (os.getenv("OPENAI_MODEL") or "").strip()

    missing = []
    if not api_key:
        missing.append("OPENAI_API_KEY")
    if not model:
        missing.append("OPENAI_MODEL")
    if missing:
        raise ConfigError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill in real values."
        )

    return Settings(api_key=api_key, base_url=base_url.rstrip("/"), model=model)
