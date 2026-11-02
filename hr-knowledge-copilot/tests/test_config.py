from pathlib import Path

import pytest

from hr_copilot.config import ConfigError, load_settings


def test_load_settings_from_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=sk-test-12345678\n"
        "OPENAI_BASE_URL=https://example.test/v1\n"
        "OPENAI_MODEL=demo-model\n",
        encoding="utf-8",
    )

    settings = load_settings(env_file)
    assert settings.model == "demo-model"
    assert settings.base_url == "https://example.test/v1"
    assert settings.api_key == "sk-test-12345678"
    assert "sk-t" in settings.masked_api_key()
    assert "12345678" not in settings.masked_api_key() or settings.masked_api_key().endswith("5678")


def test_missing_key_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_MODEL=demo-model\n", encoding="utf-8")

    with pytest.raises(ConfigError) as exc:
        load_settings(env_file)
    assert "OPENAI_API_KEY" in str(exc.value)
