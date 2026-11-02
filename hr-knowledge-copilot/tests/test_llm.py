from types import SimpleNamespace

import pytest

from hr_copilot.config import Settings
from hr_copilot.llm import LLMError, ping_llm


class FakeCompletions:
    def __init__(self, content: str | None) -> None:
        self._content = content

    def create(self, **kwargs):
        assert kwargs["model"] == "demo-model"
        message = SimpleNamespace(content=self._content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class FakeClient:
    def __init__(self, content: str | None) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(content))


def test_ping_llm_returns_text() -> None:
    settings = Settings(api_key="sk-test", base_url="https://example.test/v1", model="demo-model")
    reply = ping_llm(settings, client=FakeClient("HR Copilot LLM OK"))
    assert reply == "HR Copilot LLM OK"


def test_ping_llm_empty_message() -> None:
    settings = Settings(api_key="sk-test", base_url="https://example.test/v1", model="demo-model")
    with pytest.raises(LLMError):
        ping_llm(settings, client=FakeClient(None))
