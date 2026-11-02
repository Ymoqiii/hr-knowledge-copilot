from pathlib import Path
from types import SimpleNamespace

from hr_copilot.config import Settings
from hr_copilot.models import Chunk
from hr_copilot.qa import NO_ANSWER, answer_question


class FakeCompletions:
    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="年假每年 5 个工作日。"))])


class FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeCompletions())


def _chunk() -> Chunk:
    return Chunk("c1", str(Path("policy.txt")), "请假制度.txt", "txt", 2, 1,
                 "年假每年 5 个工作日。", 0, 15)


def test_answer_uses_retrieved_evidence_and_cites_source() -> None:
    client = FakeClient()
    answer = answer_question([_chunk()], "年假有几天", Settings("key", "url", "model"), client=client)
    assert "5 个工作日" in answer
    assert "请假制度.txt" in answer and "第 2 页" in answer
    assert "年假每年 5 个工作日" in client.chat.completions.kwargs["messages"][1]["content"]


def test_no_match_refuses_without_calling_model() -> None:
    answer = answer_question([_chunk()], "股权激励归属日历", Settings("key", "url", "model"))
    assert answer == NO_ANSWER
