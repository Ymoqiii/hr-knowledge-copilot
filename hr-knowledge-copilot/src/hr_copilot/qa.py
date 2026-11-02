"""Retrieval-grounded policy question answering."""

from __future__ import annotations

from collections.abc import Sequence

from openai import APIError, AuthenticationError, OpenAI, PermissionDeniedError, RateLimitError

from hr_copilot.config import Settings
from hr_copilot.llm import LLMError, build_client
from hr_copilot.models import Chunk, SearchResult
from hr_copilot.retrieve import search_chunks

NO_ANSWER = "知识库未找到足够的制度依据，无法回答该问题。"


def answer_question(
    chunks: Sequence[Chunk],
    question: str,
    settings: Settings,
    top_k: int = 3,
    client: OpenAI | None = None,
) -> str:
    """Answer using retrieved chunks only, with source/page citations."""
    result = search_chunks(chunks, question, top_k=top_k)
    if not result.hits:
        return NO_ANSWER

    evidence = "\n\n".join(
        f"[来源：{hit.chunk.source_name}，页码：{hit.chunk.page_number}]\n{hit.chunk.text}"
        for hit in result.hits
    )
    client = client or build_client(settings)
    try:
        response = client.chat.completions.create(
            model=settings.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是企业制度问答助手。只能依据用户提供的制度片段回答，"
                        "不得使用外部知识或补充片段未明确支持的政策细节。"
                        "证据不足时原样回答：" + NO_ANSWER +
                        "回答末尾必须列出实际使用片段的来源文件名和页码。"
                    ),
                },
                {"role": "user", "content": f"问题：{question}\n\n制度片段：\n{evidence}"},
            ],
            temperature=0,
        )
    except AuthenticationError as exc:
        raise LLMError("Authentication failed (401). Check OPENAI_API_KEY and OPENAI_BASE_URL.") from exc
    except PermissionDeniedError as exc:
        raise LLMError(f"Permission denied (403). This key may not be allowed to call model '{settings.model}'.") from exc
    except RateLimitError as exc:
        raise LLMError("Rate limited or quota exceeded. Check billing/quota on the provider.") from exc
    except APIError as exc:
        raise LLMError(f"Provider API error: {exc}") from exc
    content = response.choices[0].message.content if response.choices else None
    if not content or not content.strip():
        raise LLMError("The API returned an empty message. Check the model name and provider logs.")
    answer = content.strip()
    # Top-k may contain several chunks from the same document page. Cite each
    # unique document/page pair once while retaining distinct page references.
    source_pages = dict.fromkeys(
        (hit.chunk.source_name, hit.chunk.page_number) for hit in result.hits
    )
    citations = "\n\n来源：" + "；".join(
        f"{source_name}，第 {page_number} 页"
        for source_name, page_number in source_pages
    )
    return answer + citations
