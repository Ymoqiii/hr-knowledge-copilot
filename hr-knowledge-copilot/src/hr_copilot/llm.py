"""Thin wrapper around an OpenAI-compatible Chat Completions API."""

from __future__ import annotations

from openai import APIError, AuthenticationError, OpenAI, PermissionDeniedError, RateLimitError

from hr_copilot.config import Settings


class LLMError(RuntimeError):
    """User-facing LLM call failure."""


def build_client(settings: Settings) -> OpenAI:
    return OpenAI(api_key=settings.api_key, base_url=settings.base_url, timeout=30.0)


def ping_llm(settings: Settings, client: OpenAI | None = None) -> str:
    """Send a tiny prompt to verify credentials, endpoint, and model name.

    Returns the model reply text. Raises LLMError with a readable reason.
    """
    client = client or build_client(settings)
    try:
        response = client.chat.completions.create(
            model=settings.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a connectivity probe. Reply with one short sentence.",
                },
                {
                    "role": "user",
                    "content": "Reply exactly: HR Copilot LLM OK",
                },
            ],
            max_tokens=32,
            temperature=0,
        )
    except AuthenticationError as exc:
        raise LLMError(
            "Authentication failed (401). Check OPENAI_API_KEY and OPENAI_BASE_URL."
        ) from exc
    except PermissionDeniedError as exc:
        raise LLMError(
            f"Permission denied (403). This key may not be allowed to call model '{settings.model}'."
        ) from exc
    except RateLimitError as exc:
        raise LLMError("Rate limited or quota exceeded. Check billing/quota on the provider.") from exc
    except APIError as exc:
        raise LLMError(f"Provider API error: {exc}") from exc

    choice = response.choices[0].message.content if response.choices else None
    if not choice:
        raise LLMError("The API returned an empty message. Check the model name and provider logs.")
    return choice.strip()
