"""OpenAI chat models."""

import os
from typing import Any

from judgecheck.llm.base import BackendError, ConfigError

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIBackend:
    """One chat completion per judgment, at temperature 0 with a fixed seed."""

    provider = "openai"
    paid = True

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        max_tokens: int,
        seed: int = 0,
        timeout: float = 60.0,
        max_retries: int = 2,
        client: Any = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.seed = seed
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise ConfigError("Install the llm extra: pip install 'judgecheck[llm]'") from error
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise ConfigError("OPENAI_API_KEY is not set; refusing to fall back to a mock")
            client = OpenAI(api_key=key, timeout=timeout, max_retries=max_retries)
        self.client = client

    def settings(self) -> dict[str, Any]:
        return {"temperature": 0, "seed": self.seed, "max_tokens": self.max_tokens}

    def complete(self, system: str, user: str) -> str | None:
        import openai

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                seed=self.seed,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except (
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.InternalServerError,
        ) as error:
            raise BackendError(f"openai request failed: {error}") from error
        choice = response.choices[0]
        if choice.finish_reason == "content_filter":
            return None
        return choice.message.content or ""
