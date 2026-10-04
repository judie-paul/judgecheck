"""Claude through the Anthropic API."""

import os
from typing import Any

from judgecheck.llm.base import BackendError, ConfigError

DEFAULT_MODEL = "claude-sonnet-5-5"


class AnthropicBackend:
    """One Messages API call per judgment.

    Claude Sonnet 5.5 rejects ``temperature``, so none is sent. ``between_tools`` is its
    lowest thinking setting, which keeps the prompting strategy (not hidden reasoning) the
    only thing that differs between runs. A refusal is returned as ``None`` and counted as
    unusable rather than silently re-run on another model, which would change what is measured.
    """

    provider = "anthropic"
    paid = True

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        max_tokens: int,
        timeout: float = 60.0,
        max_retries: int = 2,
        client: Any = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        if client is None:
            try:
                from anthropic import Anthropic
            except ImportError as error:
                raise ConfigError("Install the llm extra: pip install 'judgecheck[llm]'") from error
            key = os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise ConfigError("ANTHROPIC_API_KEY is not set; refusing to fall back to a mock")
            client = Anthropic(api_key=key, timeout=timeout, max_retries=max_retries)
        self.client = client

    def settings(self) -> dict[str, Any]:
        return {"thinking": "between_tools", "max_tokens": self.max_tokens}

    def complete(self, system: str, user: str) -> str | None:
        import anthropic

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                thinking={"type": "between_tools"},
                messages=[{"role": "user", "content": user}],
            )
        except (
            anthropic.APIConnectionError,
            anthropic.RateLimitError,
            anthropic.InternalServerError,
        ) as error:
            raise BackendError(f"anthropic request failed: {error}") from error
        if response.stop_reason == "refusal":
            return None
        return "".join(block.text for block in response.content if block.type == "text")
