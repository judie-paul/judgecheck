"""A judge that asks a model backend for a verdict, through the response cache."""

import logging
from typing import Literal

from judgecheck.llm.base import Backend, BackendError
from judgecheck.llm.cache import ResponseCache, cache_key
from judgecheck.llm.prompts import MAX_TOKENS, Request, Strategy, build_request, parse_verdict
from judgecheck.results import Position
from judgecheck.schema import Comparison

logger = logging.getLogger(__name__)


class LLMJudge:
    """Judge backed by a model provider, a prompting strategy and an optional cache."""

    def __init__(
        self,
        name: str,
        backend: Backend,
        strategy: Strategy,
        cache: ResponseCache | None = None,
    ) -> None:
        self.name = name
        self.backend = backend
        self.strategy = strategy
        self.cache = cache

    @property
    def paid(self) -> bool:
        """Whether a call costs money."""
        return self.backend.paid

    @property
    def max_output_tokens(self) -> int:
        """Upper bound on generated tokens per call."""
        return MAX_TOKENS[self.strategy]

    def request(self, comparison: Comparison, shown_first: Literal["a", "b"]) -> Request:
        """The prompt for one judgment."""
        return build_request(comparison, shown_first, self.strategy)

    def key(self, request: Request) -> str:
        """Cache key for a prompt under this backend's settings."""
        return cache_key(
            self.backend.provider,
            self.backend.model,
            self.backend.settings(),
            request.system,
            request.user,
        )

    def is_cached(self, request: Request) -> bool:
        """Whether this judgment is already stored."""
        return self.cache is not None and self.cache.has(self.key(request))

    def judge(self, comparison: Comparison, shown_first: Literal["a", "b"]) -> Position | None:
        request = self.request(comparison, shown_first)
        key = self.key(request)
        text = self.cache.get(key) if self.cache else None
        if text is None:
            try:
                text = self.backend.complete(request.system, request.user)
            except BackendError as error:
                logger.warning("%s: %s on %s: %s", self.name, comparison.id, shown_first, error)
                return None
            if text is None:
                return None
            if self.cache:
                self.cache.put(
                    key,
                    text,
                    {"provider": self.backend.provider, "model": self.backend.model},
                )
        return parse_verdict(text)
