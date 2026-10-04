"""Backend protocol shared by every model provider."""

from typing import Any, Protocol


class BackendError(RuntimeError):
    """A transient failure (network, timeout, rate limit) that outlived the retries.

    The judge records the response as unusable and moves on. Configuration problems
    such as a rejected API key are not wrapped, so they stop the run instead.
    """


class ConfigError(ValueError):
    """A judge cannot run as configured, for example a missing API key."""


class Backend(Protocol):
    """Sends one system + user prompt to a model and returns its text."""

    provider: str
    model: str
    paid: bool
    max_tokens: int

    def settings(self) -> dict[str, Any]:
        """Every setting that can change the output; part of the cache key."""
        ...

    def complete(self, system: str, user: str) -> str | None:
        """Return the model's text, or ``None`` if it declined to answer.

        Raises :class:`BackendError` after bounded retries on transient failures.
        """
        ...
