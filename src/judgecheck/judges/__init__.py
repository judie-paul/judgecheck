"""Judge implementations and their construction from configuration."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from judgecheck.judges.base import Judge
from judgecheck.judges.mock import MockJudge
from judgecheck.llm.base import ConfigError
from judgecheck.llm.cache import ResponseCache
from judgecheck.llm.judge import LLMJudge
from judgecheck.llm.prompts import MAX_TOKENS, STRATEGIES, Strategy

__all__ = ["Judge", "LLMJudge", "MockJudge", "build_judge"]

LLM_KINDS = ("anthropic", "openai", "ollama")


def build_judge(
    name: str,
    kind: str,
    params: Mapping[str, Any],
    *,
    seed: int,
    min_length_ratio: float,
    cache_dir: Path | None = None,
) -> Judge:
    """Construct the judge named in a run configuration."""
    if kind == "mock":
        return MockJudge(name, seed=seed, min_length_ratio=min_length_ratio, **params)
    if kind not in LLM_KINDS:
        raise ValueError(f"unknown judge type {kind!r}")
    options = dict(params)
    strategy = options.pop("strategy", "direct")
    if strategy not in STRATEGIES:
        raise ConfigError(f"strategy must be one of {', '.join(STRATEGIES)}")
    max_tokens = MAX_TOKENS[strategy]
    try:
        backend = _backend(kind, options, max_tokens, seed)
    except TypeError as error:
        raise ConfigError(f"invalid parameters for {kind} judge {name!r}: {error}") from error
    cache = ResponseCache(cache_dir / "responses") if cache_dir else None
    return LLMJudge(name, backend, strategy_of(strategy), cache)


def strategy_of(value: str) -> Strategy:
    """Narrow a validated strategy name."""
    return next(strategy for strategy in STRATEGIES if strategy == value)


def _backend(kind: str, options: dict[str, Any], max_tokens: int, seed: int) -> Any:
    if kind == "ollama":
        from judgecheck.llm.ollama import OllamaBackend

        if "model" not in options:
            raise ConfigError("ollama judges need a model, for example qwen2.5:1.5b")
        return OllamaBackend(max_tokens=max_tokens, seed=seed, **options)
    if kind == "openai":
        from judgecheck.llm.openai_backend import OpenAIBackend

        return OpenAIBackend(max_tokens=max_tokens, seed=seed, **options)
    from judgecheck.llm.anthropic_backend import AnthropicBackend

    return AnthropicBackend(max_tokens=max_tokens, **options)
