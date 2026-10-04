"""Local models served by Ollama."""

import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from judgecheck.llm.base import BackendError

logger = logging.getLogger(__name__)

Post = Callable[[str, dict[str, Any], float], dict[str, Any]]


def http_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    """POST JSON and decode the JSON response."""
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        result = json.loads(response.read())
    if not isinstance(result, dict):
        raise ValueError("unexpected response from Ollama")
    return result


class OllamaBackend:
    """Greedy, seeded generation from a local model."""

    provider = "ollama"
    paid = False

    def __init__(
        self,
        model: str,
        *,
        host: str = "http://127.0.0.1:11434",
        max_tokens: int,
        seed: int = 0,
        num_ctx: int = 8192,
        timeout: float = 600.0,
        retries: int = 2,
        post: Post = http_post,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.max_tokens = max_tokens
        self.seed = seed
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.retries = retries
        self._post = post
        self._sleep = sleep

    def settings(self) -> dict[str, Any]:
        return {
            "temperature": 0,
            "seed": self.seed,
            "num_ctx": self.num_ctx,
            "max_tokens": self.max_tokens,
        }

    def complete(self, system: str, user: str) -> str | None:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "temperature": 0,
                "seed": self.seed,
                "num_ctx": self.num_ctx,
                "num_predict": self.max_tokens,
            },
        }
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                result = self._post(f"{self.host}/api/chat", payload, self.timeout)
                content = result["message"]["content"]
                if not isinstance(content, str):
                    raise ValueError("response has no text content")
                return content
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                ValueError,
                KeyError,
            ) as e:
                last = e
                logger.warning("ollama attempt %d failed: %s", attempt + 1, e)
                if attempt < self.retries:
                    self._sleep(2.0 * (attempt + 1))
        raise BackendError(f"ollama request failed after {self.retries + 1} attempts: {last}")
