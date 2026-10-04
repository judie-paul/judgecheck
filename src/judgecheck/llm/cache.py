"""On-disk response cache so reruns are free and interrupted runs resume."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def cache_key(provider: str, model: str, settings: dict[str, Any], system: str, user: str) -> str:
    """Hash everything that determines a response."""
    payload = json.dumps(
        {
            "provider": provider,
            "model": model,
            "settings": settings,
            "system": system,
            "user": user,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    """One JSON file per response, written atomically."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def has(self, key: str) -> bool:
        """Whether a response is stored, without reading it."""
        return self._path(key).exists()

    def get(self, key: str) -> str | None:
        """Return the stored response text, or ``None`` if absent or unreadable."""
        try:
            data = json.loads(self._path(key).read_text(encoding="utf-8"))
            text = data["text"]
        except (OSError, ValueError, KeyError):
            return None
        return text if isinstance(text, str) else None

    def put(self, key: str, text: str, meta: dict[str, Any]) -> None:
        """Store a response with the provider and model that produced it."""
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"text": text, "meta": meta}), encoding="utf-8")
        os.replace(temporary, path)
