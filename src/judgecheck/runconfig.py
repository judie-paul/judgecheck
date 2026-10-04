"""YAML run configuration."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class JudgeSpec(BaseModel):
    """One judge to run."""

    model_config = ConfigDict(extra="forbid")

    name: str
    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class RunConfig(BaseModel):
    """What to run and how to score it."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["sample", "jsonl", "hf"] = "sample"
    path: Path | None = None
    seed: int = 42
    limit: int | None = Field(default=None, ge=1)
    bootstrap: int = Field(default=1000, ge=0)
    min_length_ratio: float = Field(default=1.2, ge=1.0)
    out_dir: Path = Path("reports")
    judges: list[JudgeSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> "RunConfig":
        names = [judge.name for judge in self.judges]
        if len(set(names)) != len(names):
            raise ValueError("judge names must be unique")
        if any(
            not name.replace("-", "").replace("_", "").replace(".", "").isalnum() for name in names
        ):
            raise ValueError("judge names may contain only letters, digits, '-', '_' and '.'")
        if self.source == "jsonl" and self.path is None:
            raise ValueError("path is required when source is jsonl")
        return self


def load_config(path: Path) -> RunConfig:
    """Read and validate a YAML run configuration."""
    with path.open(encoding="utf-8") as handle:
        return RunConfig.model_validate(yaml.safe_load(handle))
