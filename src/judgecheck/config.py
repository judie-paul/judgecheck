"""Environment-driven settings."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

DATASET_ID = "lmsys/mt_bench_human_judgments"
DATASET_REVISION = "f7d2896d2cc5d80f8b55c2bbc722613555233c25"


class Settings(BaseSettings):
    """Settings read from ``JUDGECHECK_*`` environment variables or ``.env``."""

    model_config = SettingsConfigDict(env_prefix="JUDGECHECK_", env_file=".env", extra="ignore")

    dataset_revision: str = DATASET_REVISION
    cache_dir: Path = Path(".cache")
