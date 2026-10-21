"""Configuration from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_models_dir() -> Path:
    repo_models = Path(__file__).resolve().parents[2] / "models"
    if repo_models.is_dir():
        return repo_models
    return Path.home() / ".cache" / "chordanalyzer" / "models"


@dataclass
class Settings:
    models_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("MODELS_DIR") or _default_models_dir()))
    device: str = field(default_factory=lambda: os.environ.get("DEVICE", "cpu"))
    beat_tracker: str = field(default_factory=lambda: os.environ.get("BEAT_TRACKER", "auto"))
    beat_checkpoint: str = field(default_factory=lambda: os.environ.get("BEAT_CHECKPOINT", "final0"))
    max_upload_mb: int = field(default_factory=lambda: int(os.environ.get("MAX_UPLOAD_MB", "50")))
    min_duration_s: float = field(default_factory=lambda: float(os.environ.get("MIN_DURATION_S", "5")))
    max_duration_s: float = field(default_factory=lambda: float(os.environ.get("MAX_DURATION_S", "600")))
    workers: int = field(default_factory=lambda: int(os.environ.get("WORKERS", "1")))
    max_queue: int = field(default_factory=lambda: int(os.environ.get("MAX_QUEUE", "8")))
    api_keys: list[str] = field(default_factory=lambda: [
        k.strip() for k in os.environ.get("API_KEYS", "").split(",") if k.strip()])
    cache_dir: Path | None = field(default_factory=lambda: (
        Path(os.environ["CACHE_DIR"]) if os.environ.get("CACHE_DIR") else None))
    cors_origins: list[str] = field(default_factory=lambda: [
        o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()])
    torch_threads: int = field(default_factory=lambda: int(os.environ.get("TORCH_THREADS", "0")))
