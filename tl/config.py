"""All runtime settings, read from environment variables.

Every variable here is listed (with its default) in .env.example. Keep the two in sync.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    return _env(name, "true" if default else "false").strip().lower() in {"1", "true", "yes", "on"}


def _hours(value: str) -> tuple[int, int]:
    """Parse "08-22" into (8, 22). Empty string or "*" means all day."""
    value = value.strip()
    if value in {"", "*"}:
        return (0, 24)
    start, end = value.split("-", 1)
    return (int(start), int(end))


@dataclass(frozen=True)
class Settings:
    # --- web / auth ---
    domain: str = field(default_factory=lambda: _env("DOMAIN", "x.jblairkiel.com"))
    admin_username: str = field(default_factory=lambda: _env("ADMIN_USERNAME", "admin"))
    admin_password_hash: str = field(default_factory=lambda: _env("ADMIN_PASSWORD_HASH", ""))
    session_secret: str = field(default_factory=lambda: _env("SESSION_SECRET", ""))
    session_max_age_hours: int = field(default_factory=lambda: _int("SESSION_MAX_AGE_HOURS", 12))
    cookie_secure: bool = field(default_factory=lambda: _bool("COOKIE_SECURE", True))
    login_max_attempts: int = field(default_factory=lambda: _int("LOGIN_MAX_ATTEMPTS", 5))
    login_window_minutes: int = field(default_factory=lambda: _int("LOGIN_WINDOW_MINUTES", 15))
    login_global_max_attempts: int = field(default_factory=lambda: _int("LOGIN_GLOBAL_MAX_ATTEMPTS", 30))

    # --- storage ---
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "./var/data")))
    models_dir: Path = field(default_factory=lambda: Path(_env("MODELS_DIR", "./var/models")))
    secrets_dir: Path = field(default_factory=lambda: Path(_env("SECRETS_DIR", "./secrets")))
    storage_state_file: str = field(default_factory=lambda: _env("STORAGE_STATE_FILE", "x_storage_state.json"))

    # --- scraper ---
    timeline: str = field(default_factory=lambda: _env("TIMELINE", "following"))  # following | for_you
    headless: bool = field(default_factory=lambda: _bool("HEADLESS", True))
    max_tweets_per_run: int = field(default_factory=lambda: _int("SCRAPE_MAX_TWEETS_PER_RUN", 60))
    max_tweets_per_day: int = field(default_factory=lambda: _int("SCRAPE_MAX_TWEETS_PER_DAY", 300))
    max_scrolls_per_run: int = field(default_factory=lambda: _int("SCRAPE_MAX_SCROLLS_PER_RUN", 25))
    scroll_delay_min: float = field(default_factory=lambda: _float("SCROLL_DELAY_MIN_SECONDS", 4.0))
    scroll_delay_max: float = field(default_factory=lambda: _float("SCROLL_DELAY_MAX_SECONDS", 11.0))
    page_timeout_seconds: int = field(default_factory=lambda: _int("PAGE_TIMEOUT_SECONDS", 30))
    stale_scrolls_before_stop: int = field(default_factory=lambda: _int("SCRAPE_STALE_SCROLLS", 3))

    # --- schedule (worker) ---
    scrape_enabled: bool = field(default_factory=lambda: _bool("SCRAPE_ENABLED", True))
    scrape_interval_minutes: int = field(default_factory=lambda: _int("SCRAPE_INTERVAL_MINUTES", 240))
    scrape_interval_jitter_minutes: int = field(default_factory=lambda: _int("SCRAPE_INTERVAL_JITTER_MINUTES", 45))
    scrape_active_hours: tuple[int, int] = field(default_factory=lambda: _hours(_env("SCRAPE_ACTIVE_HOURS", "08-22")))
    worker_poll_seconds: int = field(default_factory=lambda: _int("WORKER_POLL_SECONDS", 20))

    # --- model ---
    model_version: str = field(default_factory=lambda: _env("MODEL_VERSION", "latest"))
    base_model: str = field(default_factory=lambda: _env("BASE_MODEL", "distilroberta-base"))
    ai_threshold: float = field(default_factory=lambda: _float("AI_THRESHOLD", 0.8))
    classify_batch_size: int = field(default_factory=lambda: _int("CLASSIFY_BATCH_SIZE", 16))
    max_length: int = field(default_factory=lambda: _int("MODEL_MAX_LENGTH", 128))

    @property
    def db_path(self) -> Path:
        return Path(_env("DB_PATH", str(self.data_dir / "tl.sqlite3")))

    @property
    def storage_state_path(self) -> Path:
        return self.secrets_dir / self.storage_state_file


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
