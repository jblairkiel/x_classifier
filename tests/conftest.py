from __future__ import annotations

from pathlib import Path

import pytest

from tl.config import Settings, reset_settings_cache

FIXTURES = Path(__file__).parent / "fixtures"
PASSWORD = "correct horse battery staple"


@pytest.fixture
def settings(tmp_path, monkeypatch) -> Settings:
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("SECRETS_DIR", str(tmp_path / "secrets"))
    monkeypatch.setenv("SCROLL_DELAY_MIN_SECONDS", "0")
    monkeypatch.setenv("SCROLL_DELAY_MAX_SECONDS", "0.05")
    monkeypatch.delenv("DB_PATH", raising=False)
    reset_settings_cache()
    yield Settings()
    reset_settings_cache()


@pytest.fixture
def fixture_html():
    def load(name: str) -> str:
        return (FIXTURES / name).read_text(encoding="utf-8")

    return load


@pytest.fixture
def app_settings(settings):
    return settings.__class__(**{
        **settings.__dict__,
        "admin_password_hash": _pw_hash(),
        "session_secret": "x" * 48,
        "login_max_attempts": 3,
        "login_global_max_attempts": 100,
    })


_HASH_CACHE: list[str] = []


def _pw_hash() -> str:
    if not _HASH_CACHE:
        from tl.web.auth import hash_password

        _HASH_CACHE.append(hash_password(PASSWORD))
    return _HASH_CACHE[0]
