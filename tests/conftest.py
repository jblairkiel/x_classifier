from __future__ import annotations

from pathlib import Path

import pytest

from tl.config import Settings, reset_settings_cache

FIXTURES = Path(__file__).parent / "fixtures"


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
