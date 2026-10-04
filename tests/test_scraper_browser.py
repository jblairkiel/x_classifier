"""End-to-end scraper run in real headless Chromium against saved fixture pages."""

import time

import pytest

from tl.db import session
from tl.db import queries as Q
from tl.scraper.browser import run_scrape
from tests.conftest import FIXTURES

pw = pytest.importorskip("playwright.sync_api")


def _chromium_available() -> bool:
    try:
        with pw.sync_playwright() as p:
            p.chromium.launch().close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _chromium_available(), reason="Chromium not installed")


def scrape(settings, name):
    return run_scrape(settings, trigger="test", home_url=(FIXTURES / name).as_uri(),
                      use_storage_state=False, sleep=lambda s: time.sleep(0.2))


def test_scrolls_and_respects_per_run_cap(settings, monkeypatch):
    s = settings.__class__(**{**settings.__dict__, "max_tweets_per_run": 20})
    out = scrape(s, "timeline_infinite.html")
    assert out.status == "ok", out
    assert out.tweets_seen == 20 and out.tweets_new == 20
    assert out.scrolls >= 1
    with session(s.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM tweets").fetchone()[0] == 20
        assert Q.get_state(conn)["status"] == "idle"


def test_daily_cap_blocks_next_run(settings):
    s = settings.__class__(**{**settings.__dict__, "max_tweets_per_run": 20, "max_tweets_per_day": 25})
    assert scrape(s, "timeline_infinite.html").tweets_seen == 20
    second = scrape(s, "timeline_infinite.html")
    assert second.tweets_seen == 5  # only the remaining daily budget
    third = scrape(s, "timeline_infinite.html")
    assert third.status == "skipped" and "daily cap" in third.reason


@pytest.mark.parametrize("name,kind", [("blocker_rate_limit.html", "rate_limit"),
                                       ("blocker_login_wall.html", "login_wall")])
def test_blocker_pauses_and_is_not_retried(settings, name, kind):
    out = scrape(settings, name)
    assert out.status == "paused" and out.reason.startswith(kind)
    with session(settings.db_path) as conn:
        state = Q.get_state(conn)
        assert state["status"] == "paused" and state["reason"].startswith(kind)
    again = scrape(settings, "timeline_infinite.html")
    assert again.status == "skipped" and "paused" in again.reason


def test_dom_drift_sets_error_state(settings):
    out = scrape(settings, "timeline_dom_changed.html")
    assert out.status == "error" and "selectors.py" in out.reason
    with session(settings.db_path) as conn:
        assert Q.get_state(conn)["status"] == "error"


def test_missing_session_file_pauses(settings):
    out = run_scrape(settings, trigger="test")
    assert out.status == "paused" and "session file missing" in out.reason
