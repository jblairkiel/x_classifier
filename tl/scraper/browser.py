"""Playwright-driven, read-only timeline reader.

The browser only navigates, clicks a timeline tab, scrolls and reads. It never posts,
likes, follows, replies or DMs, and there is no anti-detection / stealth configuration.

Stop rules (see README): on a login wall, captcha, rate-limit or suspension notice the
scraper sets state "paused" and returns. On a selector failure it sets "error". Neither is
retried automatically; a human must press Resume (UI) or run `tl resume`.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Callable

from tl.config import Settings
from tl.db import now_iso, session
from tl.db import queries as Q
from tl.scraper import selectors as S
from tl.scraper.extract import Blocker, SelectorError, detect_blocker, extract_articles

log = logging.getLogger(__name__)

ARTICLES_JS = "els => els.map(e => e.outerHTML)"


@dataclass
class ScrapeOutcome:
    status: str               # ok | skipped | paused | error
    reason: str | None = None
    tweets_seen: int = 0
    tweets_new: int = 0
    scrolls: int = 0
    run_id: int | None = None


class _Stop(Exception):
    def __init__(self, blocker: Blocker):
        super().__init__(blocker.reason())
        self.blocker = blocker


def _check_blocker(page) -> None:
    blocker = detect_blocker(page.url, page.content())
    if blocker is not None:
        raise _Stop(blocker)


def _human_pause(settings: Settings, sleep: Callable[[float], None]) -> float:
    lo, hi = sorted((settings.scroll_delay_min, settings.scroll_delay_max))
    delay = random.uniform(lo, hi)
    sleep(delay)
    return delay


def _select_timeline_tab(page, settings: Settings) -> None:
    tab_sel = S.TIMELINE_TABS.get(settings.timeline)
    if tab_sel is None:
        raise ValueError(f"TIMELINE must be one of {sorted(S.TIMELINE_TABS)}, got {settings.timeline!r}")
    tab = page.locator(tab_sel).first
    try:
        tab.wait_for(state="visible", timeout=settings.page_timeout_seconds * 1000)
    except Exception as exc:  # playwright TimeoutError
        raise SelectorError(f"timeline tab {tab_sel!r} not found. Update tl/scraper/selectors.py.") from exc
    if tab.get_attribute(S.TAB_SELECTED_ATTR) != "true":
        tab.click()


def preflight(settings: Settings, *, trigger: str) -> tuple[ScrapeOutcome | None, int]:
    """Decide whether a run may start. Returns (outcome-if-not, per-run cap)."""
    with session(settings.db_path) as conn:
        state = Q.get_state(conn)
        if state["status"] in ("paused", "error"):
            return ScrapeOutcome("skipped", f"scraper is {state['status']}: {state['reason']}"), 0
        if state["status"] == "running":
            return ScrapeOutcome("skipped", "a run is already in progress"), 0
        remaining = settings.max_tweets_per_day - Q.tweets_seen_last_24h(conn)
        if remaining <= 0:
            return ScrapeOutcome("skipped", f"daily cap of {settings.max_tweets_per_day} tweets reached"), 0
    return None, min(settings.max_tweets_per_run, remaining)


def run_scrape(
    settings: Settings,
    *,
    trigger: str = "cli",
    home_url: str | None = None,
    use_storage_state: bool = True,
    sleep: Callable[[float], None] = time.sleep,
) -> ScrapeOutcome:
    """One scrape session. `home_url` / `use_storage_state=False` exist for fixture tests."""
    blocked, cap = preflight(settings, trigger=trigger)
    if blocked is not None:
        log.info("scrape skipped: %s", blocked.reason)
        return blocked

    storage_state = None
    if use_storage_state:
        path = settings.storage_state_path
        if not path.is_file():
            with session(settings.db_path) as conn:
                Q.set_state(conn, "paused", f"session file missing: {path} (run `tl login` locally and copy it)",
                            session_file_ok=0)
            return ScrapeOutcome("paused", f"session file missing: {path}")
        storage_state = str(path)

    with session(settings.db_path) as conn:
        Q.set_state(conn, "running", None, session_file_ok=1 if use_storage_state else None)
        run_id = Q.start_run(conn, settings.timeline, trigger)

    out = ScrapeOutcome("ok", run_id=run_id)
    seen: set[str] = set()
    try:
        _scrape_loop(settings, out, seen, cap, home_url or S.HOME_URL, storage_state, sleep)
    except _Stop as stop:
        out.status, out.reason = "paused", stop.blocker.reason()
    except SelectorError as exc:
        out.status, out.reason = "error", f"selector failure: {exc}"
    except Exception as exc:  # noqa: BLE001 -- any browser failure is an error state, never retried
        log.exception("scrape failed")
        out.status, out.reason = "error", f"{type(exc).__name__}: {exc}"[:500]

    out.tweets_seen = len(seen)
    with session(settings.db_path) as conn:
        Q.finish_run(conn, run_id, out.status, out.reason, out.tweets_seen, out.tweets_new, out.scrolls)
        new_status = {"ok": "idle", "paused": "paused", "error": "error"}[out.status]
        Q.set_state(conn, new_status, out.reason, last_run_at=now_iso())
    log.info("scrape finished: %s", out)
    return out


def _scrape_loop(settings, out, seen, cap, url, storage_state, sleep) -> None:
    from playwright.sync_api import sync_playwright

    timeout_ms = settings.page_timeout_seconds * 1000
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=settings.headless)
        try:
            context = browser.new_context(storage_state=storage_state, viewport={"width": 1280, "height": 900})
            context.set_default_timeout(timeout_ms)
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded")
            sleep(random.uniform(2.0, 4.0))
            _check_blocker(page)
            _select_timeline_tab(page, settings)
            try:
                page.wait_for_selector(S.TWEET_ARTICLE, timeout=timeout_ms)
            except Exception as exc:
                _check_blocker(page)  # a blocker explains the empty page better than a selector error
                raise SelectorError(
                    f"no elements matched {S.TWEET_ARTICLE!r} within {settings.page_timeout_seconds}s. "
                    "Update tl/scraper/selectors.py."
                ) from exc

            stale = 0
            while True:
                fragments = page.eval_on_selector_all(S.TWEET_ARTICLE, ARTICLES_JS)
                records, _promoted = extract_articles(fragments)
                fresh = [r for r in records if r.id not in seen]
                fresh = fresh[: max(0, cap - len(seen))]
                if fresh:
                    with session(settings.db_path) as conn:
                        _, new = Q.upsert_tweets(conn, fresh, timeline=settings.timeline, run_id=out.run_id)
                    out.tweets_new += new
                    seen.update(r.id for r in fresh)
                    stale = 0
                else:
                    stale += 1

                if len(seen) >= cap:
                    out.reason = f"per-run cap reached ({cap})"
                    break
                if out.scrolls >= settings.max_scrolls_per_run:
                    out.reason = f"max scrolls reached ({settings.max_scrolls_per_run})"
                    break
                if stale >= settings.stale_scrolls_before_stop:
                    out.reason = "no new tweets after scrolling"
                    break

                page.mouse.wheel(0, int(900 * random.uniform(0.55, 0.95)))
                out.scrolls += 1
                _human_pause(settings, sleep)
                _check_blocker(page)
        finally:
            browser.close()
