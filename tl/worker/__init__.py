"""Background worker: scheduled scraping + classification + account features, and jobs queued by the web UI."""

from __future__ import annotations

import logging
import random
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tl.config import Settings
from tl.db import now_iso, session
from tl.db import queries as Q

log = logging.getLogger("tl.worker")

HEARTBEAT_FILE = Path("/tmp/tl-worker-heartbeat")
HEARTBEAT_EVERY = 15
HEARTBEAT_MAX_AGE = 120


def _heartbeat_loop(settings: Settings, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            HEARTBEAT_FILE.write_text(str(time.time()))
            with session(settings.db_path) as conn:
                Q.update_state_fields(conn, worker_heartbeat_at=now_iso(),
                                      session_file_ok=int(settings.storage_state_path.is_file()))
        except Exception:  # noqa: BLE001
            log.exception("heartbeat failed")
        stop.wait(HEARTBEAT_EVERY)


def healthy() -> bool:
    try:
        return time.time() - float(HEARTBEAT_FILE.read_text()) < HEARTBEAT_MAX_AGE
    except (OSError, ValueError):
        return False


def in_active_hours(settings: Settings, now: datetime | None = None) -> bool:
    start, end = settings.scrape_active_hours
    hour = (now or datetime.now().astimezone()).hour
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end  # window wraps midnight, e.g. 20-02


def next_run_after(settings: Settings, now: datetime) -> datetime:
    jitter = random.uniform(-settings.scrape_interval_jitter_minutes, settings.scrape_interval_jitter_minutes)
    return now + timedelta(minutes=max(5.0, settings.scrape_interval_minutes + jitter))


def classify_pending(settings: Settings, *, rescore: bool = False) -> str:
    """Score tweets with the active model. Returns a short message for the job log."""
    from tl.model import predict  # heavy import (torch) only when needed

    try:
        scorer = predict.load_scorer(settings)
    except predict.NoModelError as exc:
        return f"no model: {exc}"
    with session(settings.db_path) as conn:
        if rescore:
            Q.delete_predictions(conn, scorer.version)
        rows = Q.unscored_tweets(conn, scorer.version)
    rows = [r for r in rows if r["text"].strip()]
    total = 0
    for start in range(0, len(rows), 256):
        batch = rows[start : start + 256]
        scores = scorer.score([r["text"] for r in batch])
        with session(settings.db_path) as conn:
            total += Q.save_predictions(conn, scorer.version, zip([r["id"] for r in batch], scores))
    return f"scored {total} tweets with {scorer.version}"


def compute_features(settings: Settings) -> str:
    from tl.features import update_all_accounts

    n = update_all_accounts(settings)
    return f"updated {n} accounts"


def run_scrape_cycle(settings: Settings, trigger: str) -> str:
    from tl.scraper.browser import run_scrape

    out = run_scrape(settings, trigger=trigger)
    msg = f"scrape {out.status}: seen={out.tweets_seen} new={out.tweets_new} scrolls={out.scrolls}"
    if out.reason:
        msg += f" ({out.reason})"
    if out.status == "ok" and out.tweets_new:
        msg += "; " + _safe(classify_pending, settings)
        msg += "; " + _safe(compute_features, settings)
    return msg


def _safe(fn, *args, **kwargs) -> str:
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        log.exception("%s failed", fn.__name__)
        return f"{fn.__name__} failed: {exc}"


JOB_HANDLERS = {
    "scrape": lambda s: run_scrape_cycle(s, "manual"),
    "classify": lambda s: classify_pending(s),
    "reclassify": lambda s: classify_pending(s, rescore=True),
    "features": compute_features,
}


def process_jobs(settings: Settings) -> int:
    handled = 0
    while True:
        with session(settings.db_path) as conn:
            job = Q.claim_next_job(conn)
        if job is None:
            return handled
        log.info("job %s (%s) started", job["id"], job["kind"])
        try:
            msg = JOB_HANDLERS[job["kind"]](settings)
            ok = not msg.startswith(("scrape error", "scrape paused"))
        except Exception as exc:  # noqa: BLE001
            log.exception("job %s failed", job["id"])
            msg, ok = f"{type(exc).__name__}: {exc}", False
        with session(settings.db_path) as conn:
            Q.finish_job(conn, job["id"], ok, msg[:1000])
        log.info("job %s finished: %s", job["id"], msg)
        handled += 1


def maybe_scheduled_scrape(settings: Settings) -> None:
    if not settings.scrape_enabled:
        return
    now = datetime.now(timezone.utc)
    with session(settings.db_path) as conn:
        state = Q.get_state(conn)
        nxt = state["next_run_at"]
        if nxt is None:
            # First start: wait a little rather than scraping the instant the container boots.
            Q.update_state_fields(conn, next_run_at=(now + timedelta(minutes=random.uniform(2, 10))).isoformat())
            return
    if datetime.fromisoformat(nxt) > now:
        return
    if state["status"] != "idle":
        return  # paused / error wait for a human; running can't happen here (single worker)
    with session(settings.db_path) as conn:
        Q.update_state_fields(conn, next_run_at=next_run_after(settings, now).replace(microsecond=0).isoformat())
    if not in_active_hours(settings):
        log.info("outside active hours %s; skipping scheduled scrape", settings.scrape_active_hours)
        return
    log.info(run_scrape_cycle(settings, "schedule"))


def run_forever(settings: Settings) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    with session(settings.db_path) as conn:
        Q.fail_stale_running_jobs(conn)
        if Q.get_state(conn)["status"] == "running":
            Q.set_state(conn, "idle", "previous run was interrupted by a worker restart")
    stop = threading.Event()
    threading.Thread(target=_heartbeat_loop, args=(settings, stop), daemon=True).start()
    log.info("worker started (scrape_enabled=%s, interval=%smin, active_hours=%s)",
             settings.scrape_enabled, settings.scrape_interval_minutes, settings.scrape_active_hours)
    try:
        while True:
            _safe(process_jobs, settings)
            _safe(maybe_scheduled_scrape, settings)
            time.sleep(settings.worker_poll_seconds)
    finally:
        stop.set()
