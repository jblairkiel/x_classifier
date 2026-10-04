"""All SQL used by the app, grouped by table. Functions take an open connection."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence

from tl.db import now_iso
from tl.records import TweetRecord

LABEL_VALUES = ("human", "ai", "unsure")
SCRAPER_STATUSES = ("idle", "running", "paused", "error")


# --------------------------------------------------------------------------- tweets / accounts


def upsert_tweets(
    conn: sqlite3.Connection,
    records: Iterable[TweetRecord],
    *,
    timeline: str | None = None,
    run_id: int | None = None,
) -> tuple[int, int]:
    """Insert or refresh tweets, deduplicated by id. Returns (unique_seen, newly_inserted)."""
    unique: dict[str, TweetRecord] = {}
    for rec in records:
        unique[rec.id] = rec  # later sighting wins (fresher counts)
    if not unique:
        return (0, 0)

    ids = list(unique)
    existing: set[str] = set()
    for chunk_start in range(0, len(ids), 500):
        chunk = ids[chunk_start : chunk_start + 500]
        q = f"SELECT id FROM tweets WHERE id IN ({','.join('?' * len(chunk))})"
        existing.update(row[0] for row in conn.execute(q, chunk))

    now = now_iso()
    for rec in unique.values():
        conn.execute(
            """
            INSERT INTO accounts (handle, display_name, first_seen_at, last_seen_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(handle) DO UPDATE SET
                display_name = COALESCE(excluded.display_name, accounts.display_name),
                last_seen_at = excluded.last_seen_at
            """,
            (rec.author_handle, rec.display_name, now, now),
        )
        conn.execute(
            """
            INSERT INTO tweets (id, author_handle, display_name, text, created_at, reply_count,
                                repost_count, like_count, is_reply, is_repost, reposted_by, has_media,
                                lang, timeline, first_seen_at, last_seen_at, seen_count, run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            ON CONFLICT(id) DO UPDATE SET
                reply_count  = COALESCE(excluded.reply_count, tweets.reply_count),
                repost_count = COALESCE(excluded.repost_count, tweets.repost_count),
                like_count   = COALESCE(excluded.like_count, tweets.like_count),
                last_seen_at = excluded.last_seen_at,
                seen_count   = tweets.seen_count + 1
            """,
            (
                rec.id, rec.author_handle, rec.display_name, rec.text, rec.created_at,
                rec.reply_count, rec.repost_count, rec.like_count, int(rec.is_reply),
                int(rec.is_repost), rec.reposted_by, int(rec.has_media), rec.lang,
                timeline, now, now, run_id,
            ),
        )
    return (len(unique), len(unique) - len(existing))


def get_tweet(conn: sqlite3.Connection, tweet_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM tweets WHERE id = ?", (tweet_id,)).fetchone()


def tweets_for_account(conn: sqlite3.Connection, handle: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM tweets WHERE author_handle = ? ORDER BY COALESCE(created_at, first_seen_at)",
        (handle,),
    ).fetchall()


def all_account_handles(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute("SELECT handle FROM accounts ORDER BY handle")]


def get_account(conn: sqlite3.Connection, handle: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM accounts WHERE handle = ?", (handle,)).fetchone()


def set_account_features(conn: sqlite3.Connection, handle: str, score: float | None, features: dict) -> None:
    conn.execute(
        "UPDATE accounts SET botlike_score = ?, features_json = ?, features_updated_at = ? WHERE handle = ?",
        (score, json.dumps(features, sort_keys=True), now_iso(), handle),
    )


def counts(conn: sqlite3.Connection) -> dict[str, int]:
    one = lambda q, *a: conn.execute(q, a).fetchone()[0]  # noqa: E731
    today = datetime.now(timezone.utc).date().isoformat()
    return {
        "tweets": one("SELECT COUNT(*) FROM tweets"),
        "tweets_today": one("SELECT COUNT(*) FROM tweets WHERE first_seen_at >= ?", today),
        "accounts": one("SELECT COUNT(*) FROM accounts"),
        "manual_labels": one("SELECT COUNT(*) FROM current_manual_labels"),
        "unlabeled": one(
            "SELECT COUNT(*) FROM tweets t WHERE NOT EXISTS "
            "(SELECT 1 FROM labels l WHERE l.tweet_id = t.id AND l.source = 'manual')"
        ),
        "samples": one("SELECT COUNT(*) FROM samples"),
        "predictions": one("SELECT COUNT(*) FROM predictions"),
    }


# --------------------------------------------------------------------------- scraper state / runs


def get_state(conn: sqlite3.Connection) -> sqlite3.Row:
    return conn.execute("SELECT * FROM scraper_state WHERE id = 1").fetchone()


def set_state(conn: sqlite3.Connection, status: str, reason: str | None = None, **extra) -> None:
    if status not in SCRAPER_STATUSES:
        raise ValueError(f"bad scraper status {status!r}")
    allowed = {"last_run_at", "next_run_at", "session_file_ok"}
    sets = ["status = ?", "reason = ?", "updated_at = ?"]
    params: list = [status, reason, now_iso()]
    for key, value in extra.items():
        if key not in allowed:
            raise ValueError(key)
        sets.append(f"{key} = ?")
        params.append(value)
    conn.execute(f"UPDATE scraper_state SET {', '.join(sets)} WHERE id = 1", params)


def update_state_fields(conn: sqlite3.Connection, **fields) -> None:
    allowed = {"last_run_at", "next_run_at", "worker_heartbeat_at", "session_file_ok"}
    if not fields:
        return
    for key in fields:
        if key not in allowed:
            raise ValueError(key)
    sets = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(f"UPDATE scraper_state SET {sets} WHERE id = 1", list(fields.values()))


def start_run(conn: sqlite3.Connection, timeline: str, trigger: str) -> int:
    cur = conn.execute(
        "INSERT INTO scrape_runs (started_at, timeline, status, trigger) VALUES (?, ?, 'running', ?)",
        (now_iso(), timeline, trigger),
    )
    return int(cur.lastrowid)


def finish_run(
    conn: sqlite3.Connection, run_id: int, status: str, reason: str | None,
    tweets_seen: int, tweets_new: int, scrolls: int,
) -> None:
    conn.execute(
        """UPDATE scrape_runs SET finished_at = ?, status = ?, reason = ?, tweets_seen = ?,
           tweets_new = ?, scrolls = ? WHERE id = ?""",
        (now_iso(), status, reason, tweets_seen, tweets_new, scrolls, run_id),
    )


def tweets_seen_since(conn: sqlite3.Connection, since: datetime) -> int:
    """Tweets read by the scraper since `since` (sum over runs) -- used for the daily cap."""
    row = conn.execute(
        "SELECT COALESCE(SUM(tweets_seen), 0) FROM scrape_runs WHERE started_at >= ?",
        (since.astimezone(timezone.utc).replace(microsecond=0).isoformat(),),
    ).fetchone()
    return int(row[0])


def tweets_seen_last_24h(conn: sqlite3.Connection) -> int:
    return tweets_seen_since(conn, datetime.now(timezone.utc) - timedelta(hours=24))


def recent_runs(conn: sqlite3.Connection, limit: int = 10) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM scrape_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()


# --------------------------------------------------------------------------- jobs


def enqueue_job(conn: sqlite3.Connection, kind: str) -> int:
    dup = conn.execute(
        "SELECT id FROM jobs WHERE kind = ? AND status IN ('queued', 'running')", (kind,)
    ).fetchone()
    if dup:
        return int(dup[0])
    cur = conn.execute(
        "INSERT INTO jobs (kind, status, created_at) VALUES (?, 'queued', ?)", (kind, now_iso())
    )
    return int(cur.lastrowid)


def claim_next_job(conn: sqlite3.Connection) -> sqlite3.Row | None:
    row = conn.execute("SELECT * FROM jobs WHERE status = 'queued' ORDER BY id LIMIT 1").fetchone()
    if row is None:
        return None
    conn.execute("UPDATE jobs SET status = 'running', started_at = ? WHERE id = ?", (now_iso(), row["id"]))
    return row


def finish_job(conn: sqlite3.Connection, job_id: int, ok: bool, message: str | None = None) -> None:
    conn.execute(
        "UPDATE jobs SET status = ?, finished_at = ?, message = ? WHERE id = ?",
        ("done" if ok else "failed", now_iso(), message, job_id),
    )


def recent_jobs(conn: sqlite3.Connection, limit: int = 10) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()


def fail_stale_running_jobs(conn: sqlite3.Connection) -> None:
    """On worker start, anything still 'running' was interrupted."""
    conn.execute(
        "UPDATE jobs SET status = 'failed', finished_at = ?, message = 'interrupted (worker restarted)' "
        "WHERE status = 'running'",
        (now_iso(),),
    )


# --------------------------------------------------------------------------- labels


def add_manual_label(conn: sqlite3.Connection, tweet_id: str, label: str) -> None:
    if label not in LABEL_VALUES:
        raise ValueError(f"bad label {label!r}")
    if get_tweet(conn, tweet_id) is None:
        raise KeyError(tweet_id)
    conn.execute(
        "INSERT INTO labels (tweet_id, label, source, created_at) VALUES (?, ?, 'manual', ?)",
        (tweet_id, label, now_iso()),
    )


def current_label(conn: sqlite3.Connection, tweet_id: str) -> str | None:
    row = conn.execute("SELECT label FROM current_manual_labels WHERE tweet_id = ?", (tweet_id,)).fetchone()
    return row[0] if row else None


def next_unlabeled(conn: sqlite3.Connection, model_version: str | None, skip: Sequence[str] = ()) -> sqlite3.Row | None:
    """Next tweet without a manual label. Prefers uncertain predictions (score near 0.5), then newest."""
    params: list = [model_version]
    skip_sql = ""
    if skip:
        skip_sql = f"AND t.id NOT IN ({','.join('?' * len(skip))})"
        params.extend(skip)
    return conn.execute(
        f"""
        SELECT t.*, p.score FROM tweets t
        LEFT JOIN predictions p ON p.tweet_id = t.id AND p.model_version = ?
        WHERE NOT EXISTS (SELECT 1 FROM labels l WHERE l.tweet_id = t.id AND l.source = 'manual')
          AND length(trim(t.text)) > 0 {skip_sql}
        ORDER BY CASE WHEN p.score IS NULL THEN 1 ELSE 0 END,
                 ABS(COALESCE(p.score, 0.5) - 0.5),
                 t.first_seen_at DESC
        LIMIT 1
        """,
        params,
    ).fetchone()


def manual_eval_set(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Hand-labelled timeline tweets (human/ai only). Never used for training."""
    return conn.execute(
        """SELECT t.id, t.text, l.label FROM current_manual_labels l JOIN tweets t ON t.id = l.tweet_id
           WHERE l.label IN ('human', 'ai') ORDER BY t.id"""
    ).fetchall()


def label_counts(conn: sqlite3.Connection) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for row in conn.execute(
        "SELECT source, label, COUNT(*) FROM labels WHERE source != 'manual' GROUP BY source, label"
    ):
        out.setdefault(row[0], {})[row[1]] = row[2]
    for row in conn.execute("SELECT label, COUNT(*) FROM current_manual_labels GROUP BY label"):
        out.setdefault("manual", {})[row[0]] = row[1]
    return out


# --------------------------------------------------------------------------- samples (training corpus)


def insert_sample(
    conn: sqlite3.Connection, *, text: str, text_hash: str, origin: str, variant: str | None,
    meta: dict | None, split: str, label: str, source: str,
) -> bool:
    """Insert a corpus sample + its label. Returns False if the text already exists."""
    if source not in ("synthetic", "dataset"):
        raise ValueError(source)
    if label not in ("human", "ai"):
        raise ValueError(label)
    cur = conn.execute(
        """INSERT OR IGNORE INTO samples (text, text_hash, origin, variant, meta_json, split, imported_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (text, text_hash, origin, variant, json.dumps(meta) if meta else None, split, now_iso()),
    )
    if cur.rowcount == 0:
        return False
    conn.execute(
        "INSERT INTO labels (sample_id, label, source, created_at) VALUES (?, ?, ?, ?)",
        (cur.lastrowid, label, source, now_iso()),
    )
    return True


def training_samples(conn: sqlite3.Connection, split: str) -> list[sqlite3.Row]:
    """Corpus rows for a split. Manual (timeline) labels are excluded by construction:
    they never point at samples. Samples whose text also appears among manually-labelled
    timeline tweets are dropped too (leakage guard, see tl.data.common.text_hash)."""
    return conn.execute(
        """SELECT s.id, s.text, s.text_hash, s.origin, s.variant, l.label, l.source
           FROM samples s JOIN labels l ON l.sample_id = s.id
           WHERE s.split = ? AND l.source IN ('synthetic', 'dataset') AND l.label IN ('human', 'ai')
           ORDER BY s.id""",
        (split,),
    ).fetchall()


def sample_origin_counts(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT l.source, s.origin, COALESCE(s.variant, '') AS variant, l.label, COUNT(*) AS n
           FROM samples s JOIN labels l ON l.sample_id = s.id
           GROUP BY l.source, s.origin, s.variant, l.label ORDER BY l.source, s.origin"""
    ).fetchall()


# --------------------------------------------------------------------------- predictions


def unscored_tweets(conn: sqlite3.Connection, model_version: str, limit: int | None = None) -> list[sqlite3.Row]:
    sql = """SELECT t.id, t.text FROM tweets t WHERE NOT EXISTS
             (SELECT 1 FROM predictions p WHERE p.tweet_id = t.id AND p.model_version = ?)
             ORDER BY t.first_seen_at"""
    params: list = [model_version]
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    return conn.execute(sql, params).fetchall()


def save_predictions(conn: sqlite3.Connection, model_version: str, scores: Iterable[tuple[str, float]]) -> int:
    now = now_iso()
    n = 0
    for tweet_id, score in scores:
        conn.execute(
            """INSERT INTO predictions (tweet_id, model_version, score, created_at) VALUES (?, ?, ?, ?)
               ON CONFLICT(tweet_id, model_version) DO UPDATE SET score = excluded.score,
               created_at = excluded.created_at""",
            (tweet_id, model_version, float(score), now),
        )
        n += 1
    return n


def delete_predictions(conn: sqlite3.Connection, model_version: str) -> None:
    conn.execute("DELETE FROM predictions WHERE model_version = ?", (model_version,))


def prediction_versions(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT model_version, COUNT(*) AS n, MAX(created_at) AS last FROM predictions GROUP BY model_version"
    ).fetchall()


def timeline_page(
    conn: sqlite3.Connection, *, model_version: str | None, min_score: float = 0.0,
    handle: str | None = None, date_from: str | None = None, date_to: str | None = None,
    label: str | None = None, limit: int = 50, offset: int = 0,
) -> list[sqlite3.Row]:
    where = ["COALESCE(p.score, -1) >= ?"]
    params: list = [model_version, min_score]
    if handle:
        where.append("t.author_handle = ?")
        params.append(handle.lower().lstrip("@"))
    if date_from:
        where.append("COALESCE(t.created_at, t.first_seen_at) >= ?")
        params.append(date_from)
    if date_to:
        where.append("COALESCE(t.created_at, t.first_seen_at) < date(?, '+1 day')")
        params.append(date_to)
    if label == "unlabeled":
        where.append("cl.label IS NULL")
    elif label:
        where.append("cl.label = ?")
        params.append(label)
    params += [limit, offset]
    return conn.execute(
        f"""
        SELECT t.*, p.score, cl.label AS manual_label, a.botlike_score
        FROM tweets t
        LEFT JOIN predictions p ON p.tweet_id = t.id AND p.model_version = ?
        LEFT JOIN current_manual_labels cl ON cl.tweet_id = t.id
        LEFT JOIN accounts a ON a.handle = t.author_handle
        WHERE {' AND '.join(where)}
        ORDER BY p.score DESC NULLS LAST, COALESCE(t.created_at, t.first_seen_at) DESC
        LIMIT ? OFFSET ?
        """,
        params,
    ).fetchall()


def accounts_ranked(conn: sqlite3.Connection, model_version: str | None, limit: int = 200) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT a.*, COUNT(t.id) AS n_tweets, AVG(p.score) AS mean_score, MAX(p.score) AS max_score
        FROM accounts a
        LEFT JOIN tweets t ON t.author_handle = a.handle
        LEFT JOIN predictions p ON p.tweet_id = t.id AND p.model_version = ?
        GROUP BY a.handle
        ORDER BY a.botlike_score DESC NULLS LAST, mean_score DESC NULLS LAST, n_tweets DESC
        LIMIT ?
        """,
        (model_version, limit),
    ).fetchall()


def account_tweets_with_scores(conn: sqlite3.Connection, handle: str, model_version: str | None) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT t.*, p.score, cl.label AS manual_label FROM tweets t
           LEFT JOIN predictions p ON p.tweet_id = t.id AND p.model_version = ?
           LEFT JOIN current_manual_labels cl ON cl.tweet_id = t.id
           WHERE t.author_handle = ? ORDER BY COALESCE(t.created_at, t.first_seen_at) DESC""",
        (model_version, handle),
    ).fetchall()


def scores_for_account(conn: sqlite3.Connection, handle: str, model_version: str | None) -> list[float]:
    return [
        r[0]
        for r in conn.execute(
            """SELECT p.score FROM predictions p JOIN tweets t ON t.id = p.tweet_id
               WHERE t.author_handle = ? AND p.model_version = ?""",
            (handle, model_version),
        )
    ]
