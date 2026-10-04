-- tl-ai-detector schema. Applied idempotently on every connect (CREATE ... IF NOT EXISTS).
-- Timestamps are ISO-8601 UTC strings ("2026-10-04T12:34:56+00:00").

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

-- Accounts seen on the timeline. Botlike score is filled by tl.features.
CREATE TABLE IF NOT EXISTS accounts (
    handle              TEXT PRIMARY KEY,           -- lowercase, without "@"
    display_name        TEXT,
    first_seen_at       TEXT NOT NULL,
    last_seen_at        TEXT NOT NULL,
    botlike_score       REAL,                       -- 0..1, NULL until computed
    features_json       TEXT,                       -- per-signal values used for the score
    features_updated_at TEXT
);

-- Tweets scraped from the home timeline. id is X's numeric status id (as text).
CREATE TABLE IF NOT EXISTS tweets (
    id              TEXT PRIMARY KEY,
    author_handle   TEXT NOT NULL REFERENCES accounts(handle),
    display_name    TEXT,
    text            TEXT NOT NULL,
    created_at      TEXT,                           -- tweet timestamp from <time datetime>
    reply_count     INTEGER,
    repost_count    INTEGER,
    like_count      INTEGER,
    is_reply        INTEGER NOT NULL DEFAULT 0,
    is_repost       INTEGER NOT NULL DEFAULT 0,     -- shown on the timeline because someone reposted it
    reposted_by     TEXT,
    has_media       INTEGER NOT NULL DEFAULT 0,
    lang            TEXT,
    timeline        TEXT,                           -- following | for_you
    first_seen_at   TEXT NOT NULL,
    last_seen_at    TEXT NOT NULL,
    seen_count      INTEGER NOT NULL DEFAULT 1,
    run_id          INTEGER REFERENCES scrape_runs(id)
);
CREATE INDEX IF NOT EXISTS idx_tweets_author ON tweets(author_handle);
CREATE INDEX IF NOT EXISTS idx_tweets_created ON tweets(created_at);
CREATE INDEX IF NOT EXISTS idx_tweets_first_seen ON tweets(first_seen_at);

-- Training / reference corpus that is NOT from the timeline (public datasets, synthetic AI text).
CREATE TABLE IF NOT EXISTS samples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    text        TEXT NOT NULL,
    text_hash   TEXT NOT NULL UNIQUE,               -- sha1 of normalized text, dedupes imports
    origin      TEXT NOT NULL,                      -- dataset name, or generator model name
    variant     TEXT,                               -- e.g. "casual", "default"
    meta_json   TEXT,
    split       TEXT NOT NULL CHECK (split IN ('train', 'val', 'test')),
    imported_at TEXT NOT NULL
);

-- Labels. Manual labels point at timeline tweets and form the held-out eval set (never trained on).
-- synthetic/dataset labels point at samples.
CREATE TABLE IF NOT EXISTS labels (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tweet_id    TEXT REFERENCES tweets(id) ON DELETE CASCADE,
    sample_id   INTEGER REFERENCES samples(id) ON DELETE CASCADE,
    label       TEXT NOT NULL CHECK (label IN ('human', 'ai', 'unsure')),
    source      TEXT NOT NULL CHECK (source IN ('manual', 'synthetic', 'dataset')),
    created_at  TEXT NOT NULL,
    CHECK ((tweet_id IS NOT NULL) + (sample_id IS NOT NULL) = 1),
    CHECK ((source = 'manual') = (tweet_id IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS idx_labels_tweet ON labels(tweet_id);
CREATE INDEX IF NOT EXISTS idx_labels_sample ON labels(sample_id);

-- Labels are append-only (relabeling inserts a new row); this view is the latest manual label per tweet.
CREATE VIEW IF NOT EXISTS current_manual_labels AS
SELECT l.* FROM labels l
JOIN (SELECT tweet_id, MAX(id) AS max_id FROM labels WHERE source = 'manual' GROUP BY tweet_id) m
  ON l.id = m.max_id;

CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    tweet_id      TEXT NOT NULL REFERENCES tweets(id) ON DELETE CASCADE,
    model_version TEXT NOT NULL,
    score         REAL NOT NULL,                   -- calibrated P(ai)
    created_at    TEXT NOT NULL,
    UNIQUE (tweet_id, model_version)
);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_version, score);

-- Singleton row (id = 1).
CREATE TABLE IF NOT EXISTS scraper_state (
    id                 INTEGER PRIMARY KEY CHECK (id = 1),
    status             TEXT NOT NULL CHECK (status IN ('idle', 'running', 'paused', 'error')),
    reason             TEXT,
    updated_at         TEXT NOT NULL,
    last_run_at        TEXT,
    next_run_at        TEXT,
    worker_heartbeat_at TEXT,
    session_file_ok    INTEGER
);
INSERT OR IGNORE INTO scraper_state (id, status, updated_at) VALUES (1, 'idle', '1970-01-01T00:00:00+00:00');

CREATE TABLE IF NOT EXISTS scrape_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    timeline      TEXT,
    status        TEXT NOT NULL,                   -- running | ok | paused | error
    reason        TEXT,
    tweets_seen   INTEGER NOT NULL DEFAULT 0,
    tweets_new    INTEGER NOT NULL DEFAULT 0,
    scrolls       INTEGER NOT NULL DEFAULT 0,
    trigger       TEXT                             -- schedule | manual | cli
);

-- Work requested from the web UI and executed by the worker (web never runs the browser or the model).
CREATE TABLE IF NOT EXISTS jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL CHECK (kind IN ('scrape', 'classify', 'reclassify', 'features')),
    status      TEXT NOT NULL CHECK (status IN ('queued', 'running', 'done', 'failed')),
    created_at  TEXT NOT NULL,
    started_at  TEXT,
    finished_at TEXT,
    message     TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, id);
