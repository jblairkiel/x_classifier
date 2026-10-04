"""SQLite access: connection setup + schema bootstrap. Queries live in tl.db.queries."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Iterator

SCHEMA_VERSION = 1

_init_lock = threading.Lock()
_initialized: set[str] = set()


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _schema_sql() -> str:
    return resources.files("tl.db").joinpath("schema.sql").read_text(encoding="utf-8")


def init_db(path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(_schema_sql())
        if conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 0:
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        conn.commit()
    finally:
        conn.close()


def connect(path: Path | str | None = None, *, readonly: bool = False) -> sqlite3.Connection:
    """Open a connection, creating/migrating the schema on first use per process."""
    if path is None:
        from tl.config import get_settings

        path = get_settings().db_path
    key = str(Path(path).resolve())
    if not readonly:
        with _init_lock:
            if key not in _initialized:
                init_db(path)
                _initialized.add(key)
    if readonly:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
    else:
        conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


@contextmanager
def session(path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    """Connection that commits on success, rolls back on error, always closes."""
    conn = connect(path)
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()
