"""Backup / restore of the SQLite DB and model checkpoints as one .tar.gz (file or stdout/stdin)."""

from __future__ import annotations

import io
import shutil
import sqlite3
import sys
import tarfile
import tempfile
from pathlib import Path

from tl.config import Settings

DB_MEMBER = "tl.sqlite3"
MODELS_PREFIX = "models/"


def backup(settings: Settings, out: str | Path, *, include_models: bool = True) -> str:
    """Write a consistent snapshot (sqlite online-backup API, safe while web/worker run)."""
    with tempfile.TemporaryDirectory() as tmp:
        snap = Path(tmp) / DB_MEMBER
        src = sqlite3.connect(settings.db_path)
        dst = sqlite3.connect(snap)
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()

        target = sys.stdout.buffer if str(out) == "-" else open(out, "wb")
        try:
            with tarfile.open(fileobj=target, mode="w|gz") as tar:
                tar.add(snap, arcname=DB_MEMBER)
                n_models = 0
                if include_models and settings.models_dir.is_dir():
                    for d in sorted(settings.models_dir.iterdir()):
                        if d.is_dir():
                            tar.add(d, arcname=MODELS_PREFIX + d.name)
                            n_models += 1
        finally:
            if target is not sys.stdout.buffer:
                target.close()
    return f"backup written to {out} (db + {n_models} checkpoints)"


def _safe_members(tar: tarfile.TarFile):
    for m in tar:
        name = m.name
        if name.startswith("/") or ".." in Path(name).parts or not (m.isfile() or m.isdir()):
            raise ValueError(f"refusing suspicious archive member {name!r}")
        if name != DB_MEMBER and not name.startswith(MODELS_PREFIX):
            raise ValueError(f"unexpected archive member {name!r}")
        yield m


def restore(settings: Settings, src: str | Path) -> str:
    """Replace the DB (and any checkpoints contained in the archive). Stop web+worker first."""
    data = sys.stdin.buffer.read() if str(src) == "-" else Path(src).read_bytes()
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            tar.extractall(tmpdir, members=_safe_members(tar), filter="data")
        db = tmpdir / DB_MEMBER
        if not db.is_file():
            raise ValueError("archive has no database")
        with sqlite3.connect(db) as check:
            if check.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("database in archive failed integrity_check")

        settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("-wal", "-shm"):
            Path(str(settings.db_path) + suffix).unlink(missing_ok=True)
        shutil.copyfile(db, settings.db_path)

        restored = []
        models_src = tmpdir / "models"
        if models_src.is_dir():
            settings.models_dir.mkdir(parents=True, exist_ok=True)
            for d in models_src.iterdir():
                dest = settings.models_dir / d.name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(d, dest)
                restored.append(d.name)
    return f"restored database and {len(restored)} checkpoints {restored}"
