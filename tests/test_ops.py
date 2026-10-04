import tarfile

import pytest

from tl.db import session
from tl.db import queries as Q
from tl.ops import backup, restore
from tl.scraper.extract import extract_page


def test_backup_restore_roundtrip(settings, fixture_html, tmp_path):
    with session(settings.db_path) as conn:
        Q.upsert_tweets(conn, extract_page(fixture_html("timeline_following.html")))
    ck = settings.models_dir / "v1"
    ck.mkdir(parents=True)
    (ck / "COMPLETE").write_text("")
    out = tmp_path / "b.tar.gz"
    backup(settings, out)
    with session(settings.db_path) as conn:
        conn.execute("DELETE FROM tweets")
    (ck / "COMPLETE").unlink()
    restore(settings, out)
    with session(settings.db_path) as conn:
        assert Q.counts(conn)["tweets"] == 6
    assert (ck / "COMPLETE").exists()


def test_restore_rejects_path_traversal(settings, tmp_path):
    evil = tmp_path / "evil.tar.gz"
    payload = tmp_path / "p"
    payload.write_text("x")
    with tarfile.open(evil, "w:gz") as tar:
        tar.add(payload, arcname="../../etc/cron.d/x")
    with pytest.raises(ValueError, match="suspicious"):
        restore(settings, evil)
