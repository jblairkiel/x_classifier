from datetime import datetime, timedelta, timezone

from tl.db import session
from tl.db import queries as Q
from tl.scraper.extract import extract_page


def test_dedupe_within_batch_and_across_runs(settings, fixture_html):
    records = extract_page(fixture_html("timeline_following.html"))
    assert len(records) == 7  # fixture renders alice's tweet twice
    with session(settings.db_path) as conn:
        seen, new = Q.upsert_tweets(conn, records, timeline="following")
        assert (seen, new) == (6, 6)
        seen, new = Q.upsert_tweets(conn, records, timeline="following")
        assert (seen, new) == (6, 0)
        assert conn.execute("SELECT COUNT(*) FROM tweets").fetchone()[0] == 6
        row = Q.get_tweet(conn, "1842000000000000001")
        assert row["seen_count"] == 2
        assert conn.execute("SELECT COUNT(*) FROM accounts").fetchone()[0] == 6


def test_upsert_refreshes_counts_but_keeps_text(settings, fixture_html):
    rec = extract_page(fixture_html("timeline_following.html"))[0]
    with session(settings.db_path) as conn:
        Q.upsert_tweets(conn, [rec])
        rec.like_count = 999
        rec.text = "edited"
        Q.upsert_tweets(conn, [rec])
        row = Q.get_tweet(conn, rec.id)
        assert row["like_count"] == 999 and row["text"] != "edited"


def test_daily_cap_counter(settings):
    with session(settings.db_path) as conn:
        rid = Q.start_run(conn, "following", "cli")
        Q.finish_run(conn, rid, "ok", None, 40, 30, 5)
        assert Q.tweets_seen_last_24h(conn) == 40
        assert Q.tweets_seen_since(conn, datetime.now(timezone.utc) + timedelta(minutes=1)) == 0


def test_manual_labels_append_and_current_view(settings, fixture_html):
    recs = extract_page(fixture_html("timeline_following.html"))
    with session(settings.db_path) as conn:
        Q.upsert_tweets(conn, recs)
        Q.add_manual_label(conn, recs[0].id, "ai")
        Q.add_manual_label(conn, recs[0].id, "human")  # relabel
        assert Q.current_label(conn, recs[0].id) == "human"
        assert conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == 2
        assert [r["label"] for r in Q.manual_eval_set(conn)] == ["human"]
