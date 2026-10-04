from datetime import datetime, timedelta, timezone

import pytest

from tl.db import session
from tl.db import queries as Q
from tl.features import PRIOR, account_features, update_all_accounts
from tl.scraper.extract import extract_page


def tw(text, minutes, reply=False):
    t0 = datetime(2026, 10, 1, tzinfo=timezone.utc)
    return {"text": text, "created_at": (t0 + timedelta(minutes=minutes)).isoformat(), "is_reply": reply}


HUMAN = [tw("lol the bus was late again", 0), tw("anyone know a good plumber in austin? asking for a friend (me)", 95),
         tw("ok this sandwich is elite", 400), tw("why is my cat like this", 1500),
         tw("finally finished that book, the ending was wild and I need to talk about it with someone", 3000)]
BOT = [tw(f"In today's fast-paced world — consistency is key. Keep growing! #growth #mindset #success {i}", 60 * i)
       for i in range(12)]


def test_botlike_ranks_templated_regular_account_higher():
    human = account_features(HUMAN, [0.2, 0.3, 0.1, 0.2, 0.25])
    bot = account_features(BOT, [0.85] * 12)
    assert bot["botlike_score"] > 0.6 > 0.3 > human["botlike_score"]
    s = bot["signals"]
    assert s["near_duplicates"]["score"] == 1.0
    assert s["cadence"]["score"] == 1.0  # exactly hourly
    assert s["em_dash_rate"]["value"] == 1.0 and s["hashtag_rate"]["value"] == 3
    assert s["length_variance"]["score"] > 0.8
    assert bot["confidence"] == "high"


def test_few_tweets_shrink_toward_prior():
    one = account_features([tw("Embracing growth — every day! #journey", 0)], [0.95])
    assert one["confidence"] == "low"
    assert PRIOR < one["botlike_score"] < one["raw_score"]


def test_no_model_scores_uses_heuristics_only():
    f = account_features(BOT, [])
    assert f["signals"]["text_score"]["score"] is None and f["botlike_score"] is not None


def test_reply_heavy_account():
    f = account_features([tw(f"@x totally agree with this take number {i}", i * 37, reply=True) for i in range(8)], [])
    assert f["signals"]["reply_ratio"]["score"] == 1.0


def test_update_all_accounts_writes_scores(settings, fixture_html):
    with session(settings.db_path) as conn:
        Q.upsert_tweets(conn, extract_page(fixture_html("timeline_following.html")))
    assert update_all_accounts(settings) == 6
    with session(settings.db_path) as conn:
        acct = Q.get_account(conn, "bobbuilds")
        assert acct["botlike_score"] is not None and acct["features_json"]
        ranked = Q.accounts_ranked(conn, None)
        assert ranked[0]["botlike_score"] >= ranked[-1]["botlike_score"]
