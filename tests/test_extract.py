import pytest

from tl.scraper.extract import SelectorError, detect_blocker, extract_articles, extract_page, parse_count


@pytest.mark.parametrize(
    "raw,expected",
    [("12", 12), ("1,234", 1234), ("1.2K", 1200), ("48.2K", 48200), ("3M", 3_000_000), ("", 0),
     (None, None), ("abc", None)],
)
def test_parse_count(raw, expected):
    assert parse_count(raw) == expected


def by_id(records):
    return {r.id: r for r in records}


def test_extracts_core_fields(fixture_html):
    recs = by_id(extract_page(fixture_html("timeline_following.html")))
    alice = recs["1842000000000000001"]
    assert alice.author_handle == "alice_dev"
    assert alice.display_name == "Alice 🌱"
    assert alice.text.endswith("🎉 it works.")  # emoji <img alt> preserved
    assert alice.created_at == "2026-10-03T14:02:11.000Z"
    assert (alice.reply_count, alice.repost_count, alice.like_count) == (3, 12, 45)
    assert not alice.is_reply and not alice.is_repost and not alice.has_media
    assert alice.lang == "en"


def test_handle_is_lowercased_and_hashtags_kept(fixture_html):
    bob = by_id(extract_page(fixture_html("timeline_following.html")))["1842000000000000002"]
    assert bob.author_handle == "bobbuilds"
    assert "#motivation #growth" in bob.text
    assert "—" in bob.text


def test_reply_flag(fixture_html):
    carol = by_id(extract_page(fixture_html("timeline_following.html")))["1842000000000000003"]
    assert carol.is_reply
    assert carol.text == "lol no. the bus was 40 min late AGAIN"  # "Replying to" not in text


def test_repost_flag_media_and_abbreviated_counts(fixture_html):
    news = by_id(extract_page(fixture_html("timeline_following.html")))["1842000000000000004"]
    assert news.is_repost and news.reposted_by == "Alice 🌱"
    assert news.author_handle == "newsdesk"  # original author, not the reposter
    assert news.has_media
    assert news.repost_count == 12_500 and news.like_count == 48_200


def test_promoted_tweets_are_skipped(fixture_html):
    recs = by_id(extract_page(fixture_html("timeline_following.html")))
    assert "1842000000000000005" not in recs


def test_quote_tweet_text_not_mixed_in(fixture_html):
    dave = by_id(extract_page(fixture_html("timeline_following.html")))["1842000000000000006"]
    assert dave.text == "this take is wrong and here's why"
    assert dave.like_count == 1  # read from the "unlike" button (already liked by someone else's session)


def test_media_only_tweet_has_empty_text(fixture_html):
    vid = by_id(extract_page(fixture_html("timeline_following.html")))["1842000000000000007"]
    assert vid.text == "" and vid.has_media


def test_dom_drift_fails_loudly(fixture_html):
    with pytest.raises(SelectorError, match="selectors.py"):
        extract_page(fixture_html("timeline_dom_changed.html"))


def test_page_without_articles_fails_loudly():
    with pytest.raises(SelectorError):
        extract_page("<html><body><main>nothing</main></body></html>")


def test_occasional_bad_article_is_tolerated(fixture_html):
    good = fixture_html("timeline_following.html")
    from bs4 import BeautifulSoup

    arts = [str(a) for a in BeautifulSoup(good, "html.parser").select('article[data-testid="tweet"]')]
    recs, promoted = extract_articles(arts + ['<article data-testid="tweet"><div>broken</div></article>'])
    assert promoted == 1 and len(recs) == 7


@pytest.mark.parametrize(
    "name,kind",
    [("blocker_login_wall.html", "login_wall"), ("blocker_captcha.html", "captcha"),
     ("blocker_rate_limit.html", "rate_limit"), ("blocker_suspended.html", "suspended")],
)
def test_blockers_detected(fixture_html, name, kind):
    b = detect_blocker("https://x.com/home", fixture_html(name))
    assert b is not None and b.kind == kind


@pytest.mark.parametrize(
    "url,kind",
    [("https://x.com/i/flow/login?redirect_after_login=%2Fhome", "login_wall"),
     ("https://x.com/login", "login_wall"),
     ("https://x.com/account/access", "captcha")],
)
def test_blocker_by_url(url, kind):
    assert detect_blocker(url, "<html></html>").kind == kind


def test_normal_timeline_is_not_a_blocker(fixture_html):
    assert detect_blocker("https://x.com/home", fixture_html("timeline_following.html")) is None


def test_tweet_text_cannot_trigger_a_stop(fixture_html):
    html = fixture_html("not_blocker_tweet_mentions_rate_limit.html")
    assert detect_blocker("https://x.com/home", html) is None
