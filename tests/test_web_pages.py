import pytest

from tl.db import session
from tl.db import queries as Q
from tl.scraper.extract import extract_page
from tests.test_web_auth import csrf_from, login


@pytest.fixture
def authed(app_settings, fixture_html):
    from fastapi.testclient import TestClient

    from tl.web.app import create_app

    with session(app_settings.db_path) as conn:
        Q.upsert_tweets(conn, extract_page(fixture_html("timeline_following.html")), timeline="following")
    c = TestClient(create_app(app_settings), base_url="https://testserver", follow_redirects=False)
    assert login(c).status_code == 303
    return c


@pytest.mark.parametrize("path", ["/", "/timeline", "/timeline?min_score=0.5&handle=@carol&label=unlabeled",
                                  "/accounts", "/accounts/alice_dev", "/label", "/label?show_score=1", "/model"])
def test_pages_render(authed, path):
    r = authed.get(path)
    assert r.status_code == 200, r.text[:500]
    assert "<html" in r.text


def test_unknown_account_404(authed):
    assert authed.get("/accounts/nobody").status_code == 404


def test_timeline_shows_tweets_unscored_without_model(authed):
    html = authed.get("/timeline").text
    assert "No trained model yet" in html
    assert "lol no. the bus was 40 min late AGAIN" in html


def test_labeling_flow_htmx(authed, app_settings):
    page = authed.get("/label").text
    token = csrf_from(page)
    import re

    tid = re.search(r'name="tweet_id" value="(\d+)"', page).group(1)
    r = authed.post("/label", data={"tweet_id": tid, "label": "ai", "csrf_token": token},
                    headers={"HX-Request": "true"})
    assert r.status_code == 200 and "Labelled ai" in r.text
    next_tid = re.search(r'name="tweet_id" value="(\d+)"', r.text).group(1)
    assert next_tid != tid
    # skip does not label
    r = authed.post("/label", data={"tweet_id": next_tid, "label": "skip", "csrf_token": token},
                    headers={"HX-Request": "true"})
    assert next_tid not in re.search(r'name="tweet_id" value="(\d+)"', r.text).group(1)
    with session(app_settings.db_path) as conn:
        assert Q.current_label(conn, tid) == "ai"
        assert Q.current_label(conn, next_tid) is None


def test_label_without_js_redirects(authed):
    page = authed.get("/label").text
    import re

    tid = re.search(r'name="tweet_id" value="(\d+)"', page).group(1)
    r = authed.post("/label", data={"tweet_id": tid, "label": "human", "csrf_token": csrf_from(page)})
    assert r.status_code == 303 and r.headers["location"].startswith("/label?")


def test_relabel_button(authed, app_settings):
    token = csrf_from(authed.get("/timeline").text)
    r = authed.post("/tweets/1842000000000000003/label", data={"label": "human", "csrf_token": token},
                    headers={"HX-Request": "true"})
    assert r.status_code == 200 and 'aria-pressed="true"' in r.text
    authed.post("/tweets/1842000000000000003/label", data={"label": "unsure", "csrf_token": token},
                headers={"HX-Request": "true"})
    with session(app_settings.db_path) as conn:
        assert Q.current_label(conn, "1842000000000000003") == "unsure"
    assert authed.post("/tweets/1842000000000000003/label", data={"label": "bogus", "csrf_token": token}
                       ).status_code == 400


def test_pause_shows_banner_and_jobs_enqueue(authed, app_settings):
    token = csrf_from(authed.get("/").text)
    authed.post("/scraper/pause", data={"csrf_token": token})
    assert "Scraper paused" in authed.get("/timeline").text
    authed.post("/jobs/features", data={"csrf_token": token})
    authed.post("/jobs/features", data={"csrf_token": token})  # deduped while queued
    with session(app_settings.db_path) as conn:
        assert [j["kind"] for j in Q.recent_jobs(conn)] == ["features"]
    assert authed.post("/jobs/rm-rf", data={"csrf_token": token}).status_code == 404
