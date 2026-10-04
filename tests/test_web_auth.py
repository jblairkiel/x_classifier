"""Auth, CSRF, rate limiting and header tests. Unauthenticated requests must be rejected."""

import re
import time

import pytest
from fastapi.testclient import TestClient

from tl.web.auth import hash_password, safe_next

PASSWORD = "correct horse battery staple"
PW_HASH = hash_password(PASSWORD)
PUBLIC = {"/healthz", "/login", "/robots.txt"}


@pytest.fixture
def app_settings(settings):
    return settings.__class__(**{
        **settings.__dict__,
        "admin_password_hash": PW_HASH,
        "session_secret": "x" * 48,
        "login_max_attempts": 3,
        "login_global_max_attempts": 100,
    })


@pytest.fixture
def client(app_settings):
    from tl.web.app import create_app

    return TestClient(create_app(app_settings), base_url="https://testserver", follow_redirects=False)


def csrf_from(html: str) -> str:
    return re.search(r'name="csrf_token" value="([^"]+)"', html).group(1)


def login(client, password=PASSWORD, username="admin"):
    token = csrf_from(client.get("/login").text)
    return client.post("/login", data={"username": username, "password": password, "csrf_token": token,
                                       "next": "/"})


@pytest.fixture
def authed(client):
    assert login(client).status_code == 303
    return client


def all_routes(app):
    for r in app.routes:
        methods = getattr(r, "methods", None) or {"GET"}
        path = r.path.replace("{handle}", "alice").replace("{tweet_id}", "1").replace("{kind}", "scrape")
        if path.startswith("/static"):
            continue
        for m in methods:
            if m != "HEAD":
                yield m, path


def test_every_non_public_route_rejects_anonymous(client):
    checked = 0
    for method, path in all_routes(client.app):
        if path in PUBLIC:
            continue
        resp = client.request(method, path)
        checked += 1
        if method == "GET":
            assert resp.status_code == 303, (method, path)
            assert resp.headers["location"].startswith("/login?next="), path
        else:
            assert resp.status_code == 401, (method, path)
    assert checked >= 10


def test_htmx_requests_get_401_not_redirect(client):
    r = client.get("/timeline", headers={"HX-Request": "true"})
    assert r.status_code == 401 and r.headers["hx-redirect"] == "/login"


def test_public_routes(client):
    assert client.get("/healthz").json() == {"ok": True}
    r = client.get("/robots.txt")
    assert r.text == "User-agent: *\nDisallow: /\n"
    assert client.get("/login").status_code == 200


def test_noindex_and_security_headers_everywhere(client):
    for path in ["/healthz", "/login", "/robots.txt", "/", "/static/app.css", "/nope"]:
        r = client.get(path)
        assert "noindex" in r.headers["x-robots-tag"], path
        assert r.headers["x-frame-options"] == "DENY"
        assert "default-src 'self'" in r.headers["content-security-policy"]
        assert r.headers["strict-transport-security"].startswith("max-age=")


def test_static_serves_only_assets(client):
    assert client.get("/static/app.css").status_code == 200
    for path in ["/static/../../db/schema.sql", "/static/%2e%2e/%2e%2e/config.py", "/static/tl.sqlite3",
                 "/data/tl.sqlite3", "/secrets/x_storage_state.json", "/models/"]:
        r = client.get(path)
        assert r.status_code in (303, 404), path
        assert "CREATE TABLE" not in r.text and "SESSION_SECRET" not in r.text


def test_login_success_sets_hardened_cookie(client):
    r = login(client)
    assert r.status_code == 303 and r.headers["location"] == "/"
    cookie = r.headers["set-cookie"].lower()
    assert "tl_session=" in cookie and "httponly" in cookie and "secure" in cookie and "samesite=lax" in cookie
    assert client.get("/").status_code == 200


def test_wrong_password_and_wrong_user(client):
    assert login(client, password="nope").status_code == 401
    assert login(client, username="root").status_code == 401
    assert client.get("/").status_code == 303


def test_login_requires_csrf(client):
    client.get("/login")
    r = client.post("/login", data={"username": "admin", "password": PASSWORD})
    assert r.status_code == 403


def test_login_rate_limited_per_ip(client):
    for _ in range(3):
        assert login(client, password="bad").status_code == 401
    r = login(client)  # even the right password is refused while limited
    assert r.status_code == 429 and "Too many" in r.text


def test_post_requires_csrf_when_logged_in(authed):
    assert authed.post("/scraper/pause").status_code == 403
    assert authed.post("/scraper/pause", data={"csrf_token": "forged"}).status_code == 403
    token = csrf_from(authed.get("/").text)
    assert authed.post("/scraper/pause", data={"csrf_token": token}).status_code == 303
    assert authed.post("/scraper/resume", headers={"X-CSRF-Token": token}).status_code == 303


def test_cross_origin_post_rejected(authed):
    token = csrf_from(authed.get("/").text)
    r = authed.post("/scraper/pause", data={"csrf_token": token}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403


def test_logout_clears_session(authed):
    token = csrf_from(authed.get("/").text)
    assert authed.post("/logout", data={"csrf_token": token}).status_code == 303
    assert authed.get("/").status_code == 303


def test_session_expires(app_settings, monkeypatch):
    from tl.web.app import create_app

    c = TestClient(create_app(app_settings), base_url="https://testserver", follow_redirects=False)
    login(c)
    assert c.get("/").status_code == 200
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() + app_settings.session_max_age_hours * 3600 + 5)
    assert c.get("/").status_code == 303


def test_forged_cookie_rejected(client):
    client.cookies.set("tl_session", "eyJ1c2VyIjogImFkbWluIn0=.forged.sig")
    assert client.get("/").status_code == 303


@pytest.mark.parametrize("target,expected", [("/timeline?x=1", "/timeline?x=1"), ("//evil.com", "/"),
                                             ("https://evil.com", "/"), ("/\\evil.com", "/"), (None, "/")])
def test_safe_next(target, expected):
    assert safe_next(target) == expected


def test_app_refuses_to_start_without_secrets(settings):
    from tl.web.app import create_app

    with pytest.raises(RuntimeError):
        create_app(settings)  # empty hash + secret


def test_referrer_policy_keeps_same_origin_header(client):
    # "no-referrer" makes Chrome send `Origin: null` on same-site form POSTs, which the
    # origin check (correctly) rejects -- so logins from a real browser would fail.
    assert client.get("/login").headers["referrer-policy"] == "same-origin"


def test_same_origin_post_with_origin_header_allowed(client):
    token = csrf_from(client.get("/login").text)
    r = client.post("/login", data={"username": "admin", "password": PASSWORD, "csrf_token": token},
                    headers={"Origin": "https://testserver"})
    assert r.status_code == 303
    r = client.post("/login", data={"csrf_token": token}, headers={"Origin": "null"})
    assert r.status_code == 403
