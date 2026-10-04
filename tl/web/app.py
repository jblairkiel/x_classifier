"""FastAPI web UI. Everything requires the admin session except /healthz, /login, /robots.txt
and the CSS/JS under /static (which contain no data)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Iterator
from urllib.parse import quote, urlencode

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from tl.config import Settings, get_settings
from tl.db import connect
from tl.db import queries as Q
from tl.model.checkpoints import list_versions, resolve_version_or_none
from tl.web import auth

PUBLIC_PATHS = {"/healthz", "/login", "/robots.txt"}
PUBLIC_PREFIXES = ("/static/",)
UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
ROBOTS_TXT = "User-agent: *\nDisallow: /\n"

SECURITY_HEADERS = {
    b"x-robots-tag": b"noindex, nofollow, noarchive, nosnippet",
    b"x-content-type-options": b"nosniff",
    b"x-frame-options": b"DENY",
    b"referrer-policy": b"same-origin",
    b"cross-origin-opener-policy": b"same-origin",
    b"permissions-policy": b"camera=(), microphone=(), geolocation=(), interest-cohort=()",
    b"content-security-policy": (
        b"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        b"connect-src 'self'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'; object-src 'none'"
    ),
}


class SecurityHeadersMiddleware:
    """Adds security headers (incl. X-Robots-Tag: noindex) to every response."""

    def __init__(self, app: ASGIApp, *, hsts: bool):
        self.app = app
        self.hsts = hsts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = [(k, v) for k, v in message.get("headers", []) if k.lower() not in SECURITY_HEADERS]
                headers += list(SECURITY_HEADERS.items())
                if self.hsts:
                    headers.append((b"strict-transport-security", b"max-age=31536000; includeSubDomains"))
                path = scope.get("path", "")
                if not path.startswith("/static/"):
                    headers.append((b"cache-control", b"no-store"))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)


class AuthGateMiddleware:
    """Default-deny: any path not explicitly public needs a valid, unexpired admin session."""

    def __init__(self, app: ASGIApp, *, max_age_seconds: int):
        self.app = app
        self.max_age = max_age_seconds

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope["path"]
        if path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES):
            return await self.app(scope, receive, send)
        session = scope.get("session") or {}
        login_at = session.get(auth.SESSION_LOGIN_AT, 0)
        if session.get(auth.SESSION_USER) and time.time() - login_at < self.max_age:
            return await self.app(scope, receive, send)
        session.clear() if isinstance(session, dict) else None

        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        if scope["method"] == "GET" and "hx-request" not in headers:
            qs = scope.get("query_string", b"").decode()
            nxt = path + (f"?{qs}" if qs else "")
            response: Response = RedirectResponse(f"/login?next={quote(nxt)}", status_code=303)
        else:
            response = JSONResponse({"detail": "authentication required"}, status_code=401,
                                    headers={"HX-Redirect": "/login"})
        await response(scope, receive, send)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    if not settings.admin_password_hash.startswith("$argon2"):
        raise RuntimeError("ADMIN_PASSWORD_HASH must be an argon2 hash (generate with `tl hash-password`)")
    if len(settings.session_secret) < 32:
        raise RuntimeError("SESSION_SECRET must be at least 32 characters")

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    max_age = settings.session_max_age_hours * 3600
    # Order: last added = outermost. Security headers wrap everything (incl. 401/redirects).
    app.add_middleware(AuthGateMiddleware, max_age_seconds=max_age)
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.session_secret,
        session_cookie="tl_session",
        max_age=max_age,
        same_site="lax",
        https_only=settings.cookie_secure,  # Secure flag; HttpOnly is always set by Starlette
    )
    app.add_middleware(SecurityHeadersMiddleware, hsts=settings.cookie_secure)

    static_dir = Path(str(resources.files("tl.web").joinpath("static")))
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    templates = Jinja2Templates(directory=str(resources.files("tl.web").joinpath("templates")))
    templates.env.filters["dt"] = _fmt_dt
    templates.env.filters["pct"] = lambda v: "–" if v is None else f"{v * 100:.0f}%"
    templates.env.filters["num"] = lambda v, d=3: "–" if v is None else f"{v:.{d}f}"
    templates.env.filters["fromjson"] = lambda s: json.loads(s) if s else {}

    limiter = auth.LoginRateLimiter(
        settings.login_max_attempts, settings.login_global_max_attempts, settings.login_window_minutes * 60
    )
    app.state.settings = settings
    app.state.limiter = limiter

    # ------------------------------------------------------------------ dependencies

    def db() -> Iterator:
        conn = connect(settings.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    async def csrf_protect(request: Request) -> None:
        if request.method not in UNSAFE_METHODS:
            return
        origin = request.headers.get("origin")
        if origin and origin != f"{request.url.scheme}://{request.url.netloc}":
            raise HTTPException(403, "cross-origin request rejected")
        submitted = request.headers.get(auth.CSRF_HEADER)
        if not submitted:
            form = await request.form()
            submitted = form.get(auth.CSRF_FORM_FIELD)
        if not auth.csrf_valid(request.session, submitted if isinstance(submitted, str) else None):
            raise HTTPException(403, "invalid or missing CSRF token")

    app.router.dependencies.append(Depends(csrf_protect))

    def is_htmx(request: Request) -> bool:
        return request.headers.get("hx-request") == "true"

    def render(request: Request, name: str, conn=None, status_code: int = 200, **ctx) -> HTMLResponse:
        base = {
            "csrf": auth.csrf_token(request.session),
            "user": request.session.get(auth.SESSION_USER),
            "settings": settings,
            "path": request.url.path,
        }
        if conn is not None and base["user"]:
            base["state"] = Q.get_state(conn)
        base.update(ctx)
        return templates.TemplateResponse(request, name, base, status_code=status_code)

    def back(request: Request, default: str = "/") -> RedirectResponse:
        ref = request.headers.get("referer") or ""
        own = f"{request.url.scheme}://{request.url.netloc}"
        target = ref[len(own):] if ref.startswith(own + "/") else default
        return RedirectResponse(auth.safe_next(target), status_code=303)

    def active_version() -> str | None:
        return resolve_version_or_none(settings)

    # ------------------------------------------------------------------ public

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict:
        return {"ok": True}

    @app.get("/robots.txt", include_in_schema=False)
    def robots() -> PlainTextResponse:
        return PlainTextResponse(ROBOTS_TXT)

    @app.get("/login")
    def login_form(request: Request, next: str = "/"):
        if request.session.get(auth.SESSION_USER):
            return RedirectResponse(auth.safe_next(next), status_code=303)
        return render(request, "login.html", next=auth.safe_next(next), error=None)

    @app.post("/login")
    def login_submit(
        request: Request,
        username: str = Form(..., max_length=128),
        password: str = Form(..., max_length=1024),
        next: str = Form("/"),
    ):
        ip = request.client.host if request.client else "unknown"
        wait = limiter.retry_after(ip)
        if wait:
            return render(request, "login.html", next=auth.safe_next(next), status_code=429,
                          error=f"Too many failed attempts. Try again in {wait // 60 + 1} minutes.")
        if not auth.verify_credentials(username, password, expected_user=settings.admin_username,
                                       password_hash=settings.admin_password_hash):
            limiter.record_failure(ip)
            return render(request, "login.html", next=auth.safe_next(next), status_code=401,
                          error="Invalid username or password.")
        limiter.record_success(ip)
        request.session.clear()  # new session (and new CSRF token) on privilege change
        request.session[auth.SESSION_USER] = settings.admin_username
        request.session[auth.SESSION_LOGIN_AT] = int(time.time())
        auth.csrf_token(request.session)
        return RedirectResponse(auth.safe_next(next), status_code=303)

    @app.post("/logout")
    def logout(request: Request):
        request.session.clear()
        return RedirectResponse("/login", status_code=303)

    # ------------------------------------------------------------------ dashboard / scraper

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request, conn=Depends(db)):
        version = active_version()
        return render(
            request, "dashboard.html", conn,
            counts=Q.counts(conn),
            seen_24h=Q.tweets_seen_last_24h(conn),
            runs=Q.recent_runs(conn, 8),
            jobs=Q.recent_jobs(conn, 8),
            version=version,
            flagged=Q.timeline_page(conn, model_version=version, min_score=settings.ai_threshold, limit=5)
            if version else [],
            top_accounts=[a for a in Q.accounts_ranked(conn, version, limit=5) if a["botlike_score"] is not None],
            now=datetime.now(timezone.utc),
        )

    @app.post("/scraper/pause")
    def scraper_pause(request: Request, conn=Depends(db)):
        Q.set_state(conn, "paused", "paused from the web UI")
        return back(request)

    @app.post("/scraper/resume")
    def scraper_resume(request: Request, conn=Depends(db)):
        prev = Q.get_state(conn)
        Q.set_state(conn, "idle", f"resumed from {prev['status']} via web UI")
        return back(request)

    @app.post("/jobs/{kind}")
    def enqueue(kind: str, request: Request, conn=Depends(db)):
        if kind not in ("scrape", "classify", "reclassify", "features"):
            raise HTTPException(404)
        Q.enqueue_job(conn, kind)
        return back(request)

    # ------------------------------------------------------------------ timeline

    @app.get("/timeline", response_class=HTMLResponse)
    def timeline(
        request: Request, conn=Depends(db), min_score: float | None = None, handle: str = "",
        date_from: str = "", date_to: str = "", label: str = "", page: int = 1,
    ):
        version = active_version()
        if min_score is None:
            min_score = settings.ai_threshold if version else 0.0
        min_score = max(0.0, min(1.0, min_score))
        page = max(1, page)
        per_page = 50
        rows = Q.timeline_page(
            conn, model_version=version, min_score=min_score if version else -1, handle=handle or None,
            date_from=_date_or_none(date_from), date_to=_date_or_none(date_to),
            label=label if label in ("human", "ai", "unsure", "unlabeled") else None,
            limit=per_page + 1, offset=(page - 1) * per_page,
        )
        filters = {"min_score": min_score, "handle": handle, "date_from": date_from, "date_to": date_to,
                   "label": label}
        return render(
            request, "timeline.html", conn, tweets=rows[:per_page], version=version, filters=filters,
            page=page, has_next=len(rows) > per_page,
            next_qs=urlencode({**filters, "page": page + 1}), prev_qs=urlencode({**filters, "page": page - 1}),
        )

    # ------------------------------------------------------------------ accounts

    @app.get("/accounts", response_class=HTMLResponse)
    def accounts(request: Request, conn=Depends(db)):
        version = active_version()
        return render(request, "accounts.html", conn, accounts=Q.accounts_ranked(conn, version), version=version)

    @app.get("/accounts/{handle}", response_class=HTMLResponse)
    def account_detail(handle: str, request: Request, conn=Depends(db)):
        acct = Q.get_account(conn, handle.lower())
        if acct is None:
            raise HTTPException(404)
        version = active_version()
        return render(
            request, "account.html", conn, account=acct,
            features=json.loads(acct["features_json"]) if acct["features_json"] else None,
            tweets=Q.account_tweets_with_scores(conn, acct["handle"], version), version=version,
        )

    # ------------------------------------------------------------------ labeling

    def _label_card(request: Request, conn, skip: list[str], show_score: bool, flash: str | None = None):
        version = active_version()
        tweet = Q.next_unlabeled(conn, version, skip[-200:])
        return render(
            request, "_label_card.html", conn, tweet=tweet, skip=",".join(skip[-200:]), show_score=show_score,
            counts=Q.counts(conn), label_counts=Q.label_counts(conn).get("manual", {}), flash=flash,
        )

    @app.get("/label", response_class=HTMLResponse)
    def label_page(request: Request, conn=Depends(db), skip: str = "", show_score: int = 0):
        skip_ids = [s for s in skip.split(",") if s.isdigit()]
        card = _label_card(request, conn, skip_ids, bool(show_score)).body.decode()
        return render(request, "label.html", conn, card=card, show_score=bool(show_score))

    @app.post("/label", response_class=HTMLResponse)
    def label_submit(
        request: Request, conn=Depends(db), tweet_id: str = Form(...), label: str = Form(...),
        skip: str = Form(""), show_score: int = Form(0),
    ):
        skip_ids = [s for s in skip.split(",") if s.isdigit()]
        flash = None
        if label == "skip":
            skip_ids.append(tweet_id)
        else:
            try:
                Q.add_manual_label(conn, tweet_id, label)
                flash = f"Labelled {label}"
            except (ValueError, KeyError):
                raise HTTPException(400, "bad label or tweet")
        if is_htmx(request):
            return _label_card(request, conn, skip_ids, bool(show_score), flash)
        qs = urlencode({"skip": ",".join(skip_ids[-200:]), "show_score": show_score})
        return RedirectResponse(f"/label?{qs}", status_code=303)

    @app.post("/tweets/{tweet_id}/label", response_class=HTMLResponse)
    def relabel(tweet_id: str, request: Request, conn=Depends(db), label: str = Form(...)):
        try:
            Q.add_manual_label(conn, tweet_id, label)
        except (ValueError, KeyError):
            raise HTTPException(400, "bad label or tweet")
        if is_htmx(request):
            return render(request, "_label_controls.html", tweet={"id": tweet_id, "manual_label": label})
        return back(request)

    # ------------------------------------------------------------------ model

    @app.get("/model", response_class=HTMLResponse)
    def model_page(request: Request, conn=Depends(db)):
        versions = list_versions(settings.models_dir)
        return render(
            request, "model.html", conn, versions=versions, active=active_version(),
            configured=settings.model_version, pred_versions=Q.prediction_versions(conn),
            label_counts=Q.label_counts(conn), samples=Q.sample_origin_counts(conn), jobs=Q.recent_jobs(conn, 5),
        )

    return app


def _fmt_dt(value) -> str:
    if not value:
        return "–"
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return str(value)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _date_or_none(value: str) -> str | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date().isoformat() if value else None
    except ValueError:
        return None
