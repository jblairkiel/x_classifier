"""Single-admin authentication, CSRF and login rate limiting."""

from __future__ import annotations

import hmac
import secrets
import threading
import time
from collections import defaultdict, deque

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

_hasher = PasswordHasher()  # argon2id, library defaults (m=64MiB, t=3, p=4)
# Verified against when the username is wrong, so timing does not reveal valid usernames.
_DUMMY_HASH = _hasher.hash(secrets.token_urlsafe(16))

SESSION_USER = "user"
SESSION_CSRF = "csrf"
SESSION_LOGIN_AT = "login_at"
CSRF_FORM_FIELD = "csrf_token"
CSRF_HEADER = "x-csrf-token"


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_credentials(username: str, password: str, *, expected_user: str, password_hash: str) -> bool:
    user_ok = hmac.compare_digest(username.encode(), expected_user.encode())
    try:
        pw_ok = _hasher.verify(password_hash if user_ok else _DUMMY_HASH, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        pw_ok = False
    return user_ok and pw_ok


def csrf_token(session: dict) -> str:
    tok = session.get(SESSION_CSRF)
    if not tok:
        tok = secrets.token_urlsafe(32)
        session[SESSION_CSRF] = tok
    return tok


def csrf_valid(session: dict, submitted: str | None) -> bool:
    expected = session.get(SESSION_CSRF)
    return bool(expected and submitted) and hmac.compare_digest(expected, submitted)


class LoginRateLimiter:
    """Sliding-window limit on FAILED logins, per client IP and globally.

    In-memory, so it resets when the web container restarts; that's acceptable because each
    attempt also pays an argon2 verification and the global cap bounds distributed guessing.
    """

    def __init__(self, per_ip: int, global_max: int, window_seconds: int):
        self.per_ip = per_ip
        self.global_max = global_max
        self.window = window_seconds
        self._by_ip: dict[str, deque[float]] = defaultdict(deque)
        self._global: deque[float] = deque()
        self._lock = threading.Lock()

    def _prune(self, q: deque[float], now: float) -> None:
        while q and now - q[0] > self.window:
            q.popleft()

    def retry_after(self, ip: str) -> int:
        """0 if allowed now, else seconds until the next attempt is allowed."""
        now = time.monotonic()
        with self._lock:
            q = self._by_ip[ip]
            self._prune(q, now)
            self._prune(self._global, now)
            waits = []
            if len(q) >= self.per_ip:
                waits.append(self.window - (now - q[0]))
            if len(self._global) >= self.global_max:
                waits.append(self.window - (now - self._global[0]))
            return int(max(waits)) + 1 if waits else 0

    def record_failure(self, ip: str) -> None:
        now = time.monotonic()
        with self._lock:
            self._by_ip[ip].append(now)
            self._global.append(now)

    def record_success(self, ip: str) -> None:
        with self._lock:
            self._by_ip.pop(ip, None)


def safe_next(target: str | None) -> str:
    """Only allow same-site relative redirects after login."""
    if not target or not target.startswith("/") or target.startswith("//") or "\\" in target:
        return "/"
    return target
