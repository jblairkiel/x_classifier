"""Pure HTML -> TweetRecord extraction and stop-condition detection.

No browser here: Playwright hands us `outerHTML` strings, and the unit tests feed saved
fixtures through the exact same functions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, NavigableString, Tag

from tl.records import TweetRecord
from tl.scraper import selectors as S


class SelectorError(RuntimeError):
    """A required selector stopped matching. The DOM probably changed; fix selectors.py."""


class MissingField(ValueError):
    """One article lacks a required field (counted; too many -> SelectorError)."""


@dataclass
class Blocker:
    kind: str     # login_wall | captcha | rate_limit | suspended
    detail: str

    def reason(self) -> str:
        return f"{self.kind}: {self.detail}"


# --------------------------------------------------------------------------- helpers


def parse_count(raw: str | None) -> int | None:
    """'1,234' -> 1234, '1.2K' -> 1200, '3M' -> 3000000, '' -> 0, None -> None."""
    if raw is None:
        return None
    s = raw.strip().replace(",", "").replace(" ", "")
    if s == "":
        return 0
    mult = 1
    if s[-1] in "Kk":
        mult, s = 1_000, s[:-1]
    elif s[-1] in "Mm":
        mult, s = 1_000_000, s[:-1]
    try:
        return int(round(float(s) * mult))
    except ValueError:
        return None


def _text_of(node: Tag) -> str:
    """Visible text with emoji <img alt> preserved, as X renders it."""
    parts: list[str] = []
    for el in node.descendants:
        if isinstance(el, NavigableString):
            parts.append(str(el))
        elif isinstance(el, Tag) and el.name == "img" and el.get("alt"):
            parts.append(el["alt"])
        elif isinstance(el, Tag) and el.name == "br":
            parts.append("\n")
    text = "".join(parts)
    text = re.sub(r"[ \t ]+", " ", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def _count_for(article: Tag, selectors: list[str]) -> int | None:
    for sel in selectors:
        btn = article.select_one(sel)
        if btn is None:
            continue
        label = btn.get("aria-label") or ""
        m = S.COUNT_ARIA_RE.match(label)
        if m:
            return parse_count(m.group("num"))
        visible = btn.select_one(S.COUNT_VISIBLE_TEXT)
        if visible is not None:
            return parse_count(visible.get_text())
        return 0  # button present, no number rendered => zero
    return None


def _is_promoted(article: Tag) -> bool:
    if any(article.select_one(sel) for sel in S.PROMOTED):
        return True
    return any(
        S.PROMOTED_TEXT_RE.match(sp.get_text(strip=True)) and not _inside_tweet_text(sp)
        for sp in article.find_all("span")
    )


def _inside_tweet_text(node) -> bool:
    """True if `node` is part of a tweet body (whose words must not drive flags)."""
    return any(
        isinstance(p, Tag) and p.get("data-testid") == S.TWEET_TEXT_TESTID for p in node.parents
    )


# --------------------------------------------------------------------------- extraction


def extract_article(article: Tag) -> TweetRecord:
    """Extract one tweet. Raises MissingField when a required selector does not match."""
    user_block = article.select_one(S.USER_NAME_BLOCK)
    if user_block is None:
        raise MissingField(f"no {S.USER_NAME_BLOCK}")

    tweet_id = handle = created_at = None
    for link in article.select(S.STATUS_LINK):
        m = S.STATUS_HREF_RE.match(link.get("href", ""))
        if m and link.select_one(S.TIME) is not None:
            tweet_id, handle = m.group("id"), m.group("handle")
            created_at = link.select_one(S.TIME)["datetime"]
            break
    if tweet_id is None:
        raise MissingField(f"no {S.STATUS_LINK} wrapping {S.TIME}")

    display_name = None
    for link in user_block.select(S.USER_NAME_LINKS):
        txt = _text_of(link)
        if txt.startswith("@"):
            handle = txt[1:]
        elif txt and display_name is None:
            display_name = txt

    text_node = article.select_one(S.TWEET_TEXT)
    text = _text_of(text_node) if text_node is not None else ""
    lang = text_node.get("lang") if text_node is not None else None

    social = article.select_one(S.SOCIAL_CONTEXT)
    social_text = _text_of(social) if social is not None else ""
    is_repost = bool(S.REPOST_CONTEXT_RE.search(social_text))
    reposted_by = None
    if is_repost:
        reposted_by = S.REPOST_CONTEXT_RE.split(social_text)[0].strip() or None

    is_reply = any(
        not _inside_tweet_text(s) for s in article.find_all(string=S.REPLYING_TO_RE)
    )

    return TweetRecord(
        id=tweet_id,
        author_handle=handle.lower(),
        display_name=display_name,
        text=text,
        created_at=created_at,
        reply_count=_count_for(article, S.COUNT_BUTTONS["reply_count"]),
        repost_count=_count_for(article, S.COUNT_BUTTONS["repost_count"]),
        like_count=_count_for(article, S.COUNT_BUTTONS["like_count"]),
        is_reply=is_reply,
        is_repost=is_repost,
        reposted_by=reposted_by,
        has_media=any(article.select_one(sel) is not None for sel in S.MEDIA),
        lang=lang,
    )


def extract_articles(html_fragments: list[str], *, strict: bool = True) -> tuple[list[TweetRecord], int]:
    """Extract from a list of article outerHTML strings (what the browser hands us).

    Returns (records, skipped_promoted). Raises SelectorError if too many articles miss
    required fields -- partial garbage is worse than a loud failure.
    """
    records: list[TweetRecord] = []
    misses: list[str] = []
    promoted = 0
    for frag in html_fragments:
        soup = BeautifulSoup(frag, "html.parser")
        article = soup.select_one(S.TWEET_ARTICLE) or soup
        if _is_promoted(article):
            promoted += 1
            continue
        try:
            records.append(extract_article(article))
        except MissingField as exc:
            misses.append(str(exc))
    considered = len(html_fragments) - promoted
    if strict and considered and len(misses) / considered > S.MAX_REQUIRED_MISS_RATIO:
        raise SelectorError(
            f"{len(misses)}/{considered} tweet articles missed required selectors "
            f"(first: {misses[0]}). Update tl/scraper/selectors.py."
        )
    return records, promoted


def extract_page(html: str, *, strict: bool = True) -> list[TweetRecord]:
    """Extract every tweet in a full page HTML (used for fixtures and debugging)."""
    soup = BeautifulSoup(html, "html.parser")
    articles = soup.select(S.TWEET_ARTICLE)
    if strict and not articles:
        raise SelectorError(f"no elements matched {S.TWEET_ARTICLE!r}. Update tl/scraper/selectors.py.")
    records, _ = extract_articles([str(a) for a in articles], strict=strict)
    return records


# --------------------------------------------------------------------------- stop conditions


def detect_blocker(url: str, html: str) -> Blocker | None:
    """Login wall, captcha, rate limit or suspension notice on the current page, if any."""
    if S.LOGIN_WALL_URL_RE.search(url or ""):
        return Blocker("login_wall", f"redirected to {url}")
    if S.CAPTCHA_URL_RE.search(url or ""):
        return Blocker("captcha", f"redirected to {url}")

    soup = BeautifulSoup(html or "", "html.parser")
    for art in soup.select(S.TWEET_ARTICLE):
        art.decompose()  # tweet content must never trigger a stop

    for sel in S.CAPTCHA_SELECTORS:
        if soup.select_one(sel) is not None:
            return Blocker("captcha", f"matched {sel}")
    for sel in S.LOGIN_WALL_SELECTORS:
        if soup.select_one(sel) is not None:
            return Blocker("login_wall", f"matched {sel}")

    for bad in soup(["script", "style", "noscript"]):
        bad.decompose()
    text = soup.get_text(" ", strip=True).lower()
    text = re.sub(r"\s+", " ", text)
    for phrase in S.SUSPENSION_TEXT:
        if phrase in text:
            return Blocker("suspended", f"page says {phrase!r}")
    for phrase in S.RATE_LIMIT_TEXT:
        if phrase in text:
            return Blocker("rate_limit", f"page says {phrase!r}")
    return None
