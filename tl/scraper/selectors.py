"""Every DOM selector and page-text pattern the scraper depends on.

X changes its markup without notice. When extraction breaks, this is the only file that
should need editing. Each selector is used in exactly one role; the extractor raises
`SelectorError` (and the scraper goes to state "error") when a *required* selector stops
matching, instead of silently writing partial rows.

CSS selectors here are consumed by BeautifulSoup/soupsieve (tl.scraper.extract) and/or by
Playwright (tl.scraper.browser). Playwright-only selectors (``:has-text``) are marked.
"""

from __future__ import annotations

import re

HOME_URL = "https://x.com/home"
LOGIN_URL = "https://x.com/i/flow/login"

# ---- timeline page -------------------------------------------------------------------------

# One tweet card. REQUIRED: if none match on a loaded home page -> SelectorError.
TWEET_ARTICLE = 'article[data-testid="tweet"]'

# Timeline tabs (Playwright-only: uses :has-text).
TIMELINE_TABS = {
    "following": '[role="tab"]:has-text("Following")',
    "for_you": '[role="tab"]:has-text("For you")',
}
TAB_SELECTED_ATTR = "aria-selected"

# ---- inside one tweet article -------------------------------------------------------------

# Block holding display name, @handle and the timestamp link. REQUIRED.
USER_NAME_BLOCK = 'div[data-testid="User-Name"]'
# Links inside the User-Name block; the one whose text starts with "@" is the handle.
USER_NAME_LINKS = 'a[href^="/"]'
# Link to the tweet itself, wrapping the <time>. REQUIRED (gives the id).
STATUS_LINK = 'a[href*="/status/"]'
STATUS_HREF_RE = re.compile(r"^/(?P<handle>[A-Za-z0-9_]{1,15})/status/(?P<id>\d+)")
TIME = "time[datetime]"
# Tweet body. Optional: media-only tweets have no text.
TWEET_TEXT_TESTID = "tweetText"
TWEET_TEXT = f'div[data-testid="{TWEET_TEXT_TESTID}"]'
# Engagement buttons. Count is read from aria-label ("12 Replies. Reply") or the visible span.
COUNT_BUTTONS = {
    "reply_count": ['[data-testid="reply"]'],
    "repost_count": ['[data-testid="retweet"]', '[data-testid="unretweet"]'],
    "like_count": ['[data-testid="like"]', '[data-testid="unlike"]'],
}
COUNT_VISIBLE_TEXT = '[data-testid="app-text-transition-container"]'
COUNT_ARIA_RE = re.compile(r"^\s*(?P<num>[\d][\d,.]*\s*[KkMm]?)\s+\w+")
# "Alice reposted" header above a repost.
SOCIAL_CONTEXT = '[data-testid="socialContext"]'
REPOST_CONTEXT_RE = re.compile(r"\breposted\b", re.IGNORECASE)
# "Replying to @bob" line on replies (matched on article text, not a test id).
REPLYING_TO_RE = re.compile(r"^\s*Replying to\b", re.IGNORECASE)
# Media containers; any match sets has_media.
MEDIA = [
    'div[data-testid="tweetPhoto"]',
    'div[data-testid="videoPlayer"]',
    'div[data-testid="videoComponent"]',
    'div[data-testid="card.layoutLarge.media"]',
]
# Promoted tweets are skipped (they are ads, not the timeline).
PROMOTED = ['div[data-testid="placementTracking"]']
PROMOTED_TEXT_RE = re.compile(r"^(Ad|Promoted)$")

# If more than this fraction of articles on a page miss a required field, raise SelectorError.
MAX_REQUIRED_MISS_RATIO = 0.5

# ---- stop conditions (pause scraper, never auto-retry) -------------------------------------

LOGIN_WALL_URL_RE = re.compile(r"/(i/flow/login|login|i/flow/signup|logout)(\b|/|\?|$)")
LOGIN_WALL_SELECTORS = [
    '[data-testid="loginButton"]',
    '[data-testid="login"]',
    'input[autocomplete="username"]',
    'a[href="/login"]',
]
CAPTCHA_URL_RE = re.compile(r"/account/access|/i/flow/consent|challenge", re.IGNORECASE)
CAPTCHA_SELECTORS = [
    'iframe[src*="arkoselabs"]',
    'iframe[src*="captcha"]',
    "#arkose_iframe",
]
# Matched case-insensitively against visible page text *outside* tweet articles
# (so a tweet that quotes these phrases does not trigger a pause).
RATE_LIMIT_TEXT = [
    "rate limit exceeded",
    "you are over the daily limit",
    "something went wrong. try reloading.",
]
SUSPENSION_TEXT = [
    "your account is suspended",
    "your account has been locked",
    "account is temporarily restricted",
    "caution: this account is temporarily restricted",
    "we've temporarily limited some of your account features",
]
