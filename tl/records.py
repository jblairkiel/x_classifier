"""Plain data records shared by the scraper, DB layer and tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class TweetRecord:
    id: str
    author_handle: str            # lowercase, no "@"
    display_name: str | None
    text: str
    created_at: str | None        # ISO-8601 from <time datetime>
    reply_count: int | None
    repost_count: int | None
    like_count: int | None
    is_reply: bool
    is_repost: bool
    reposted_by: str | None
    has_media: bool
    lang: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)
