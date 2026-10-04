"""Regenerates the saved timeline HTML fixtures in this directory.

The markup mirrors the structure X's web client renders (data-testid attributes, nested
spans, aria-labels on engagement buttons). When X changes its DOM, save a real page
(DevTools -> Copy outerHTML of the timeline) next to these and add a test for it.

    python tests/fixtures/make_fixtures.py
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

HERE = Path(__file__).parent


def article(
    *, tid, handle, name, text_html, dt, replies=0, reposts=0, likes=0, social=None,
    replying_to=None, media=None, promoted=False, quote=None, lang="en", liked=False,
):
    def btn(testid, n, noun, verb):
        label = f"{n} {noun}. {verb}" if n else verb
        shown = "" if not n else f'<span data-testid="app-text-transition-container"><span><span>{escape(str(n))}</span></span></span>'
        return f'<button role="button" data-testid="{testid}" aria-label="{escape(label)}"><div dir="ltr"><div><svg></svg></div>{shown}</div></button>'

    def short(n):
        if isinstance(n, int) and n >= 10_000:
            return f"{n/1000:.1f}K".replace(".0K", "K")
        return f"{n:,}" if isinstance(n, int) else n

    social_html = (
        f'<div><span data-testid="socialContext">{escape(social)}</span></div>' if social else ""
    )
    reply_html = (
        f'<div dir="ltr" class="css-146c3p1">Replying to <div class="r-1q142lx"><a href="/{replying_to}" role="link">'
        f'<span>@{replying_to}</span></a></div></div>' if replying_to else ""
    )
    media_html = {
        None: "",
        "photo": '<div aria-label="Image" data-testid="tweetPhoto"><img alt="Image" src="data:,"></div>',
        "video": '<div data-testid="videoPlayer"><video></video></div>',
    }[media]
    promoted_html = (
        '<div data-testid="placementTracking"><span>Ad</span></div>' if promoted else ""
    )
    quote_html = ""
    if quote:
        quote_html = (
            '<div role="link" tabindex="0" class="quote">'
            f'<div data-testid="User-Name"><div><span>{escape(quote["name"])}</span></div>'
            f'<div><span>@{quote["handle"]}</span></div><time datetime="{quote["dt"]}">Sep 1</time></div>'
            f'<div data-testid="tweetText" lang="en"><span>{escape(quote["text"])}</span></div></div>'
        )
    like_testid = "unlike" if liked else "like"
    return f"""
<div data-testid="cellInnerDiv" style="transform: translateY(0px);"><div><div><article aria-labelledby="id__{tid}" role="article" tabindex="0" data-testid="tweet">
 <div><div>{social_html}</div>
 <div class="r-18u37iz">
  <div class="avatar"><a href="/{handle}" role="link"><img alt="" src="data:,"></a></div>
  <div>
   <div data-testid="User-Name" class="r-1wbh5a2">
    <div><a href="/{handle}" role="link"><div dir="ltr"><span><span>{name}</span></span></div><div><svg aria-label="Verified account"></svg></div></a></div>
    <div><div dir="ltr"><a href="/{handle}" role="link" tabindex="-1"><div dir="ltr"><span>@{handle}</span></div></a></div>
     <div aria-hidden="true"><span>·</span></div>
     <div><a href="/{handle}/status/{tid}" dir="ltr" aria-label="{dt}" role="link"><time datetime="{dt}">Oct 3</time></a></div></div>
   </div>
   {promoted_html}
   {reply_html}
   <div data-testid="tweetText" lang="{lang}" dir="auto" class="css-146c3p1">{text_html}</div>
   {media_html}{quote_html}
   <div aria-label="{replies} replies, {reposts} reposts, {likes} likes" role="group">
    {btn("reply", short(replies), "Replies", "Reply")}
    {btn("retweet", short(reposts), "reposts", "Repost")}
    {btn(like_testid, short(likes), "Likes", "Like")}
    <a href="/{handle}/status/{tid}/analytics" aria-label="1203 views. View post analytics" role="link"></a>
   </div>
  </div>
 </div></div>
</article></div></div></div>"""


def page(articles: list[str], *, extra_body: str = "", scripts: str = "") -> str:
    return f"""<!DOCTYPE html>
<html dir="ltr" lang="en"><head><meta charset="utf-8"><title>Home / X</title></head>
<body><div id="react-root"><main role="main">
<div role="tablist" data-testid="ScrollSnap-List">
  <div role="presentation"><a href="/home" role="tab" aria-selected="false"><div><span>For you</span></div></a></div>
  <div role="presentation"><a href="/home" role="tab" aria-selected="true"><div><span>Following</span></div></a></div>
</div>
{extra_body}
<section role="region"><h1>Your Home Timeline</h1>
<div aria-label="Timeline: Your Home Timeline"><div id="timeline">
{''.join(articles)}
</div></div></section>
</main></div>{scripts}</body></html>
"""


TWEETS = [
    dict(tid="1842000000000000001", handle="alice_dev", name="Alice 🌱",
         text_html='<span>Shipped the new release today. Took three tries to get the migration right but </span>'
                   '<img alt="🎉" src="data:,"><span> it works.</span>',
         dt="2026-10-03T14:02:11.000Z", replies=3, reposts=12, likes=45),
    dict(tid="1842000000000000002", handle="BobBuilds", name="Bob",
         text_html='<span>In today\'s fast-paced world, it\'s important to remember that consistency is key. '
                   'Whether you\'re building a startup or a habit — every small step matters. </span>'
                   '<a href="/hashtag/motivation" role="link">#motivation</a><span> </span>'
                   '<a href="/hashtag/growth" role="link">#growth</a>',
         dt="2026-10-03T13:40:00.000Z", replies=0, reposts=1, likes=2),
    dict(tid="1842000000000000003", handle="carol", name="Carol",
         text_html="<span>lol no. the bus was 40 min late AGAIN</span>",
         dt="2026-10-03T12:10:00.000Z", replies=1, reposts=0, likes=7, replying_to="metro_transit"),
    dict(tid="1842000000000000004", handle="newsdesk", name="News Desk",
         text_html="<span>Photos from this morning's launch.</span>",
         dt="2026-10-03T11:00:00.000Z", replies=210, reposts=12500, likes=48213,
         social="Alice 🌱 reposted", media="photo"),
    dict(tid="1842000000000000005", handle="adco", name="Ad Co",
         text_html="<span>Buy our thing.</span>", dt="2026-10-03T10:00:00.000Z", promoted=True),
    dict(tid="1842000000000000006", handle="dave", name="Dave",
         text_html="<span>this take is wrong and here's why</span>", dt="2026-10-03T09:00:00.000Z",
         replies=4, reposts=0, likes=1, liked=True,
         quote=dict(handle="erin", name="Erin", dt="2026-09-01T00:00:00.000Z",
                    text="Quoted tweet text that must not be extracted as Dave's text")),
    dict(tid="1842000000000000007", handle="vidguy", name="Vid Guy",
         text_html="", dt="2026-10-03T08:00:00.000Z", media="video", likes=3),
]


def main() -> None:
    arts = [article(**t) for t in TWEETS]
    # The same tweet rendered twice (X re-renders cells while scrolling) -> dedupe test.
    (HERE / "timeline_following.html").write_text(page(arts + [arts[0]]), encoding="utf-8")

    # DOM drift: test ids renamed, so required selectors stop matching.
    drift = "".join(arts).replace('data-testid="User-Name"', 'data-testid="UserName2"').replace(
        "/status/", "/s/"
    )
    (HERE / "timeline_dom_changed.html").write_text(page([drift]), encoding="utf-8")

    # Stop conditions.
    (HERE / "blocker_login_wall.html").write_text(
        page([], extra_body='<div><a href="/login" data-testid="login">Log in</a></div>'), encoding="utf-8")
    (HERE / "blocker_captcha.html").write_text(
        page([], extra_body='<iframe id="arkose_iframe" src="https://client-api.arkoselabs.com/fc"></iframe>'),
        encoding="utf-8")
    (HERE / "blocker_rate_limit.html").write_text(
        page(arts[:1], extra_body="<div><span>Something went wrong. Try reloading.</span><button>Retry</button></div>"),
        encoding="utf-8")
    (HERE / "blocker_suspended.html").write_text(
        page([], extra_body="<div><span>Your account is suspended</span></div>"), encoding="utf-8")
    # A tweet *quoting* a rate-limit phrase must NOT trigger a stop.
    (HERE / "not_blocker_tweet_mentions_rate_limit.html").write_text(
        page([article(tid="1842000000000000099", handle="sysadmin", name="Sys",
                      text_html="<span>Rate limit exceeded again on our API lol</span>",
                      dt="2026-10-03T08:00:00.000Z")]), encoding="utf-8")

    # Infinite-scroll page for the Playwright integration test: 30 tweets in batches of 3,
    # a new batch is appended whenever the user scrolls near the bottom.
    many = [
        article(tid=str(1843000000000000000 + i), handle=f"user{i % 7}", name=f"User {i % 7}",
                text_html=f"<span>Scrolling fixture tweet number {i}. Some words here.</span>",
                dt=f"2026-10-02T{(i % 24):02d}:00:00.000Z", likes=i)
        for i in range(30)
    ]
    batches = [many[i : i + 3] for i in range(0, len(many), 3)]
    script = (
        "<script>\nconst BATCHES = " + json.dumps(["".join(b) for b in batches[1:]]) + ";\n"
        "let next = 0;\n"
        "document.head.insertAdjacentHTML('beforeend', '<style>article{display:block;min-height:420px}</style>');\n"
        "window.addEventListener('scroll', () => {\n"
        "  if (next < BATCHES.length && window.innerHeight + window.scrollY >= document.body.scrollHeight - 2000) {\n"
        "    document.getElementById('timeline').insertAdjacentHTML('beforeend', BATCHES[next++]);\n"
        "  }\n"
        "});\n</script>"
    )
    (HERE / "timeline_infinite.html").write_text(page(["".join(batches[0])], scripts=script), encoding="utf-8")


if __name__ == "__main__":
    main()
