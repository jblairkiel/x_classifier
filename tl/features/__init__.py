"""Account-level signals combined with text-model scores into a "botlike" score.

Each signal is mapped to a 0..1 "botlike contribution" by a simple, documented rule, then
combined as a weighted mean over the signals that could be computed, and finally shrunk toward a
prior for accounts with few tweets. These are heuristics, not a trained model -- the per-signal
breakdown on the account page is there so you can see *why* an account ranks high.

Caveat: the scraper only sees what reaches the bot's home timeline, so cadence is estimated from
a sample, not the account's full history.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from datetime import datetime
from itertools import combinations

from tl.config import Settings
from tl.db import session
from tl.db import queries as Q
from tl.model.checkpoints import resolve_version_or_none

# Weights for the weighted mean. The text model carries the most weight; everything else is a nudge.
WEIGHTS = {
    "text_score": 0.45,
    "cadence": 0.12,
    "near_duplicates": 0.15,
    "reply_ratio": 0.05,
    "hashtag_rate": 0.05,
    "em_dash_rate": 0.08,
    "length_variance": 0.10,
}
PRIOR = 0.25              # where an account with almost no evidence sits
SHRINK_K = 5              # confidence = n / (n + SHRINK_K)
MAX_TWEETS = 60           # most recent tweets per account considered
NEAR_DUP_JACCARD = 0.5
HASHTAG_RE = re.compile(r"(?<!\w)#\w+")
EM_DASH = "—"


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def _shingles(text: str, k: int = 5) -> set[str]:
    t = re.sub(r"\s+", " ", re.sub(r"https?://\S+|@\w+", "", text.lower())).strip()
    return {t[i : i + k] for i in range(max(1, len(t) - k + 1))} if t else set()


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _signal(label, value, display, score, help_):
    return {"label": label, "value": value, "display": display, "score": score, "help": help_}


def account_features(tweets: list[dict], scores: list[float]) -> dict:
    """Pure function: tweets = dicts with text/created_at/is_reply; scores = P(AI) of scored tweets."""
    tweets = tweets[-MAX_TWEETS:]
    texts = [t["text"] for t in tweets if (t.get("text") or "").strip()]
    n = len(tweets)
    signals: dict[str, dict] = {}

    if scores:
        mean = statistics.fmean(scores)
        signals["text_score"] = _signal("Mean text P(AI)", mean, f"{mean:.2f} over {len(scores)}", mean,
                                        "Average calibrated P(AI) from the text model.")
    else:
        signals["text_score"] = _signal("Mean text P(AI)", None, "no scores", None, "No model scores yet.")

    times = sorted(ts for ts in (_parse_ts(t.get("created_at")) for t in tweets) if ts)
    if len(times) >= 4:
        gaps = [(b - a).total_seconds() for a, b in zip(times, times[1:])]
        span_days = max((times[-1] - times[0]).total_seconds() / 86400, 1 / 24)
        per_day = (len(times) - 1) / span_days
        mean_gap = statistics.fmean(gaps)
        cv = statistics.pstdev(gaps) / mean_gap if mean_gap > 0 else 0.0
        regularity = _clip((0.5 - cv) / 0.5)          # metronome-like posting
        volume = _clip(math.log10(max(per_day, 1)) / 2)  # 100+/day -> 1
        signals["cadence"] = _signal(
            "Posting cadence", {"per_day": per_day, "interval_cv": cv},
            f"{per_day:.1f}/day, interval CV {cv:.2f}", max(regularity, volume),
            "Very regular gaps between posts (low coefficient of variation) or very high volume.")
    else:
        signals["cadence"] = _signal("Posting cadence", None, "need ≥4 timestamps", None, "")

    if len(texts) >= 3:
        sh = [_shingles(t) for t in texts]
        has_dup = [False] * len(texts)
        for i, j in combinations(range(len(texts)), 2):
            a, b = sh[i], sh[j]
            if a and b and len(a & b) / len(a | b) >= NEAR_DUP_JACCARD:
                has_dup[i] = has_dup[j] = True
        openers = Counter(" ".join(t.lower().split()[:3]) for t in texts if len(t.split()) >= 3)
        templated = sum(c for c in openers.values() if c > 1) / len(texts)
        dup_rate = sum(has_dup) / len(texts)
        signals["near_duplicates"] = _signal(
            "Near-duplicate phrasing", {"dup_rate": dup_rate, "templated_openers": templated},
            f"{dup_rate:.0%} near-dup, {templated:.0%} shared openers", _clip(max(dup_rate, templated)),
            "Share of tweets that closely match another tweet (char 5-gram Jaccard ≥ 0.5) or reuse the same first three words.")
    else:
        signals["near_duplicates"] = _signal("Near-duplicate phrasing", None, "need ≥3 tweets", None, "")

    if n:
        replies = sum(1 for t in tweets if t.get("is_reply"))
        ratio = replies / max(n - replies, 1)
        share = replies / n
        signals["reply_ratio"] = _signal("Reply-to-post ratio", ratio, f"{replies} replies : {n - replies} posts",
                                         _clip((share - 0.5) / 0.5),
                                         "Accounts that almost only reply (engagement farming) score higher.")
    if texts:
        tags = statistics.fmean(len(HASHTAG_RE.findall(t)) for t in texts)
        dash = sum(1 for t in texts if EM_DASH in t) / len(texts)
        signals["hashtag_rate"] = _signal("Hashtags per tweet", tags, f"{tags:.2f}", _clip(tags / 3),
                                          "Heavy hashtag use.")
        signals["em_dash_rate"] = _signal("Em-dash rate", dash, f"{dash:.0%} of tweets", _clip(dash / 0.5),
                                          "Share of tweets containing an em dash (—), common in LLM output.")
    if len(texts) >= 3:
        lengths = [len(t) for t in texts]
        cv = statistics.pstdev(lengths) / statistics.fmean(lengths)
        signals["length_variance"] = _signal("Length variance", cv, f"CV {cv:.2f}", _clip((0.35 - cv) / 0.35),
                                             "Unusually uniform tweet lengths (low coefficient of variation).")
    else:
        signals["length_variance"] = _signal("Length variance", None, "need ≥3 tweets", None, "")

    for key, sig in signals.items():
        sig["weight"] = WEIGHTS[key]
    usable = {k: s for k, s in signals.items() if s["score"] is not None}
    if not usable:
        return {"signals": signals, "n_tweets": n, "n_scored": len(scores), "raw_score": None,
                "botlike_score": None, "confidence": "none"}
    wsum = sum(WEIGHTS[k] for k in usable)
    raw = sum(WEIGHTS[k] * s["score"] for k, s in usable.items()) / wsum
    evidence = len(scores) if scores else n
    conf = evidence / (evidence + SHRINK_K)
    botlike = conf * raw + (1 - conf) * PRIOR
    return {
        "signals": signals,
        "n_tweets": n,
        "n_scored": len(scores),
        "raw_score": raw,
        "botlike_score": botlike,
        "confidence": "low" if evidence < 3 else "medium" if evidence < 10 else "high",
    }


def update_all_accounts(settings: Settings) -> int:
    version = resolve_version_or_none(settings)
    with session(settings.db_path) as conn:
        handles = Q.all_account_handles(conn)
        for handle in handles:
            tweets = [dict(r) for r in Q.tweets_for_account(conn, handle)]
            scores = Q.scores_for_account(conn, handle, version) if version else []
            feats = account_features(tweets, scores)
            feats["model_version"] = version
            Q.set_account_features(conn, handle, feats["botlike_score"], feats)
    return len(handles)
