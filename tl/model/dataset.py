"""Builds train/val/test example lists from the DB, with the leakage guard applied."""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass

from tl.data.common import text_hash
from tl.db import queries as Q

LABEL2ID = {"human": 0, "ai": 1}


@dataclass
class Example:
    text: str
    label: int          # 1 = ai, 0 = human
    source: str         # synthetic | dataset
    origin: str
    variant: str | None


def timeline_hashes(conn: sqlite3.Connection) -> set[str]:
    """Hashes of every scraped timeline tweet. Corpus samples matching one are never trained or
    tested on, so the hand-labelled timeline eval set can't leak into training."""
    return {text_hash(r[0]) for r in conn.execute("SELECT text FROM tweets WHERE length(text) > 0")}


def load_split(conn: sqlite3.Connection, split: str, *, exclude: set[str] | None = None) -> tuple[list[Example], int]:
    exclude = exclude if exclude is not None else timeline_hashes(conn)
    out, dropped = [], 0
    for r in Q.training_samples(conn, split):
        if r["text_hash"] in exclude:
            dropped += 1
            continue
        out.append(Example(r["text"], LABEL2ID[r["label"]], r["source"], r["origin"], r["variant"]))
    return out, dropped


def stratified_cap(examples: list[Example], cap: int | None, seed: int) -> list[Example]:
    if not cap or len(examples) <= cap:
        return examples
    rng = random.Random(seed)
    by_label: dict[int, list[Example]] = {}
    for ex in examples:
        by_label.setdefault(ex.label, []).append(ex)
    out: list[Example] = []
    for exs in by_label.values():
        k = max(1, round(cap * len(exs) / len(examples)))
        out.extend(rng.sample(exs, min(k, len(exs))))
    rng.shuffle(out)
    return out


def manual_examples(conn: sqlite3.Connection) -> list[Example]:
    return [Example(r["text"], LABEL2ID[r["label"]], "manual", "timeline", None) for r in Q.manual_eval_set(conn)]
