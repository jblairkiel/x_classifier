"""Import a public HUMAN-written tweet dataset from a local file you already downloaded.

Supported: CSV (with or without header), JSONL, or TXT (one tweet per line). The text column is
auto-detected from common names, or pass --text-column (a name, or a 0-based index for
headerless CSVs such as Sentiment140, whose text is column 5).

Nothing is downloaded. See docs/DATA_FORMATS.md.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Iterator

from tl.data.common import RT_PREFIX_RE, assign_split, clean_text, text_hash
from tl.db import session
from tl.db import queries as Q

TEXT_COLUMN_CANDIDATES = ("text", "tweet", "full_text", "content", "body", "tweet_text", "message")
MIN_CHARS = 15
MAX_CHARS = 1000

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def _detect_format(path: Path, fmt: str) -> str:
    if fmt != "auto":
        return fmt
    ext = path.suffix.lower()
    if ext in (".jsonl", ".ndjson", ".json"):
        return "jsonl"
    if ext in (".csv", ".tsv"):
        return "csv"
    return "txt"


def _pick_column(keys: list[str], text_column: str | None) -> str:
    if text_column:
        if text_column not in keys:
            raise ValueError(f"column {text_column!r} not found; available: {keys}")
        return text_column
    lower = {k.lower(): k for k in keys}
    for cand in TEXT_COLUMN_CANDIDATES:
        if cand in lower:
            return lower[cand]
    raise ValueError(f"could not detect the text column among {keys}; pass --text-column")


def iter_texts(path: Path, *, fmt: str = "auto", text_column: str | None = None) -> Iterator[str]:
    fmt = _detect_format(path, fmt)
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        if fmt == "txt":
            for line in fh:
                yield line.rstrip("\n")
        elif fmt == "jsonl":
            col = None
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, str):
                    yield obj
                    continue
                col = col or _pick_column(list(obj.keys()), text_column)
                val = obj.get(col)
                if isinstance(val, str):
                    yield val
        elif fmt == "csv":
            sample = fh.read(65536)
            fh.seek(0)
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;") if sample else csv.excel
            if text_column is not None and text_column.isdigit():
                idx = int(text_column)  # headerless CSV, positional column
                for row in csv.reader(fh, dialect):
                    if len(row) > idx:
                        yield row[idx]
            else:
                reader = csv.DictReader(fh, dialect=dialect)
                col = _pick_column(list(reader.fieldnames or []), text_column)
                for row in reader:
                    if row.get(col):
                        yield row[col]
        else:
            raise ValueError(f"unknown format {fmt}")


def keep(text: str) -> bool:
    if not (MIN_CHARS <= len(text) <= MAX_CHARS):
        return False
    if RT_PREFIX_RE.match(text):
        return False  # retweets duplicate someone else's text
    return True


def import_human_dataset(
    db_path, path: Path, *, fmt: str = "auto", text_column: str | None = None, name: str | None = None,
    limit: int | None = None,
) -> dict:
    origin = name or path.stem
    stats = {"origin": origin, "read": 0, "imported": 0, "duplicates": 0, "filtered": 0}
    with session(db_path) as conn:
        for raw in iter_texts(path, fmt=fmt, text_column=text_column):
            stats["read"] += 1
            text = clean_text(raw)
            if not keep(text):
                stats["filtered"] += 1
                continue
            h = text_hash(text)
            ok = Q.insert_sample(conn, text=text, text_hash=h, origin=origin, variant=None, meta=None,
                                 split=assign_split(h), label="human", source="dataset")
            stats["imported" if ok else "duplicates"] += 1
            if limit and stats["imported"] >= limit:
                break
    return stats
