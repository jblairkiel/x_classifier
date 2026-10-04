"""Import synthetic AI-written tweets that YOU generated with various models.

Readers are pluggable: each file extension maps to a function yielding SyntheticItem. Add a format
with ``@register_reader(".ext")``. Formats (full spec in docs/DATA_FORMATS.md):

* ``.jsonl`` -- one object per line: {"text": ..., "model": ..., "variant": ..., "prompt": ..., ...}
  Only "text" is required; any other keys are kept as metadata.
* ``.csv``   -- header row with a ``text`` column; optional ``model``, ``variant`` columns.
* ``.txt``   -- one tweet per line; or multi-line tweets separated by a line containing only ``---``.

Model/variant resolution order: per-row value > CLI flag (--model/--variant) > file name convention
``<model>__<variant>.<ext>`` (e.g. ``llama-3.1-8b__casual.jsonl``) > "unknown"/"default".
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from tl.data.common import assign_split, clean_text, text_hash
from tl.data.human import MAX_CHARS
from tl.db import session
from tl.db import queries as Q

MIN_CHARS = 10


@dataclass
class SyntheticItem:
    text: str
    model: str | None = None
    variant: str | None = None
    meta: dict = field(default_factory=dict)


Reader = Callable[[Path], Iterator[SyntheticItem]]
READERS: dict[str, Reader] = {}


def register_reader(*exts: str) -> Callable[[Reader], Reader]:
    def deco(fn: Reader) -> Reader:
        for ext in exts:
            READERS[ext.lower()] = fn
        return fn

    return deco


@register_reader(".jsonl", ".ndjson")
def read_jsonl(path: Path) -> Iterator[SyntheticItem]:
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
            if isinstance(obj, str):
                yield SyntheticItem(obj)
                continue
            if not isinstance(obj.get("text"), str):
                raise ValueError(f"{path}:{lineno}: missing string field 'text'")
            meta = {k: v for k, v in obj.items() if k not in ("text", "model", "variant")}
            yield SyntheticItem(obj["text"], obj.get("model"), obj.get("variant"), meta)


@register_reader(".csv")
def read_csv(path: Path) -> Iterator[SyntheticItem]:
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if "text" not in (reader.fieldnames or []):
            raise ValueError(f"{path}: CSV needs a 'text' column (got {reader.fieldnames})")
        for row in reader:
            meta = {k: v for k, v in row.items() if k not in ("text", "model", "variant") and v}
            yield SyntheticItem(row["text"] or "", row.get("model") or None, row.get("variant") or None, meta)


@register_reader(".txt")
def read_txt(path: Path) -> Iterator[SyntheticItem]:
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    if any(line.strip() == "---" for line in lines):
        block: list[str] = []
        for line in lines + ["---"]:
            if line.strip() == "---":
                if "".join(block).strip():
                    yield SyntheticItem("\n".join(block))
                block = []
            else:
                block.append(line)
    else:
        for line in lines:
            if line.strip():
                yield SyntheticItem(line)


def _from_filename(path: Path) -> tuple[str | None, str | None]:
    stem = path.stem
    if "__" in stem:
        model, variant = stem.split("__", 1)
        return model or None, variant or None
    return None, None


def import_synthetic_files(db_path, paths: list[Path], *, model: str | None = None, variant: str | None = None) -> dict:
    files = []
    for p in paths:
        files.extend(sorted(x for x in p.rglob("*") if x.is_file()) if p.is_dir() else [p])
    stats: dict = {"files": 0, "read": 0, "imported": 0, "duplicates": 0, "filtered": 0, "by_origin": {}}
    with session(db_path) as conn:
        for path in files:
            reader = READERS.get(path.suffix.lower())
            if reader is None:
                raise ValueError(f"{path}: no reader for {path.suffix!r}; known: {sorted(READERS)}")
            fn_model, fn_variant = _from_filename(path)
            stats["files"] += 1
            for item in reader(path):
                stats["read"] += 1
                text = clean_text(item.text)
                if not (MIN_CHARS <= len(text) <= MAX_CHARS):
                    stats["filtered"] += 1
                    continue
                origin = item.model or model or fn_model or "unknown"
                var = item.variant or variant or fn_variant or "default"
                meta = {**item.meta, "file": path.name}
                h = text_hash(text)
                ok = Q.insert_sample(conn, text=text, text_hash=h, origin=origin, variant=var, meta=meta,
                                     split=assign_split(h), label="ai", source="synthetic")
                if ok:
                    stats["imported"] += 1
                    key = f"{origin}/{var}"
                    stats["by_origin"][key] = stats["by_origin"].get(key, 0) + 1
                else:
                    stats["duplicates"] += 1
    return stats
