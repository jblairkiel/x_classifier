"""Text normalization, hashing and split assignment shared by importers, training and inference."""

from __future__ import annotations

import hashlib
import re
import unicodedata

URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b[a-z0-9-]+\.(?:com|net|org|io|co|ly|me)/\S*)", re.IGNORECASE)
MENTION_RE = re.compile(r"(?<![\w@])@[A-Za-z0-9_]{1,15}\b")
RT_PREFIX_RE = re.compile(r"^RT\s+@[A-Za-z0-9_]{1,15}:\s*")
WS_RE = re.compile(r"[ \t ​]+")


def clean_text(text: str) -> str:
    """Light cleanup for storage: NFC, collapse spaces, trim. Keeps emoji, punctuation, em dashes."""
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = WS_RE.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def normalize_for_model(text: str) -> str:
    """What the classifier sees, at training AND inference time.

    URLs and @mentions are replaced by placeholders so the model can't learn shortcuts such as
    "public human datasets contain @mentions, my synthetic AI tweets don't".
    """
    text = clean_text(text)
    text = RT_PREFIX_RE.sub("", text)
    text = URL_RE.sub("http", text)
    text = MENTION_RE.sub("@user", text)
    return text


def text_hash(text: str) -> str:
    """Stable identity for dedupe and train/eval leakage checks."""
    norm = normalize_for_model(text).lower()
    norm = re.sub(r"\s+", " ", norm)
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def assign_split(h: str, *, val_pct: int = 10, test_pct: int = 10) -> str:
    """Deterministic split from the text hash, so re-imports never move a sample between splits."""
    bucket = int(h[:8], 16) % 100
    if bucket < test_pct:
        return "test"
    if bucket < test_pct + val_pct:
        return "val"
    return "train"
