"""Evaluate a checkpoint.

Reports, separately:
  * "test"            -- held-out split of the imported/synthetic corpus (same distribution as training)
  * "manual_timeline" -- YOUR hand labels on real timeline tweets (the number that matters)
  * "manual_timeline@<AI_THRESHOLD>" -- same set at the UI's flag threshold
plus per-origin breakdowns of the test split (e.g. how often "casual" variants slip through).
Results are written to <checkpoint>/eval.json for the Model page.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

from tl.config import Settings
from tl.db import session
from tl.model import metrics as M
from tl.model.checkpoints import resolve_version, write_json
from tl.model.dataset import load_split, manual_examples

log = logging.getLogger("tl.eval")


def _by_origin(examples, probs, threshold) -> list[dict]:
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, ex in enumerate(examples):
        groups[(ex.source, ex.origin, ex.variant or "")].append(i)
    rows = []
    for (source, origin, variant), idx in sorted(groups.items()):
        p = np.asarray([probs[i] for i in idx])
        y = np.asarray([examples[i].label for i in idx])
        rows.append({
            "source": source, "origin": origin, "variant": variant, "n": len(idx),
            "accuracy": float(((p >= threshold).astype(int) == y).mean()),
            "mean_p_ai": float(p.mean()),
        })
    return rows


def evaluate(settings: Settings, *, version: str | None = None, device: str = "auto", scorer=None,
             threshold: float = 0.5) -> dict:
    if scorer is None:
        from tl.model.predict import load_scorer

        scorer = load_scorer(settings, version, device)
    name, path = resolve_version(settings, version or scorer.version)

    with session(settings.db_path) as conn:
        test_ex, dropped = load_split(conn, "test")
        manual = manual_examples(conn)

    sets: dict[str, dict] = {}
    by_origin: list[dict] = []
    if test_ex:
        probs = scorer.score([e.text for e in test_ex])
        sets["test"] = M.classification_report([e.label for e in test_ex], probs, threshold)
        by_origin = _by_origin(test_ex, probs, threshold)
    if manual:
        probs = scorer.score([e.text for e in manual])
        y = [e.label for e in manual]
        sets["manual_timeline"] = M.classification_report(y, probs, threshold)
        sets[f"manual_timeline@{settings.ai_threshold}"] = M.classification_report(y, probs, settings.ai_threshold)

    report = {
        "version": name,
        "evaluated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "threshold": threshold,
        "sets": sets,
        "test_by_origin": by_origin,
        "test_dropped_overlap": dropped,
        "warnings": [],
    }
    if not manual:
        report["warnings"].append("No hand-labelled timeline tweets yet: label some on the Label page; "
                                  "corpus test metrics alone overstate real-world accuracy.")
    elif len(manual) < 100:
        report["warnings"].append(f"Only {len(manual)} hand-labelled tweets: timeline metrics are noisy.")
    try:
        write_json(path / "eval.json", report)
    except OSError as exc:  # e.g. models volume mounted read-only in this container
        log.warning("could not write %s: %s", path / "eval.json", exc)
    return report
