"""eval.py with a stand-in scorer (no torch needed): the report keeps the hand-labelled timeline
set separate from the corpus test split and is written to the checkpoint dir."""

import json

from tl.data.synthetic import import_synthetic_files
from tl.data.human import import_human_dataset
from tl.db import session
from tl.db import queries as Q
from tl.model.checkpoints import list_versions, resolve_version
from tl.model.eval import evaluate
from tl.scraper.extract import extract_page
from tests.conftest import FIXTURES


class FakeScorer:
    version = "vtest"

    def score(self, texts):  # "AI" if it sounds like a LinkedIn post
        keys = ("journey", "growth", "essential", "excited", "grateful", "fast-paced", "embracing")
        return [0.9 if any(k in t.lower() for k in keys) else 0.2 for t in texts]


def make_checkpoint(settings, name="vtest"):
    d = settings.models_dir / name
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"created_at": "2026-10-04T00:00:00+00:00", "temperature": 1.3}))
    (d / "COMPLETE").write_text(name)
    return d


def test_eval_report_separates_manual_set(settings, fixture_html):
    d = make_checkpoint(settings)
    import_human_dataset(settings.db_path, FIXTURES / "data" / "human_with_header.csv")
    import_synthetic_files(settings.db_path, [FIXTURES / "data" / "model-a__default.jsonl"])
    with session(settings.db_path) as conn:
        conn.execute("UPDATE samples SET split = 'test'")  # tiny fixture: put everything in test
        recs = extract_page(fixture_html("timeline_following.html"))
        Q.upsert_tweets(conn, recs)
        Q.add_manual_label(conn, "1842000000000000002", "ai")      # "fast-paced world" tweet
        Q.add_manual_label(conn, "1842000000000000003", "human")
        Q.add_manual_label(conn, "1842000000000000001", "unsure")  # excluded from metrics
    rep = evaluate(settings, scorer=FakeScorer())
    assert set(rep["sets"]) == {"test", "manual_timeline", "manual_timeline@0.8"}
    assert rep["sets"]["manual_timeline"]["n"] == 2 and rep["sets"]["manual_timeline"]["accuracy"] == 1.0
    assert rep["sets"]["test"]["n"] == 7
    assert {r["origin"] for r in rep["test_by_origin"]} == {"human_with_header", "model-a", "model-a-override"}
    assert any("noisy" in w for w in rep["warnings"])
    saved = json.loads((d / "eval.json").read_text())
    assert saved["sets"]["manual_timeline"]["confusion"] == {"tp": 1, "fp": 0, "tn": 1, "fn": 0}
    assert list_versions(settings.models_dir)[0]["eval"]["version"] == "vtest"


def test_checkpoint_resolution(settings):
    make_checkpoint(settings, "v1")
    (settings.models_dir / "v-partial").mkdir()  # no COMPLETE marker -> ignored
    assert resolve_version(settings)[0] == "v1"
    import pytest

    with pytest.raises(ValueError):
        resolve_version(settings, "../etc")


def test_classify_job_without_model_is_graceful(settings):
    from tl.worker import classify_pending

    assert classify_pending(settings).startswith("no model")
