import json

import pytest

from tl.data.common import assign_split, normalize_for_model, text_hash
from tl.data.human import import_human_dataset
from tl.data.synthetic import READERS, SyntheticItem, import_synthetic_files, register_reader
from tl.db import session
from tl.db import queries as Q
from tl.model.dataset import load_split, manual_examples
from tl.scraper.extract import extract_page
from tests.conftest import FIXTURES

DATA = FIXTURES / "data"


def samples(settings):
    with session(settings.db_path) as conn:
        return conn.execute(
            "SELECT s.*, l.label, l.source FROM samples s JOIN labels l ON l.sample_id = s.id ORDER BY s.id"
        ).fetchall()


def test_normalization_masks_urls_and_mentions():
    assert normalize_for_model("@bob look https://t.co/x  ok") == "@user look http ok"
    assert normalize_for_model("RT @a: hello") == "hello"
    assert text_hash("Hello  @bob") == text_hash("hello @alice")  # same shape -> same identity


def test_split_is_deterministic_and_roughly_80_10_10():
    hs = [text_hash(f"tweet {i}") for i in range(3000)]
    splits = [assign_split(h) for h in hs]
    assert splits == [assign_split(h) for h in hs]
    assert 0.75 < splits.count("train") / 3000 < 0.85
    assert 0.07 < splits.count("test") / 3000 < 0.13


def test_human_csv_autodetects_column_filters_and_dedupes(settings):
    stats = import_human_dataset(settings.db_path, DATA / "human_with_header.csv")
    assert stats == {"origin": "human_with_header", "read": 7, "imported": 4, "duplicates": 1, "filtered": 2}
    rows = samples(settings)
    assert {r["label"] for r in rows} == {"human"} and {r["source"] for r in rows} == {"dataset"}
    # re-import is idempotent
    assert import_human_dataset(settings.db_path, DATA / "human_with_header.csv")["imported"] == 0


def test_human_headerless_csv_by_index(settings):
    stats = import_human_dataset(settings.db_path, DATA / "sentiment140_style.csv", text_column="5",
                                 name="sentiment140")
    assert stats["imported"] == 3
    assert all(r["origin"] == "sentiment140" for r in samples(settings))


def test_human_jsonl_and_txt(settings):
    assert import_human_dataset(settings.db_path, DATA / "human.jsonl")["imported"] == 2
    st = import_human_dataset(settings.db_path, DATA / "human.txt")
    assert st["imported"] == 2 and st["filtered"] == 1


def test_human_bad_column_is_reported(settings):
    with pytest.raises(ValueError, match="not found"):
        import_human_dataset(settings.db_path, DATA / "human_with_header.csv", text_column="tweet_body")


def test_synthetic_formats_and_model_variant_resolution(settings):
    stats = import_synthetic_files(settings.db_path, [DATA / "model-a__default.jsonl", DATA / "model-b__casual.txt",
                                                      DATA / "synthetic.csv"])
    assert stats["imported"] == 8 and stats["files"] == 3
    assert stats["by_origin"] == {
        "model-a/default": 2, "model-a-override/default": 1, "model-b/casual": 3,
        "model-c/default": 1, "model-c/casual": 1,
    }
    rows = samples(settings)
    assert {r["label"] for r in rows} == {"ai"} and {r["source"] for r in rows} == {"synthetic"}
    multi = [r for r in rows if r["text"].startswith("lowkey")][0]
    assert multi["text"] == "lowkey can't believe it's already october?? where did the year go\ntime is fake honestly"
    meta = json.loads([r for r in rows if r["origin"] == "model-a-override"][0]["meta_json"])
    assert meta == {"temperature": 0.9, "file": "model-a__default.jsonl"}


def test_synthetic_cli_flags_fill_missing_model(settings, tmp_path):
    f = tmp_path / "plain.txt"
    f.write_text("one generated tweet about the weather today\n")
    import_synthetic_files(settings.db_path, [f], model="my-model", variant="casual")
    assert (samples(settings)[0]["origin"], samples(settings)[0]["variant"]) == ("my-model", "casual")


def test_synthetic_directory_and_bad_jsonl(settings, tmp_path):
    (tmp_path / "x.jsonl").write_text('{"text": "fine tweet text here"}\n{"nope": 1}\n')
    with pytest.raises(ValueError, match="x.jsonl:2"):
        import_synthetic_files(settings.db_path, [tmp_path])


def test_synthetic_reader_is_pluggable(settings, tmp_path):
    @register_reader(".tsvx")
    def read_tsvx(path):
        for line in path.read_text().splitlines():
            model, text = line.split("\t", 1)
            yield SyntheticItem(text, model=model)

    try:
        f = tmp_path / "gen.tsvx"
        f.write_text("m1\tthis is a generated sentence one\nm2\tthis is a generated sentence two\n")
        assert import_synthetic_files(settings.db_path, [f])["by_origin"] == {"m1/default": 1, "m2/default": 1}
    finally:
        READERS.pop(".tsvx")


def test_manual_labels_never_reach_training(settings, fixture_html):
    """Hand labels live on timeline tweets; a corpus copy of a timeline tweet is excluded too."""
    recs = extract_page(fixture_html("timeline_following.html"))
    with session(settings.db_path) as conn:
        Q.upsert_tweets(conn, recs)
        Q.add_manual_label(conn, recs[1].id, "ai")
        # the same text also shows up in an imported corpus
        Q.insert_sample(conn, text=recs[1].text + " ", text_hash=text_hash(recs[1].text), origin="leak",
                        variant=None, meta=None, split="train", label="ai", source="synthetic")
        Q.insert_sample(conn, text="an unrelated generated tweet", text_hash=text_hash("an unrelated generated tweet"),
                        origin="ok", variant=None, meta=None, split="train", label="ai", source="synthetic")
        train, dropped = load_split(conn, "train")
        assert [e.origin for e in train] == ["ok"] and dropped == 1
        assert [e.source for e in manual_examples(conn)] == ["manual"]
        assert all(r["source"] != "manual" for r in Q.training_samples(conn, "train"))
