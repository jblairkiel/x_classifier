"""End-to-end train -> checkpoint -> predict -> eval with a tiny randomly initialised RoBERTa built
locally (no downloads). Skipped when torch/transformers aren't installed."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from tl.data.human import import_human_dataset  # noqa: E402
from tl.data.synthetic import import_synthetic_files  # noqa: E402
from tl.db import session  # noqa: E402
from tl.model.checkpoints import list_versions  # noqa: E402
from tests.conftest import FIXTURES  # noqa: E402


def tiny_base_model(path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaForSequenceClassification

    words = sorted({w.lower() for f in (FIXTURES / "data").iterdir() for w in f.read_text().split()})
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, **{w: i + 4 for i, w in enumerate(words)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", unk_token="<unk>", bos_token="<s>",
                            eos_token="</s>", model_max_length=128).save_pretrained(path)
    cfg = RobertaConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                        intermediate_size=64, max_position_embeddings=160, pad_token_id=0, type_vocab_size=1)
    RobertaForSequenceClassification(cfg).save_pretrained(path)
    return str(path)


def test_train_predict_eval_roundtrip(settings, tmp_path):
    from tl.model.eval import evaluate
    from tl.model.predict import load_scorer
    from tl.model.train import train

    for f in ("human_with_header.csv", "human.jsonl", "human.txt"):
        import_human_dataset(settings.db_path, FIXTURES / "data" / f)
    import_synthetic_files(settings.db_path, [FIXTURES / "data" / f for f in
                                              ("model-a__default.jsonl", "model-b__casual.txt", "synthetic.csv")])
    with session(settings.db_path) as conn:  # tiny data: use everything for train, and test on it too
        conn.execute("UPDATE samples SET split = 'train'")
    version = train(settings, base_model=tiny_base_model(tmp_path / "base"), epochs=2, batch_size=4,
                    version="vsmoke", device="cpu")
    assert version == "vsmoke" and list_versions(settings.models_dir)[0]["version"] == "vsmoke"
    scorer = load_scorer(settings)
    scores = scorer.score(["ngl this slaps", "Embracing change is essential for growth! 🚀"])
    assert len(scores) == 2 and all(0.0 <= s <= 1.0 for s in scores)
    assert evaluate(settings, scorer=scorer)["version"] == "vsmoke"
