"""Fine-tune a small transformer (DistilRoBERTa by default) for human-vs-AI tweet classification.

Training data = imported datasets (human) + your synthetic files (AI), train split only.
Your hand labels from the timeline are NEVER used here: they don't live in `samples`, and any
sample whose text matches a scraped timeline tweet is dropped (see tl.model.dataset).

Runs on CPU, CUDA or Apple MPS. Writes a versioned checkpoint atomically:
models/.partial-<version>/ -> models/<version>/ with a COMPLETE marker written last.
"""

from __future__ import annotations

import logging
import math
import platform
import random
import shutil
import time
from collections import Counter
from datetime import datetime, timezone

import numpy as np

from tl.config import Settings
from tl.data.common import normalize_for_model
from tl.db import session
from tl.model import metrics as M
from tl.model.checkpoints import COMPLETE_MARKER, new_version_name, validate_version, write_json
from tl.model.dataset import LABEL2ID, Example, load_split, stratified_cap, timeline_hashes

log = logging.getLogger("tl.train")


def _batches(examples: list[Example], batch_size: int, shuffle: bool, rng: random.Random):
    idx = list(range(len(examples)))
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        yield [examples[i] for i in idx[start : start + batch_size]]


def _margins(model, tokenizer, examples, *, device, batch_size, max_length, torch) -> np.ndarray:
    model.eval()
    out = []
    with torch.inference_mode():
        for batch in _batches(examples, batch_size * 2, False, random.Random(0)):
            enc = tokenizer([normalize_for_model(e.text) for e in batch], truncation=True, max_length=max_length,
                            padding=True, return_tensors="pt").to(device)
            logits = model(**enc).logits.float()
            out.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
    model.train()
    return np.concatenate(out) if out else np.zeros(0)


def train(
    settings: Settings,
    *,
    base_model: str | None = None,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 3e-5,
    max_train: int | None = None,
    version: str | None = None,
    device: str = "auto",
    seed: int = 13,
) -> str:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    from tl.model.predict import pick_device

    base_model = base_model or settings.base_model
    version = validate_version(version or new_version_name())
    final_dir = settings.models_dir / version
    if final_dir.exists():
        raise FileExistsError(f"{final_dir} already exists")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)

    with session(settings.db_path) as conn:
        exclude = timeline_hashes(conn)
        train_ex, dropped_train = load_split(conn, "train", exclude=exclude)
        val_ex, dropped_val = load_split(conn, "val", exclude=exclude)
    train_ex = stratified_cap(train_ex, max_train, seed)
    counts = Counter(e.label for e in train_ex)
    if len(counts) < 2:
        raise RuntimeError(
            f"training needs both classes; have {dict(counts)} (0=human, 1=ai). "
            "Import a human dataset (tl import-human) and synthetic AI text (tl import-synthetic)."
        )
    if not val_ex:
        log.warning("validation split is empty; using 10%% of train for validation")
        rng.shuffle(train_ex)
        cut = max(1, len(train_ex) // 10)
        val_ex, train_ex = train_ex[:cut], train_ex[cut:]

    dev = pick_device(device)
    log.info("version=%s base=%s device=%s train=%d %s val=%d (dropped %d/%d overlapping timeline tweets)",
             version, base_model, dev, len(train_ex), dict(counts), len(val_ex), dropped_train, dropped_val)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=2, id2label={v: k for k, v in LABEL2ID.items()}, label2id=LABEL2ID,
    ).to(dev)

    # Class-balanced loss: synthetic and human sets rarely have equal sizes.
    n = len(train_ex)
    weights = torch.tensor([n / (2 * counts[0]), n / (2 * counts[1])], dtype=torch.float32, device=dev)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    params = [
        {"params": [p for nme, p in model.named_parameters() if not any(nd in nme for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for nme, p in model.named_parameters() if any(nd in nme for nd in no_decay)], "weight_decay": 0.0},
    ]
    optim = torch.optim.AdamW(params, lr=lr)
    steps_per_epoch = math.ceil(len(train_ex) / batch_size)
    total_steps = steps_per_epoch * epochs
    sched = get_linear_schedule_with_warmup(optim, int(0.06 * total_steps), total_steps)

    val_y = np.array([e.label for e in val_ex])
    best = {"score": -1.0, "epoch": 0, "state": None, "metrics": None}
    history = []
    model.train()
    started = time.time()
    for epoch in range(1, epochs + 1):
        running, seen = 0.0, 0
        for step, batch in enumerate(_batches(train_ex, batch_size, True, rng), 1):
            enc = tokenizer([normalize_for_model(e.text) for e in batch], truncation=True,
                            max_length=settings.max_length, padding=True, return_tensors="pt").to(dev)
            labels = torch.tensor([e.label for e in batch], device=dev)
            loss = loss_fn(model(**enc).logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            running += loss.item() * len(batch)
            seen += len(batch)
            if step % 50 == 0 or step == steps_per_epoch:
                log.info("epoch %d step %d/%d loss %.4f", epoch, step, steps_per_epoch, running / seen)

        margins = _margins(model, tokenizer, val_ex, device=dev, batch_size=batch_size,
                           max_length=settings.max_length, torch=torch)
        rep = M.classification_report(val_y, M.sigmoid(margins))
        rep.pop("reliability", None)
        score = rep["roc_auc"] if rep["roc_auc"] is not None else rep["accuracy"]
        history.append({"epoch": epoch, "train_loss": running / max(seen, 1), "val": rep, "val_nll": M.nll(val_y, margins)})
        log.info("epoch %d val roc_auc=%s f1=%s acc=%.4f", epoch, rep["roc_auc"], rep["f1"], rep["accuracy"])
        if score > best["score"]:
            best = {"score": score, "epoch": epoch, "metrics": rep,
                    "state": {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}}

    model.load_state_dict(best["state"])
    val_margins = _margins(model, tokenizer, val_ex, device=dev, batch_size=batch_size,
                           max_length=settings.max_length, torch=torch)
    temperature = M.fit_temperature(val_y, val_margins)
    calibrated = M.sigmoid(val_margins / temperature)
    log.info("best epoch %d; temperature %.3f (val ECE %.4f -> %.4f)", best["epoch"], temperature,
             M.ece(val_y, M.sigmoid(val_margins)), M.ece(val_y, calibrated))

    settings.models_dir.mkdir(parents=True, exist_ok=True)
    partial = settings.models_dir / f".partial-{version}"
    if partial.exists():
        shutil.rmtree(partial)
    model.to("cpu")
    model.save_pretrained(partial)
    tokenizer.save_pretrained(partial)
    origins = Counter(f"{e.source}:{e.origin}:{e.variant or ''}" for e in train_ex)
    write_json(partial / "metadata.json", {
        "version": version,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "base_model": base_model,
        "label2id": LABEL2ID,
        "max_length": settings.max_length,
        "temperature": temperature,
        "epochs": epochs, "best_epoch": best["epoch"], "batch_size": batch_size, "lr": lr, "seed": seed,
        "device": dev, "train_seconds": round(time.time() - started, 1),
        "n_train": len(train_ex), "n_val": len(val_ex),
        "class_counts": {"human": counts[0], "ai": counts[1]},
        "train_origins": dict(origins.most_common()),
        "dropped_overlap": {"train": dropped_train, "val": dropped_val},
        "val_metrics": best["metrics"],
        "val_ece_after_temperature": M.ece(val_y, calibrated),
        "history": history,
        "versions": {"torch": torch.__version__, "python": platform.python_version()},
        "note": "Manual timeline labels are never used for training.",
    })
    (partial / COMPLETE_MARKER).write_text(version)
    partial.rename(final_dir)
    log.info("saved %s", final_dir)
    return version
