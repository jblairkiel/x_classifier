"""CPU inference with a versioned checkpoint. Returns calibrated P(AI) per text."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from tl.config import Settings
from tl.data.common import normalize_for_model
from tl.model.checkpoints import resolve_version
from tl.model.metrics import sigmoid


class NoModelError(RuntimeError):
    """No usable checkpoint (none trained yet, or torch/transformers not installed)."""


def pick_device(device: str = "auto") -> str:
    import torch

    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Scorer:
    def __init__(self, path: Path, version: str, *, device: str = "cpu", batch_size: int = 16,
                 max_length: int | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.path = path
        self.version = version
        self.meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        self.temperature = float(self.meta.get("temperature", 1.0))
        self.ai_index = int(self.meta.get("label2id", {}).get("ai", 1))
        self.human_index = 1 - self.ai_index
        self.max_length = int(max_length or self.meta.get("max_length", 128))
        self.batch_size = batch_size
        self.device = pick_device(device)
        if self.device == "cpu":
            torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", min(4, os.cpu_count() or 1))))
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
        self.model.to(self.device).eval()

    def margins(self, texts: list[str]) -> np.ndarray:
        """logit(ai) - logit(human), before temperature scaling."""
        torch = self._torch
        out: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = [normalize_for_model(t) for t in texts[start : start + self.batch_size]]
                enc = self.tokenizer(batch, truncation=True, max_length=self.max_length, padding=True,
                                     return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits.float()
                out.append((logits[:, self.ai_index] - logits[:, self.human_index]).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def score(self, texts: list[str]) -> list[float]:
        return [float(p) for p in sigmoid(self.margins(texts) / self.temperature)]


def load_scorer(settings: Settings, version: str | None = None, device: str = "cpu") -> Scorer:
    try:
        name, path = resolve_version(settings, version)
    except (FileNotFoundError, ValueError) as exc:
        raise NoModelError(str(exc)) from exc
    try:
        import torch  # noqa: F401  (availability check)
        import transformers  # noqa: F401
    except ImportError as exc:
        raise NoModelError(f"torch/transformers not installed in this container ({exc})") from exc
    return Scorer(path, name, device=device, batch_size=settings.classify_batch_size,
                  max_length=settings.max_length)
