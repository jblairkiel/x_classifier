"""Versioned checkpoint directory layout. Pure Python (no torch) so the web UI can read it.

    $MODELS_DIR/
      v20261004-0130/
        config.json, model.safetensors, tokenizer.json, ...   (Hugging Face save_pretrained)
        metadata.json    training config, data counts, temperature, base model
        eval.json        metrics written by `tl eval`
        COMPLETE         written last; a directory without it is ignored (crashed/partial run)
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from tl.config import Settings

VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
COMPLETE_MARKER = "COMPLETE"


def new_version_name() -> str:
    return "v" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def validate_version(name: str) -> str:
    if not VERSION_RE.match(name) or name in {".", ".."}:
        raise ValueError(f"invalid model version name {name!r}")
    return name


def list_versions(models_dir: Path) -> list[dict]:
    """Complete checkpoints, newest first, with their metadata/eval (if present)."""
    out = []
    if not models_dir.is_dir():
        return out
    for d in models_dir.iterdir():
        if not d.is_dir() or not (d / COMPLETE_MARKER).exists() or not VERSION_RE.match(d.name):
            continue
        meta = _read_json(d / "metadata.json") or {}
        out.append({
            "version": d.name,
            "path": d,
            "created_at": meta.get("created_at", ""),
            "metadata": meta,
            "eval": _read_json(d / "eval.json"),
        })
    out.sort(key=lambda v: (v["created_at"], v["version"]), reverse=True)
    return out


def resolve_version(settings: Settings, requested: str | None = None) -> tuple[str, Path]:
    """Map MODEL_VERSION ("latest" or a name) to an existing checkpoint directory."""
    name = requested or settings.model_version
    if name == "latest":
        versions = list_versions(settings.models_dir)
        if not versions:
            raise FileNotFoundError(f"no complete checkpoints in {settings.models_dir}")
        return versions[0]["version"], versions[0]["path"]
    validate_version(name)
    path = settings.models_dir / name
    if not (path / COMPLETE_MARKER).exists():
        raise FileNotFoundError(f"checkpoint {path} missing or incomplete")
    return name, path


def resolve_version_or_none(settings: Settings) -> str | None:
    try:
        return resolve_version(settings)[0]
    except (FileNotFoundError, ValueError):
        return None


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
