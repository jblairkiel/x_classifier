"""`tl` command line."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from tl.config import get_settings


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """tl-ai-detector: read a bot account's home timeline and score tweets as AI vs human."""


# --------------------------------------------------------------------------- scraping


@main.command()
@click.option("--out", "out", type=click.Path(path_type=Path), default=None,
              help="Where to write the storage state (default: $SECRETS_DIR/$STORAGE_STATE_FILE).")
def login(out: Path | None) -> None:
    """Log in LOCALLY in a visible browser and save the session (never your password)."""
    from tl.scraper.login import interactive_login

    settings = get_settings()
    path = interactive_login(out or settings.storage_state_path)
    click.echo(f"\nSaved session to {path} (mode 600).")
    click.echo("Copy it to the server's secrets volume, e.g.:  make push-session SESSION=" + str(path))


@main.command()
@click.option("--fixture", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
              help="Scrape a saved HTML page instead of x.com (testing only).")
def scrape(fixture: Path | None) -> None:
    """Run one scrape session now (respects caps and the paused state)."""
    import logging

    from tl.scraper.browser import run_scrape

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()
    if fixture:
        out = run_scrape(settings, trigger="cli", home_url=fixture.resolve().as_uri(), use_storage_state=False)
    else:
        out = run_scrape(settings, trigger="cli")
    click.echo(json.dumps(out.__dict__, indent=2))
    sys.exit(0 if out.status in ("ok", "skipped") else 2)


@main.command()
def resume() -> None:
    """Clear a paused/error state so scheduled scraping can continue."""
    from tl.db import session
    from tl.db import queries as Q

    with session(get_settings().db_path) as conn:
        prev = Q.get_state(conn)
        Q.set_state(conn, "idle", f"resumed from {prev['status']} via CLI")
    click.echo(f"was {prev['status']} ({prev['reason']}); now idle")


@main.command()
@click.option("--reason", default="paused via CLI")
def pause(reason: str) -> None:
    """Pause scheduled scraping."""
    from tl.db import session
    from tl.db import queries as Q

    with session(get_settings().db_path) as conn:
        Q.set_state(conn, "paused", reason)
    click.echo("paused")


@main.command()
def status() -> None:
    """Show scraper state, counts and recent runs."""
    from tl.db import session
    from tl.db import queries as Q

    with session(get_settings().db_path) as conn:
        click.echo(json.dumps(
            {
                "state": dict(Q.get_state(conn)),
                "counts": Q.counts(conn),
                "seen_last_24h": Q.tweets_seen_last_24h(conn),
                "recent_runs": [dict(r) for r in Q.recent_runs(conn, 5)],
            },
            indent=2,
        ))


@main.command()
def worker() -> None:
    """Run the scheduler loop (used by the worker container)."""
    from tl.worker import run_forever

    run_forever(get_settings())


@main.command("worker-health")
def worker_health() -> None:
    """Exit 0 if the worker heartbeat is fresh (Docker healthcheck)."""
    from tl.worker import healthy

    sys.exit(0 if healthy() else 1)


# --------------------------------------------------------------------------- web


@main.command()
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=8000, type=int)
@click.option("--proxy-headers/--no-proxy-headers", default=False,
              help="Trust X-Forwarded-* (only behind the Caddy proxy).")
def serve(host: str, port: int, proxy_headers: bool) -> None:
    """Run the web UI."""
    import uvicorn

    from tl.web.app import create_app

    uvicorn.run(create_app(), host=host, port=port, proxy_headers=proxy_headers,
                forwarded_allow_ips="*" if proxy_headers else None, server_header=False,
                access_log=True)


@main.command("hash-password")
@click.option("--password", prompt=True, hide_input=True, confirmation_prompt=True)
def hash_password(password: str) -> None:
    """Print an argon2 hash for ADMIN_PASSWORD_HASH."""
    from tl.web.auth import hash_password as _hash

    if len(password) < 12:
        raise click.BadParameter("use at least 12 characters")
    click.echo(_hash(password))


# --------------------------------------------------------------------------- labeling


@main.command()
@click.option("--limit", default=50, help="Stop after this many labels.")
def label(limit: int) -> None:
    """Label unlabeled timeline tweets in the terminal: h=human a=ai u=unsure s=skip q=quit."""
    from tl.db import session
    from tl.db import queries as Q
    from tl.model.checkpoints import resolve_version_or_none

    settings = get_settings()
    version = resolve_version_or_none(settings)
    skipped: list[str] = []
    done = 0
    while done < limit:
        with session(settings.db_path) as conn:
            row = Q.next_unlabeled(conn, version, skipped)
        if row is None:
            click.echo("Nothing left to label.")
            break
        click.echo("\n" + "-" * 70)
        score = f"  model P(ai)={row['score']:.2f}" if row["score"] is not None else ""
        click.echo(f"@{row['author_handle']}  {row['created_at'] or ''}{score}")
        click.echo(row["text"])
        key = click.prompt("[h]uman [a]i [u]nsure [s]kip [q]uit", type=click.Choice(list("hausq")),
                           show_choices=False)
        if key == "q":
            break
        if key == "s":
            skipped.append(row["id"])
            continue
        with session(settings.db_path) as conn:
            Q.add_manual_label(conn, row["id"], {"h": "human", "a": "ai", "u": "unsure"}[key])
        done += 1
    click.echo(f"labelled {done}")


# --------------------------------------------------------------------------- data


@main.command("import-human")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--format", "fmt", type=click.Choice(["auto", "csv", "jsonl", "txt"]), default="auto")
@click.option("--text-column", default=None, help="Column/key holding the tweet text (default: auto-detect).")
@click.option("--name", default=None, help="Dataset name recorded as origin (default: file name).")
@click.option("--limit", type=int, default=None)
def import_human(path: Path, fmt: str, text_column: str | None, name: str | None, limit: int | None) -> None:
    """Import a public HUMAN tweet dataset from a local file."""
    from tl.data.human import import_human_dataset

    stats = import_human_dataset(get_settings().db_path, path, fmt=fmt, text_column=text_column,
                                 name=name, limit=limit)
    click.echo(json.dumps(stats, indent=2))


@main.command("import-synthetic")
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--model", "model_name", default=None, help="Generator model name if not in the file.")
@click.option("--variant", default=None, help="Prompt variant (e.g. casual) if not in the file.")
def import_synthetic(paths: tuple[Path, ...], model_name: str | None, variant: str | None) -> None:
    """Import synthetic AI tweets (see docs/DATA_FORMATS.md)."""
    from tl.data.synthetic import import_synthetic_files

    stats = import_synthetic_files(get_settings().db_path, list(paths), model=model_name, variant=variant)
    click.echo(json.dumps(stats, indent=2))


@main.command("data-stats")
def data_stats() -> None:
    """Counts of training samples by source/origin/label and of manual labels."""
    from tl.db import session
    from tl.db import queries as Q

    with session(get_settings().db_path) as conn:
        for r in Q.sample_origin_counts(conn):
            click.echo(f"{r['source']:9} {r['origin']:30} {r['variant']:10} {r['label']:6} {r['n']}")
        click.echo(json.dumps(Q.label_counts(conn), indent=2))


# --------------------------------------------------------------------------- model


@main.command()
@click.option("--base-model", default=None, help="HF model id or local path (default $BASE_MODEL).")
@click.option("--epochs", default=3, type=int)
@click.option("--batch-size", default=16, type=int)
@click.option("--lr", default=3e-5, type=float)
@click.option("--max-train", default=None, type=int, help="Cap training rows (quick runs).")
@click.option("--version", "version", default=None, help="Checkpoint name (default: timestamp).")
@click.option("--device", default="auto", type=click.Choice(["auto", "cpu", "cuda", "mps"]))
@click.option("--seed", default=13, type=int)
def train(**kwargs) -> None:
    """Fine-tune the classifier on synthetic + dataset samples (never on manual labels)."""
    from tl.model.train import train as _train

    version = _train(get_settings(), **kwargs)
    click.echo(f"trained {version}; run `tl eval --version {version}`")


@main.command("eval")
@click.option("--version", "version", default=None, help="Checkpoint (default $MODEL_VERSION).")
@click.option("--device", default="auto", type=click.Choice(["auto", "cpu", "cuda", "mps"]))
def eval_cmd(version: str | None, device: str) -> None:
    """Evaluate on the held-out test split AND, separately, on your hand-labelled timeline tweets."""
    from tl.model.eval import evaluate

    report = evaluate(get_settings(), version=version, device=device)
    click.echo(json.dumps(report, indent=2))


@main.command()
@click.option("--rescore", is_flag=True, help="Delete and recompute scores for the active model.")
def classify(rescore: bool) -> None:
    """Score unscored timeline tweets with the active model (CPU)."""
    from tl.worker import classify_pending

    click.echo(classify_pending(get_settings(), rescore=rescore))


@main.command()
def features() -> None:
    """Recompute account-level signals and botlike scores."""
    from tl.worker import compute_features

    click.echo(compute_features(get_settings()))


@main.command()
@click.argument("text")
def predict(text: str) -> None:
    """Score a single piece of text."""
    from tl.model.predict import load_scorer

    scorer = load_scorer(get_settings())
    click.echo(f"{scorer.version}: P(ai) = {scorer.score([text])[0]:.3f}")


# --------------------------------------------------------------------------- ops


@main.command()
@click.argument("dest", type=click.Path(file_okay=False, path_type=Path))
@click.option("--models/--no-models", default=True)
def backup(dest: Path, models: bool) -> None:
    """Consistent copy of the SQLite DB (+ model checkpoints) into DEST."""
    from tl.ops import backup as _backup

    click.echo(_backup(get_settings(), dest, include_models=models))


@main.command()
@click.argument("src", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--yes", is_flag=True, help="Overwrite without asking.")
def restore(src: Path, yes: bool) -> None:
    """Restore a directory produced by `tl backup` (stop web+worker first)."""
    from tl.ops import restore as _restore

    if not yes:
        click.confirm(f"Overwrite the current DB/models with {src}?", abort=True)
    click.echo(_restore(get_settings(), src))


if __name__ == "__main__":
    main()
