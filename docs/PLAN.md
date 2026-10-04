# tl-ai-detector — plan & layout

## Plan

A read-only bot account's home timeline is scrolled slowly by a headless Chromium
(Playwright) running inside the `worker` container, using a storage-state file
produced locally by `tl login`. Tweets land in SQLite. A small fine-tuned
transformer (DistilRoBERTa by default) scores each tweet's text for "AI-generated",
account-level heuristics are combined with those scores into a "botlike" score,
and a FastAPI + Jinja2 + HTMX UI (behind Caddy, single admin login) shows it all.

Key decisions

| Area | Decision | Why |
|---|---|---|
| Frontend | Server-rendered Jinja2 + vendored HTMX, no build step | smallest moving parts; strict CSP (`script-src 'self'`) |
| Extraction | Playwright locates `article` nodes and hands their `outerHTML` to a pure-Python BeautifulSoup parser | the same parser runs against saved HTML fixtures in unit tests, no browser needed |
| Selectors | every CSS selector / text pattern lives in `tl/scraper/selectors.py`; misses raise `SelectorError` and the scraper goes to `error` | fail loudly, single place to fix |
| Safety stops | login wall / captcha / rate-limit / suspension → state `paused` + UI banner; no automatic retry; resume is manual | requirement |
| Jobs | web never touches the browser or the model; it enqueues rows in a `jobs` table (`scrape`, `classify`, `features`) that the worker polls | web image stays tiny (no torch, no browser), secrets volume is not mounted in web at all |
| Labels | one `labels` table, `source ∈ {manual, synthetic, dataset}`; manual labels point at timeline tweets, synthetic/dataset labels point at rows of a separate `samples` corpus | manual labels are an eval-only set; training queries exclude `manual` *and* any sample whose normalized text matches a manually-labelled tweet |
| Model | custom PyTorch loop (no `accelerate`), temperature scaling fitted on the validation split, checkpoints in `models/<version>/` with `metadata.json` + `eval.json` | versioned, calibrated, CPU or GPU |
| Metrics | implemented with numpy (P/R/F1, ROC-AUC, Brier, ECE + reliability bins) | avoid scikit-learn in the images |
| Auth | argon2 hash from env, signed cookie session (HttpOnly, Secure, SameSite=Lax), synchronizer CSRF token, per-IP + global login rate limiting | requirement |
| Caddy | runs non-root on 8080/8443 inside the container, host 80/443 mapped to them | no capabilities needed |

## Repo layout

```
tl/                         python package (CLI entry point: `tl`)
  config.py                 every setting, read from env (.env.example mirrors it)
  cli.py                    tl login|scrape|label|train|eval|classify|serve|worker|...
  db/                       schema.sql + thin sqlite helpers / queries
  scraper/
    selectors.py            ALL DOM selectors + blocker patterns
    extract.py              pure HTML -> TweetRecord parser
    browser.py              Playwright session: open timeline, scroll, caps, stop conditions
    login.py                headed local login -> storage state file
  worker/                   scheduler loop, job runner, healthcheck
  data/                     human-dataset importer, synthetic-AI importer (pluggable formats)
  model/                    train.py, eval.py, predict.py, metrics.py, checkpoints.py
  features/                 account-level signals + botlike score
  web/                      FastAPI app, auth, CSRF, templates/, static/
tests/                      pytest; fixtures/ has saved timeline HTML + sample datasets
docker/                     Dockerfiles (web, worker, train), Caddyfile
docker-compose.yml          web, worker, proxy (+ train under --profile train)
docker-compose.local.yml    localhost override for testing
Makefile, scripts/          build/up/down/logs/backup/restore
```

## Stages
1. scraper + DB + worker
2. auth + web UI shell + Docker/Caddy
3. labeling UI
4. training / eval
5. features + full dashboard
