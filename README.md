# tl-ai-detector

Reads a bot account's X (Twitter) home timeline through a logged-in browser session, stores the
tweets in SQLite, and scores each one as likely AI-generated or human-written with a small
transformer **you train yourself**. It also ranks accounts by a "botlike" score. Everything runs on
your own server: no paid APIs, no external inference.

> [!WARNING]
> **Scores are probabilistic and produce false positives.** A high score means the text model sees
> patterns it learned from your synthetic AI samples. It is not proof that a person used AI, and a
> low score doesn't prove they didn't. Plenty of humans write in a polished, "LLM-ish" style, and
> modern models can write casually. Never use these scores to accuse, report or harass anyone.
> Check the `manual_timeline` metrics on the Model page before trusting anything.

> [!CAUTION]
> **Automated browsing may violate X's Terms of Service** and can get the account restricted or
> suspended. Use a **throwaway account**, keep the volume low (the defaults are deliberately
> conservative), and never use your main account. The scraper is read-only (it never posts, likes,
> follows, replies or DMs) and has no stealth or anti-detection features. When X shows a login
> wall, captcha, rate-limit or suspension notice, it stops and waits for you.

---

## Contents

1. [How it works](#how-it-works)
2. [Deploying to x.jblairkiel.com](#deploying-to-xjblairkielcom)
3. [Logging in to X (`tl login`, run locally)](#logging-in-to-x-tl-login-run-locally)
4. [Scraper behaviour, caps and the paused state](#scraper-behaviour-caps-and-the-paused-state)
5. [Labeling](#labeling)
6. [Training data](#training-data)
7. [Training and evaluation](#training-and-evaluation)
8. [Account botlike score](#account-botlike-score)
9. [Backups](#backups)
10. [Security notes](#security-notes)
11. [Development and tests](#development-and-tests)
12. [CLI and Makefile reference](#cli-and-makefile-reference)
13. [Troubleshooting](#troubleshooting)

## How it works

```
             Internet ──80/443──► proxy (Caddy, auto-HTTPS)
                                     │  internal network "edge" (no internet)
                                     ▼
                                   web (FastAPI + Jinja2 + htmx)       ── reads/writes ──┐
                                     ▲  labels, pause/resume, queues jobs                 │
                                     │                                         tl_data (SQLite)
   x.com ◄── headless Chromium ── worker (scheduler + scraper + CPU inference) ──────────┘
              (read-only)            │ reads tl_secrets (session, read-only) + tl_models (read-only)
                                     ▼
                         train (opt-in profile, CPU or GPU) ── writes ──► tl_models
```

* `web` serves the UI. It has **no internet access, no browser, no torch**, and never sees the X
  session file. Buttons such as "Run scrape now" and "Reclassify" queue jobs that the worker picks up.
* `worker` scrapes on a schedule within your caps, scores new tweets with the active checkpoint
  (CPU), and recomputes account features.
* `proxy` is the only container with published ports.
* `train` only runs when you ask for it (`--profile train`).

Repo layout and design decisions are in [docs/PLAN.md](docs/PLAN.md).

## Deploying to x.jblairkiel.com

### 1. DNS

At your DNS provider, create an **A record**:

| Type | Name | Value | TTL |
|---|---|---|---|
| A | `x` (i.e. `x.jblairkiel.com`) | your server's public IPv4 | 300 |

If the server has IPv6, add an AAAA record too, or leave it out entirely (a wrong AAAA record breaks
certificate issuance). Check it with `dig +short x.jblairkiel.com` before starting Caddy, which
requests the Let's Encrypt certificate on first start.

### 2. Firewall

| Port | Why |
|---|---|
| 22/tcp | SSH (better: restrict it to your IP) |
| 80/tcp | ACME HTTP challenge + redirect to HTTPS |
| 443/tcp | HTTPS |
| 443/udp | HTTP/3 (optional) |

Everything else stays closed. Example with ufw:

```bash
sudo ufw default deny incoming && sudo ufw default allow outgoing
sudo ufw allow 22/tcp && sudo ufw allow 80/tcp && sudo ufw allow 443/tcp && sudo ufw allow 443/udp
sudo ufw enable
```

Docker-published ports bypass ufw's INPUT rules. That's fine here, because only the proxy
publishes ports (80/443). Don't add `ports:` to the other services.

### 3. Configure and start

Requirements: Docker Engine 24+ with the Compose plugin, about 6 GB of disk for images, and 2 GB+ RAM.

```bash
git clone <this repo> tl-ai-detector && cd tl-ai-detector
cp .env.example .env && chmod 600 .env
make build                 # builds web, worker (Playwright + CPU torch) and proxy images
make hash-password         # paste the output into ADMIN_PASSWORD_HASH='...' (keep the single quotes)
python3 -c "import secrets; print(secrets.token_urlsafe(48))"   # -> SESSION_SECRET
$EDITOR .env               # DOMAIN, ACME_EMAIL, the two secrets, TZ, caps, schedule
make up
make ps                    # web + worker should become "healthy"
```

Open https://x.jblairkiel.com and log in with `ADMIN_USERNAME` and your password. The scraper stays
**paused with "session file missing"** until you complete the next section.

`make logs` (or `make logs S=worker`) follows the logs. Containers restart automatically
(`unless-stopped`), and logs are rotated (json-file, 10 MB × 5 per container).

### Trying it locally first

```bash
cp .env.example .env    # set ADMIN_PASSWORD_HASH and SESSION_SECRET; DOMAIN is overridden to localhost
make local-up           # https://localhost:8443 (Caddy's internal CA; accept the browser warning)
docker compose -f docker-compose.yml -f docker-compose.local.yml run --rm worker \
    tl scrape --fixture /fixtures/timeline_infinite.html     # scrape a saved page instead of x.com
make local-down
```

## Logging in to X (`tl login`, run locally)

Docker can't do an interactive login, so you create the browser session **on your own computer**
and copy only the resulting session file to the server. The tool never asks for, sees or stores
your password: you type it into X's own login page in a normal browser window.

```bash
# on your laptop (Python 3.11+)
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[scraper]"
playwright install chromium
tl login                     # opens a Chromium window on x.com/login
#   -> log in to the THROWAWAY bot account, wait for the home timeline, press Enter in the terminal
#   -> writes ./secrets/x_storage_state.json (mode 600, git-ignored)
```

Copy it to the server and load it into the secrets volume:

```bash
scp secrets/x_storage_state.json you@server:~/tl-ai-detector/secrets/
ssh you@server
cd ~/tl-ai-detector
make push-session                       # or: make push-session SESSION=/path/to/file.json
shred -u secrets/x_storage_state.json   # don't leave a plaintext copy around
```

Then press **Resume** on the dashboard. Treat this file like a password: anyone who has it is
logged in as the bot account. It is git-ignored and docker-ignored, so it never enters an image,
and it lives only in the `tl_secrets` volume, which only the worker mounts (read-only).

Sessions expire eventually. When that happens the scraper hits a login wall, pauses, and the UI
shows a banner. Run `tl login` again and push the new file.

## Scraper behaviour, caps and the paused state

* Opens `x.com/home`, selects the **Following** tab (`TIMELINE=following`) or **For you**
  (`for_you`), scrolls with a random pause between `SCROLL_DELAY_MIN_SECONDS` and
  `SCROLL_DELAY_MAX_SECONDS`, and extracts id, author handle, display name, text, timestamp,
  reply/repost/like counts, reply flag, repost flag (with who reposted), media flag and language.
  Promoted tweets are skipped. Tweets are deduplicated by id, and later sightings refresh the counts.
* **Caps:** `SCRAPE_MAX_TWEETS_PER_RUN`, `SCRAPE_MAX_SCROLLS_PER_RUN`, and a rolling 24-hour
  `SCRAPE_MAX_TWEETS_PER_DAY` across all runs, manual ones included. A run also stops after
  `SCRAPE_STALE_SCROLLS` scrolls in a row with nothing new.
* **Schedule:** every `SCRAPE_INTERVAL_MINUTES` ± `SCRAPE_INTERVAL_JITTER_MINUTES`, only inside
  `SCRAPE_ACTIVE_HOURS` (local time, `TZ`). "Run scrape now" on the dashboard queues a run that
  obeys the same caps.
* **Stop conditions:** a login wall, captcha, rate-limit message ("Something went wrong. Try
  reloading." counts), or a suspension/lock notice → state **paused** with the reason. Text inside
  tweets never triggers a stop.
* **Selector failures:** every DOM selector lives in `tl/scraper/selectors.py`. If the page has no
  tweet articles, or most articles miss required fields, the run fails loudly with state **error**
  rather than writing partial rows. Fix the selectors, add a fixture/test, rebuild, then Resume.
* **Nothing is retried automatically.** In paused or error state the schedule does nothing until you
  press **Resume** (or run `make resume`).

## Labeling

The **Label** page shows one unlabeled tweet at a time with big **Human / AI / Unsure** buttons
(keys <kbd>H</kbd> <kbd>A</kbd> <kbd>U</kbd>, <kbd>S</kbd> to skip). It works on a phone. The model's
score is hidden by default to avoid anchoring you, and tweets the model is least sure about come
first. Every tweet elsewhere in the UI has relabel chips. Relabeling appends a new label, and the
latest one wins.

These hand labels are your **held-out evaluation set. They are never used for training**, and
evaluation reports them separately as `manual_timeline`. `unsure` labels are excluded from metrics.
There's also a terminal version: `docker compose exec worker tl label`.

## Training data

See **[docs/DATA_FORMATS.md](docs/DATA_FORMATS.md)** for the formats, the pluggable reader API, the
normalization step and the leakage guard. In short:

```bash
# put files on the server, e.g. /srv/import, then:
docker compose --profile ops run --rm -v /srv/import:/import:ro ops import-human /import/human.csv
docker compose --profile ops run --rm -v /srv/import:/import:ro ops import-synthetic /import/synthetic/
docker compose --profile ops run --rm ops data-stats
```

Name synthetic files `<model>__<variant>.jsonl` (for example `llama-3.1-8b__casual.jsonl`) or put
`model`/`variant` in each row. Then the evaluation breaks results down per generator and
per prompt style.

## Training and evaluation

The model is `distilroberta-base` (`BASE_MODEL`, which can be any Hugging Face id or local path)
fine-tuned for binary classification with a class-balanced loss. The best epoch is chosen by
validation ROC-AUC, and a temperature is fitted on the validation split so scores are calibrated
probabilities. Checkpoints are versioned in the `tl_models` volume as `models/<version>/`, with
`metadata.json`, `eval.json` and a `COMPLETE` marker written last, so a crashed run is never
picked up.

### Option A: in Docker on the server (opt-in profile)

```bash
make train                                  # CPU (slow but fine for a few thousand samples)
make train ARGS="--epochs 4 --batch-size 32 --max-train 20000"
make train-gpu                              # NVIDIA GPU; needs the NVIDIA Container Toolkit
make eval                                   # newest checkpoint; or ARGS="--version v20261004-..."
```

The first run downloads the base model from Hugging Face into the `tl_hf_cache` volume. The web
and worker containers never train; they only run inference on CPU.

### Option B: train on another machine and copy the checkpoint in

```bash
# server: export the DB
make backup                                        # backups/tl-<stamp>.tar.gz (DB + checkpoints)
# laptop/GPU box:
scp server:~/tl-ai-detector/backups/tl-<stamp>.tar.gz . && mkdir -p var/data && tar -xzf tl-*.tar.gz -C var/data tl.sqlite3
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # or /whl/cpu, or plain `pip install torch` on macOS
pip install -e ".[ml]"
DATA_DIR=var/data MODELS_DIR=models tl train --device auto
DATA_DIR=var/data MODELS_DIR=models tl eval
# copy the checkpoint to the server and load it into the volume:
rsync -a models/v20261004-120000 server:~/tl-ai-detector/models/
ssh server 'cd ~/tl-ai-detector && make push-model VERSION=v20261004-120000'
```

### Activating a checkpoint

`MODEL_VERSION=latest` (the default) uses the newest complete checkpoint. Pin one with
`MODEL_VERSION=v20261004-120000` in `.env` and run `make up` again. After a new checkpoint arrives,
press **Reclassify everything** on the Model page (or `make classify`). New tweets are scored
automatically after each scrape.

### Reading the metrics

`tl eval` (and the Model page) report:

* **test**: the held-out 10% of your imported and synthetic corpus. This is usually optimistic,
  because it comes from the same distribution as training.
* **manual_timeline**: your hand labels on real timeline tweets, at threshold 0.5 and at
  `AI_THRESHOLD`. **This is the number that matters.** Below ~100 labels it's noisy.
* Precision, recall, F1, false-positive rate, ROC-AUC, Brier score, ECE (expected calibration
  error) and reliability bins, plus per-origin/variant accuracy on the test split.

## Account botlike score

`tl/features` computes, per account (most recent 60 tweets seen):

| Signal | Botlike when… | Weight |
|---|---|---|
| Mean text P(AI) | the model scores their tweets high | 0.45 |
| Posting cadence | gaps are metronome-regular (low CV) or volume is very high | 0.12 |
| Near-duplicate phrasing | tweets share char 5-grams (Jaccard ≥ 0.5) or reuse the same opener | 0.15 |
| Length variance | lengths are unusually uniform | 0.10 |
| Em-dash rate | many tweets contain "—" | 0.08 |
| Reply-to-post ratio | almost everything is a reply | 0.05 |
| Hashtags per tweet | heavy hashtag use | 0.05 |

The score is the weighted mean of the signals that can be computed, shrunk toward 0.25 when there
are few tweets (`confidence = n / (n + 5)`). It's a hand-tuned heuristic, not a trained model. The
account page shows each signal's value and contribution so you can judge it yourself. Cadence is
estimated only from tweets that reached the bot's timeline.

## Backups

```bash
make backup                                    # -> backups/tl-<UTC timestamp>.tar.gz
make restore FILE=backups/tl-20261004-030000.tar.gz   # stops web+worker, restores, starts them
```

The DB is copied with SQLite's online-backup API, so backups are consistent while everything runs.
The archive contains the database and all checkpoints. It does **not** include the X session file
(re-run `tl login` instead) or Caddy's certificates (re-issued automatically). Restore checks
`PRAGMA integrity_check` and refuses archives containing unexpected paths. Copy backups off the
server, e.g. with a cron job running `make backup && rclone copy backups remote:tl`.

## Security notes

* **Every route requires login** except `/healthz` (returns `{"ok": true}`), `/login`,
  `/robots.txt` and `/static/*`, which contains only CSS/JS. A default-deny middleware enforces
  this, and a test walks every registered route anonymously to check it.
* Single admin user. The password is an argon2id hash from `ADMIN_PASSWORD_HASH`; the app refuses
  to start without it or with a `SESSION_SECRET` shorter than 32 characters.
* Session cookie `tl_session`: signed, `HttpOnly`, `Secure`, `SameSite=Lax`, with an absolute
  expiry of `SESSION_MAX_AGE_HOURS`. The session is regenerated at login. Rotating `SESSION_SECRET`
  logs everyone out.
* CSRF: a synchronizer token on every POST (hidden form field or `X-CSRF-Token` header for htmx),
  plus an `Origin` check.
* Login rate limiting: `LOGIN_MAX_ATTEMPTS` failures per IP and `LOGIN_GLOBAL_MAX_ATTEMPTS` overall
  per `LOGIN_WINDOW_MINUTES`. It's in memory, so restarting `web` resets it.
* Headers: `X-Robots-Tag: noindex, nofollow`, a strict CSP (`script-src 'self'`, no inline JS),
  `X-Frame-Options: DENY`, HSTS, `Referrer-Policy: same-origin`, and `Cache-Control: no-store` on
  pages. `robots.txt` disallows everything.
* Scraped data, the DB, checkpoints and the session file are never served statically. They live
  in volumes that aren't mounted under any static route, and `.dockerignore` keeps them out of the
  images.
* Containers run as non-root (uid 1001 for web/worker/train, 1002 for Caddy) with read-only root
  filesystems, `cap_drop: ALL` and `no-new-privileges`. Caddy listens on unprivileged ports
  8080/8443 inside its container, so it needs no capabilities. Base images are pinned by tag
  **and** digest.

## Development and tests

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[web,scraper,dev]"
playwright install chromium     # browser tests are skipped if Chromium is missing
make test
```

The tests cover extraction against the saved HTML fixtures in `tests/fixtures/` (regenerate them
with `python tests/fixtures/make_fixtures.py`), dedupe, blocker detection, the pause and no-retry
behaviour, caps (a real headless Chromium scrolling a fixture page), dataset loaders, the leakage
guard, metrics and calibration, eval reporting, account features, backup/restore, and auth (every
non-public route must reject anonymous requests). `tests/test_train_smoke.py` trains a tiny
randomly initialised model end to end when torch and transformers are installed.

When X changes its markup: save the real timeline HTML (DevTools → copy the `outerHTML` of the
timeline), add it as a fixture with a test, update `tl/scraper/selectors.py` until the test passes,
then rebuild the worker.

## CLI and Makefile reference

| `tl …` | |
|---|---|
| `login` | local, headed browser login → storage-state file |
| `scrape [--fixture page.html]` | one scrape run now (respects caps and the paused state) |
| `resume` / `pause` / `status` | scraper state |
| `label` | label in the terminal |
| `import-human`, `import-synthetic`, `data-stats` | training data |
| `train`, `eval`, `classify [--rescore]`, `predict "text"` | model |
| `features` | recompute account scores |
| `serve`, `worker`, `worker-health` | what the containers run |
| `hash-password`, `install-session`, `backup`, `restore` | ops |

`make help` lists the Makefile targets: build, up, down, restart, logs, ps, status, resume,
push-session, push-model, hash-password, backup, restore, train, train-gpu, eval, classify,
features, test, local-up and local-down.

Every setting is documented in [.env.example](.env.example).

## Troubleshooting

* **Banner "session file missing"**: run `tl login` locally, then `make push-session`, then Resume.
* **Banner "login_wall"**: the session expired or X logged the bot out. Same fix.
* **Banner "rate_limit" / "captcha" / "suspended"**: stop. Wait (hours to days), check the account
  in a normal browser, consider lowering the caps or raising the interval, then Resume. Don't push
  through captchas.
* **State "error: selector failure"**: X changed its DOM. See [Development and tests](#development-and-tests).
* **Caddy can't get a certificate**: check the A record, make sure ports 80/443 are open and that
  nothing else listens on them, then read `make logs S=proxy`.
* **"no model" in job messages**: no complete checkpoint in `tl_models` yet. Train one or `make push-model`.
* **Building behind a TLS-intercepting proxy**: pass its CA as a BuildKit secret, e.g.
  `docker build --secret id=extra_ca,src=/path/ca.crt ...`. The Dockerfiles use it only for pip.
