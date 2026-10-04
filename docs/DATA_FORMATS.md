# Data formats

Training uses two kinds of text, both stored in the `samples` table and never mixed with the
timeline tweets you label by hand:

| Kind | Command | Label | `labels.source` |
|---|---|---|---|
| Public human-written tweets | `tl import-human PATH` | human | `dataset` |
| AI text you generated | `tl import-synthetic PATH...` | ai | `synthetic` |
| Your hand labels on timeline tweets | Label page / `tl label` | human / ai / unsure | `manual` (eval only) |

Nothing is downloaded by these commands; point them at files you already have. Inside Docker,
mount the directory with the files, for example:

```bash
docker compose --profile ops run --rm -v /srv/import:/import:ro ops import-human /import/tweets.csv
docker compose --profile ops run --rm -v /srv/import:/import:ro ops import-synthetic /import/synthetic/
docker compose --profile ops run --rm ops data-stats
```

Every sample gets a deterministic split (≈80/10/10 train/val/test) from a hash of its normalized
text, so re-importing never moves a sample between splits. Exact duplicates (after normalization)
are skipped.

## Human datasets (`tl import-human`)

* **CSV with a header**: the text column is auto-detected from `text`, `tweet`, `full_text`,
  `content`, `body`, `tweet_text`, `message` (case-insensitive), or set with `--text-column NAME`.
* **CSV without a header**: pass the 0-based column index, e.g. Sentiment140:
  `tl import-human training.1600000.processed.noemoticon.csv --text-column 5 --name sentiment140`.
  (That file is Latin-1. Convert it with `iconv -f latin1 -t utf-8` first, otherwise undecodable
  bytes are replaced.)
* **JSONL**: one object per line; text key auto-detected as above, or `--text-column`.
* **TXT**: one tweet per line.

Filters: 15–1000 characters, and `RT @user:` retweets are dropped. `--limit N` caps the number of
imported rows, and `--name` sets the origin recorded for the rows (the default is the file name).

## Synthetic AI text (`tl import-synthetic`)

Pass files or directories (directories are searched recursively). Files are read by extension:

### `.jsonl` (recommended)

```json
{"text": "Embracing change is the first step toward growth! 🚀", "model": "llama-3.1-8b", "variant": "default", "prompt": "write a tweet about change"}
{"text": "ngl mondays hit different when u actually slept lol", "model": "llama-3.1-8b", "variant": "casual", "temperature": 1.1}
```

Only `text` is required. `model` and `variant` are optional. Every other key (prompt,
temperature, seed, …) is kept as metadata.

### `.csv`

A header row with a `text` column, plus optional `model` and `variant` columns. Other columns are
kept as metadata.

### `.txt`

One tweet per line. For multi-line tweets, separate the tweets with a line containing only `---`.

### Model and variant resolution

For each row the first available value wins:

1. the row's own `model` / `variant` field
2. the `--model` / `--variant` flags
3. the file name convention `<model>__<variant>.<ext>`, e.g. `gpt-x__casual.jsonl`,
   `mistral-7b__default.txt`
4. `unknown` / `default`

Use a distinct `variant` for your "write casually like a human" prompts (for example `casual`).
`tl eval` reports accuracy per origin and variant on the test split, so you can see whether
casual-style generations slip through.

### Adding a format

Readers are pluggable. In your own module (imported before running the importer), register a
generator that yields `SyntheticItem`s:

```python
from tl.data.synthetic import SyntheticItem, register_reader

@register_reader(".parquet")
def read_parquet(path):
    import pandas as pd
    for row in pd.read_parquet(path).itertuples():
        yield SyntheticItem(row.text, model=row.model, variant=row.variant, meta={"id": row.id})
```

## What the model sees

`tl.data.common.normalize_for_model` runs at both training and inference time. It replaces URLs
with `http` and @mentions with `@user`, and strips `RT @x:` prefixes. Without this, a model can
learn shortcuts like "public human datasets contain @mentions, my synthetic text doesn't".

Other shortcuts are still possible: length, emoji, era-specific slang (an old dataset is full of
2009 slang), and topic. Try to match topics and lengths between your human and synthetic sets.
Trust the **manual_timeline** eval numbers over the corpus test split.

## Leakage guard

Manual labels can only point at timeline tweets, and training only reads `samples`. On top of
that, any sample whose normalized text matches *any* scraped timeline tweet is dropped from
train, val and test (`tl.model.dataset.timeline_hashes`).
