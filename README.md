# RAG-application

## Web UIs

* **Loader (`app_loader.py`)** – Gradio UI mirroring `pipeline.py`’s options. Upload or point to a PDF, select mini/small/medium, tweak chunking/Azure overrides, and truncate/dry-run before writing. Launch with `uv run app_loader.py`; set `PROFILE` before launching if you want a different default target table (`md` is the default profile, so the loader starts on medium unless you override it).
* **Chat (`app_query.py`)** – Tool-enabled assistant that queries the populated tables. It checks `PROFILE` (`mini`, `sm`, or `md`, default `md`) to decide which table/embedding pair to hit and reads the corresponding embedding model from `DEPLOY_MINI`, `DEPLOY_SMALL`, or `DEPLOY_MEDIUM`. Run `uv run app_query.py` after loading data. To launch the UI with a specific profile, prefix the command with `PROFILE=mini uv run ...` (or `sm`/`md`).

Both UIs and the CLI share the same env-loading order: `project.env`, `.env`, then `--env-file` (pipeline default is `.env`).

## Local data population (single command)

```bash
uv run pipeline.py
```

That run truncates `knowledge_base_mini`, `knowledge_base_sm`, and `knowledge_base_md` (unless you set `--skip-empty`) and populates the mini/small tables by default.

## Prerequisites

Start the local DB/migrations:

```bash
docker compose up -d database
docker compose run --rm database-migrations
```

Install dependencies (one-time):

```bash
uv sync
```

## CLI options

```bash
uv run pipeline.py --help
```

### Table targeting

- `--mini`: populate only `knowledge_base_mini`
- `--small`: populate only `knowledge_base_sm`
- `--medium`: populate only `knowledge_base_md` (Azure creds required)
- `--all`: run mini + small + medium
- `--skip-empty`: skip the initial truncation step
- `--dry-run`: preprocess without writing

### Input overrides

- `--pdf-path`: override the PDF path
- `--source`: override the source name stored in the DB
- `--env-file`: extra env file loaded last (defaults to `.env`)

## Environment variables

All runners load env vars from `project.env`, then `.env`, then `--env-file` (if supplied).

### Local Postgres

```env
PGHOST=localhost
PGPORT=5431
PGDATABASE=postgres
PGUSER=postgres
PGPASSWORD=password
PGSSLMODE=disable
```

### Chunking and preprocessing defaults

```env
PDF_PATH=data/euaiact.pdf
SOURCE_NAME=euaiact.pdf

CHUNKER=spacy
SPACY_MODEL=en_core_web_sm
MAX_SENTENCES=5

CHUNKING_PROVIDER=local
CHUNKING_LOCAL_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNKING_DEPLOYMENT=
BREAKPOINT_THRESHOLD_TYPE=percentile

MAX_EMBED_TOKENS=2000
SPLIT_OVERLAP_TOKENS=80
```

### Azure embedding overrides (for `--medium`/`--all` or the medium loader step)

```env
AZURE_ENDPOINT=...
AZURE_API_KEY=...
DEPLOY_MEDIUM=...
AZURE_API_VERSION=2025-03-01-preview
```

### Embedding profile hints

Set `PROFILE` (`mini`, `sm`, or `md`) to tell `app_query.py` which table/embedding pair to use. The chat manager then pulls the embedding model from the matching `DEPLOY_*` env var, so swap in gated models as needed.

### Hugging Face

Provide `HF_TOKEN` to avoid Hugging Face rate-limit warnings, especially when using gated medium embeddings:

```env
HF_TOKEN=...
```
