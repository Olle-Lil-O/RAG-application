# RAG-application

## Local data population (single command)

Use `pipeline.py` from the project root. This is the recommended cross-platform flow for everyone.

```bash
uv run pipeline.py
```

The pipeline targets your local Docker PostgreSQL (via `.env`) and populates:

- `knowledge_base_mini` (local embeddings)
- `knowledge_base_sm` (local embeddings)

It also truncates `knowledge_base_mini`, `knowledge_base_sm`, and `knowledge_base_md` before loading (unless disabled with options below).

Note: The same dataset/database setup is also available on a remote Azure PostgreSQL Flexible Server. Connection details can be requested from Paavo.

## Prerequisites

Start local DB + migrations:

```bash
docker compose up -d database
docker compose run --rm database-migrations
```

Install deps (first time only):

```bash
uv sync
```

## CLI options

```bash
uv run pipeline.py --help
```

Available options:

- `--env-file <path>`: extra env file loaded last (default: `.env`)
- `--pdf-path <path>`: override input PDF path
- `--source <name>`: override source name stored in DB
- `--skip-empty`: do not truncate tables before loading
- `--dry-run`: run preprocessing without DB writes
- `--mini`: populate only `knowledge_base_mini`
- `--small`: populate only `knowledge_base_sm`
- `--medium`: populate only `knowledge_base_md`
- `--all`: populate mini + small + medium

## Common examples

Default local run:

```bash
uv run pipeline.py
```

Non-destructive check:

```bash
uv run pipeline.py --dry-run
```

Use a custom PDF:

```bash
uv run pipeline.py --pdf-path data/mydoc.pdf --source mydoc.pdf
```

Populate only medium (keep existing mini/small rows):

```bash
uv run pipeline.py --medium --skip-empty
```

Populate all three tables in one run:

```bash
uv run pipeline.py --all
```

## Environment variables

`pipeline.py` loads env files in this order:

1. `project.env`
2. `.env`
3. `--env-file` (defaults to `.env`)

### Required for local container DB

Set these in `.env`:

```dotenv
PGHOST=localhost
PGPORT=5431
PGDATABASE=postgres
PGUSER=postgres
PGPASSWORD=password
PGSSLMODE=disable
```

### Chunking and preprocessing

Optional, with defaults shown:

```dotenv
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

### Azure variables (required when using `--medium` or `--all`)

```dotenv
AZURE_ENDPOINT=...
AZURE_API_KEY=...
DEPLOY_MEDIUM=...
AZURE_API_VERSION=2025-03-01-preview
```

