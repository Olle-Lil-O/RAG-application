#!/usr/bin/env bash
set -euo pipefail

PDF_PATH="${PDF_PATH:-data/euaiact.pdf}"
SOURCE_NAME="${SOURCE_NAME:-$(basename "$PDF_PATH")}"
SPACY_MODEL="${SPACY_MODEL:-en_core_web_sm}"
MAX_SENTENCES="${MAX_SENTENCES:-5}"
CHUNKER="${CHUNKER:-spacy}"
BREAKPOINT_THRESHOLD_TYPE="${BREAKPOINT_THRESHOLD_TYPE:-percentile}"
CHUNKING_PROVIDER="${CHUNKING_PROVIDER:-local}"
CHUNKING_LOCAL_MODEL="${CHUNKING_LOCAL_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
CHUNKING_DEPLOYMENT="${CHUNKING_DEPLOYMENT:-${DEPLOY_MEDIUM:-}}"
MAX_EMBED_TOKENS="${MAX_EMBED_TOKENS:-2000}"
SPLIT_OVERLAP_TOKENS="${SPLIT_OVERLAP_TOKENS:-80}"

PGHOST="${PGHOST:-localhost}"
PGPORT="${PGPORT:-5431}"
PGDATABASE="${PGDATABASE:-postgres}"
PGUSER="${PGUSER:-postgres}"
PGPASSWORD="${PGPASSWORD:-password}"

NO_EXEC="${NO_EXEC:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EMPTY="${SKIP_EMPTY:-0}"

base_args=(
  --pdf-path "$PDF_PATH"
  --source "$SOURCE_NAME"
  --spacy-model "$SPACY_MODEL"
  --max-sentences "$MAX_SENTENCES"
  --chunker "$CHUNKER"
  --chunking-provider "$CHUNKING_PROVIDER"
  --chunking-local-model "$CHUNKING_LOCAL_MODEL"
  --chunking-deployment "$CHUNKING_DEPLOYMENT"
  --breakpoint-threshold-type "$BREAKPOINT_THRESHOLD_TYPE"
  --max-embed-tokens "$MAX_EMBED_TOKENS"
  --split-overlap-tokens "$SPLIT_OVERLAP_TOKENS"
  --pg-host "$PGHOST"
  --pg-port "$PGPORT"
  --pg-database "$PGDATABASE"
  --pg-user "$PGUSER"
  --pg-password "$PGPASSWORD"
)

if [[ "$DRY_RUN" == "1" ]]; then
  base_args+=(--dry-run)
fi

run_step() {
  echo "> Running: $1"
  if [[ "$NO_EXEC" == "1" ]]; then
    echo "  (NO_EXEC enabled; command not executed)"
    return 0
  fi
  "$@"
}

if [[ "$SKIP_EMPTY" != "1" ]]; then
  run_step bash scripts/empty_tables.sh
fi

run_step uv run python scripts/preprocess.py "${base_args[@]}" \
  --table knowledge_base_mini \
  --provider local \
  --local-model sentence-transformers/all-MiniLM-L6-v2

run_step uv run python scripts/preprocess.py "${base_args[@]}" \
  --table knowledge_base_sm \
  --provider local \
  --local-model BAAI/bge-large-en-v1.5

if [[ "$NO_EXEC" != "1" ]]; then
  : "${AZURE_ENDPOINT:?AZURE_ENDPOINT is required for Azure embeddings}"
  : "${AZURE_API_KEY:?AZURE_API_KEY is required for Azure embeddings}"
  : "${DEPLOY_MEDIUM:?DEPLOY_MEDIUM is required for knowledge_base_md}"
fi

AZURE_ENDPOINT_VAL="${AZURE_ENDPOINT:-NO_EXEC_ENDPOINT}"
AZURE_API_KEY_VAL="${AZURE_API_KEY:-NO_EXEC_KEY}"
DEPLOY_MEDIUM_VAL="${DEPLOY_MEDIUM:-NO_EXEC_DEPLOY_MEDIUM}"

run_step uv run python scripts/preprocess.py "${base_args[@]}" \
  --table knowledge_base_md \
  --provider azure \
  --deployment "$DEPLOY_MEDIUM_VAL" \
  --azure-endpoint "$AZURE_ENDPOINT_VAL" \
  --azure-api-key "$AZURE_API_KEY_VAL"

echo "Populate-all pipeline completed."
