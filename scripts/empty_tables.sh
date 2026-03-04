#!/usr/bin/env bash
set -euo pipefail

PGHOST="${PGHOST:-localhost}"
PGPORT="${PGPORT:-5431}"
PGDATABASE="${PGDATABASE:-postgres}"
PGUSER="${PGUSER:-postgres}"
PGPASSWORD="${PGPASSWORD:-password}"
PGSSLMODE="${PGSSLMODE:-disable}"

SQL="TRUNCATE TABLE knowledge_base_mini, knowledge_base_sm, knowledge_base_md RESTART IDENTITY;"
CONN="host=${PGHOST} port=${PGPORT} dbname=${PGDATABASE} user=${PGUSER} password=${PGPASSWORD} sslmode=${PGSSLMODE}"

if [[ "${NO_EXEC:-0}" == "1" ]]; then
  echo "[NO_EXEC] Would run: psql \"${CONN}\" -c \"${SQL}\""
  exit 0
fi

psql "${CONN}" -v ON_ERROR_STOP=1 -c "${SQL}"
echo "Tables truncated successfully."
