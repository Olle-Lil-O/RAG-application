#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: source scripts/use-env.sh <local|remote|path-to-env-file>"
  return 1 2>/dev/null || exit 1
fi

profile="$1"
case "$profile" in
  local)
    env_file=".env"
    ;;
  remote)
    env_file="remote/remote.env"
    ;;
  *)
    env_file="$profile"
    ;;
esac

if [[ ! -f "$env_file" ]]; then
  echo "Env file not found: $env_file"
  return 1 2>/dev/null || exit 1
fi

set -a
source "$env_file"
if [[ -f ".env" && "$env_file" != ".env" ]]; then
  source ".env"
fi
set +a

echo "Loaded environment from: $env_file"
echo "Active DB target: ${PGUSER:-?}@${PGHOST:-?}:${PGPORT:-?}/${PGDATABASE:-?} (sslmode=${PGSSLMODE:-unset})"
