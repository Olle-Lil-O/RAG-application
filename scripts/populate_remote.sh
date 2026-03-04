#!/usr/bin/env bash
set -euo pipefail

set -a
source remote/remote.env
if [[ -f ".env" ]]; then
  source .env
fi
set +a

bash scripts/populate_all.sh
