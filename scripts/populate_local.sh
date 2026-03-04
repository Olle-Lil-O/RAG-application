#!/usr/bin/env bash
set -euo pipefail

set -a
source .env
set +a

bash scripts/populate_all.sh
