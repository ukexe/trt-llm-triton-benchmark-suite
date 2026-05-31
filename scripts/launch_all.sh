#!/usr/bin/env bash
# Bring up serving backends + monitoring via docker compose profiles.
#
# Usage:
#   scripts/launch_all.sh                 # defaults: vllm + monitoring
#   scripts/launch_all.sh triton monitoring
#   scripts/launch_all.sh all             # everything
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/../infra/docker-compose.yml"

profiles=("$@")
if [ "${#profiles[@]}" -eq 0 ]; then
  profiles=(vllm monitoring)
fi

args=()
for p in "${profiles[@]}"; do
  args+=(--profile "$p")
done

echo "Starting docker compose profiles: ${profiles[*]}"
exec docker compose -f "$COMPOSE_FILE" "${args[@]}" up -d
