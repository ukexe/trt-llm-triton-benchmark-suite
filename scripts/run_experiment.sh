#!/usr/bin/env bash
# Run one benchmark config, then attach a cost estimate for single experiments.
#
# Usage:
#   scripts/run_experiment.sh [CONFIG] [extra args for the controller...]
#   GPU=H100-80GB PROVIDER=spheron scripts/run_experiment.sh benchmarks/configs/single_gpu_baseline.yaml
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${1:-benchmarks/configs/single_gpu_baseline.yaml}"
OUT="${OUT:-results}"
GPU="${GPU:-A100-80GB}"
PROVIDER="${PROVIDER:-runpod}"
shift || true
extra=("$@")

python -m benchmarks.runner.client --config "$CONFIG" --out "$OUT" "${extra[@]}"

# Cost analysis works on a single summary; sweeps produce many, so skip if the
# single-name summary is absent.
name="$(python -c "import sys,yaml;print(yaml.safe_load(open('${CONFIG}')).get('name','experiment'))")"
summary="${OUT}/${name}/summary.json"
if [ -f "$summary" ]; then
  python -m cost.cost_analysis --summary "$summary" --gpu "$GPU" --provider "$PROVIDER"
else
  echo "No single summary at ${summary} (sweep?); run cost analysis per sub-experiment."
fi
