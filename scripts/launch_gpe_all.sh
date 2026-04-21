#!/usr/bin/env bash
# Launch GPE ablation with controlled parallelism.
# Runs 3 dataset workers in parallel (one per dataset).
# Each worker processes all rhos × variants × seeds sequentially.
# This keeps GPU usage at 3 concurrent processes max (~15GB each, fits in 48GB).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

SEEDS="${1:-0 1 2}"

TS=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="runs/gpe_${TS}"
mkdir -p "$OUT_ROOT"
echo "[launcher] output: $OUT_ROOT"
echo "[launcher] seeds: $SEEDS"

# Activate your conda/mamba environment (edit CONDA_ENV if needed)
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
source "$CONDA_SH"
conda activate "${CONDA_ENV:-pact}"

# 3 parallel workers: one per dataset, each handles all rhos sequentially
PIDS=()
for ds in cora_full dblp pubmed; do
    (
        for rho in 5 10 15 20 30; do
            bash scripts/run_gpe_parallel.sh "$ds" "$rho" "$OUT_ROOT" "$SEEDS"
        done
    ) > "$OUT_ROOT/${ds}.worker.log" 2>&1 &
    PIDS+=($!)
    echo "[launcher] started $ds worker (pid $!)"
done

echo "[launcher] ${#PIDS[@]} workers launched, waiting..."

FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAILED=$((FAILED + 1))
    fi
done

TOTAL=$(find "$OUT_ROOT" -name "*.json" | wc -l)
echo "[launcher] done. $TOTAL/216 json files. $FAILED workers failed."
echo "[launcher] results: $OUT_ROOT"
