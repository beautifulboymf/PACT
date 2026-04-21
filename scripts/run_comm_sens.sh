#!/usr/bin/env bash
# Community-detection sensitivity: rerun GPE pipeline on DBLP ρ=10 with alternative
# community-detection algorithms (Leiden, Infomap) and confirm GPE's PEHE delta is
# not Louvain-specific.
#
# Grid: 2 models (tarnet, x) × {vanilla, +GPE} × 3 community algs (louvain, leiden, infomap)
#       × 5 seeds = 60 runs.
# The louvain row is redundant with Table 1's DBLP ρ=10 column but included so the new
# supplementary table is self-contained.
#
# Prereq (run once):
#   pip install leidenalg python-igraph infomap
#
# Usage: bash scripts/run_comm_sens.sh [SEEDS]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

SEEDS="${1:-0 1 2 3 4}"
EPOCHS="${EPOCHS:-200}"
CACHE_DIR="${CACHE_DIR:-runs/data_cache}"

TS=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="runs/comm_sens_${TS}"
mkdir -p "$OUT_ROOT"
echo "[comm-sens] writing to $OUT_ROOT"

# Activate your conda/mamba environment (edit CONDA_ENV if needed)
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
source "$CONDA_SH"
conda activate "${CONDA_ENV:-pact}"

SPECS=(
    "tarnet|--learner tarnet --no-gpe"
    "tarnet_gpe|--learner tarnet"
    "x|--learner x --no-gpe"
    "x_gpe|--learner x"
)

for alg in louvain leiden infomap; do
    for seed in $SEEDS; do
        for spec in "${SPECS[@]}"; do
            IFS='|' read -r tag flags <<< "$spec"
            run_name="${tag}__comm${alg}__seed${seed}"
            json_file="$OUT_ROOT/${run_name}.json"
            log_file="$OUT_ROOT/${run_name}.log"
            if [[ -f "$json_file" ]]; then
                echo "[skip] $run_name"
                continue
            fi
            CACHE_FLAG=""
            [[ -d "$CACHE_DIR" ]] && CACHE_FLAG="--cache-dir $CACHE_DIR"
            echo "[run ] $run_name"
            python -u -m pact.main \
                --config pact/configs/server/dblp.yaml \
                --device cuda \
                --rho 10 \
                --seed "$seed" \
                --epochs "$EPOCHS" \
                --community-detection "$alg" \
                $flags \
                $CACHE_FLAG \
                --tag "$run_name" \
                --out-json "$json_file" \
                > "$log_file" 2>&1 \
                || { echo "  [FAIL] see $log_file"; continue; }
            tail -n 4 "$log_file" | sed 's/^/   | /'
        done
    done
done

echo "[comm-sens] done. 60 result JSONs under $OUT_ROOT"
