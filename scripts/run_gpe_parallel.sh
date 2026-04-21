#!/usr/bin/env bash
# Parallel GPE ablation: runs one (dataset, rho) combo.
# Designed to be launched as multiple instances in separate tmux panes.
#
# Usage: bash run_gpe_parallel.sh <dataset> <rho> <out_root> [seeds]
# Example: bash run_gpe_parallel.sh cora_full 10 runs/gpe_20260409 "0 1 2"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

DS="$1"
RHO="$2"
OUT_ROOT="$3"
SEEDS="${4:-0 1 2}"
EPOCHS="${EPOCHS:-200}"
CACHE_DIR="${CACHE_DIR:-runs/data_cache}"

# Activate your conda/mamba environment (edit CONDA_ENV if needed)
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
source "$CONDA_SH"
conda activate "${CONDA_ENV:-pact}"

cfg="pact/configs/server/${DS}.yaml"
out_ds="$OUT_ROOT/$DS"
mkdir -p "$out_ds"

SPECS=(
    "bnn|--learner bnn --no-gpe"
    "bnn_gpe|--learner bnn"
    "tarnet|--learner tarnet --no-gpe"
    "tarnet_gpe|--learner tarnet"
    "x|--learner x --no-gpe"
    "x_gpe|--learner x"
    "pact_nogpe|--learner pact --no-gpe"
    "pact|--learner pact"
)

for seed in $SEEDS; do
    for spec in "${SPECS[@]}"; do
        IFS='|' read -r tag flags <<< "$spec"
        run_name="${tag}__rho${RHO}__seed${seed}"
        json_file="$out_ds/${run_name}.json"
        log_file="$out_ds/${run_name}.log"
        if [[ -f "$json_file" ]]; then
            echo "[skip] $DS/$run_name"
            continue
        fi
        CACHE_FLAG=""
        [[ -d "$CACHE_DIR" ]] && CACHE_FLAG="--cache-dir $CACHE_DIR"
        echo "[run ] $DS/$run_name"
        python -u -m pact.main \
            --config "$cfg" \
            --device cuda \
            --rho "$RHO" \
            --seed "$seed" \
            --epochs "$EPOCHS" \
            $flags \
            $CACHE_FLAG \
            --tag "$DS/$run_name" \
            --out-json "$json_file" \
            > "$log_file" 2>&1 \
            || { echo "  [FAIL] $log_file"; continue; }
        echo "  [done] $(tail -1 "$log_file")"
    done
done
echo "[worker] $DS rho=$RHO finished"
