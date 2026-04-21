#!/usr/bin/env bash
# GPE plug-and-play ablation on graph baselines.
# Tests: "GPE improves ANY graph meta-learner"
# Grid: {bnn, tarnet, x, pact} × {±GPE} × {rho 5,10,15,20,30} × seeds
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

SEEDS="${1:-0 1 2}"
DATASETS="${2:-cora_full,dblp,pubmed}"
IFS=',' read -ra RHOS <<< "${RHOS:-5,10,15,20,30}"
EPOCHS="${EPOCHS:-200}"
CACHE_DIR="${CACHE_DIR:-runs/data_cache}"

IFS=',' read -ra DS_LIST <<< "$DATASETS"

# Each spec: tag|flags
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

TS=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="runs/gpe_${TS}"
mkdir -p "$OUT_ROOT"
echo "[gpe-runner] writing to $OUT_ROOT"
echo "[gpe-runner] datasets=${DS_LIST[*]} rhos=${RHOS[*]} seeds=$SEEDS"

# Activate your conda/mamba environment (edit CONDA_ENV if needed)
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
source "$CONDA_SH"
conda activate "${CONDA_ENV:-pact}"

for ds in "${DS_LIST[@]}"; do
    cfg="pact/configs/server/${ds}.yaml"
    out_ds="$OUT_ROOT/$ds"
    mkdir -p "$out_ds"
    for rho in "${RHOS[@]}"; do
        for seed in $SEEDS; do
            for spec in "${SPECS[@]}"; do
                IFS='|' read -r tag flags <<< "$spec"
                run_name="${tag}__rho${rho}__seed${seed}"
                log_file="$out_ds/${run_name}.log"
                json_file="$out_ds/${run_name}.json"
                if [[ -f "$json_file" ]]; then
                    echo "[skip] $ds/$run_name"
                    continue
                fi
                CACHE_FLAG=""
                [[ -d "$CACHE_DIR" ]] && CACHE_FLAG="--cache-dir $CACHE_DIR"
                echo "[run ] $ds/$run_name"
                python -u -m pact.main \
                    --config "$cfg" \
                    --device cuda \
                    --rho "$rho" \
                    --seed "$seed" \
                    --epochs "$EPOCHS" \
                    $flags \
                    $CACHE_FLAG \
                    --tag "$ds/$run_name" \
                    --out-json "$json_file" \
                    > "$log_file" 2>&1 \
                    || { echo "  [FAIL] see $log_file"; continue; }
                tail -n 4 "$log_file" | sed 's/^/   | /'
            done
        done
    done
done

echo "[gpe-runner] done. results under $OUT_ROOT"
