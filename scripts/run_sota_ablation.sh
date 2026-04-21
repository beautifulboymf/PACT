#!/usr/bin/env bash
# SOTA baseline ablation: {netdeconf, gial, gnum, gdc} × {vanilla, +GPE} on our datasets
# Runs 3 dataset workers in parallel (one per dataset).
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
    "netdeconf|--learner netdeconf --no-gpe"
    "netdeconf_gpe|--learner netdeconf"
    "gial|--learner gial --no-gpe"
    "gial_gpe|--learner gial"
    "gnum|--learner gnum --no-gpe"
    "gnum_gpe|--learner gnum"
    "gdc|--learner gdc --no-gpe"
    "gdc_gpe|--learner gdc"
)

TS=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="runs/sota_${TS}"
mkdir -p "$OUT_ROOT"
echo "[sota-runner] writing to $OUT_ROOT"
echo "[sota-runner] datasets=${DS_LIST[*]} rhos=${RHOS[*]} seeds=$SEEDS"

# Activate your conda/mamba environment (edit CONDA_ENV if needed)
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
source "$CONDA_SH"
conda activate "${CONDA_ENV:-pact}"

# Run one dataset at a time to avoid GPU OOM with GPE models
for ds in "${DS_LIST[@]}"; do
    cfg="pact/configs/server/${ds}.yaml"
    out_ds="$OUT_ROOT/$ds"
    mkdir -p "$out_ds"
    for rho in "${RHOS[@]}"; do
        for seed in $SEEDS; do
            for spec in "${SPECS[@]}"; do
                IFS='|' read -r tag flags <<< "$spec"
                run_name="${tag}__rho${rho}__seed${seed}"
                json_file="$out_ds/${run_name}.json"
                log_file="$out_ds/${run_name}.log"
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

echo "[sota-runner] done. results under $OUT_ROOT"
