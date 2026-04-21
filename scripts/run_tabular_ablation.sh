#!/usr/bin/env bash
# Tabular uplift plug-in ablation: {learner} × {±variance} × datasets × seeds
# Tests: "variance weighting improves any base meta-learner"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

SEEDS="${1:-0 1 2 3 4}"
DATASETS="${2:-hillstrom,criteo,retailhero,hillstrom_spend}"

IFS=',' read -ra DS_LIST <<< "$DATASETS"

# Each spec: tag|flags
# ite_mode for S/T learners is "s" (mu1-mu0), for X is "x" (tau heads)
SPECS=(
    "s|--learner s --no-variance"
    "s_var|--learner s --use-variance"
    "t|--learner t --no-variance"
    "t_var|--learner t --use-variance"
    "x|--learner x --no-variance"
    "x_var|--learner x --use-variance"
)

TS=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="runs/tab_${TS}"
mkdir -p "$OUT_ROOT"
echo "[tab-runner] writing to $OUT_ROOT"
echo "[tab-runner] datasets=${DS_LIST[*]} seeds=$SEEDS"

# Activate your conda/mamba environment (edit CONDA_ENV if needed)
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
source "$CONDA_SH"
conda activate "${CONDA_ENV:-pact}"

for ds in "${DS_LIST[@]}"; do
    cfg="pact/configs/server/${ds}.yaml"
    out_ds="$OUT_ROOT/$ds"
    mkdir -p "$out_ds"
    for seed in $SEEDS; do
        for spec in "${SPECS[@]}"; do
            IFS='|' read -r tag flags <<< "$spec"
            run_name="${tag}__seed${seed}"
            log_file="$out_ds/${run_name}.log"
            json_file="$out_ds/${run_name}.json"
            if [[ -f "$json_file" ]]; then
                echo "[skip] $ds/$run_name"
                continue
            fi
            echo "[run ] $ds/$run_name"
            python -u -m pact.main \
                --mode tabular \
                --config "$cfg" \
                --device cuda \
                --seed "$seed" \
                --no-gpe \
                $flags \
                --tag "$ds/$run_name" \
                --out-json "$json_file" \
                > "$log_file" 2>&1 \
                || { echo "  [FAIL] see $log_file"; continue; }
            tail -n 4 "$log_file" | sed 's/^/   | /'
        done
    done
done

echo "[tab-runner] done. results under $OUT_ROOT"
