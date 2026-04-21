#!/usr/bin/env bash
# Tune the variance-weighted learning δ (noise floor) on the in-scope
# tabular benchmarks of §5.4 (Hillstrom-visit, Criteo, Lenta). Sweeps
# δ across 4 values × 3 seeds × 3 datasets × the strongest base learner per
# dataset (selected on the vanilla-baseline Qini).
#
# Datasets and chosen base (strongest vanilla Qini from existing runs):
#   hillstrom-visit -> T-learner (S is broken, T has real signal)
#   criteo          -> T-learner (S/T tied, T slightly stronger)
#   lenta           -> X-learner (X has highest vanilla baseline)
#
# Runs on autodl server. Writes per-run json to runs/tune_vwl_tabular_{ts}/.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"
# Activate your conda/mamba environment (edit CONDA_ENV if needed)
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
source "$CONDA_SH"
conda activate "${CONDA_ENV:-pact}"

TS=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="runs/tune_vwl_${TS}"
mkdir -p "$OUT_ROOT"
echo "[tune] out=$OUT_ROOT"

SEEDS="0 1 2"
DELTAS="1e-4 1e-3 1e-2 1e-1"

# Paired (dataset, base-learner)
declare -a PAIRS=(
    "hillstrom|t"
    "criteo|t"
    "lenta|x"
)

for pair in "${PAIRS[@]}"; do
    IFS='|' read -r ds learner <<< "$pair"
    cfg="pact/configs/server/${ds}.yaml"
    out_ds="$OUT_ROOT/${ds}_${learner}"
    mkdir -p "$out_ds"
    for seed in $SEEDS; do
        # Baseline (no Var) — one run per seed
        base_json="$out_ds/base_seed${seed}.json"
        if [[ ! -f "$base_json" ]]; then
            echo "[run ] $ds/$learner base seed=$seed"
            python -u -m pact.main --mode tabular \
                --config "$cfg" --learner "$learner" --no-variance \
                --seed "$seed" --out-json "$base_json" --device cuda \
                > "$out_ds/base_seed${seed}.log" 2>&1
        fi
        for d in $DELTAS; do
            tag="var_d${d}_seed${seed}"
            json="$out_ds/${tag}.json"
            if [[ -f "$json" ]]; then
                echo "[skip] $ds/$learner $tag"
                continue
            fi
            echo "[run ] $ds/$learner $tag"
            python -u -m pact.main --mode tabular \
                --config "$cfg" --learner "$learner" --use-variance \
                --delta "$d" --seed "$seed" \
                --out-json "$json" --device cuda \
                > "$out_ds/${tag}.log" 2>&1
        done
    done
done

echo "[tune] done."
