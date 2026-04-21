#!/usr/bin/env bash
# Full ablation: ALL baselines × {vanilla, +GPE, +Var, +GPE+Var} on ALL graph datasets.
# Covers: CoraFull/DBLP/PubMed (our DGP) + BlogCatalog/Flickr (WSDM benchmarks).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

SEEDS="${1:-0 1 2}"
MODE="${2:-all}"  # "our" for CoraFull/DBLP/PubMed, "wsdm" for Blog/Flickr, "all" for both
EPOCHS="${EPOCHS:-200}"
CACHE_DIR="${CACHE_DIR:-runs/data_cache}"

# Activate your conda/mamba environment (edit CONDA_ENV if needed)
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
source "$CONDA_SH"
conda activate "${CONDA_ENV:-pact}"

TS=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="runs/full_${TS}"
mkdir -p "$OUT_ROOT"
echo "[full-runner] output: $OUT_ROOT"

# Learner specs: tag|flags
# vanilla = no GPE, no variance
# _gpe = with GPE
# _var = with variance (no GPE)
# _gpe_var = both GPE and variance
SPECS=(
    "netdeconf|--learner netdeconf --no-gpe --no-variance"
    "netdeconf_gpe|--learner netdeconf --no-variance"
    "netdeconf_var|--learner netdeconf --no-gpe --use-variance"
    "netdeconf_gpe_var|--learner netdeconf --use-variance"
    "gial|--learner gial --no-gpe --no-variance"
    "gial_gpe|--learner gial --no-variance"
    "gial_var|--learner gial --no-gpe --use-variance"
    "gial_gpe_var|--learner gial --use-variance"
    "gnum|--learner gnum --no-gpe --no-variance"
    "gnum_gpe|--learner gnum --no-variance"
    "gnum_var|--learner gnum --no-gpe --use-variance"
    "gnum_gpe_var|--learner gnum --use-variance"
    "gdc|--learner gdc --no-gpe --no-variance"
    "gdc_gpe|--learner gdc --no-variance"
    "gdc_var|--learner gdc --no-gpe --use-variance"
    "gdc_gpe_var|--learner gdc --use-variance"
)

run_one() {
    local ds="$1" cfg="$2" bias_flag="$3" seed="$4" tag="$5" flags="$6"
    local out_ds="$OUT_ROOT/$ds"
    mkdir -p "$out_ds"
    local run_name="${tag}__${bias_flag}__seed${seed}"
    local json_file="$out_ds/${run_name}.json"
    local log_file="$out_ds/${run_name}.log"
    if [[ -f "$json_file" ]]; then
        echo "[skip] $ds/$run_name"
        return
    fi
    local CACHE_FLAG=""
    [[ -d "$CACHE_DIR" ]] && CACHE_FLAG="--cache-dir $CACHE_DIR"
    echo "[run ] $ds/$run_name"
    python -u -m pact.main \
        --config "$cfg" \
        --device cuda \
        --seed "$seed" \
        --epochs "$EPOCHS" \
        $bias_flag \
        $flags \
        $CACHE_FLAG \
        --tag "$ds/$run_name" \
        --out-json "$json_file" \
        > "$log_file" 2>&1 \
        || { echo "  [FAIL] see $log_file"; return; }
    tail -n 2 "$log_file" | sed 's/^/   | /'
}

# --- Our datasets (CoraFull/DBLP/PubMed) ---
if [[ "$MODE" == "our" || "$MODE" == "all" ]]; then
    for ds in cora_full dblp pubmed; do
        cfg="pact/configs/server/${ds}.yaml"
        for rho in 5 10 15 20 30; do
            for seed in $SEEDS; do
                for spec in "${SPECS[@]}"; do
                    IFS='|' read -r tag flags <<< "$spec"
                    run_one "$ds" "$cfg" "--rho $rho" "$seed" "$tag" "$flags"
                done
            done
        done
    done
fi

# --- WSDM benchmarks (BlogCatalog/Flickr) ---
if [[ "$MODE" == "wsdm" || "$MODE" == "all" ]]; then
    for ds_name in blogcatalog flickr; do
        # Prefer user-customized server/ configs; fall back to public templates
        if [ -f "pact/configs/server/${ds_name}.yaml" ]; then
            cfg="pact/configs/server/${ds_name}.yaml"
        else
            cfg="pact/configs/${ds_name}.yaml"
        fi
        # WSDM benchmarks are reported at the default confounding-bias
        # (kappa=1, Guo et al.). Earlier iterations swept {0.5, 1, 2}; the
        # paper reports kappa=1 only and the sweep wrapper is removed.
            # Override extra_str via a temp config (wraps the base YAML for each kappa)
            for seed in $SEEDS; do
                for spec in "${SPECS[@]}"; do
                    IFS='|' read -r tag flags <<< "$spec"
                    # Pass kappa as a dataset override
                    run_one "${ds_name}" "$cfg" "--seed $seed" "$seed" "$tag" "$flags"
                done
            done
            # Only need one kappa pass per seed since extra_str is in config
            # Actually we need to override extra_str. Use python directly:
    done
fi

TOTAL=$(find "$OUT_ROOT" -name "*.json" | wc -l)
echo "[full-runner] done. $TOTAL json files in $OUT_ROOT"
