#!/bin/bash
# 5-seed expansion: seeds 3,4 for all remaining model/dataset combos
# This script is idempotent: it skips any run whose output JSON already exists.

set -e
# Activate your conda/mamba environment (edit CONDA_ENV if needed)
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
source "$CONDA_SH"
conda activate "${CONDA_ENV:-pact}"
cd ${REPO_ROOT}

OUTDIR=runs/5seed_full
CACHEDIR=runs/data_cache
EPOCHS=200

run_one() {
    local ds=$1 cfg=$2 tag=$3 seed=$4 rho=$5 learner=$6
    shift 6
    local extra_flags="$@"

    local json
    if [ -n "$rho" ]; then
        json="${OUTDIR}/${ds}/${tag}__rho${rho}__seed${seed}.json"
    else
        json="${OUTDIR}/${ds}/${tag}__seed${seed}.json"
    fi

    if [ -f "$json" ]; then
        echo "[SKIP] $json already exists"
        return 0
    fi

    local log="${json%.json}.log"
    mkdir -p "$(dirname "$json")"

    local rho_flag=""
    if [ -n "$rho" ]; then
        rho_flag="--rho $rho"
    fi

    local cache_flag=""
    # Only use cache for DGP datasets (cora_full, dblp, pubmed)
    if [ "$ds" = "cora_full" ] || [ "$ds" = "dblp" ] || [ "$ds" = "pubmed" ]; then
        cache_flag="--cache-dir $CACHEDIR"
    fi

    local outtag
    if [ -n "$rho" ]; then
        outtag="${ds}/${tag}__rho${rho}__seed${seed}"
    else
        outtag="${ds}/${tag}__seed${seed}"
    fi

    echo "[RUN] $json  (learner=$learner $extra_flags)"
    python -u -m pact.main \
        --config "$cfg" \
        --device cuda \
        --seed "$seed" \
        --epochs "$EPOCHS" \
        --learner "$learner" \
        $rho_flag \
        $extra_flags \
        $cache_flag \
        --tag "$outtag" \
        --out-json "$json" \
        > "$log" 2>&1

    if [ $? -eq 0 ] && [ -f "$json" ]; then
        echo "[OK]  $json"
    else
        echo "[FAIL] $json — see $log"
    fi
}

###############################################################################
# Model specs: tag|learner|extra_flags
# The 12 variants needed (excluding tarnet/gdc which are already done for cora/dblp)
###############################################################################
MODELS_NO_TARNET_GDC=(
    "bnn|bnn|--no-gpe --no-variance"
    "bnn_gpe|bnn|--no-variance"
    "x|x|--no-gpe --no-variance"
    "x_gpe|x|--no-variance"
    "netdeconf|netdeconf|--no-gpe --no-variance"
    "netdeconf_gpe|netdeconf|--no-variance"
    "gial|gial|--no-gpe --no-variance"
    "gial_gpe|gial|--no-variance"
    "gnum|gnum|--no-gpe --no-variance"
    "gnum_gpe|gnum|--no-variance"
    "pact_nogpe|pact|--no-gpe"
    "pact|pact|"
)

# All 16 variants (adds tarnet, tarnet_gpe, gdc, gdc_gpe)
MODELS_ALL=(
    "${MODELS_NO_TARNET_GDC[@]}"
    "tarnet|tarnet|--no-gpe --no-variance"
    "tarnet_gpe|tarnet|--no-variance"
    "gdc|gdc|--no-gpe --no-variance"
    "gdc_gpe|gdc|--no-variance"
)

RHOS="5 10 15 20 30"
SEEDS="3 4"

total=0
done_count=0
fail=0

echo "========================================="
echo " 5-SEED FULL EXPANSION: seeds 3,4"
echo " Started: $(date)"
echo "========================================="

###############################################################################
# PART 1: CoraFull and DBLP — only models not already done (no tarnet/gdc)
###############################################################################
echo ""
echo "=== PART 1: CoraFull + DBLP (12 models x 3 rhos x 2 seeds = 144 runs) ==="
for ds in cora_full dblp; do
    cfg="pact/configs/server/${ds}.yaml"
    for rho in $RHOS; do
        for seed in $SEEDS; do
            for spec in "${MODELS_NO_TARNET_GDC[@]}"; do
                IFS='|' read -r tag learner flags <<< "$spec"
                total=$((total + 1))
                run_one "$ds" "$cfg" "$tag" "$seed" "$rho" "$learner" $flags || fail=$((fail + 1))
                done_count=$((done_count + 1))
                echo "  Progress: $done_count / ~304 total"
            done
        done
    done
done

###############################################################################
# PART 2: PubMed — ALL 16 models (including tarnet/gdc)
###############################################################################
echo ""
echo "=== PART 2: PubMed (16 models x 3 rhos x 2 seeds = 96 runs) ==="
cfg="pact/configs/server/pubmed.yaml"
for rho in $RHOS; do
    for seed in $SEEDS; do
        for spec in "${MODELS_ALL[@]}"; do
            IFS='|' read -r tag learner flags <<< "$spec"
            total=$((total + 1))
            run_one "pubmed" "$cfg" "$tag" "$seed" "$rho" "$learner" $flags || fail=$((fail + 1))
            done_count=$((done_count + 1))
            echo "  Progress: $done_count / ~304 total"
        done
    done
done

###############################################################################
# PART 3: BlogCatalog + Flickr — ALL 16 models, no rho
###############################################################################
echo ""
echo "=== PART 3: BlogCatalog + Flickr (16 models x 2 datasets x 2 seeds = 64 runs) ==="
for ds in blogcatalog flickr; do
    cfg="pact/configs/server/${ds}.yaml"
    for seed in $SEEDS; do
        for spec in "${MODELS_ALL[@]}"; do
            IFS='|' read -r tag learner flags <<< "$spec"
            total=$((total + 1))
            run_one "$ds" "$cfg" "$tag" "$seed" "" "$learner" $flags || fail=$((fail + 1))
            done_count=$((done_count + 1))
            echo "  Progress: $done_count / ~304 total"
        done
    done
done

echo ""
echo "========================================="
echo " COMPLETED: $(date)"
echo " Total: $total, Done: $done_count, Failed: $fail"
echo "========================================="
