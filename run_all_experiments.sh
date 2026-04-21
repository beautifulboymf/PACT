#!/bin/bash
# Activate your conda/mamba environment (edit CONDA_ENV if needed)
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
source "$CONDA_SH"
conda activate "${CONDA_ENV:-pact}"
cd ${REPO_ROOT}

echo "=== Phase 1: GDC 5-seed ablation (W7) ==="
python -m pact.experiments.run_gdc_5seed_ablation --device cuda --cache-dir runs/data_cache 2>&1 | tee results/gdc_ablation_5seed/run.log

echo ""
echo "=== Phase 2: No-community DGP ablation (W1) ==="  
python -m pact.experiments.run_no_community_ablation --device cuda 2>&1 | tee results/no_community_ablation/run.log

echo ""
echo "=== Phase 3: Graphformer comparison (W3) ==="
python -m pact.experiments.run_graphformer_comparison --device cuda --cache-dir runs/data_cache 2>&1 | tee results/graphformer_comparison/run.log

echo ""
echo "=== ALL EXPERIMENTS COMPLETE ==="
date
