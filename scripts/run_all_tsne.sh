#!/usr/bin/env bash
# Run t-SNE visualizations for all 5 graph datasets.
#
# Usage: bash scripts/run_all_tsne.sh
#
# Outputs: runs/tsne_all/tsne_<dataset>.png for each of
#          cora_full, dblp, pubmed, blogcatalog, flickr.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

# Activate your conda/mamba environment (override via CONDA_ENV / CONDA_SH)
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
source "$CONDA_SH"
conda activate "${CONDA_ENV:-pact}"

python supplementary/run_all_tsne.py --device "${DEVICE:-cuda}"
