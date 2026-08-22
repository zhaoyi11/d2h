#!/usr/bin/env bash
set -eo pipefail

TRAIN_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TRAIN_REPO_ROOT"
export PYTHONPATH="$TRAIN_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available; initialize Conda before running this script." >&2
  exit 1
fi
TRAIN_CONDA_BASE="$(conda info --base)"
source "$TRAIN_CONDA_BASE/etc/profile.d/conda.sh"
conda activate env_isaaclab
set -u

python3 scripts/rsl_rl/train.py \
  --task Cupcake_on_Plate-v0 \
  --num_envs 4096 \
  --max_iterations 15000 \
  --headless \
  "$@"
