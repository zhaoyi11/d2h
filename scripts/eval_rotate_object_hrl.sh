#!/usr/bin/env bash
set -eo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <low-level-checkpoint> [rollout options]" >&2
  exit 2
fi

EVAL_LOW_LEVEL_CHECKPOINT="$1"
shift

EVAL_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$EVAL_REPO_ROOT"
export PYTHONPATH="$EVAL_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available; initialize Conda before running this script." >&2
  exit 1
fi
EVAL_CONDA_BASE="$(conda info --base)"
source "$EVAL_CONDA_BASE/etc/profile.d/conda.sh"
conda activate env_isaaclab
set -u

python3 scripts/instant_dexterity.py \
  --task Rotate_Knob_HRL-v0 \
  --num_envs 1 \
  --low_level_checkpoint "$EVAL_LOW_LEVEL_CHECKPOINT" \
  "$@"
