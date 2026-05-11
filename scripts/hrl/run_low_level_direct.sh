#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  echo "Could not find conda.sh at ${CONDA_ROOT}/etc/profile.d/conda.sh" >&2
  echo "Set CONDA_ROOT to your conda installation root and rerun." >&2
  exit 1
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate env_isaaclab
set -u

TASK="${TASK:-Pick_Insert_Debug_HRL-v0}"
NUM_ENVS="${NUM_ENVS:-1}"
LOW_LEVEL_CHECKPOINT="${LOW_LEVEL_CHECKPOINT:-${REPO_ROOT}/logs/rsl_rl/anyreorient/model_14999.pt}"
LOW_LEVEL_OBS_GROUP="${LOW_LEVEL_OBS_GROUP:-low_level}"

python scripts/hrl/play_low_level_direct.py \
  --task "${TASK}" \
  --num_envs "${NUM_ENVS}" \
  --low_level_checkpoint "${LOW_LEVEL_CHECKPOINT}" \
  --low_level_obs_group "${LOW_LEVEL_OBS_GROUP}" \
  "$@"
