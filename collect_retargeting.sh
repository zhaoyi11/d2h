#!/usr/bin/env bash
set -euo pipefail

cd /home/yizhao/yi/D2H

# Conda activation scripts may read unset variables such as ADDR2LINE, which
# conflicts with bash nounset. Re-enable nounset immediately after activation.
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env_isaaclab
set -u

PYTHON_BIN="$CONDA_PREFIX/bin/python"
"$PYTHON_BIN" -c 'import numpy' >/dev/null

CHECKPOINT="/home/yizhao/yi/D2H/logs/rsl_rl/anyreorient/model_14999.pt"
OUT_DIR="/home/yizhao/yi/D2H/datasets/Reorient_Play-v0/retargeting_anyreorient_model_14999"

NUM_ENVS="${NUM_ENVS:-64}"
NUM_EPISODES="${NUM_EPISODES:-1024}"
SEED="${SEED:-0}"
ACTION_NOISE_STD_MAX="${ACTION_NOISE_STD_MAX:-0.0}"

"$PYTHON_BIN" -m src.policy.data_collection_for_retargeting \
  --task Reorient_Play-v0 \
  --checkpoint "$CHECKPOINT" \
  --num_envs "$NUM_ENVS" \
  --num_episodes "$NUM_EPISODES" \
  --output_dir "$OUT_DIR" \
  --headless \
  --seed "$SEED" \
  --action_noise_std_max "$ACTION_NOISE_STD_MAX" \
  --action_term_name joint_pos \
  --contact_force_threshold 0.25 \
  --actor_only_checkpoint_load
