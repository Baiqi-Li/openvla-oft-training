#!/usr/bin/env bash
#
# Start the OpenVLA-OFT inference server for the panda-40k checkpoint.
#
# Run this ON THE INFERENCE SERVER. Serves POST /act on $SERVER_PORT.
#
# Every flag below is pinned to how the checkpoint was actually trained (see
# `training/commands/train_panda.sbatch`). Two of them override defaults that would
# otherwise be silently wrong:
#
#   --num_images_in_input 2   (DeployConfig defaults to 3)
#   --use_film True           (DeployConfig defaults to False)
#
# With use_film=False the FiLM wrapper is never applied, so `vision_backbone--40000_checkpoint.pt`
# is never loaded and the model quietly runs on base OpenVLA vision weights. The
# `--expected_*` flags make both of these fail at startup instead.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

require_var PRETRAINED_CHECKPOINT "e.g. PRETRAINED_CHECKPOINT=/data/ckpts/panda_40k"
require_var SERVER_REPO_DIR

if [[ ! -d "$PRETRAINED_CHECKPOINT" ]]; then
  echo "error: PRETRAINED_CHECKPOINT is not a directory: $PRETRAINED_CHECKPOINT" >&2
  echo "       Run ./download_checkpoint.sh first." >&2
  exit 1
fi

if [[ -f "$CONDA_SH" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
fi

# `get_vla` -> `check_model_logic_mismatch` walks "./prismatic/" and copies this repo's
# modeling_prismatic.py / configuration_prismatic.py into the checkpoint directory
# (backing up the originals as *.back). Both the walk and the sync require the repo root
# as cwd; running from anywhere else warns and skips, leaving stale modeling code in play.
cd "$SERVER_REPO_DIR"

# Pin the robot platform explicitly. Left to itself, `prismatic/vla/constants.py` sniffs
# sys.argv for the substring "panda" and otherwise falls back to LIBERO constants
# (chunk 8, action dim 7) without raising.
export ROBOT_PLATFORM=PANDA

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# The checkpoint is already local; keep the hub out of the startup path.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "repo       : $SERVER_REPO_DIR"
echo "checkpoint : $PRETRAINED_CHECKPOINT"
echo "gpu        : $CUDA_VISIBLE_DEVICES"
echo "listening  : 0.0.0.0:$SERVER_PORT  (POST /act)"
echo "started    : $(date)"
echo

exec python vla-scripts/deploy.py \
  --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
  --unnorm_key panda_2026_02_09_all \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film True \
  --num_images_in_input 2 \
  --use_proprio True \
  --center_crop True \
  --lora_rank 32 \
  --load_in_8bit False \
  --load_in_4bit False \
  --expected_robot_platform PANDA \
  --expected_num_actions_chunk 15 \
  --expected_action_dim 8 \
  --expected_proprio_dim 8 \
  --expected_num_images_in_input 2 \
  --expected_use_film True \
  --host 0.0.0.0 \
  --port "$SERVER_PORT"
