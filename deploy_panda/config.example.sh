# Deployment settings for the panda-40k OpenVLA-OFT policy.
#
#   cp config.example.sh config.sh   # then fill in the values marked <<< FILL IN
#
# The scripts under `server/` source `config.sh` if it exists. Every value can also be
# overridden per-invocation from the environment, e.g.
#
#   PRETRAINED_CHECKPOINT=/data/ckpts/other ./server/deploy_panda.sh
#
# `config.sh` is git-ignored, so machine-specific paths never end up in a commit.

# ---------------------------------------------------------------------------
# Inference server
# ---------------------------------------------------------------------------

# Where the checkpoint lives ON THE INFERENCE SERVER. `download_checkpoint.sh` writes
# here; `deploy_panda.sh` reads from here. Needs ~19 GB free.
#
# NOTE: this must be a local directory, NOT the Hugging Face repo id. Passing a hub id to
# `--pretrained_checkpoint` trips a hardcoded `moojink/...` allow-list in
# `experiments/robot/openvla_utils.py` and fails with "Unsupported HF Hub pretrained
# checkpoint found!".
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-}"    # <<< FILL IN, e.g. /data/ckpts/panda_40k

# Hugging Face repo the checkpoint is downloaded from.
HF_REPO_ID="${HF_REPO_ID:-BaiqiL/openvla-oft-panda-40k-steps}"

# Conda environment on the inference server that has openvla-oft installed.
CONDA_ENV="${CONDA_ENV:-openvla-oft}"

# Path to conda's profile script on the inference server. Leave as-is if conda is already
# on PATH in non-interactive shells.
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"

# Repo root ON THE INFERENCE SERVER (the rsync destination). `deploy_panda.sh` cds here.
SERVER_REPO_DIR="${SERVER_REPO_DIR:-$HOME/openvla-oft}"

# Port the FastAPI server listens on. Must be open in the server's firewall.
SERVER_PORT="${SERVER_PORT:-8777}"

# Which GPU to serve from (single card is plenty: bf16 7B + FiLM + 2 images is ~20 GB).
CUDA_DEVICE="${CUDA_DEVICE:-0}"

# ---------------------------------------------------------------------------
# rsync target (used by sync_to_server.sh, which runs on the TRAINING machine)
# ---------------------------------------------------------------------------

SERVER_USER_HOST="${SERVER_USER_HOST:-}"              # <<< FILL IN, e.g. baiqili@10.0.0.5

# ---------------------------------------------------------------------------
# Robot host (used by the scripts under `robot/`, passed as --host / --port)
# ---------------------------------------------------------------------------

SERVER_HOST="${SERVER_HOST:-}"                        # <<< FILL IN: IP the robot host dials
