#!/usr/bin/env bash
#
# Copy this repo from the training machine to the inference server.
#
# Run this ON THE TRAINING MACHINE (the one holding this checkout).
#
# Why rsync rather than `git clone`: this checkout carries uncommitted changes that
# upstream does not have, and one of them is required for inference --
# `prismatic/vla/constants.py` is the only place `NUM_ACTIONS_CHUNK=15` / `ACTION_DIM=8`
# are defined for the panda platform. A fresh clone of moojink/openvla-oft would serve
# LIBERO constants (chunk 8, dim 7) instead, and nothing would raise.
#
# Usage:
#   ./sync_to_server.sh              # sync
#   DRY_RUN=1 ./sync_to_server.sh    # show what would be transferred

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

require_var SERVER_USER_HOST "e.g. SERVER_USER_HOST=baiqili@10.0.0.5"
require_var SERVER_REPO_DIR

RSYNC_FLAGS=(-av --delete
  --exclude '.git/'
  --exclude '*.egg-info/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude 'deploy_panda/config.sh'   # machine-specific; the server gets its own
  --exclude 'wandb/'
  --exclude 'runs/'
)
[[ -n "${DRY_RUN:-}" ]] && RSYNC_FLAGS+=(--dry-run)

echo "Local  repo : $REPO_ROOT"
echo "Remote repo : $SERVER_USER_HOST:$SERVER_REPO_DIR"

# Show what differs from upstream, so it is visible that a plain clone is not equivalent.
if git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
  echo
  echo "Uncommitted changes being carried over (these do NOT exist upstream):"
  git -C "$REPO_ROOT" status --short || true
fi

echo
rsync "${RSYNC_FLAGS[@]}" "$REPO_ROOT/" "$SERVER_USER_HOST:$SERVER_REPO_DIR/"

cat <<EOF

Done. Next, on the inference server:

  cd $SERVER_REPO_DIR
  cp deploy_panda/config.example.sh deploy_panda/config.sh   # fill in PRETRAINED_CHECKPOINT
  ./deploy_panda/server/download_checkpoint.sh
  ./deploy_panda/server/deploy_panda.sh
EOF
