#!/usr/bin/env bash
#
# Fetch the panda-40k checkpoint from Hugging Face onto the inference server.
#
# Run this ON THE INFERENCE SERVER. Downloads ~19 GB into $PRETRAINED_CHECKPOINT.
#
# The download must land in a local directory: passing the hub repo id straight to
# `deploy.py --pretrained_checkpoint` hits a hardcoded `moojink/...` allow-list in
# `get_proprio_projector` / `get_action_head` (experiments/robot/openvla_utils.py) and
# dies with "Unsupported HF Hub pretrained checkpoint found!". From a local directory,
# `find_checkpoint_file` globs the `*--40000_checkpoint.pt` files correctly.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

require_var PRETRAINED_CHECKPOINT "e.g. PRETRAINED_CHECKPOINT=/data/ckpts/panda_40k"
require_var HF_REPO_ID

# The repo is public, but HF_HUB_OFFLINE may be inherited from a training profile.
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

if [[ -f "$CONDA_SH" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
fi

echo "Downloading $HF_REPO_ID -> $PRETRAINED_CHECKPOINT"
mkdir -p "$PRETRAINED_CHECKPOINT"

# `hf` is the current CLI; `huggingface-cli` is the older name for the same thing.
if command -v hf >/dev/null 2>&1; then
  hf download "$HF_REPO_ID" --local-dir "$PRETRAINED_CHECKPOINT"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$HF_REPO_ID" --local-dir "$PRETRAINED_CHECKPOINT"
else
  python -c "
from huggingface_hub import snapshot_download
snapshot_download('$HF_REPO_ID', local_dir='$PRETRAINED_CHECKPOINT')
"
fi

echo
echo "Verifying expected files..."
missing=0
for f in \
  config.json \
  dataset_statistics.json \
  model.safetensors.index.json \
  action_head--40000_checkpoint.pt \
  proprio_projector--40000_checkpoint.pt \
  vision_backbone--40000_checkpoint.pt \
  lora_adapter/adapter_model.safetensors
do
  if [[ -e "$PRETRAINED_CHECKPOINT/$f" ]]; then
    echo "  ok      $f"
  else
    echo "  MISSING $f"
    missing=1
  fi
done

# The three `*--40000_checkpoint.pt` globs must each match exactly one file:
# `find_checkpoint_file` asserts on the count, so a stale 30k/50k copy in the same
# directory would break startup with a confusing message.
for pattern in action_head proprio_projector vision_backbone; do
  count=$(find "$PRETRAINED_CHECKPOINT" -maxdepth 1 -name "${pattern}*checkpoint*.pt" | wc -l)
  if [[ "$count" -ne 1 ]]; then
    echo "  ERROR: found $count '${pattern}*checkpoint*.pt' files, expected exactly 1."
    echo "         find_checkpoint_file() asserts on this. Remove the extras."
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  echo
  echo "Download incomplete. Re-run this script." >&2
  exit 1
fi

echo
echo "Checkpoint ready at $PRETRAINED_CHECKPOINT"
du -sh "$PRETRAINED_CHECKPOINT"
