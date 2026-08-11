# Deploying the panda-40k OpenVLA-OFT policy across two hosts

Serves [`BaiqiL/openvla-oft-panda-40k-steps`](https://huggingface.co/BaiqiL/openvla-oft-panda-40k-steps)
on an inference server and drives a DROID Panda from a separate robot host.

```
[robot host]                                  [inference server]
 DROID RobotEnv                                openvla-oft + panda-40k
 evaluate_openvla_oft_baiqi.py  ─── POST ───►  FastAPI /act  :8777
 15 Hz control loop             ◄── (15,8) ──  1x GPU, ~20 GB bf16
```

Roughly 460 KB per request (two 320x180 images base64-encoded), about 1.2 requests/second
at `--open_loop_horizon 12`, so on the order of **4.5 Mbit/s**.

## Layout

```
deploy_panda/
├── config.example.sh          copy to config.sh and fill in
├── server/
│   ├── _common.sh             shared bootstrap (sourced, not run)
│   ├── sync_to_server.sh      run on the TRAINING machine
│   ├── download_checkpoint.sh run on the INFERENCE SERVER
│   └── deploy_panda.sh        run on the INFERENCE SERVER
└── robot/                     self-contained; scp this directory to the robot host
    ├── openvla_oft_client.py  transport layer
    ├── evaluate_openvla_oft_baiqi.py
    ├── smoke_test_client.py
    ├── tasks.json
    └── requirements.txt
```

Two files outside this directory were also modified:

| File | Change |
| --- | --- |
| `prismatic/vla/constants.py` | `ROBOT_PLATFORM` env var now overrides the argv sniffing |
| `vla-scripts/deploy.py` | `--expected_*` startup self-checks; logs the first chunk's shape |

Both are additive and leave the LIBERO / ALOHA / BRIDGE paths unchanged.

## Setup

### 1. Training machine — push the repo

```bash
cp deploy_panda/config.example.sh deploy_panda/config.sh   # fill in SERVER_USER_HOST
./deploy_panda/server/sync_to_server.sh
```

**Use this, not `git clone`.** This checkout carries uncommitted changes that upstream does
not have, and `prismatic/vla/constants.py` is the only place the panda constants
(`NUM_ACTIONS_CHUNK=15`, `ACTION_DIM=8`) are defined. A fresh clone of
`moojink/openvla-oft` would serve LIBERO constants (chunk 8, action dim 7) and nothing
would raise.

### 2. Inference server — environment

Python 3.10, torch 2.2.0, then `pip install -e .` (see `SETUP.md`). Flash Attention is
**not** needed: `get_vla` has `attn_implementation` commented out.

Two things not to substitute:

- `transformers` must be the `moojink/transformers-openvla-oft` fork — parallel decoding
  needs its bidirectional attention.
- `tensorflow==2.15.0` is on the inference path (`resize_image_for_policy` uses `tf.image`).

The protobuf / tensorflow-datasets / wandb versions in the training env are a load-bearing
combination. Reproducing it beats reinstalling from scratch:

```bash
# on the training machine
conda env export -n openvla-oft > openvla-oft-env.yml
# on the inference server
conda env create -f openvla-oft-env.yml
```

### 3. Inference server — checkpoint

```bash
cp deploy_panda/config.example.sh deploy_panda/config.sh   # fill in PRETRAINED_CHECKPOINT
./deploy_panda/server/download_checkpoint.sh               # ~19 GB
```

It must land in a **local directory**. Passing the hub repo id straight to
`--pretrained_checkpoint` hits a hardcoded `moojink/...` allow-list in
`get_proprio_projector` / `get_action_head` and dies with
`Unsupported HF Hub pretrained checkpoint found!`.

### 4. Inference server — serve

```bash
./deploy_panda/server/deploy_panda.sh
```

Open port 8777 in the firewall. Startup takes a couple of minutes; check the banner:

```
robot platform        = PANDA
num_actions_chunk     = 15
action_dim            = 8
proprio_dim           = 8
num_images_in_input   = 2
use_film              = True
```

The launcher pins all of these via `--expected_*`, so a mismatch raises before any weights
load rather than producing quietly wrong actions.

### 5. Robot host

```bash
scp -r deploy_panda/robot/ <robot-host>:~/openvla_oft_eval/
```

Nothing to install. The client uses `urllib` instead of `requests` and carries its own
copy of the json-numpy wire format, so the environment that already runs
`evaluate_openpi_baiqi.py` is sufficient.

## Bring-up

Run the smoke test twice — once locally on the server to isolate model problems from
network problems, then from the robot host to get the latency the control loop will see.

```bash
# on the inference server
python smoke_test_client.py --host 127.0.0.1

# on the robot host
python smoke_test_client.py --host <server-ip>
```

It checks the chunk shape, checks the action ranges, confirms that different observations
produce different chunks (a policy ignoring its inputs looks fine otherwise), and prints a
recommended `--open_loop_horizon` derived from the p90 latency.

Then, hand on the e-stop:

```bash
python evaluate_openvla_oft_baiqi.py \
    --remote_host <server-ip> \
    --task_id grasp_coke \
    --max_timesteps 30 \
    --open_loop_horizon <from the smoke test>
```

Watch that the arm moves in a sensible direction before running a full evaluation. The
script pings the server before constructing `RobotEnv`, so a server that is down or still
loading fails before the robot is touched.

## How this differs from the openpi client

The state machine, scoring prompts, CSV and MP4 output, camera ids, 15 Hz pacing and
action post-processing are unchanged. What differs:

| | openpi | openvla-oft |
| --- | --- | --- |
| transport | websocket, port 8000 | HTTP POST `/act`, port 8777 |
| payload keys | `observation/exterior_image_1_left`, `observation/wrist_image_left`, `observation/joint_position`, `observation/gripper_position`, `prompt` | `full_image`, `wrist_image`, `state`, `instruction` |
| proprio | two separate keys | one raw 8-vector, `[joint_positions(7), gripper_position(1)]` |
| images | `resize_with_pad(224, 224)` | plain resize to 320x180, no padding |
| chunk | `(H, 8)` | `(15, 8)` |

Actions carry identical semantics in both — 7 joint velocities in DROID's normalized
`[-1, 1]` space plus one absolute gripper position — so
`RobotEnv(action_space="joint_velocity", gripper_action_space="position")`,
`np.clip(action, -1, 1)` and the `gripper < 0.1 -> 0` rule carry over unchanged.

### Why the images must not be padded

Training frames were 320x180 natively, and the pipeline squashed them to 224x224 without
preserving aspect ratio. The server reproduces exactly that
(`resize_image_for_policy`: JPEG round-trip, lanczos3, then a center crop because training
used `--image_aug`). `resize_with_pad` would introduce black bars the model never saw.

The client downsamples to 320x180 and lets the server do the 224x224 step, which keeps the
resize bit-identical to training while cutting the wire payload from ~3.7 MB to ~230 KB per
image.

### Why proprio is sent raw

The server normalizes `state` with the BOUNDS_Q99 proprio statistics from
`dataset_statistics.json`. Normalizing client-side too would apply it twice.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| Chunk comes back `(8, 7)` | Server resolved LIBERO constants. Set `ROBOT_PLATFORM=PANDA` and restart. |
| `Unsupported HF Hub pretrained checkpoint found!` | `--pretrained_checkpoint` is a hub repo id. Download it locally first. |
| `Expected exactly 1 <name> checkpoint but found N` | Multiple `*--*_checkpoint.pt` for one component in the checkpoint dir. `find_checkpoint_file` asserts on the count; remove the extras. |
| Server returns the bare string `"error"` | `get_server_action` swallows the exception and logs it to the **server's** console. The traceback is there, not on the robot host. |
| Policy runs but performs badly | Check `use_film=True` and `num_images_in_input=2` in the startup banner. With `use_film=False` the FiLM wrapper is skipped, so `vision_backbone--40000_checkpoint.pt` is never loaded and the model silently falls back to base OpenVLA vision weights. |
| Control loop stutters at chunk boundaries | Inference latency exceeds `open_loop_horizon / 15` seconds. Raise the horizon (max 15) or move the server closer. |
| Warning about stale modeling code | `deploy_panda.sh` cds to the repo root because `check_model_logic_mismatch` walks `./prismatic/` and syncs `modeling_prismatic.py` / `configuration_prismatic.py` into the checkpoint directory (backing the originals up as `*.back`). Run from anywhere else and it warns and skips. |

## Provenance

Serving flags are pinned to `training/commands/train_panda.sbatch`:

```
--use_l1_regression True  --use_diffusion False  --use_film True
--num_images_in_input 2   --use_proprio True     --lora_rank 32
--image_aug True  (=> --center_crop True at eval)
dataset: panda_2026_02_09_all  (228 demos, 60,369 frames, 24 tasks)
```

Instructions in `tasks.json` are copied verbatim from
`training/2026-02-09-all/meta/tasks.jsonl`; all 24 are covered and none were reworded.
