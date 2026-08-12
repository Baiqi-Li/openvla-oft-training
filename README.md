> ## About this fork
>
> This repository is a fork of **[moojink/openvla-oft](https://github.com/moojink/openvla-oft)**,
> branched at commit [`e4287e9`](https://github.com/moojink/openvla-oft/commit/e4287e94541f459edc4feabc4e181f537cd569a8)
> ("Update pyproject.toml: Pin diffusers version", 2025-09-08). That project is itself
> built on **[openvla/openvla](https://github.com/openvla/openvla)**.
>
> All of the code below, the model architecture, the training recipe and the paper are the
> work of the original authors. This fork adds only the following, for fine-tuning and
> deploying OpenVLA-OFT on a single-arm Franka Panda:
>
> | Added here | What it is |
> | --- | --- |
> | `prismatic/vla/constants.py` | `PANDA_CONSTANTS` (15-action chunks, 8-dim actions/proprio) and a `ROBOT_PLATFORM` env var for selecting the platform explicitly |
> | `prismatic/vla/datasets/rlds/oxe/` | The `panda_2026_02_09_all` / `panda_2026_03_27` datasets (converted from LeRobot) and `ActionEncoding.JOINT_VEL` support |
> | `vla-scripts/deploy.py` | Optional `--expected_*` startup self-checks (disabled by default; upstream behavior is unchanged) |
> | [`deploy_panda/`](deploy_panda/) | Two-host deployment: an inference server plus a DROID robot-host client |
>
> To pull in upstream changes:
>
> ```bash
> git remote add moojink https://github.com/moojink/openvla-oft.git
> git fetch moojink && git merge moojink/main
> ```
>
> Licensed under the original [MIT license](LICENSE) (Copyright (c) 2025 Moo Jin Kim,
> Chelsea Finn, Percy Liang). **Please direct questions about the fork-specific additions
> to this repository's issues, not to the upstream authors.**

---

# Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success

**Project website: https://openvla-oft.github.io/**

**Paper: https://arxiv.org/abs/2502.19645**

**Summary video: https://youtu.be/T3Zkkr_NTSA**

## System Requirements

Inference:
* 1 GPU with ~16 GB VRAM for LIBERO sim benchmark tasks
* 1 GPU with ~18 GB VRAM for ALOHA robot tasks

Training:
* Between 1-8 GPUs with 27-80 GB, depending on the desired training setup (with default bfloat16 data type). See [this FAQ on our project website](https://openvla-oft.github.io/#train-compute) for details.

## Quick Start

First, set up a conda environment (see instructions in [SETUP.md](SETUP.md)).

Then, run the Python script below to download a pretrained OpenVLA-OFT checkpoint and run inference to generate an action chunk:

```python
import pickle
from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

# Instantiate config (see class GenerateConfig in experiments/robot/libero/run_libero_eval.py for definitions)
cfg = GenerateConfig(
    pretrained_checkpoint = "moojink/openvla-7b-oft-finetuned-libero-spatial",
    use_l1_regression = True,
    use_diffusion = False,
    use_film = False,
    num_images_in_input = 2,
    use_proprio = True,
    load_in_8bit = False,
    load_in_4bit = False,
    center_crop = True,
    num_open_loop_steps = NUM_ACTIONS_CHUNK,
    unnorm_key = "libero_spatial_no_noops",
)

# Load OpenVLA-OFT policy and inputs processor
vla = get_vla(cfg)
processor = get_processor(cfg)

# Load MLP action head to generate continuous actions (via L1 regression)
action_head = get_action_head(cfg, llm_dim=vla.llm_dim)

# Load proprio projector to map proprio to language embedding space
proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

# Load sample observation:
#   observation (dict): {
#     "full_image": primary third-person image,
#     "wrist_image": wrist-mounted camera image,
#     "state": robot proprioceptive state,
#     "task_description": task description,
#   }
with open("experiments/robot/libero/sample_libero_spatial_observation.pkl", "rb") as file:
    observation = pickle.load(file)

# Generate robot action chunk (sequence of future actions)
actions = get_vla_action(cfg, vla, processor, observation, observation["task_description"], action_head, proprio_projector)
print("Generated action chunk:")
for act in actions:
    print(act)
```

## Installation

See [SETUP.md](SETUP.md) for instructions on setting up the conda environment.

## Training and Evaluation

See [LIBERO.md](LIBERO.md) for fine-tuning/evaluating on LIBERO simulation benchmark task suites.

See [ALOHA.md](ALOHA.md) for fine-tuning/evaluating on real-world ALOHA robot tasks.

## Support

> **Fork note:** the paragraph below is the upstream project's support policy and applies to
> upstream OpenVLA-OFT. For anything specific to this fork -- the panda datasets, the
> `ROBOT_PLATFORM` selection or `deploy_panda/` -- please open an issue here instead of
> contacting the upstream authors.

If you run into any issues, please open a new GitHub issue. If you do not receive a response within 2 business days, please email Moo Jin Kim (moojink@cs.stanford.edu) to bring the issue to his attention.

## Citation

If you use our code in your work, please cite [our paper](https://arxiv.org/abs/2502.19645):

```bibtex
@article{kim2025fine,
  title={Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success},
  author={Kim, Moo Jin and Finn, Chelsea and Liang, Percy},
  journal={arXiv preprint arXiv:2502.19645},
  year={2025}
}
```
