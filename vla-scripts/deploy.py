"""
deploy.py

Starts VLA server which the client can query to get robot actions.
"""

import os.path

# ruff: noqa: E402
import json_numpy

json_numpy.patch()
import json
import logging
import numpy as np
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import draccus
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from experiments.robot.openvla_utils import (
    get_vla,
    get_vla_action,
    get_action_head,
    get_processor,
    get_proprio_projector,
)
from experiments.robot.robot_utils import (
    get_image_resize_size,
)
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_TOKEN_BEGIN_IDX,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
    ROBOT_PLATFORM,
    STOP_INDEX,
)


def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


# === Server Interface ===
class OpenVLAServer:
    def __init__(self, cfg) -> Path:
        """
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given observation + instruction.
        """
        self.cfg = cfg

        # Get expected image dimensions
        self.resize_size = get_image_resize_size(cfg)

        # Validate the serving configuration before spending minutes loading weights
        self._run_startup_checks()

        # Load model
        self.vla = get_vla(cfg)

        # Load proprio projector
        self.proprio_projector = None
        if cfg.use_proprio:
            self.proprio_projector = get_proprio_projector(cfg, self.vla.llm_dim, PROPRIO_DIM)

        # Load continuous action head
        self.action_head = None
        if cfg.use_l1_regression or cfg.use_diffusion:
            self.action_head = get_action_head(cfg, self.vla.llm_dim)

        # Check that the model contains the action un-normalization key
        assert cfg.unnorm_key in self.vla.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

        # Get Hugging Face processor
        self.processor = None
        self.processor = get_processor(cfg)

        # Log the shape of the first predicted action chunk, so a client can be verified against it
        self._logged_action_shape = False

    def _run_startup_checks(self) -> None:
        """
        Print the resolved serving configuration and fail fast on any mismatch the launcher declared.

        Several `DeployConfig` defaults do not match every fine-tune (`num_images_in_input=3` and
        `use_film=False` in particular), and the robot-platform constants are picked up implicitly.
        Getting either wrong yields wrong-shaped or silently degraded actions rather than an error,
        so a launcher can pin the values it expects via the `expected_*` fields.
        """
        print("=" * 80)
        print("OpenVLA-OFT server configuration")
        print("-" * 80)
        print(f"  checkpoint            = {self.cfg.pretrained_checkpoint}")
        print(f"  unnorm_key            = {self.cfg.unnorm_key}")
        print(f"  robot platform        = {ROBOT_PLATFORM}")
        print(f"  num_actions_chunk     = {NUM_ACTIONS_CHUNK}")
        print(f"  action_dim            = {ACTION_DIM}")
        print(f"  proprio_dim           = {PROPRIO_DIM}")
        print(f"  num_images_in_input   = {self.cfg.num_images_in_input}")
        print(f"  use_film              = {self.cfg.use_film}")
        print(f"  use_proprio           = {self.cfg.use_proprio}")
        print(f"  use_l1_regression     = {self.cfg.use_l1_regression}")
        print(f"  use_diffusion         = {self.cfg.use_diffusion}")
        print(f"  center_crop           = {self.cfg.center_crop}")
        print(f"  lora_rank             = {self.cfg.lora_rank}")
        print(f"  image size            = {self.resize_size}")
        print("=" * 80)

        expectations = [
            ("robot platform", ROBOT_PLATFORM, self.cfg.expected_robot_platform.strip().upper()),
            ("num_actions_chunk", NUM_ACTIONS_CHUNK, self.cfg.expected_num_actions_chunk),
            ("action_dim", ACTION_DIM, self.cfg.expected_action_dim),
            ("proprio_dim", PROPRIO_DIM, self.cfg.expected_proprio_dim),
            ("num_images_in_input", self.cfg.num_images_in_input, self.cfg.expected_num_images_in_input),
        ]
        mismatches = [
            f"{name}: expected {expected!r}, got {actual!r}"
            for name, actual, expected in expectations
            if expected and expected != actual
        ]
        if self.cfg.expected_use_film is not None and self.cfg.expected_use_film != self.cfg.use_film:
            mismatches.append(f"use_film: expected {self.cfg.expected_use_film}, got {self.cfg.use_film}")
        if mismatches:
            raise ValueError(
                "Serving configuration does not match what the launcher declared:\n  "
                + "\n  ".join(mismatches)
                + "\n\nFor the robot-platform constants, set the `ROBOT_PLATFORM` env var "
                "(see `prismatic/vla/constants.py`). For the rest, pass the matching "
                "`--<name>` flag -- it must agree with how the checkpoint was trained."
            )

    def get_server_action(self, payload: Dict[str, Any]) -> str:
        try:
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            observation = payload
            instruction = observation["instruction"]

            action = get_vla_action(
                self.cfg, self.vla, self.processor, observation, instruction, action_head=self.action_head, proprio_projector=self.proprio_projector, use_film=self.cfg.use_film,
            )

            if not self._logged_action_shape:
                self._logged_action_shape = True
                chunk = np.asarray(action)
                print(f"First action chunk: shape={chunk.shape}, dtype={chunk.dtype}")
                print(f"  instruction: {instruction!r}")
                print(f"  per-dim min: {np.round(chunk.min(axis=0), 4).tolist()}")
                print(f"  per-dim max: {np.round(chunk.max(axis=0), 4).tolist()}")

            if double_encode:
                return JSONResponse(json_numpy.dumps(action))
            else:
                return JSONResponse(action)
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'observation': dict, 'instruction': str}\n"
            )
            return "error"

    def run(self, host: str = "0.0.0.0", port: int = 8777) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.get_server_action)
        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployConfig:
    # fmt: off

    # Server Configuration
    host: str = "0.0.0.0"                                               # Host IP Address
    port: int = 8777                                                    # Host Port

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 3                     # Number of images in the VLA input (default: 3)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key
    use_relative_actions: bool = False               # Whether to use relative actions (delta joint angles)

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # Startup self-checks (0 / "" disables the individual check)
    #
    # `num_images_in_input` and `use_film` above default to values that do NOT match every
    # fine-tune, and the robot-platform constants are resolved implicitly at import time. Pin
    # what this checkpoint needs here so a mismatch fails at startup instead of silently
    # producing wrong-shaped actions.
    #################################################################################################################
    expected_robot_platform: str = ""                # e.g. "PANDA" (see prismatic/vla/constants.py)
    expected_num_actions_chunk: int = 0              # e.g. 15
    expected_action_dim: int = 0                     # e.g. 8
    expected_proprio_dim: int = 0                    # e.g. 8
    expected_num_images_in_input: int = 0            # e.g. 2
    expected_use_film: Optional[bool] = None         # e.g. True

    #################################################################################################################
    # Utils
    #################################################################################################################
    seed: int = 7                                    # Random Seed (for reproducibility)
    # fmt: on


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = OpenVLAServer(cfg)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()
