"""
Important constants for VLA training and evaluation.

Attempts to automatically identify the correct constants to set based on the Python command used to launch
training or evaluation. If it is unclear, defaults to using the LIBERO simulation benchmark constants.
"""
import os
import sys
from enum import Enum

# Llama 2 token constants
IGNORE_INDEX = -100
ACTION_TOKEN_BEGIN_IDX = 31743
STOP_INDEX = 2  # '</s>'


# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


# Define constants for each robot platform
LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

ALOHA_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 25,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

BRIDGE_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 5,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 7,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

# Single-arm Franka/Panda datasets converted from LeRobot (15 fps, DROID-style actions):
#   action = joint velocity (7, rad/s) + absolute gripper position (1)
#   proprio = joint positions (7) + gripper position (1)
# BOUNDS_Q99 is used (rather than ALOHA's BOUNDS) because the first 7 action dims are
# velocities with outliers worth clipping, not absolute joint angles the policy must be
# able to reach exactly.
PANDA_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 15,  # 15 fps => 1 second of actions per chunk
    "ACTION_DIM": 8,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}


PLATFORM_CONSTANTS = {
    "LIBERO": LIBERO_CONSTANTS,
    "ALOHA": ALOHA_CONSTANTS,
    "BRIDGE": BRIDGE_CONSTANTS,
    "PANDA": PANDA_CONSTANTS,
}


# Function to detect robot platform from command line arguments
def detect_robot_platform():
    # An explicit `ROBOT_PLATFORM` env var always wins. The argv sniffing below silently
    # falls back to LIBERO (chunk 8, action dim 7) whenever the platform name happens not
    # to appear on the command line -- e.g. a checkpoint directory renamed to something
    # without "panda" in it -- which produces wrong-shaped actions with no error.
    env_platform = os.environ.get("ROBOT_PLATFORM", "").strip().upper()
    if env_platform:
        if env_platform not in PLATFORM_CONSTANTS:
            raise ValueError(
                f"ROBOT_PLATFORM={env_platform!r} is not one of {sorted(PLATFORM_CONSTANTS)}."
            )
        return env_platform, "ROBOT_PLATFORM env var"

    cmd_args = " ".join(sys.argv).lower()

    if "libero" in cmd_args:
        return "LIBERO", "command line"
    elif "aloha" in cmd_args:
        return "ALOHA", "command line"
    elif "bridge" in cmd_args:
        return "BRIDGE", "command line"
    elif "panda" in cmd_args:
        return "PANDA", "command line"
    else:
        # Default to LIBERO if unclear
        return "LIBERO", "default (nothing matched)"


# Determine which robot platform to use
ROBOT_PLATFORM, ROBOT_PLATFORM_SOURCE = detect_robot_platform()

# Set the appropriate constants based on the detected platform
constants = PLATFORM_CONSTANTS[ROBOT_PLATFORM]

# Assign constants to global variables
NUM_ACTIONS_CHUNK = constants["NUM_ACTIONS_CHUNK"]
ACTION_DIM = constants["ACTION_DIM"]
PROPRIO_DIM = constants["PROPRIO_DIM"]
ACTION_PROPRIO_NORMALIZATION_TYPE = constants["ACTION_PROPRIO_NORMALIZATION_TYPE"]

# Print which robot platform constants are being used (for debugging)
print(f"Using {ROBOT_PLATFORM} constants (selected via {ROBOT_PLATFORM_SOURCE}):")
print(f"  NUM_ACTIONS_CHUNK = {NUM_ACTIONS_CHUNK}")
print(f"  ACTION_DIM = {ACTION_DIM}")
print(f"  PROPRIO_DIM = {PROPRIO_DIM}")
print(f"  ACTION_PROPRIO_NORMALIZATION_TYPE = {ACTION_PROPRIO_NORMALIZATION_TYPE}")
print("If needed, manually set the correct constants in `prismatic/vla/constants.py`!")
