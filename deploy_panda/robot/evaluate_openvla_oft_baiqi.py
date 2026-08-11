# ruff: noqa
"""
Evaluate the panda-40k OpenVLA-OFT policy on the real robot.

Runs ON THE ROBOT HOST, against an inference server started with
`deploy_panda/server/deploy_panda.sh`.

Ported from `evaluate_openpi_baiqi.py`. The task/step state machine, scoring prompts, CSV
and merged-MP4 output, camera ids, control frequency and action post-processing are all
unchanged; only the policy transport differs:

  openpi                                    openvla-oft
  ------------------------------------      ------------------------------------
  websocket, port 8000                      HTTP POST /act, port 8777
  observation/exterior_image_1_left         full_image
  observation/wrist_image_left              wrist_image
  observation/joint_position   (7)   \\      state (8, concatenated, raw)
  observation/gripper_position (1)   /
  prompt                                    instruction
  image_tools.resize_with_pad(224)          plain resize to 320x180 (no padding)
  -> (H, 8) chunk                           -> (15, 8) chunk

Actions are identical in both cases: 7 joint velocities in DROID's normalized [-1, 1]
space plus one absolute gripper position, so the env setup and clipping carry over as-is.
"""

import contextlib
import dataclasses
import datetime
import faulthandler
import json
import os
import signal
import time
from moviepy.editor import ImageSequenceClip
from typing import Optional, Union
import numpy as np
import pandas as pd
from PIL import Image
from droid.robot_env import RobotEnv
import tqdm
import tyro

from openvla_oft_client import OpenVLAOFTClient, EXPECTED_CHUNK_LEN

faulthandler.enable()

# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 15

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclasses.dataclass
class Args:
    # Hardware parameters
    left_camera_id: str = "36087771"  # e.g., "24259877"
    right_camera_id: str = ""  # e.g., "24514023"
    wrist_camera_id: str = "16478870"  # e.g., "13062452"

    # Policy parameters
    external_camera: str = "left"

    # Rollout parameters
    max_timesteps: int = 600
    # How many actions to execute from a predicted action chunk before querying policy
    # server again. The chunk holds 15 actions (1 second at 15 Hz).
    #
    # 12 rather than openpi's 8: a 7B model with two images on an RTX 6000 Ada takes
    # roughly 300-500 ms per forward pass, and at 8 the next chunk is due every 533 ms,
    # which leaves almost no headroom once network time is added. Run
    # `smoke_test_client.py` to measure the real number and size this accordingly -- it
    # prints a recommendation.
    open_loop_horizon: int = 12

    # Remote server parameters
    remote_host: str = "0.0.0.0"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = 8777  # default server port for openvla-oft deploy.py
    request_timeout: float = 30.0

    model: str = "openvla-oft-panda-40k"

    # Task definitions JSON, layout: {task_id: {"description": str, "steps": [str, ...]}}
    tasks_json: str = os.path.join(SCRIPT_DIR, "tasks.json")
    # If empty, will prompt interactively after listing available ids.
    task_id: str = ""
    # Output directory for csv + merged mp4
    output_dir: str = os.path.join(SCRIPT_DIR, "rollouts")


@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def load_tasks(tasks_json_path: str) -> dict:
    if not os.path.isfile(tasks_json_path):
        raise FileNotFoundError(
            f"tasks JSON not found at {tasks_json_path}. "
            f"Create one with the schema {{task_id: {{description, steps: [str,...]}}}}."
        )
    with open(tasks_json_path, "r") as f:
        tasks = json.load(f)
    # Drop the leading comment block, if present (tasks.json ships with one).
    tasks = {tid: entry for tid, entry in tasks.items() if not tid.startswith("_")}
    # Validate
    for tid, entry in tasks.items():
        if not isinstance(entry, dict) or "steps" not in entry or not isinstance(entry["steps"], list):
            raise ValueError(f"Task {tid} must be an object with a 'steps' list.")
    return tasks


def pick_task_id(tasks: dict, preset: str) -> str:
    if preset:
        if preset not in tasks:
            raise KeyError(f"task_id '{preset}' not in tasks JSON. Available: {list(tasks.keys())}")
        return preset
    print("\nAvailable tasks:")
    for tid, entry in tasks.items():
        desc = entry.get("description", "")
        n_steps = len(entry["steps"])
        print(f"  {tid:<20} ({n_steps} steps)  {desc}")
    while True:
        chosen = input("Enter task_id: ").strip()
        if chosen in tasks:
            return chosen
        print(f"  '{chosen}' not found. Try again.")


def ask_success() -> float:
    while True:
        raw = input(
            "Did the rollout succeed? (y=100%, n=0%, or a numeric value 0-100): "
        ).strip()
        if raw == "y":
            return 1.0
        if raw == "n":
            return 0.0
        try:
            val = float(raw) / 100
            if 0 <= val <= 1:
                return val
            print(f"  Must be in [0, 100], got {val * 100}")
        except ValueError:
            print("  Invalid input.")


def ask_next_action() -> str:
    """Returns 'continue', 'retry', or 'abort'."""
    while True:
        raw = input("Next? (c=continue next step / r=retry this step / a=abort task): ").strip().lower()
        if raw in ("c", "continue", ""):
            return "continue"
        if raw in ("r", "retry"):
            return "retry"
        if raw in ("a", "abort"):
            return "abort"
        print("  Please enter c, r, or a.")


def run_step(env, policy_client: OpenVLAOFTClient, args: Args, instruction: str, rollout_dir: str, save_first_obs: bool):
    """Run a single step rollout. Returns (frames_list, t_step, latencies)."""
    actions_from_chunk_completed = 0
    pred_action_chunk = None
    frames = []
    latencies = []
    bar = tqdm.tqdm(range(args.max_timesteps))
    print(f"Running step: '{instruction}'. Press Ctrl+C to stop early.")
    first_chunk = True
    t_step = 0

    for t_step in bar:
        start_time = time.time()
        try:
            curr_obs = _extract_observation(
                args,
                env.get_observation(),
                save_to_disk=save_first_obs and t_step == 0,
                rollout_dir=rollout_dir,
            )

            frames.append(curr_obs[f"{args.external_camera}_image"])

            if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                actions_from_chunk_completed = 0

                # Raw 8-dim proprio: the server applies the training BOUNDS_Q99
                # normalization, so it must not be normalized here as well.
                state = np.concatenate([curr_obs["joint_position"], curr_obs["gripper_position"]])

                with prevent_keyboard_interrupt():
                    pred_action_chunk = policy_client.infer(
                        full_image=curr_obs[f"{args.external_camera}_image"],
                        wrist_image=curr_obs["wrist_image"],
                        state=state,
                        instruction=instruction,
                    )
                latencies.append(policy_client.last_latency)
                assert pred_action_chunk.shape == (EXPECTED_CHUNK_LEN, 8), pred_action_chunk.shape

                if first_chunk:
                    print(f"Predicted action chunk shape: {pred_action_chunk.shape}")
                    print(f"Server round-trip: {policy_client.last_latency * 1000:.0f} ms")
                    first_chunk = False

            action = pred_action_chunk[actions_from_chunk_completed]
            actions_from_chunk_completed += 1

            if action[-1].item() < 0.1:
                action = np.concatenate([action[:-1], np.zeros((1,))])

            action = np.clip(action, -1, 1)
            env.step(action)

            elapsed_time = time.time() - start_time
            if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
        except KeyboardInterrupt:
            break

    return frames, t_step, latencies


def main(args: Args):
    tasks = load_tasks(args.tasks_json)
    task_id = pick_task_id(tasks, args.task_id)
    task_entry = tasks[task_id]
    steps = task_entry["steps"]
    description = task_entry.get("description", "")
    print(f"\nTask {task_id}: {description}")
    print(f"  {len(steps)} steps loaded.\n")

    policy_client = OpenVLAOFTClient(
        args.remote_host, args.remote_port, timeout=args.request_timeout
    )

    # Confirm the server answers before touching the robot, so a misconfigured or
    # still-loading server surfaces here rather than mid-rollout.
    print(f"Contacting policy server at {policy_client.endpoint} ...")
    reachable, message = policy_client.ping()
    if not reachable:
        raise SystemExit(
            f"Policy server is not usable: {message}\n"
            f"Start it with deploy_panda/server/deploy_panda.sh and check that port "
            f"{args.remote_port} is open."
        )
    print(f"Policy server {message}")

    budget_ms = 1000 * args.open_loop_horizon / DROID_CONTROL_FREQUENCY
    if policy_client.last_latency * 1000 > budget_ms:
        print(
            f"WARNING: round-trip ({policy_client.last_latency * 1000:.0f} ms) exceeds the "
            f"{budget_ms:.0f} ms budget implied by open_loop_horizon={args.open_loop_horizon}. "
            f"The control loop will stall on every chunk boundary. Raise --open_loop_horizon "
            f"(max {EXPECTED_CHUNK_LEN})."
        )

    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    print("Created the droid env!")

    os.makedirs(args.output_dir, exist_ok=True)

    while True:
        # One outer iteration == one full attempt at the whole task.
        run_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        run_basename = f"{task_id}_{run_timestamp}"
        merged_frames = []
        step_rows = []

        step_idx = 0
        while step_idx < len(steps):
            instruction = steps[step_idx]
            attempt = 0
            while True:
                step_frames, t_step, latencies = run_step(
                    env,
                    policy_client,
                    args,
                    instruction,
                    rollout_dir=args.output_dir,
                    save_first_obs=(step_idx == 0 and attempt == 0),
                )
                success = ask_success()
                decision = ask_next_action()

                if decision == "retry":
                    print("Retrying current step. Resetting robot...")
                    env.reset()
                    attempt += 1
                    continue

                # Accepted: keep frames + log row.
                merged_frames.extend(step_frames)
                step_rows.append(
                    {
                        "task_id": task_id,
                        "step_idx": step_idx,
                        "instruction": instruction,
                        "success": success,
                        "duration": t_step,
                        "retries": attempt,
                        "model": args.model,
                        "mean_latency_ms": round(1000 * float(np.mean(latencies)), 1) if latencies else None,
                        "max_latency_ms": round(1000 * float(np.max(latencies)), 1) if latencies else None,
                    }
                )
                break

            if decision == "abort":
                print(f"Aborting task {task_id} at step {step_idx}.")
                step_idx = len(steps)  # exit the steps loop
            else:
                step_idx += 1

        # Save merged video for this task attempt.
        if merged_frames:
            video_path = os.path.join(args.output_dir, f"{run_basename}.mp4")
            video = np.stack(merged_frames)
            ImageSequenceClip(list(video), fps=10).write_videofile(video_path, codec="libx264")
            print(f"Saved video: {video_path}")
        else:
            video_path = ""
            print("No frames collected; skipping video write.")

        # Save CSV: one row per accepted step.
        csv_path = os.path.join(args.output_dir, f"{run_basename}.csv")
        df = pd.DataFrame(step_rows)
        df["video_filename"] = os.path.basename(video_path)
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV:   {csv_path}")

        if input("Do one more eval? (enter y or n): ").lower() != "y":
            break
        env.reset()


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False, rollout_dir: str):
    image_observations = obs_dict["image"]
    left_image, wrist_image = None, None
    for key in image_observations:
        if args.left_camera_id in key and "left" in key:
            left_image = image_observations[key]
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = image_observations[key]

    left_image = left_image[..., :3]
    wrist_image = wrist_image[..., :3]

    left_image = left_image[..., ::-1]
    wrist_image = wrist_image[..., ::-1]

    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    if save_to_disk:
        combined_image = np.concatenate([left_image, wrist_image], axis=1)
        combined_image = Image.fromarray(combined_image)
        combined_image.save(os.path.join(rollout_dir, "robot_camera_views.png"))

    return {
        "left_image": left_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
