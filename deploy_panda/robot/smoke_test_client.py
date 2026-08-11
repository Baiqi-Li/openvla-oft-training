#!/usr/bin/env python
"""
Verify the inference server without touching the robot.

Sends synthetic observations, checks the shape and range of what comes back, and measures
round-trip latency so `--open_loop_horizon` can be sized from a real number.

Run it twice:

  1. On the inference server, to separate model problems from network problems:
       python smoke_test_client.py --host 127.0.0.1
  2. On the robot host, to get the latency the control loop will actually see:
       python smoke_test_client.py --host <server-ip>

Depends only on numpy, Pillow and the standard library -- no droid, no tyro.
"""

import argparse
import sys
import time

import numpy as np

from openvla_oft_client import (
    EXPECTED_ACTION_DIM,
    EXPECTED_CHUNK_LEN,
    EXPECTED_PROPRIO_DIM,
    TRAIN_IMAGE_HEIGHT,
    TRAIN_IMAGE_WIDTH,
    OpenVLAOFTClient,
    _HAVE_JSON_NUMPY,
    prepare_image,
)

DROID_CONTROL_FREQUENCY = 15

# Must be word-for-word one of the instructions in tasks.json. Anything else is out of
# distribution and makes the output meaningless as a check.
DEFAULT_INSTRUCTION = "grasp the coke"


def synthetic_observation(camera_height: int, camera_width: int, seed: int):
    """A plausible-looking observation. Content is irrelevant; shapes and ranges are not."""
    rng = np.random.default_rng(seed)
    full_image = rng.integers(0, 256, (camera_height, camera_width, 3), dtype=np.uint8)
    wrist_image = rng.integers(0, 256, (camera_height, camera_width, 3), dtype=np.uint8)
    # Roughly a Panda home pose plus an open gripper, so proprio normalization lands
    # somewhere inside the training distribution.
    state = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.8, 0.0], dtype=np.float32)
    state[:7] += rng.normal(0, 0.05, 7)
    return full_image, wrist_image, state


def check_chunk(chunk: np.ndarray) -> list:
    """Range checks on one action chunk. Returns a list of warning strings."""
    warnings = []

    joint_vel, gripper = chunk[:, :7], chunk[:, 7]

    if np.abs(joint_vel).max() > 1.05:
        warnings.append(
            f"joint velocities reach {np.abs(joint_vel).max():.3f}, outside the [-1, 1] "
            "range the training actions were bounded to"
        )
    if gripper.min() < -0.05 or gripper.max() > 1.05:
        warnings.append(
            f"gripper spans [{gripper.min():.3f}, {gripper.max():.3f}], outside [0, 1]"
        )
    if np.abs(joint_vel).max() < 1e-4:
        warnings.append(
            "joint velocities are all ~0 -- suspicious for a random image, but not "
            "impossible; compare against a real observation before reading anything into it"
        )
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", required=True, help="Inference server IP (127.0.0.1 when run on the server)")
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--num_requests", type=int, default=10, help="Timed requests after the warmup")
    parser.add_argument("--timeout", type=float, default=120.0, help="Generous: the first request compiles kernels")
    parser.add_argument("--camera_height", type=int, default=720, help="Simulates the robot camera's native size")
    parser.add_argument("--camera_width", type=int, default=1280)
    args = parser.parse_args()

    print("=" * 72)
    print("OpenVLA-OFT smoke test")
    print("=" * 72)
    print(f"  endpoint     : http://{args.host}:{args.port}/act")
    print(f"  instruction  : {args.instruction!r}")
    print(f"  camera frame : {args.camera_height}x{args.camera_width} -> "
          f"{TRAIN_IMAGE_HEIGHT}x{TRAIN_IMAGE_WIDTH} on the wire")
    print(f"  json_numpy   : {'installed' if _HAVE_JSON_NUMPY else 'not installed, using the built-in codec'}")
    print()

    client = OpenVLAOFTClient(args.host, args.port, timeout=args.timeout, retries=1)

    full_image, wrist_image, state = synthetic_observation(args.camera_height, args.camera_width, seed=0)

    # Payload size, measured rather than assumed -- it sets the bandwidth floor.
    from openvla_oft_client import numpy_dumps

    payload_bytes = len(numpy_dumps({
        "full_image": prepare_image(full_image),
        "wrist_image": prepare_image(wrist_image),
        "state": state,
        "instruction": args.instruction,
    }))
    print(f"Request payload: {payload_bytes / 1024:.0f} KB")

    print("\n[1/3] Warmup request (first call is slow: CUDA kernels + caches)...")
    t0 = time.time()
    try:
        chunk = client.infer(full_image, wrist_image, state, args.instruction)
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAILED: {exc}")
        print("\nIf this is a connection error, check that the server is running and that "
              f"port {args.port} is open in its firewall.")
        print("If the server returned an error, its traceback is on the SERVER's console.")
        return 1
    print(f"      ok, {time.time() - t0:.1f} s, chunk {chunk.shape} {chunk.dtype}")

    print("\n[2/3] Shape and range checks")
    ok = True
    if chunk.shape == (EXPECTED_CHUNK_LEN, EXPECTED_ACTION_DIM):
        print(f"      PASS  chunk shape is ({EXPECTED_CHUNK_LEN}, {EXPECTED_ACTION_DIM})")
    else:
        ok = False
        print(f"      FAIL  chunk shape is {chunk.shape}, expected "
              f"({EXPECTED_CHUNK_LEN}, {EXPECTED_ACTION_DIM})")
        if chunk.shape == (8, 7):
            print("            (8, 7) is the LIBERO default. The server did not resolve the "
                  "PANDA constants -- set ROBOT_PLATFORM=PANDA and restart it.")

    print(f"      joint velocities [:, :7]  min {chunk[:, :7].min():+.3f}  max {chunk[:, :7].max():+.3f}")
    print(f"      gripper          [:,  7]  min {chunk[:, 7].min():+.3f}  max {chunk[:, 7].max():+.3f}")
    for warning in check_chunk(chunk):
        print(f"      WARN  {warning}")

    # A policy that ignores its inputs returns the same chunk for every observation, which
    # is what a mis-wired proprio projector or a dropped image looks like.
    other_full, other_wrist, other_state = synthetic_observation(
        args.camera_height, args.camera_width, seed=1
    )
    other_chunk = client.infer(other_full, other_wrist, other_state, args.instruction)
    if np.allclose(chunk, other_chunk):
        ok = False
        print("      FAIL  two different observations produced an identical chunk; the "
              "policy is not reading its inputs")
    else:
        delta = np.abs(chunk - other_chunk).mean()
        print(f"      PASS  different observations produce different chunks (mean |delta| {delta:.4f})")

    print(f"\n[3/3] Latency over {args.num_requests} requests")
    latencies = []
    for i in range(args.num_requests):
        obs = synthetic_observation(args.camera_height, args.camera_width, seed=100 + i)
        client.infer(*obs, args.instruction)
        latencies.append(client.last_latency)
        print(f"      {i + 1:>2}/{args.num_requests}  {client.last_latency * 1000:6.0f} ms")

    latencies = np.array(latencies)
    p50, p90, worst = np.percentile(latencies, 50), np.percentile(latencies, 90), latencies.max()
    print(f"\n      p50 {p50 * 1000:.0f} ms   p90 {p90 * 1000:.0f} ms   max {worst * 1000:.0f} ms")

    # An open-loop horizon of H buys H / 15 seconds before the next chunk is due. Size it
    # off p90 rather than p50: the control loop stalls on the slow requests, not the median.
    step_ms = 1000.0 / DROID_CONTROL_FREQUENCY
    needed = int(np.ceil(p90 * 1000 / step_ms))
    recommended = min(max(needed + 2, 4), EXPECTED_CHUNK_LEN)

    print()
    print(f"      One control step is {step_ms:.0f} ms; horizon H gives H x {step_ms:.0f} ms of slack.")
    print(f"      p90 needs H >= {needed}; recommended --open_loop_horizon {recommended} "
          f"(+2 margin, capped at the {EXPECTED_CHUNK_LEN}-action chunk)")
    if needed > EXPECTED_CHUNK_LEN:
        ok = False
        print(f"      FAIL  even H={EXPECTED_CHUNK_LEN} cannot cover a {p90 * 1000:.0f} ms p90. "
              "The loop will stall at every chunk boundary. Move the server closer, give it "
              "a faster GPU, or accept a control rate below 15 Hz.")

    print()
    print("=" * 72)
    print("SMOKE TEST PASSED" if ok else "SMOKE TEST FAILED")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
