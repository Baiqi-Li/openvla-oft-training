"""
Transport layer between the robot host and the OpenVLA-OFT inference server.

Deliberately depends on nothing beyond numpy, Pillow and the standard library, so it can
be dropped onto a robot host without touching that machine's environment. `json_numpy` is
used when importable and reimplemented inline otherwise; HTTP goes through `urllib`
rather than `requests`.

Wire format (matches `vla-scripts/deploy.py`):

    POST http://<host>:<port>/act
    request  {"full_image": <ndarray>, "wrist_image": <ndarray>,
              "state": <ndarray (8,)>, "instruction": <str>}
    response [<ndarray (8,)>, ... 15 of them]

    ndarrays travel as {"__numpy__": <base64 of raw buffer>, "dtype": "<f4", "shape": [...]}
"""

import base64
import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

# Native resolution of the frames the policy was trained on. The RLDS conversion kept the
# LeRobot frames at 320x180 and the training pipeline resized those to 224x224.
TRAIN_IMAGE_WIDTH = 320
TRAIN_IMAGE_HEIGHT = 180

# `prismatic/vla/constants.py` PANDA_CONSTANTS. Used only to validate what comes back.
EXPECTED_CHUNK_LEN = 15
EXPECTED_ACTION_DIM = 8
EXPECTED_PROPRIO_DIM = 8


# === numpy <-> JSON ==========================================================
#
# Same encoding as the `json-numpy` package, reimplemented so the robot host does not
# need it installed. If the real package is present we defer to it.

try:
    import json_numpy as _json_numpy

    _HAVE_JSON_NUMPY = True
except ImportError:  # pragma: no cover - depends on the robot host's environment
    _json_numpy = None
    _HAVE_JSON_NUMPY = False


def _numpy_default(obj: Any) -> Dict[str, Any]:
    """JSON encoder hook: ndarray/scalar -> base64 envelope."""
    if isinstance(obj, (np.ndarray, np.generic)):
        arr = np.ascontiguousarray(obj)
        return {
            "__numpy__": base64.b64encode(arr.data).decode(),
            "dtype": arr.dtype.str,
            "shape": arr.shape,
        }
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _numpy_object_hook(dct: Dict[str, Any]) -> Any:
    """JSON decoder hook: base64 envelope -> ndarray."""
    if "__numpy__" in dct:
        arr = np.frombuffer(base64.b64decode(dct["__numpy__"]), dtype=np.dtype(dct["dtype"]))
        shape = dct["shape"]
        return arr.reshape(shape) if shape else arr[0]
    return dct


def numpy_dumps(obj: Any) -> bytes:
    if _HAVE_JSON_NUMPY:
        return _json_numpy.dumps(obj).encode("utf-8")
    return json.dumps(obj, default=_numpy_default).encode("utf-8")


def numpy_loads(raw: bytes) -> Any:
    if _HAVE_JSON_NUMPY:
        return _json_numpy.loads(raw.decode("utf-8"))
    return json.loads(raw.decode("utf-8"), object_hook=_numpy_object_hook)


# === Image preprocessing =====================================================


def prepare_image(img: np.ndarray) -> np.ndarray:
    """
    Downsample a camera frame to the resolution the policy was trained at.

    Plain resize, NOT a letterbox/pad resize. Training squashed 320x180 frames to 224x224
    without preserving aspect ratio; padding here would put black bars into an input
    distribution that never contained them. (This is the one place where the openpi client
    and this one genuinely disagree -- openpi used `image_tools.resize_with_pad`.)

    The server resizes 320x180 -> 224x224 itself, using the exact training-time path
    (`resize_image_for_policy`: JPEG round-trip + lanczos3). Handing it 320x180 rather
    than a full-resolution frame is purely a bandwidth decision: ~230 KB vs ~3.7 MB per
    image once base64-encoded.

    Args:
        img: uint8 HxWx3 RGB frame. Alpha stripping and BGR->RGB are the caller's job,
             since those depend on the camera driver.

    Returns:
        uint8 180x320x3 RGB frame.
    """
    if img.dtype != np.uint8:
        raise ValueError(f"expected a uint8 image, got {img.dtype}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 RGB image, got shape {img.shape}")

    # The caller's BGR->RGB flip (`img[..., ::-1]`) leaves a negative-stride view, which
    # neither PIL nor the base64 encoder can consume directly.
    img = np.ascontiguousarray(img)

    if img.shape[:2] == (TRAIN_IMAGE_HEIGHT, TRAIN_IMAGE_WIDTH):
        return img

    resized = Image.fromarray(img).resize(
        (TRAIN_IMAGE_WIDTH, TRAIN_IMAGE_HEIGHT), resample=Image.LANCZOS
    )
    return np.asarray(resized, dtype=np.uint8)


# === Client ==================================================================


class OpenVLAOFTClient:
    """
    Stateless HTTP client for the `/act` endpoint.

    Every call is an independent POST, so a server restart mid-rollout costs one timed-out
    request rather than a dead connection that has to be noticed and rebuilt.
    """

    def __init__(
        self,
        host: str,
        port: int = 8777,
        timeout: float = 30.0,
        retries: int = 2,
        retry_backoff: float = 0.25,
    ):
        self.endpoint = f"http://{host}:{port}/act"
        self.timeout = timeout
        self.retries = retries
        self.retry_backoff = retry_backoff
        self.last_latency: Optional[float] = None

    def infer(
        self,
        full_image: np.ndarray,
        wrist_image: np.ndarray,
        state: np.ndarray,
        instruction: str,
        *,
        preprocess: bool = True,
        validate: bool = True,
    ) -> np.ndarray:
        """
        Query the policy for one action chunk.

        Args:
            full_image:  uint8 HxWx3 RGB exterior camera frame.
            wrist_image: uint8 HxWx3 RGB wrist camera frame.
            state:       float (8,) = [joint_positions (7), gripper_position (1)], RAW.
                         The server normalizes it with the training BOUNDS_Q99 proprio
                         statistics, so normalizing here as well would double-apply.
            instruction: Task string. Must be word-for-word one of the 24 instructions the
                         policy was trained on -- see tasks.json.
            preprocess:  Resize the images to 320x180 first. Pass False only if the caller
                         already did it.
            validate:    Check the returned chunk's shape and range.

        Returns:
            float32 (15, 8) action chunk:
              [:, :7] joint velocities in DROID's normalized [-1, 1] space
              [:,  7] absolute gripper position in [0, 1]
        """
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if validate and state.shape[0] != EXPECTED_PROPRIO_DIM:
            raise ValueError(
                f"state must be {EXPECTED_PROPRIO_DIM}-dim "
                f"[joint_positions(7), gripper_position(1)], got shape {state.shape}"
            )

        if preprocess:
            full_image = prepare_image(full_image)
            wrist_image = prepare_image(wrist_image)

        payload = {
            "full_image": full_image,
            "wrist_image": wrist_image,
            "state": state,
            "instruction": instruction,
        }

        response = self._post(payload)

        # The server catches its own exceptions and returns the bare string "error", with
        # the traceback only on its stdout. Translate that into something actionable here.
        if isinstance(response, str):
            raise RuntimeError(
                f"Inference server returned an error for instruction {instruction!r}. "
                "The traceback is on the SERVER's console -- check it there. Most often "
                "this is a payload-key or state-shape mismatch."
            )

        chunk = np.asarray(response, dtype=np.float32)
        if validate:
            self._validate_chunk(chunk)
        return chunk

    def _post(self, payload: Dict[str, Any]) -> Any:
        body = numpy_dumps(payload)
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        last_error: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            start = time.time()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read()
                self.last_latency = time.time() - start
                return numpy_loads(raw)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(self.retry_backoff * (attempt + 1))

        raise RuntimeError(
            f"Could not reach the inference server at {self.endpoint} "
            f"after {self.retries + 1} attempts: {last_error}"
        ) from last_error

    @staticmethod
    def _validate_chunk(chunk: np.ndarray) -> None:
        if chunk.shape != (EXPECTED_CHUNK_LEN, EXPECTED_ACTION_DIM):
            raise ValueError(
                f"Expected an action chunk of shape ({EXPECTED_CHUNK_LEN}, {EXPECTED_ACTION_DIM}), "
                f"got {chunk.shape}. A (8, 7) chunk means the server resolved LIBERO constants "
                "instead of PANDA -- set ROBOT_PLATFORM=PANDA on the server and restart."
            )
        if not np.all(np.isfinite(chunk)):
            raise ValueError("Action chunk contains NaN or inf.")

    def ping(self) -> Tuple[bool, str]:
        """Send one dummy request. Returns (reachable, message). For preflight checks."""
        dummy_image = np.zeros((TRAIN_IMAGE_HEIGHT, TRAIN_IMAGE_WIDTH, 3), dtype=np.uint8)
        dummy_state = np.zeros(EXPECTED_PROPRIO_DIM, dtype=np.float32)
        try:
            chunk = self.infer(
                dummy_image, dummy_image, dummy_state, "grasp the coke", preprocess=False
            )
        except Exception as exc:  # noqa: BLE001 - the message is the return value
            return False, str(exc)
        return True, f"ok: chunk {chunk.shape}, {self.last_latency * 1000:.0f} ms"
