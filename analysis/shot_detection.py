"""
Shot detection: classify each rim approach as a made or missed basket.

The detector scans a sequence of YOLOv8 result frames, groups consecutive
frames where the ball is near the rim into *shot attempts*, then evaluates
each attempt by checking whether the ball's vertical trajectory passes through
the rim (made) or diverges before doing so (missed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class ShotEvent:
    """A single detected shot attempt.

    Attributes:
        frame_index: Representative frame number (midpoint of the approach).
        result: ``"made"`` or ``"missed"``.
        ball_position: ``(cx, cy)`` pixel coordinates of the ball at the
            representative frame.
        rim_position: ``(cx, cy)`` pixel coordinates of the rim at the
            representative frame.
    """
    frame_index: int
    result: str
    ball_position: Tuple[float, float]
    rim_position: Tuple[float, float]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_best_box(result, class_name: str) -> Optional[np.ndarray]:
    """Return the highest-confidence xyxy box for *class_name*, or ``None``."""
    names = result.names
    class_id = next(
        (k for k, v in names.items() if v.lower() == class_name.lower()), None
    )
    if class_id is None:
        return None

    mask = (result.boxes.cls == class_id).cpu().numpy().astype(bool)
    if not mask.any():
        return None

    xyxy = result.boxes.xyxy.cpu().numpy()[mask]
    confs = result.boxes.conf.cpu().numpy()[mask]
    return xyxy[int(np.argmax(confs))]


def _box_center(box: np.ndarray) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _is_ball_near_rim(
    ball_cx: float,
    ball_cy: float,
    rim_box: np.ndarray,
    h_scale: float = 3.0,
    w_scale: float = 2.0,
) -> bool:
    """True when the ball centre is within *w_scale* × rim-width horizontally
    and *h_scale* × rim-height vertically of the rim centre."""
    rim_cx, rim_cy = _box_center(rim_box)
    rim_w = rim_box[2] - rim_box[0]
    rim_h = rim_box[3] - rim_box[1]
    return abs(ball_cx - rim_cx) < rim_w * w_scale and abs(ball_cy - rim_cy) < rim_h * h_scale


def _evaluate_attempt(frames: list) -> Optional[ShotEvent]:
    """Classify a shot attempt as ``"made"`` or ``"missed"``.

    A shot is considered **made** when the ball centre starts *above* the rim
    centre (lower pixel-y value) and ends *below* it — i.e. the ball travelled
    downward through the hoop.  All other trajectories are ``"missed"``.

    Args:
        frames: Non-empty list of per-frame dicts collected by :func:`detect_shots`.

    Returns:
        A :class:`ShotEvent`, or ``None`` if *frames* is empty.
    """
    if not frames:
        return None

    first_ball_y = frames[0]["ball_center"][1]
    last_ball_y = frames[-1]["ball_center"][1]
    rim_cy = frames[0]["rim_center"][1]  # rim is roughly static

    passed_through = first_ball_y < rim_cy < last_ball_y

    mid = frames[len(frames) // 2]
    return ShotEvent(
        frame_index=mid["frame"],
        result="made" if passed_through else "missed",
        ball_position=mid["ball_center"],
        rim_position=mid["rim_center"],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_shots(
    results: list,
    min_approach_frames: int = 3,
) -> List[ShotEvent]:
    """Detect made and missed shots from a sequence of YOLOv8 result frames.

    Algorithm
    ---------
    1. For each frame, locate the ball and rim boxes.
    2. Accumulate consecutive frames where the ball is spatially near the rim
       into a *shot-attempt window*.
    3. When the window ends (ball leaves the rim area, or detections drop out),
       evaluate it: if it contains at least *min_approach_frames* frames it is
       classified as made or missed via :func:`_evaluate_attempt`.

    Args:
        results: Ordered list of YOLOv8 ``Results`` objects, one per frame.
        min_approach_frames: Minimum consecutive near-rim frames required to
            count as a shot attempt (filters incidental proximity).

    Returns:
        List of :class:`ShotEvent` objects, one per detected attempt, in
        chronological order.
    """
    events: List[ShotEvent] = []
    window: list = []

    for frame_idx, result in enumerate(results):
        ball_box = _get_best_box(result, "ball")
        rim_box = _get_best_box(result, "rim")

        if ball_box is None or rim_box is None:
            if len(window) >= min_approach_frames:
                event = _evaluate_attempt(window)
                if event:
                    events.append(event)
            window = []
            continue

        ball_cx, ball_cy = _box_center(ball_box)
        rim_cx, rim_cy = _box_center(rim_box)

        if _is_ball_near_rim(ball_cx, ball_cy, rim_box):
            window.append({
                "frame": frame_idx,
                "ball_center": (ball_cx, ball_cy),
                "rim_center": (rim_cx, rim_cy),
            })
        else:
            if len(window) >= min_approach_frames:
                event = _evaluate_attempt(window)
                if event:
                    events.append(event)
            window = []

    # Handle an attempt that extends to the very last frame
    if len(window) >= min_approach_frames:
        event = _evaluate_attempt(window)
        if event:
            events.append(event)

    return events
