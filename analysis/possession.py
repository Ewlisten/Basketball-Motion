"""
Possession detection: determine which player has the ball in a given frame.

Uses bounding-box overlap (IoU) as the primary signal, falling back to
nearest-centroid distance when the ball box doesn't intersect any player box.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_boxes_by_class(result, class_name: str) -> np.ndarray:
    """Return all bounding boxes (xyxy) for *class_name* in *result*.

    Args:
        result: A YOLOv8 ``Results`` object for a single frame.
        class_name: Case-insensitive class label (e.g. ``"ball"``, ``"player"``).

    Returns:
        NumPy array of shape ``(N, 4)`` in ``[x1, y1, x2, y2]`` format.
        Empty array when the class is absent.
    """
    names = result.names  # {int: str}
    class_id = next(
        (k for k, v in names.items() if v.lower() == class_name.lower()), None
    )
    if class_id is None:
        return np.empty((0, 4))

    mask = (result.boxes.cls == class_id).cpu().numpy().astype(bool)
    if not mask.any():
        return np.empty((0, 4))

    return result.boxes.xyxy.cpu().numpy()[mask]


def _box_center(box: np.ndarray):
    """Return ``(cx, cy)`` for a box in xyxy format."""
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Intersection-over-Union for two xyxy boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0.0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0.0 else 0.0


def _center_distance(box1: np.ndarray, box2: np.ndarray) -> float:
    """Euclidean distance between the centres of two xyxy boxes."""
    c1 = _box_center(box1)
    c2 = _box_center(box2)
    return float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_possession(result) -> Optional[int]:
    """Determine which player has possession of the ball in a single frame.

    The algorithm:
    1. Find the ball box (highest-confidence detection labelled ``"ball"``).
    2. If any player box **overlaps** the ball box (IoU > 0), return the index
       of the player with the highest IoU.
    3. Otherwise return the index of the **nearest** player by centroid
       distance — but only when that player is within 1.5× their own bounding-
       box height of the ball (loose proximity guard).
    4. Return ``None`` when the ball or players cannot be detected, or when no
       player is close enough to claim possession.

    Args:
        result: A YOLOv8 ``Results`` object for a single frame.

    Returns:
        Zero-based index into the array of detected player boxes, or ``None``.
    """
    ball_boxes = _get_boxes_by_class(result, "ball")
    player_boxes = _get_boxes_by_class(result, "player")

    if len(ball_boxes) == 0 or len(player_boxes) == 0:
        return None

    # Use the first detected ball (YOLOv8 sorts by confidence descending)
    ball_box = ball_boxes[0]

    # --- Phase 1: IoU overlap ---
    best_idx: Optional[int] = None
    best_iou = 0.0

    for i, player_box in enumerate(player_boxes):
        iou = _iou(ball_box, player_box)
        if iou > best_iou:
            best_iou = iou
            best_idx = i

    if best_iou > 0.0:
        return best_idx

    # --- Phase 2: nearest centroid with proximity guard ---
    best_idx = None
    best_dist = float("inf")

    for i, player_box in enumerate(player_boxes):
        dist = _center_distance(ball_box, player_box)
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    if best_idx is not None:
        player_height = player_boxes[best_idx][3] - player_boxes[best_idx][1]
        if best_dist <= player_height * 1.5:
            return best_idx

    return None
