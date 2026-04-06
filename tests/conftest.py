"""
Shared helpers for building lightweight mock YOLOv8 Result objects.

YOLOv8 Results have the structure:
    result.names          -> {int: str}
    result.boxes.cls      -> tensor of class IDs  (shape N)
    result.boxes.xyxy     -> tensor of boxes      (shape N x 4)
    result.boxes.conf     -> tensor of confidences(shape N)

We replicate this with plain Python objects so tests run without a GPU
or a real model checkpoint.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np


class _FakeTensor:
    """Wraps a NumPy array and exposes ``.cpu().numpy()`` like a PyTorch tensor."""

    def __init__(self, array: np.ndarray):
        self._array = np.asarray(array, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self._array

    # Support equality comparison used in ``boxes.cls == class_id``
    def __eq__(self, other):
        return _FakeTensor(self._array == float(other))

    def __len__(self):
        return len(self._array)


def make_result(
    names: Dict[int, str],
    detections: List[Tuple[str, float, float, float, float, float]],
) -> SimpleNamespace:
    """Build a mock YOLOv8 ``Results`` object.

    Args:
        names: Class-ID-to-name mapping, e.g. ``{0: "ball", 1: "player", 2: "rim"}``.
        detections: List of ``(class_name, x1, y1, x2, y2, confidence)`` tuples.
            Pass an empty list to simulate a frame with no detections.

    Returns:
        A ``SimpleNamespace`` that mimics the ``result.names`` / ``result.boxes``
        interface used by the analysis modules.
    """
    name_to_id = {v.lower(): k for k, v in names.items()}

    cls_list, xyxy_list, conf_list = [], [], []
    for class_name, x1, y1, x2, y2, conf in detections:
        cls_list.append(float(name_to_id[class_name.lower()]))
        xyxy_list.append([x1, y1, x2, y2])
        conf_list.append(conf)

    boxes = SimpleNamespace(
        cls=_FakeTensor(np.array(cls_list)),
        xyxy=_FakeTensor(np.array(xyxy_list).reshape(-1, 4)),
        conf=_FakeTensor(np.array(conf_list)),
    )

    return SimpleNamespace(names=names, boxes=boxes)


# Convenience: standard class mapping matching the Roboflow dataset
STANDARD_NAMES = {0: "ball", 1: "player", 2: "rim"}
