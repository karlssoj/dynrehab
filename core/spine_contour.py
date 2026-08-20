"""Back-contour geometry for estimating spine curvature from a segmentation
mask, given only hip/shoulder keypoints (no dedicated spine landmarks).

See docs/superpowers/specs/2026-08-20-yolo-spine-curvature-design.md.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def facing_left(keypoints: dict) -> bool | None:
    """True if the person's face points toward the image-left half (nose
    x-coordinate is less than shoulder x-coordinate). None if insufficient
    visibility to tell."""
    nose = keypoints.get("nose")
    shoulder = keypoints.get("left_shoulder") or keypoints.get("right_shoulder")
    if not nose or not shoulder or min(nose[3], shoulder[3]) < 0.3:
        return None
    return nose[0] < shoulder[0]
