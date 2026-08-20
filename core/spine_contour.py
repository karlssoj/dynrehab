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


_ROI_MARGIN_FRAC = 0.3    # crop half-width, as a fraction of hip-shoulder chord length
_ROI_END_PAD_FRAC = 0.15  # extra length past hip and past shoulder, as a fraction of chord length
_MIN_CHORD_PX = 40        # skip if hip-shoulder pixel distance is smaller than this


def crop_and_rotate_roi(frame: np.ndarray, hip_px: tuple[float, float],
                         shoulder_px: tuple[float, float]
                         ) -> tuple[np.ndarray, tuple[float, float], tuple[float, float], float] | None:
    """Rotate/crop `frame` so the hip->shoulder segment becomes vertical and
    centered. Returns (rotated_image, hip_point, shoulder_point, chord_len) in
    the rotated image's coordinate space, or None if the chord is too short
    to be reliable."""
    hip_x, hip_y = hip_px
    shoulder_x, shoulder_y = shoulder_px
    dx = shoulder_x - hip_x
    dy = shoulder_y - hip_y
    chord_len = math.hypot(dx, dy)
    if chord_len < _MIN_CHORD_PX:
        return None

    angle_deg = math.degrees(math.atan2(dx, -dy))
    mid_x = (hip_x + shoulder_x) / 2.0
    mid_y = (hip_y + shoulder_y) / 2.0

    out_w = round(chord_len * (1 + 2 * _ROI_MARGIN_FRAC))
    out_h = round(chord_len * (1 + 2 * _ROI_END_PAD_FRAC))

    M = cv2.getRotationMatrix2D((mid_x, mid_y), angle_deg, 1.0)
    M[0, 2] += out_w / 2.0 - mid_x
    M[1, 2] += out_h / 2.0 - mid_y

    rotated = cv2.warpAffine(frame, M, (out_w, out_h))

    hip_point = (out_w / 2.0, out_h / 2.0 + chord_len / 2.0)
    shoulder_point = (out_w / 2.0, out_h / 2.0 - chord_len / 2.0)
    return rotated, hip_point, shoulder_point, chord_len
