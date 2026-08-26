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
    if not nose or not shoulder or min(nose[3], shoulder[3]) < 0.2:
        return None
    return nose[0] < shoulder[0]


_ROI_MARGIN_FRAC = 0.3    # crop half-width, as a fraction of hip-shoulder chord length
_ROI_END_PAD_FRAC = 0.15  # extra length past hip and past shoulder, as a fraction of chord length
_MIN_CHORD_PX = 30        # skip if hip-shoulder pixel distance is smaller than this

_SPINE_ZONE_FRACTION = 1.0 / 3.0  # fraction of the profile assigned to each of the
# thoracic (shoulder-end) and lumbar (hip-end) zones. Informed starting point, not
# validated — thirds give each zone an equal share of the sampled points, but the
# true thoracic/lumbar apex locations are not guaranteed to sit at even geometric
# thirds of the hip-shoulder chord for all body proportions. Override via the
# SPINE_ZONE_FRACTION env var (read by PoseEngine, passed into spine_zone_curvature).


def crop_and_rotate_roi(frame: np.ndarray, hip_px: tuple[float, float],
                         shoulder_px: tuple[float, float]
                         ) -> tuple[np.ndarray, tuple[float, float], tuple[float, float], float, np.ndarray] | None:
    """Rotate/crop `frame` so the hip->shoulder segment becomes vertical and
    centered. Returns (rotated_image, hip_point, shoulder_point, chord_len, M)
    where hip_point/shoulder_point are in the rotated image's coordinate
    space and M is the affine matrix used to produce it (invert with
    cv2.invertAffineTransform to map ROI-space points back to the original
    frame, e.g. for drawing the back contour on the source video). Returns
    None if the chord is too short to be reliable."""
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
    return rotated, hip_point, shoulder_point, chord_len, M


def extract_back_contour(mask: np.ndarray, hip_point: tuple[float, float],
                          shoulder_point: tuple[float, float]
                          ) -> tuple[list[float], list[float]] | None:
    """Scan each row between shoulder and hip for the mask's edge distance on
    each side of the vertical centerline (x = hip_point[0] ==
    shoulder_point[0]). Returns (left_profile, right_profile), or None if
    fewer than 3 rows have the centerline inside the mask."""
    center_x = int(round(hip_point[0]))
    y_start = int(round(shoulder_point[1]))
    y_end = int(round(hip_point[1]))
    h, w = mask.shape[:2]

    left_profile: list[float] = []
    right_profile: list[float] = []
    for y in range(max(0, y_start), min(h, y_end + 1)):
        row = mask[y]
        if center_x < 0 or center_x >= w or row[center_x] == 0:
            return None

        x = center_x
        while x > 0 and row[x] > 0:
            x -= 1
        left_profile.append(float(center_x - x))

        x = center_x
        while x < w - 1 and row[x] > 0:
            x += 1
        right_profile.append(float(x - center_x))

    if len(left_profile) < 3:
        return None
    return left_profile, right_profile


def _bucket_indices(length: int, n: int) -> list[list[int]]:
    """Partition range(length) into n contiguous, roughly-equal, non-empty
    buckets covering every index exactly once (the last bucket absorbs any
    remainder). Returns n buckets when 0 < n <= length; returns `length`
    single-index buckets (i.e. n is effectively clamped to length) when
    n > length, since a bucket cannot be created from zero indices. Returns
    [] if length == 0 or n <= 0."""
    if length == 0 or n <= 0:
        return []
    if n >= length:
        return [[i] for i in range(length)]
    base_size = length // n
    remainder = length % n
    buckets = []
    start = 0
    for i in range(n):
        size = base_size + (1 if i < remainder else 0)
        buckets.append(list(range(start, start + size)))
        start += size
    return buckets


def signed_curvature_profile(back_profile: list[float], chord_len: float,
                              n: int) -> list[float] | None:
    """Per-point signed deviation of `back_profile` from the straight line
    between its own first and last element, normalized by chord_len,
    reduced to n points by partitioning the (typically much longer)
    raw profile into n contiguous buckets and taking the largest-magnitude
    deviation within each bucket (NOT a single sampled row — a real
    sustained bulge/cave spans many rows, but if a single evenly-spaced
    row were sampled instead, the deformation could fall entirely between
    two sampled rows and be missed completely; bucket-max guarantees every
    raw row is accounted for by exactly one of the n output points). Same
    sign convention as signed_curvature_ratio: positive = outward bulge,
    negative = inward cave. Index 0 corresponds to the shoulder end of
    back_profile, index n-1 to the hip end (profile ordering is unchanged
    from extract_back_contour). Returns None if back_profile is empty or
    chord_len is not positive."""
    n_raw = len(back_profile)
    if n_raw == 0:
        return None
    if chord_len <= 0:
        return None

    start, end = back_profile[0], back_profile[-1]

    def straight_at(i: int) -> float:
        if n_raw <= 1:
            return start
        t = i / (n_raw - 1)
        return start + t * (end - start)

    ratio_deviations = [(back_profile[i] - straight_at(i)) / chord_len for i in range(n_raw)]
    return [max((ratio_deviations[i] for i in bucket), key=abs)
            for bucket in _bucket_indices(n_raw, n)]


def signed_curvature_ratio(left_profile: list[float], right_profile: list[float],
                            chord_len: float, facing_left: bool) -> float | None:
    """Signed, chord-length-normalized peak deviation of the back-side
    silhouette from the straight line between its own two endpoints.
    Positive = outward bulge (excessive rounding / "bula"). Negative =
    inward cave (excessive arch / "svank"). None if the back-side profile
    is empty or chord_len is not positive."""
    back_profile = right_profile if facing_left else left_profile
    if not back_profile or chord_len <= 0:
        return None
    full_profile = signed_curvature_profile(back_profile, chord_len, len(back_profile))
    return max(full_profile, key=abs)


def spine_zone_curvature(profile: list[float],
                          zone_fraction: float = _SPINE_ZONE_FRACTION
                          ) -> tuple[float | None, float | None]:
    """Aggregate a (already time-smoothed) per-point signed curvature
    profile, ordered shoulder-end (index 0) to hip-end (index -1), into
    (thoracic, lumbar) zone values. Thoracic = largest-magnitude value
    among the first zone_fraction of points (nearest the shoulder); lumbar
    = largest-magnitude value among the last zone_fraction of points
    (nearest the hip). The middle portion is excluded from both zones.
    Each point already represents the worst (largest-magnitude) deviation
    within its own slice of the back (see signed_curvature_profile's
    bucket-max reduction) -- averaging those already-extremal values back
    together would dilute a real, spatially-concentrated deformation with
    its more-neutral neighboring points, so the zone value takes the max
    again rather than the mean. Returns (None, None) if profile is empty.
    If len(profile) < 3, both zones fall back to the whole profile's max."""
    n = len(profile)
    if n == 0:
        return None, None
    if n < 3:
        peak = max(profile, key=abs)
        return peak, peak
    zone_size = max(1, round(n * zone_fraction))
    thoracic = max(profile[:zone_size], key=abs)
    lumbar = max(profile[-zone_size:], key=abs)
    return thoracic, lumbar


def back_contour_points_in_frame(M: np.ndarray, hip_point: tuple[float, float],
                                  shoulder_point: tuple[float, float],
                                  back_profile: list[float],
                                  on_right_side: bool) -> list[tuple[float, float]]:
    """Reconstruct each back_profile row's pixel position in the rotated
    ROI's coordinate space, then map every point back into the original
    (pre-rotation) frame using the inverse of M — the same affine matrix
    crop_and_rotate_roi built to produce that ROI. on_right_side: True if
    back_profile is right_profile (image-right of the ROI's vertical
    centerline), False if it's left_profile (image-left). Returns an empty
    list if back_profile is empty."""
    if not back_profile:
        return []

    sign = 1.0 if on_right_side else -1.0
    roi_points = np.array([
        [[hip_point[0] + sign * dist, shoulder_point[1] + i]]
        for i, dist in enumerate(back_profile)
    ], dtype=np.float32)

    M_inv = cv2.invertAffineTransform(M)
    original_points = cv2.transform(roi_points, M_inv).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in original_points]


def _evenly_spaced_indices(length: int, n: int) -> list[int]:
    """Indices for picking n evenly-spaced items from a sequence of `length`
    items, always including index 0 and index length-1. Returns
    list(range(length)) unchanged if length <= n. Returns [] if length == 0
    or n <= 0."""
    if length == 0 or n <= 0:
        return []
    if length <= n:
        return list(range(length))
    if n == 1:
        return [length // 2]
    return [round(i * (length - 1) / (n - 1)) for i in range(n)]


def downsample_points(points: list[tuple[float, float]], n: int) -> list[tuple[float, float]]:
    """Pick n evenly-spaced points from `points`, always including both
    endpoints. Returns `points` unchanged if it already has n or fewer
    points. Returns an empty list if `points` is empty or n <= 0."""
    if not points or n <= 0:
        return []
    return [points[i] for i in _evenly_spaced_indices(len(points), n)]
