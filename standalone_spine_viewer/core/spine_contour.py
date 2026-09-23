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

_SPINE_ZONE_FRACTION = 1.0 / 3.0  # fraction of the profile used for the ANGLE
# MEASUREMENT'S vertex search (spine_zone_bend_angles / spine_zone_curvature) — do
# NOT raise this to make the on-screen zone-coloring look wider (see
# _SPINE_ZONE_DISPLAY_FRACTION in pose_engine.py for that instead). This value is
# load-bearing for correctness, not just cosmetic: any non-linear taper between the
# shoulder and hip widths (present in essentially every real body — narrower waist
# than shoulders/hips) has its OWN deviation from the shoulder-hip chord grow
# steadily larger the closer a row is to the profile's midpoint, entirely from
# ordinary body shape, with no spine curvature involved. Confirmed by direct
# simulation: growing this fraction from 1/3 to just 0.35 pulls in one more bucket
# from that region and is enough to flip a genuine, sustained 20px thoracic bulge
# from a correct +24° reading to a wrong -14° — a sharp discontinuity, not a
# gradual drift, that gets WORSE (stays wrong) all the way through 0.45. Smaller
# values (down to ~0.2-0.25) stay clear of that contaminated region and read
# correctly. Override via the SPINE_ZONE_FRACTION env var (read by PoseEngine,
# passed into spine_zone_curvature/spine_zone_bend_angles) — but see the warning
# above before raising it.

_SPINE_BEND_BOUNDARY_FRACTION = 0.5  # fraction of the way from shoulder-end (0.0) to
# hip-end (1.0) used as the shared "hinge" reference point between the thoracic and
# lumbar Cobb-style angle triangles (see spine_zone_bend_angles). Deliberately NOT
# tied to _SPINE_ZONE_FRACTION — this is a single fixed anatomical landmark stand-in
# (a rough analogue of the clinical T12 thoracolumbar junction), not a zone boundary.
# Informed starting point, not validated. Override via SPINE_BEND_BOUNDARY_FRACTION
# env var (read by PoseEngine, passed into spine_zone_bend_angles).


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


def _pixel_deviations(back_profile: list[float]) -> list[float] | None:
    """Per-row signed deviation of `back_profile` from the straight line
    between its own first and last element, in the SAME pixel units as
    back_profile itself (not normalized by chord_len). Exactly 0 at index 0
    and index len(back_profile)-1 by construction -- this is what makes it
    safe to use as a self-referencing (x, row_index) coordinate for a
    Cobb-angle-style triangle (see spine_zone_bend_angles): unlike raw
    back_profile values (which encode ordinary body width -- e.g. narrower
    waist than shoulders/hips -- and can shift tick-to-tick as stance
    changes, such as during a squat, for reasons that have nothing to do
    with spine shape), the deviation is always anchored to exactly 0 at
    both the shoulder and hip ends on every tick, so only genuine curvature
    beyond a straight taper between this tick's own shoulder/hip readings
    shows up. Returns None if back_profile is empty."""
    n_raw = len(back_profile)
    if n_raw == 0:
        return None

    start, end = back_profile[0], back_profile[-1]

    def straight_at(i: int) -> float:
        if n_raw <= 1:
            return start
        t = i / (n_raw - 1)
        return start + t * (end - start)

    return [back_profile[i] - straight_at(i) for i in range(n_raw)]


def _deviation_ratios(back_profile: list[float], chord_len: float) -> list[float] | None:
    """Same as _pixel_deviations, normalized by chord_len. Same sign
    convention as signed_curvature_ratio: positive = outward bulge,
    negative = inward cave. Returns None if back_profile is empty or
    chord_len is not positive. Shared by signed_curvature_profile and
    signed_curvature_profile_indices so the deviation math lives in one
    place."""
    pixel_devs = _pixel_deviations(back_profile)
    if pixel_devs is None:
        return None
    if chord_len <= 0:
        return None
    return [d / chord_len for d in pixel_devs]


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
    ratio_deviations = _deviation_ratios(back_profile, chord_len)
    if ratio_deviations is None:
        return None
    return [max((ratio_deviations[i] for i in bucket), key=abs)
            for bucket in _bucket_indices(len(back_profile), n)]


def signed_curvature_profile_indices(back_profile: list[float], chord_len: float,
                                      n: int) -> list[tuple[float, int]] | None:
    """Same bucket-max reduction as signed_curvature_profile, but each of
    the n output elements is (ratio_value, raw_index) instead of just the
    value — raw_index is the back_profile row that produced that bucket's
    largest-magnitude deviation (ties broken the same way Python's
    max(..., key=abs) breaks them: first occurrence within the bucket).
    Used by spine_zone_bend_angles to locate the vertex of each zone's
    Cobb-style bend angle. Returns None under the same conditions as
    signed_curvature_profile."""
    ratio_deviations = _deviation_ratios(back_profile, chord_len)
    if ratio_deviations is None:
        return None
    return [max(((ratio_deviations[i], i) for i in bucket), key=lambda pair: abs(pair[0]))
            for bucket in _bucket_indices(len(back_profile), n)]


def _angle_2d(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    """Angle at point b (degrees) between vectors b->a and b->c. 180 =
    collinear/straight, decreasing as the angle at b sharpens. Same
    degenerate-case guard as the _angle_2d pattern documented for
    LLM-generated exercise code in llm_service.py: returns 180.0 (i.e. "no
    bend detected") rather than raising when either vector is
    near-zero-length, since that means the vertex coincides with one of
    its own reference points -- insufficient geometry to measure a bend,
    not a real straight line."""
    vax, vay = a[0] - b[0], a[1] - b[1]
    vcx, vcy = c[0] - b[0], c[1] - b[1]
    mag = math.hypot(vax, vay) * math.hypot(vcx, vcy)
    if mag < 1e-10:
        return 180.0
    cosine = (vax * vcx + vay * vcy) / mag
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _weighted_zone_vertex(pairs: list[tuple[float, int]]) -> tuple[float, float]:
    """Magnitude-weighted centroid (signed_value, row_index) across a
    zone's bucket-representative (value, index) pairs, instead of a hard
    argmax. A hard argmax can flip between two nearby buckets whose peak
    magnitudes differ by a fraction of a pixel (routine under segmentation
    noise), producing a large, discontinuous jump in the resulting angle
    -- including a sign flip when the two candidates point in opposite
    directions -- even though the underlying deviation profile barely
    changed tick to tick. The weighted centroid instead blends smoothly as
    the true peak shifts: a genuinely large, isolated deviation still
    dominates the weighted sum (near-zero neighbors contribute almost
    nothing to it, so a real concentrated deformation is not diluted --
    same non-dilution guarantee the bucket-max reduction already gives),
    but two similar-magnitude candidates average into a stable blend
    instead of an arbitrary, noise-driven binary choice. row_index is a
    float (may be fractional) since it's itself a weighted average."""
    weights = [abs(v) for v, _ in pairs]
    total = sum(weights)
    if total < 1e-9:
        return 0.0, float(pairs[len(pairs) // 2][1])
    val = sum(v * w for (v, _), w in zip(pairs, weights)) / total
    idx = sum(i * w for (_, i), w in zip(pairs, weights)) / total
    return val, idx


def _spine_bend_geometry(back_profile: list[float], profile_indices: list[tuple[float, int]],
                          zone_fraction: float, boundary_fraction: float
                          ) -> tuple[float, float, float, float, int] | None:
    """Shared computation behind spine_zone_bend_angles and
    spine_bend_anchor_indices: (thoracic_val, thoracic_idx, lumbar_val,
    lumbar_idx, boundary_idx), where thoracic_idx/lumbar_idx are the
    (possibly fractional) weighted-centroid row positions from
    _weighted_zone_vertex. None if profile_indices is empty or
    back_profile has fewer than 3 rows.

    boundary_idx is boundary_fraction of the way through THIS tick's own
    raw profile length (n_raw), recomputed fresh every tick. A previous
    version tried anchoring it to a time-smoothed reference length
    instead, to fight perceived hinge-point drift between postures -- that
    made things worse: n_raw tracks the person's apparent on-screen scale
    (camera distance/zoom), which can change quickly (e.g. through a squat
    at ~207->119 raw rows), and a smoothed reference lags that change, so
    boundary_fraction ends up applied against a stale scale instead of the
    current one (measured case: n_raw=222 but the smoothed reference was
    still 161.8, landing the hinge at 36% of the profile instead of 50%).
    Recomputing from the current n_raw every tick is scale-invariant by
    construction and has no such lag."""
    n_raw = len(back_profile)
    n = len(profile_indices)
    if n_raw < 3 or n == 0:
        return None

    if n < 3:
        thoracic_pairs = lumbar_pairs = profile_indices
    else:
        zone_size = max(1, round(n * zone_fraction))
        thoracic_pairs = profile_indices[:zone_size]
        lumbar_pairs = profile_indices[-zone_size:]

    thoracic_val, thoracic_idx = _weighted_zone_vertex(thoracic_pairs)
    lumbar_val, lumbar_idx = _weighted_zone_vertex(lumbar_pairs)
    boundary_idx = min(n_raw - 1, max(0, round(boundary_fraction * (n_raw - 1))))
    return thoracic_val, thoracic_idx, lumbar_val, lumbar_idx, boundary_idx


def spine_zone_bend_angles(back_profile: list[float], chord_len: float,
                            profile_indices: list[tuple[float, int]],
                            zone_fraction: float = _SPINE_ZONE_FRACTION,
                            boundary_fraction: float = _SPINE_BEND_BOUNDARY_FRACTION
                            ) -> tuple[float | None, float | None]:
    """Cobb-angle-style signed bend, in degrees, for the thoracic
    (shoulder-end) and lumbar (hip-end) zones. Three anchor points define
    each zone's angle triangle: the profile's own shoulder-end point
    (back_profile[0]), a single shared "hinge" point at boundary_fraction
    of the way from shoulder to hip (a fixed stand-in for the anatomical
    thoracolumbar junction, independent of zone_fraction), and the
    profile's own hip-end point (back_profile[-1]). The thoracic angle is
    measured at the thoracic zone's vertex (a magnitude-weighted centroid
    of the zone's candidate peaks, see _weighted_zone_vertex -- not a hard
    argmax, for tick-to-tick stability), between rays to the shoulder-end
    point and the hinge point; the lumbar angle is measured at the lumbar
    zone's vertex, between rays to the hinge point and the hip-end point.

    Same 0=straight/higher=more-bent magnitude convention as every other
    _bend_2d field (bend = 180 - raw angle), but SIGNED (unlike every
    other _bend_2d field) using math.copysign against the same weighted
    deviation value that located the vertex: positive = outward bulge
    ("bula"), negative = inward cave ("svank") -- a pure 3-point angle
    magnitude cannot distinguish the two, so the sign is recovered from
    the already-computed deviation rather than re-derived.

    IMPORTANT: the (x, y) anchor/vertex points used for the angle triangles
    are built from back_profile's deviation-from-its-own-chord (see
    _pixel_deviations), NOT from the raw back_profile values themselves.
    Raw widths encode ordinary body shape (narrower waist than
    shoulders/hips is universal human anatomy, not spine curvature) and
    shift tick-to-tick with stance (e.g. hip/thigh prominence changes a lot
    between standing and a deep squat) for reasons unrelated to the spine
    -- using them directly would report a "bend" from body shape and
    stance alone, indistinguishable from real curvature, and would not
    reliably cancel out even when compared against a baseline captured at
    a different stance. Deviation-from-chord is exactly 0 at the shoulder
    and hip ends on every single tick by construction, so it only reflects
    genuine curvature beyond a straight taper between that tick's own
    shoulder/hip readings -- the same self-referencing principle that
    signed_curvature_ratio already relies on.

    Returns (None, None) if profile_indices is empty, back_profile has
    fewer than 3 rows, or chord_len is not positive. If
    len(profile_indices) < 3, both zones fall back to the whole profile's
    weighted centroid (mirrors spine_zone_curvature's whole-profile
    fallback)."""
    if chord_len <= 0:
        return None, None
    geometry = _spine_bend_geometry(back_profile, profile_indices, zone_fraction, boundary_fraction)
    if geometry is None:
        return None, None
    thoracic_val, thoracic_idx, lumbar_val, lumbar_idx, boundary_idx = geometry
    n_raw = len(back_profile)

    pixel_devs = _pixel_deviations(back_profile)

    # IMPORTANT: the vertex's x-coordinate is the already-computed weighted
    # deviation VALUE (converted from a chord_len-normalized ratio back to
    # pixel units), NOT a re-sampled/interpolated pixel_devs lookup at the
    # weighted row index. When a zone's two candidate peaks sit far apart
    # (e.g. one near its shoulder-side edge, one near its far edge), their
    # weighted-average ROW INDEX can land in the flat valley between them
    # -- interpolating the profile AT that in-between row would read
    # whatever (often near-zero) value happens to sit there, silently
    # discarding the very deviation magnitude the weighting was meant to
    # preserve. The weighted VALUE has no such problem: it is a direct
    # blend of the two real deviation magnitudes, never re-derived from an
    # arbitrary intermediate row.
    shoulder_pt = (0.0, 0.0)
    hip_pt = (0.0, float(n_raw - 1))
    boundary_pt = (pixel_devs[boundary_idx], float(boundary_idx))
    thoracic_vertex = (thoracic_val * chord_len, thoracic_idx)
    lumbar_vertex = (lumbar_val * chord_len, lumbar_idx)

    thoracic_bend = math.copysign(180.0 - _angle_2d(shoulder_pt, thoracic_vertex, boundary_pt), thoracic_val)
    lumbar_bend = math.copysign(180.0 - _angle_2d(boundary_pt, lumbar_vertex, hip_pt), lumbar_val)
    return thoracic_bend, lumbar_bend


def spine_bend_anchor_indices(back_profile: list[float],
                               profile_indices: list[tuple[float, int]],
                               zone_fraction: float = _SPINE_ZONE_FRACTION,
                               boundary_fraction: float = _SPINE_BEND_BOUNDARY_FRACTION
                               ) -> tuple[int, int, int, int, int] | None:
    """The five raw back_profile row indices spine_zone_bend_angles's
    measurement triangles are anchored at, rounded to the nearest whole
    row for drawing/inspection purposes: (shoulder_idx, thoracic_idx,
    boundary_idx, lumbar_idx, hip_idx). shoulder_idx is always 0 and
    hip_idx is always len(back_profile)-1; thoracic_idx/lumbar_idx are the
    per-zone weighted-centroid vertices (see _weighted_zone_vertex),
    rounded to an int row here since spine_zone_bend_angles itself uses
    the exact fractional position (interpolated) for the actual angle
    math -- this rounded version exists for callers that need a concrete
    row to draw or look up (e.g. an on-screen overlay), where sub-pixel
    precision doesn't matter. Returns None under the same conditions as
    spine_zone_bend_angles."""
    geometry = _spine_bend_geometry(back_profile, profile_indices, zone_fraction, boundary_fraction)
    if geometry is None:
        return None
    _, thoracic_idx, _, lumbar_idx, boundary_idx = geometry
    n_raw = len(back_profile)
    return 0, round(thoracic_idx), boundary_idx, round(lumbar_idx), n_raw - 1


def spine_zone_for_index(index: int, n_raw: int,
                          zone_fraction: float = _SPINE_ZONE_FRACTION) -> str:
    """Classifies a raw back_profile row index into "thoracic" (first
    zone_fraction of rows, nearest the shoulder), "lumbar" (last
    zone_fraction, nearest the hip), or "middle" (the excluded portion in
    between) -- for labeling/coloring on-screen points by zone. Uses the
    same zone_fraction split as spine_zone_curvature/spine_zone_bend_angles,
    applied directly to raw row position rather than to a bucketed profile,
    so it works for any index (e.g. the points chosen for drawing, which
    are sampled differently than the buckets used for the angle math)."""
    zone_size = max(1, round(n_raw * zone_fraction))
    if index < zone_size:
        return "thoracic"
    if index >= n_raw - zone_size:
        return "lumbar"
    return "middle"


def sparse_points_in_frame(M: np.ndarray, hip_point: tuple[float, float],
                            shoulder_point: tuple[float, float],
                            back_profile: list[float], indices: list[int],
                            on_right_side: bool) -> list[tuple[float, float]]:
    """Like back_contour_points_in_frame, but for an arbitrary (possibly
    non-contiguous) list of raw back_profile row indices instead of the
    whole profile in order -- used to map specific measurement points (e.g.
    spine_bend_anchor_indices' five anchor/vertex rows) into original-frame
    pixel coordinates for drawing. Returns an empty list if indices is empty."""
    if not indices:
        return []
    sign = 1.0 if on_right_side else -1.0
    roi_points = np.array([
        [[hip_point[0] + sign * back_profile[i], shoulder_point[1] + i]]
        for i in indices
    ], dtype=np.float32)
    M_inv = cv2.invertAffineTransform(M)
    original_points = cv2.transform(roi_points, M_inv).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in original_points]


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
