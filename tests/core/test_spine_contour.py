import pytest
import numpy as np
import cv2

import math

from core.spine_contour import (facing_left, crop_and_rotate_roi, extract_back_contour,
                                  signed_curvature_ratio, signed_curvature_profile,
                                  signed_curvature_profile_indices, spine_zone_bend_angles,
                                  spine_bend_anchor_indices, spine_zone_for_index,
                                  sparse_points_in_frame,
                                  spine_zone_curvature, back_contour_points_in_frame,
                                  downsample_points, _evenly_spaced_indices, _bucket_indices,
                                  _angle_2d,
                                  _MIN_CHORD_PX, _ROI_MARGIN_FRAC, _ROI_END_PAD_FRAC,
                                  _SPINE_ZONE_FRACTION)


def test_facing_left_true_when_nose_left_of_shoulder():
    keypoints = {
        "nose": (0.3, 0.2, 0.0, 0.9),
        "left_shoulder": (0.5, 0.3, 0.0, 0.9),
    }
    assert facing_left(keypoints) is True


def test_facing_left_false_when_nose_right_of_shoulder():
    keypoints = {
        "nose": (0.7, 0.2, 0.0, 0.9),
        "right_shoulder": (0.5, 0.3, 0.0, 0.9),
    }
    assert facing_left(keypoints) is False


def test_facing_left_none_when_nose_missing():
    keypoints = {"left_shoulder": (0.5, 0.3, 0.0, 0.9)}
    assert facing_left(keypoints) is None


def test_facing_left_none_when_low_visibility():
    keypoints = {
        "nose": (0.3, 0.2, 0.0, 0.1),
        "left_shoulder": (0.5, 0.3, 0.0, 0.9),
    }
    assert facing_left(keypoints) is None


def test_crop_and_rotate_roi_returns_none_for_short_chord():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    hip_px = (100.0, 100.0)
    shoulder_px = (100.0, 100.0 + _MIN_CHORD_PX / 2)
    assert crop_and_rotate_roi(frame, hip_px, shoulder_px) is None


def test_crop_and_rotate_roi_output_canvas_size():
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    hip_px = (200.0, 300.0)
    shoulder_px = (200.0, 150.0)  # chord_len = 150, already vertical
    result = crop_and_rotate_roi(frame, hip_px, shoulder_px)
    assert result is not None
    rotated, hip_point, shoulder_point, chord_len, M = result
    assert chord_len == pytest.approx(150.0, abs=0.5)
    expected_w = round(chord_len * (1 + 2 * _ROI_MARGIN_FRAC))
    expected_h = round(chord_len * (1 + 2 * _ROI_END_PAD_FRAC))
    assert rotated.shape[1] == pytest.approx(expected_w, abs=1)
    assert rotated.shape[0] == pytest.approx(expected_h, abs=1)


def test_crop_and_rotate_roi_hip_and_shoulder_points_are_vertically_aligned():
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    hip_px = (150.0, 300.0)
    shoulder_px = (250.0, 150.0)  # tilted chord
    result = crop_and_rotate_roi(frame, hip_px, shoulder_px)
    assert result is not None
    rotated, hip_point, shoulder_point, chord_len, M = result
    assert hip_point[0] == pytest.approx(shoulder_point[0], abs=0.5)
    assert hip_point[1] > shoulder_point[1]
    assert abs(hip_point[1] - shoulder_point[1]) == pytest.approx(chord_len, abs=0.5)


def test_crop_and_rotate_roi_marker_lands_at_hip_point():
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    hip_px = (150.0, 300.0)
    shoulder_px = (150.0, 150.0)
    cv2.circle(frame, (int(hip_px[0]), int(hip_px[1])), 3, (255, 255, 255), -1)
    result = crop_and_rotate_roi(frame, hip_px, shoulder_px)
    assert result is not None
    rotated, hip_point, shoulder_point, chord_len, M = result
    region = rotated[int(hip_point[1]) - 4:int(hip_point[1]) + 4,
                      int(hip_point[0]) - 4:int(hip_point[0]) + 4]
    assert region.max() > 200


def test_crop_and_rotate_roi_both_markers_land_at_expected_points_for_tilted_chord():
    # A genuinely tilted chord (dx != 0 and dy != 0), with markers stamped at
    # both the hip and shoulder source points. This exercises the actual
    # rotation direction against real pixel content -- a flipped rotation
    # sign would miss both markers even though the arithmetic hip_point /
    # shoulder_point values are unchanged.
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    hip_px = (150.0, 300.0)
    shoulder_px = (250.0, 150.0)  # tilted chord
    cv2.circle(frame, (int(hip_px[0]), int(hip_px[1])), 3, (255, 255, 255), -1)
    cv2.circle(frame, (int(shoulder_px[0]), int(shoulder_px[1])), 3, (255, 255, 255), -1)
    result = crop_and_rotate_roi(frame, hip_px, shoulder_px)
    assert result is not None
    rotated, hip_point, shoulder_point, chord_len, M = result

    hip_region = rotated[int(hip_point[1]) - 4:int(hip_point[1]) + 4,
                          int(hip_point[0]) - 4:int(hip_point[0]) + 4]
    shoulder_region = rotated[int(shoulder_point[1]) - 4:int(shoulder_point[1]) + 4,
                               int(shoulder_point[0]) - 4:int(shoulder_point[0]) + 4]
    assert hip_region.max() > 200
    assert shoulder_region.max() > 200


def test_crop_and_rotate_roi_at_min_chord_boundary_is_not_none():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    hip_px = (100.0, 100.0)
    shoulder_px = (100.0, 100.0 + _MIN_CHORD_PX)  # exactly at the boundary
    result = crop_and_rotate_roi(frame, hip_px, shoulder_px)
    assert result is not None
    _, _, _, chord_len, M = result
    assert chord_len == pytest.approx(_MIN_CHORD_PX, abs=0.01)


def test_crop_and_rotate_roi_matrix_round_trips_hip_and_shoulder_points():
    # M must be invertible and map hip_point/shoulder_point (rotated-ROI
    # space) back to hip_px/shoulder_px (original frame space) -- this is
    # exactly what the back-contour visualisation needs to draw the contour
    # on the original video frame instead of the rotated ROI.
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    hip_px = (150.0, 300.0)
    shoulder_px = (250.0, 150.0)
    result = crop_and_rotate_roi(frame, hip_px, shoulder_px)
    assert result is not None
    rotated, hip_point, shoulder_point, chord_len, M = result

    M_inv = cv2.invertAffineTransform(M)
    pts = np.array([[hip_point], [shoulder_point]], dtype=np.float32)
    back = cv2.transform(pts, M_inv).reshape(-1, 2)

    assert back[0][0] == pytest.approx(hip_px[0], abs=0.5)
    assert back[0][1] == pytest.approx(hip_px[1], abs=0.5)
    assert back[1][0] == pytest.approx(shoulder_px[0], abs=0.5)
    assert back[1][1] == pytest.approx(shoulder_px[1], abs=0.5)


def _make_straight_mask(width=100, height=100, half_width=20, center_x=50):
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:, center_x - half_width:center_x + half_width] = 255
    return mask


def test_extract_back_contour_straight_mask_gives_constant_profiles():
    mask = _make_straight_mask()
    hip_point = (50.0, 90.0)
    shoulder_point = (50.0, 10.0)
    result = extract_back_contour(mask, hip_point, shoulder_point)
    assert result is not None
    left_profile, right_profile = result
    assert len(left_profile) == len(right_profile) == 81
    assert all(v == pytest.approx(20, abs=1) for v in left_profile)
    assert all(v == pytest.approx(20, abs=1) for v in right_profile)


def test_extract_back_contour_bulge_on_right_side():
    mask = _make_straight_mask()
    mask[40:60, 70:85] = 255  # bulge outward on the right side, middle rows
    hip_point = (50.0, 90.0)
    shoulder_point = (50.0, 10.0)
    left_profile, right_profile = extract_back_contour(mask, hip_point, shoulder_point)
    assert max(right_profile[30:50]) > 20  # rows 40-59 -> indices 30-49 (offset by shoulder_point y=10)
    assert all(v == pytest.approx(20, abs=1) for v in left_profile)


def test_extract_back_contour_returns_none_for_empty_mask():
    mask = np.zeros((100, 100), dtype=np.uint8)
    result = extract_back_contour(mask, (50.0, 90.0), (50.0, 10.0))
    assert result is None


def test_extract_back_contour_returns_none_when_centerline_gap_within_range():
    mask = _make_straight_mask()
    mask[50, 45:55] = 0  # background gap covering the center column at row 50
    hip_point = (50.0, 90.0)
    shoulder_point = (50.0, 10.0)
    result = extract_back_contour(mask, hip_point, shoulder_point)
    assert result is None


def test_signed_curvature_ratio_straight_profile_near_zero():
    profile = [20.0] * 10
    ratio = signed_curvature_ratio(profile, profile, chord_len=100.0, facing_left=True)
    assert ratio == pytest.approx(0.0, abs=0.01)


def test_signed_curvature_ratio_positive_for_outward_bulge():
    straight = [20.0] * 10
    bulged = [20.0, 20.0, 20.0, 25.0, 30.0, 25.0, 20.0, 20.0, 20.0, 20.0]
    # facing_left=True -> back_profile = right_profile = bulged
    ratio = signed_curvature_ratio(straight, bulged, chord_len=100.0, facing_left=True)
    assert ratio > 0


def test_signed_curvature_ratio_negative_for_inward_cave():
    straight = [20.0] * 10
    caved = [20.0, 20.0, 20.0, 15.0, 10.0, 15.0, 20.0, 20.0, 20.0, 20.0]
    ratio = signed_curvature_ratio(straight, caved, chord_len=100.0, facing_left=True)
    assert ratio < 0


def test_signed_curvature_ratio_sign_flips_with_facing_direction():
    straight = [20.0] * 10
    bulged = [20.0, 20.0, 20.0, 25.0, 30.0, 25.0, 20.0, 20.0, 20.0, 20.0]
    ratio_facing_left = signed_curvature_ratio(straight, bulged, chord_len=100.0, facing_left=True)
    ratio_facing_right = signed_curvature_ratio(straight, bulged, chord_len=100.0, facing_left=False)
    assert ratio_facing_left > 0
    assert ratio_facing_right == pytest.approx(0.0, abs=0.01)


def test_signed_curvature_ratio_none_for_empty_profile():
    assert signed_curvature_ratio([], [], chord_len=100.0, facing_left=True) is None


def test_signed_curvature_ratio_none_for_nonpositive_chord_len():
    profile = [20.0] * 10
    assert signed_curvature_ratio(profile, profile, chord_len=0.0, facing_left=True) is None


def test_facing_left_and_profile_split_agree_end_to_end():
    """End-to-end synthetic check that facing_left() (computed in the
    ORIGINAL, unrotated frame) and the left/right profile split from
    extract_back_contour() (computed in the ROTATED ROI frame) agree on
    which side is the person's back.

    Setup: a genuinely tilted hip->shoulder chord (dx=+40, dy=-180, not
    vertical, not horizontal) and a nose placed clearly to the image-left
    of the shoulder in the ORIGINAL frame (nose_x=190 < shoulder_x=220),
    so facing_left(keypoints) is True.

    Reasoning (verified by direct computation, not just assumed):
    decompose the nose's position relative to the hip->shoulder chord
    into a component along the perpendicular of that chord. Projecting
    the same perpendicular through crop_and_rotate_roi's rotation shows
    the nose sits on the side of the chord that maps to the LEFT half
    (x < center_x) of the rotated ROI. Since a 2D rotation preserves
    chirality, "left of chord in the original frame" == "left half of the
    rotated ROI" for this chord/rotation. So: nose (face) is on the LEFT
    side of the rotated ROI -> the back must be on the RIGHT side
    (x > center_x) of the rotated ROI -> right_profile is the back-side
    profile. This matches signed_curvature_ratio's documented convention:
    back_profile = right_profile if facing_left else left_profile, and
    facing_left is True here.

    Therefore: if we place a deliberate outward bulge on the RIGHT side of
    the rotated mask (x > center_x), it must be picked up by
    right_profile, be selected as the back profile (facing_left is True),
    and produce a POSITIVE signed_curvature_ratio (outward bulge =
    positive, per the documented sign convention). If the two coordinate
    frames were mismatched (a real, previously-undetected bug), the bulge
    would instead end up in left_profile -- which is NOT selected as the
    back when facing_left is True -- and the ratio would come out ~0
    instead of clearly positive.
    """
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    hip_px = (180.0, 300.0)
    shoulder_px = (220.0, 120.0)
    # Nose clearly to the image-left of the shoulder in the ORIGINAL frame.
    nose_px = (shoulder_px[0] - 30, shoulder_px[1] - 20)

    keypoints = {
        "nose": (nose_px[0] / 400.0, nose_px[1] / 400.0, 0.0, 0.9),
        "left_shoulder": (shoulder_px[0] / 400.0, shoulder_px[1] / 400.0, 0.0, 0.9),
        "left_hip": (hip_px[0] / 400.0, hip_px[1] / 400.0, 0.0, 0.9),
    }

    result = crop_and_rotate_roi(frame, hip_px, shoulder_px)
    assert result is not None
    rotated, hip_point, shoulder_point, chord_len, M = result

    out_h, out_w = rotated.shape[:2]
    center_x = int(round(hip_point[0]))
    y_start = int(round(shoulder_point[1]))
    y_end = int(round(hip_point[1]))
    half_width = 20

    mask = np.zeros((out_h, out_w), dtype=np.uint8)
    mask[:, max(0, center_x - half_width):center_x + half_width] = 255
    # Deliberate outward bulge on the RIGHT side (x > center_x) of the
    # rotated centerline, in the middle of the hip-shoulder span.
    mid_y = (y_start + y_end) // 2
    mask[mid_y - 10:mid_y + 10, center_x + half_width:center_x + half_width + 15] = 255

    contour = extract_back_contour(mask, hip_point, shoulder_point)
    assert contour is not None
    left_profile, right_profile = contour

    fl = facing_left(keypoints)
    assert fl is True

    ratio = signed_curvature_ratio(left_profile, right_profile, chord_len, fl)
    assert ratio is not None
    # Per the reasoning above: bulge is on the right of the rotated
    # centerline -> right_profile carries it -> facing_left True selects
    # right_profile as back_profile -> ratio must be positive.
    assert ratio > 0


def test_back_contour_points_in_frame_maps_right_side_points_back_to_original_frame():
    # Vertical, untilted chord -> crop_and_rotate_roi's M is a pure
    # translation (no rotation), so the round trip has simple, exact
    # expected values: for on_right_side=True, orig_x = hip_px[0] + dist_i,
    # orig_y = shoulder_px[1] + i (row i is `i` pixels below the shoulder).
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    hip_px = (200.0, 300.0)
    shoulder_px = (200.0, 150.0)
    result = crop_and_rotate_roi(frame, hip_px, shoulder_px)
    assert result is not None
    _, hip_point, shoulder_point, chord_len, M = result

    back_profile = [10.0, 12.0, 14.0]
    points = back_contour_points_in_frame(M, hip_point, shoulder_point, back_profile,
                                           on_right_side=True)

    assert len(points) == 3
    assert points[0] == (pytest.approx(210.0, abs=0.5), pytest.approx(150.0, abs=0.5))
    assert points[1] == (pytest.approx(212.0, abs=0.5), pytest.approx(151.0, abs=0.5))
    assert points[2] == (pytest.approx(214.0, abs=0.5), pytest.approx(152.0, abs=0.5))


def test_back_contour_points_in_frame_maps_left_side_points_back_to_original_frame():
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    hip_px = (200.0, 300.0)
    shoulder_px = (200.0, 150.0)
    result = crop_and_rotate_roi(frame, hip_px, shoulder_px)
    assert result is not None
    _, hip_point, shoulder_point, chord_len, M = result

    back_profile = [10.0, 12.0, 14.0]
    points = back_contour_points_in_frame(M, hip_point, shoulder_point, back_profile,
                                           on_right_side=False)

    assert len(points) == 3
    assert points[0] == (pytest.approx(190.0, abs=0.5), pytest.approx(150.0, abs=0.5))
    assert points[1] == (pytest.approx(188.0, abs=0.5), pytest.approx(151.0, abs=0.5))
    assert points[2] == (pytest.approx(186.0, abs=0.5), pytest.approx(152.0, abs=0.5))


def test_back_contour_points_in_frame_empty_profile_gives_empty_points():
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    hip_px = (200.0, 300.0)
    shoulder_px = (200.0, 150.0)
    result = crop_and_rotate_roi(frame, hip_px, shoulder_px)
    assert result is not None
    _, hip_point, shoulder_point, chord_len, M = result

    assert back_contour_points_in_frame(M, hip_point, shoulder_point, [], on_right_side=True) == []


def test_downsample_points_picks_evenly_spaced_indices():
    points = [(float(i), float(i)) for i in range(21)]  # 21 points, indices 0..20
    result = downsample_points(points, 10)
    assert len(result) == 10
    assert result[0] == points[0]
    assert result[-1] == points[-1]
    # Evenly spaced: consecutive gaps should all be close to 20/9 ~= 2.22
    xs = [p[0] for p in result]
    gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    assert all(1.5 <= g <= 3.0 for g in gaps)


def test_downsample_points_returns_unchanged_when_fewer_than_n():
    points = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    assert downsample_points(points, 10) == points


def test_downsample_points_empty_list_gives_empty_list():
    assert downsample_points([], 10) == []


def test_downsample_points_n_equals_one_gives_single_point():
    points = [(float(i), float(i)) for i in range(21)]
    result = downsample_points(points, 1)
    assert len(result) == 1


def test_downsample_points_n_zero_or_negative_gives_empty_list():
    points = [(0.0, 0.0), (1.0, 1.0)]
    assert downsample_points(points, 0) == []
    assert downsample_points(points, -5) == []


# --- _evenly_spaced_indices -------------------------------------------------

def test_evenly_spaced_indices_picks_evenly_spaced_including_endpoints():
    indices = _evenly_spaced_indices(21, 10)
    assert len(indices) == 10
    assert indices[0] == 0
    assert indices[-1] == 20


def test_evenly_spaced_indices_returns_full_range_when_fewer_than_n():
    assert _evenly_spaced_indices(3, 10) == [0, 1, 2]


def test_evenly_spaced_indices_empty_when_length_zero():
    assert _evenly_spaced_indices(0, 10) == []


def test_evenly_spaced_indices_n_equals_one_gives_middle_index():
    assert _evenly_spaced_indices(21, 1) == [10]


def test_evenly_spaced_indices_n_zero_or_negative_gives_empty_list():
    assert _evenly_spaced_indices(10, 0) == []
    assert _evenly_spaced_indices(10, -5) == []


# --- _bucket_indices ----------------------------------------------------------

def test_bucket_indices_covers_every_index_exactly_once():
    buckets = _bucket_indices(23, 5)
    assert len(buckets) == 5
    flat = [i for bucket in buckets for i in bucket]
    assert flat == list(range(23))


def test_bucket_indices_roughly_equal_sizes():
    buckets = _bucket_indices(10, 3)
    sizes = sorted(len(b) for b in buckets)
    assert sizes == [3, 3, 4]


def test_bucket_indices_n_greater_than_length_gives_single_index_buckets():
    buckets = _bucket_indices(3, 10)
    assert buckets == [[0], [1], [2]]


def test_bucket_indices_empty_when_length_zero_or_n_nonpositive():
    assert _bucket_indices(0, 5) == []
    assert _bucket_indices(5, 0) == []
    assert _bucket_indices(5, -1) == []


# --- signed_curvature_profile ------------------------------------------------

def test_signed_curvature_profile_straight_profile_all_near_zero():
    profile = [20.0] * 10
    result = signed_curvature_profile(profile, chord_len=100.0, n=10)
    assert result is not None
    assert all(v == pytest.approx(0.0, abs=0.01) for v in result)


def test_signed_curvature_profile_downsamples_to_n_points_including_endpoints():
    profile = [20.0 + i for i in range(21)]
    result = signed_curvature_profile(profile, chord_len=100.0, n=10)
    assert result is not None
    assert len(result) == 10


def test_signed_curvature_profile_none_for_empty_profile():
    assert signed_curvature_profile([], chord_len=100.0, n=10) is None


def test_signed_curvature_profile_none_for_nonpositive_chord_len():
    profile = [20.0] * 10
    assert signed_curvature_profile(profile, chord_len=0.0, n=10) is None


def test_signed_curvature_profile_sign_matches_bulge_and_cave():
    bulged = [20.0, 20.0, 20.0, 25.0, 30.0, 25.0, 20.0, 20.0, 20.0, 20.0]
    result = signed_curvature_profile(bulged, chord_len=100.0, n=10)
    assert result is not None
    assert max(result) > 0

    caved = [20.0, 20.0, 20.0, 15.0, 10.0, 15.0, 20.0, 20.0, 20.0, 20.0]
    result = signed_curvature_profile(caved, chord_len=100.0, n=10)
    assert result is not None
    assert min(result) < 0


def test_signed_curvature_profile_never_misses_a_sustained_deformation_at_any_offset():
    """Regression test for a real aliasing bug: with a high-resolution raw
    profile (e.g. ~200 rows for a typical torso), naively sampling n=10
    evenly-spaced rows can fall entirely between a real, sustained
    deformation (a deliberate hunch spanning many consecutive rows) and
    miss it completely, regardless of how pronounced the deformation is.
    Bucket-max must catch it no matter which rows it occupies."""
    n_raw = 200
    baseline_val = 20.0
    bump = 15.0
    # Slide a 14-row-wide sustained bump across interior start offsets (not
    # touching row 0 or the last row -- those coincide with the profile's own
    # endpoints, which by construction anchor the straight-line reference and
    # so can never show a deviation from themselves; that's an inherent
    # property of the endpoint-anchored method, not the aliasing bug under
    # test here) and verify it is never fully missed.
    for start in range(10, n_raw - 24, 7):
        profile = [baseline_val] * n_raw
        for i in range(start, start + 14):
            profile[i] = baseline_val + bump
        result = signed_curvature_profile(profile, chord_len=200.0, n=10)
        assert result is not None
        assert max(result, key=abs) > 0.0, f"deformation at rows {start}-{start+14} was missed"


def test_signed_curvature_ratio_matches_signed_curvature_profile_peak():
    fixtures = [
        [20.0] * 10,
        [20.0, 20.0, 20.0, 25.0, 30.0, 25.0, 20.0, 20.0, 20.0, 20.0],
        [20.0, 20.0, 20.0, 15.0, 10.0, 15.0, 20.0, 20.0, 20.0, 20.0],
    ]
    for back_profile in fixtures:
        expected = max(signed_curvature_profile(back_profile, 100.0, len(back_profile)), key=abs)
        actual = signed_curvature_ratio(back_profile, back_profile, chord_len=100.0, facing_left=True)
        assert actual == pytest.approx(expected)


# --- spine_zone_curvature -----------------------------------------------------

def test_spine_zone_curvature_thirds_split():
    # Bulge (positive) concentrated in the first third (shoulder end),
    # cave (negative) concentrated in the last third (hip end).
    profile = [0.3, 0.3, 0.3, 0.0, 0.0, 0.0, -0.3, -0.3, -0.3]
    thoracic, lumbar = spine_zone_curvature(profile)
    assert thoracic == pytest.approx(0.3)
    assert lumbar == pytest.approx(-0.3)


def test_spine_zone_curvature_empty_profile_gives_none_none():
    assert spine_zone_curvature([]) == (None, None)


def test_spine_zone_curvature_short_profile_falls_back_to_whole_profile_peak():
    profile = [0.2, 0.4]
    thoracic, lumbar = spine_zone_curvature(profile)
    assert thoracic == pytest.approx(0.4)
    assert lumbar == pytest.approx(0.4)


def test_spine_zone_curvature_uses_peak_not_mean_within_zone():
    # A single spatially-concentrated deformation (0.6) surrounded by
    # near-neutral neighbors in the same zone must NOT be diluted by
    # averaging -- the zone value must reflect the worst point found,
    # since each input point already represents its own slice's peak
    # (see signed_curvature_profile's bucket-max reduction).
    profile = [0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    thoracic, lumbar = spine_zone_curvature(profile)
    assert thoracic == pytest.approx(0.6)
    assert lumbar == pytest.approx(0.0)


def test_spine_zone_curvature_custom_zone_fraction():
    profile = [1.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0]
    thoracic, lumbar = spine_zone_curvature(profile, zone_fraction=2.0 / 7.0)
    assert thoracic == pytest.approx(1.0)
    assert lumbar == pytest.approx(-1.0)


# --- _angle_2d -----------------------------------------------------------

def test_angle_2d_collinear_points_gives_180():
    assert _angle_2d((0.0, 0.0), (5.0, 5.0), (10.0, 10.0)) == pytest.approx(180.0)


def test_angle_2d_right_angle_gives_90():
    assert _angle_2d((1.0, 0.0), (0.0, 0.0), (0.0, 1.0)) == pytest.approx(90.0)


def test_angle_2d_degenerate_zero_length_vector_gives_180():
    # b coincides with a -> vector b->a is zero-length -> degenerate guard fires.
    assert _angle_2d((3.0, 4.0), (3.0, 4.0), (0.0, 1.0)) == pytest.approx(180.0)


# --- signed_curvature_profile_indices -------------------------------------

def test_signed_curvature_profile_indices_matches_signed_curvature_profile_values():
    back_profile = [20.0 + i for i in range(21)]
    values_only = signed_curvature_profile(back_profile, chord_len=100.0, n=7)
    indexed = signed_curvature_profile_indices(back_profile, chord_len=100.0, n=7)
    assert indexed is not None
    assert [v for v, i in indexed] == pytest.approx(values_only)


def test_signed_curvature_profile_indices_index_points_at_the_true_spike_row():
    n_raw = 21
    profile = [10.0] * n_raw
    profile[3] = 25.0  # a single spike inside what will be the first bucket
    indexed = signed_curvature_profile_indices(profile, chord_len=100.0, n=7)
    assert indexed is not None
    value, index = indexed[0]  # bucket 0 covers raw indices 0-2... or wherever row 3 lands
    # Whichever bucket contains raw row 3 must report index == 3.
    matching = [pair for pair in indexed if pair[1] == 3]
    assert len(matching) == 1
    assert matching[0][0] != 0.0


def test_signed_curvature_profile_indices_none_for_empty_or_nonpositive_chord():
    assert signed_curvature_profile_indices([], chord_len=100.0, n=5) is None
    assert signed_curvature_profile_indices([1.0, 2.0], chord_len=0.0, n=5) is None


# --- spine_zone_bend_angles ------------------------------------------------

def test_spine_zone_bend_angles_straight_profile_gives_near_zero():
    back_profile = [10.0 + 0.1 * i for i in range(180)]  # perfectly linear
    indices = signed_curvature_profile_indices(back_profile, chord_len=180.0, n=10)
    thoracic, lumbar = spine_zone_bend_angles(back_profile, 180.0, indices)
    assert thoracic == pytest.approx(0.0, abs=1e-6)
    assert lumbar == pytest.approx(0.0, abs=1e-6)


def test_spine_zone_bend_angles_known_synthetic_case_matches_hand_computed_degrees():
    # 21-row profile, flat baseline of 10.0, with a clear outward spike in the
    # thoracic zone (row 3) and an inward spike in the lumbar zone (row 17).
    n_raw = 21
    back_profile = [10.0] * n_raw
    back_profile[3] = 25.0
    back_profile[17] = -5.0
    chord_len = 20.0

    indices = signed_curvature_profile_indices(back_profile, chord_len, n=7)
    thoracic, lumbar = spine_zone_bend_angles(back_profile, chord_len, indices)
    assert thoracic is not None and lumbar is not None

    # Independently hand-derived expected values: with zone_fraction=1/3 and
    # n=7, zone_size=round(7/3)=2 -> thoracic zone = raw rows 0-6 aggregated
    # into buckets 0-1 (2 of 7 buckets), lumbar zone = the last 2 buckets
    # (raw rows covering 14-20). boundary_fraction=0.5 -> boundary_idx =
    # round(0.5*20) = 10. Anchor points: shoulder=(10,0), boundary=(10,10),
    # hip=(10,20); vertices at the true spike rows (25,3) and (-5,17).
    def expected_bend(vertex, ref_a, ref_b, sign_source):
        vax, vay = ref_a[0] - vertex[0], ref_a[1] - vertex[1]
        vcx, vcy = ref_b[0] - vertex[0], ref_b[1] - vertex[1]
        mag = math.hypot(vax, vay) * math.hypot(vcx, vcy)
        cosine = max(-1.0, min(1.0, (vax * vcx + vay * vcy) / mag))
        angle = math.degrees(math.acos(cosine))
        return math.copysign(180.0 - angle, sign_source)

    expected_thoracic = expected_bend((25.0, 3), (10.0, 0), (10.0, 10), 1.0)
    expected_lumbar = expected_bend((-5.0, 17), (10.0, 10), (10.0, 20), -1.0)

    assert thoracic == pytest.approx(expected_thoracic, abs=0.01)
    assert lumbar == pytest.approx(expected_lumbar, abs=0.01)
    assert thoracic > 0  # outward spike -> bula -> positive
    assert lumbar < 0    # inward spike -> svank -> negative


def test_spine_zone_bend_angles_degenerate_vertex_at_boundary_returns_zero():
    # 7-row profile, zone_fraction=1/3 -> zone_size=2 -> thoracic zone = rows
    # 0-1. boundary_fraction=0.1 -> boundary_idx = round(0.1*6) = 1, which
    # coincides with the thoracic zone's own peak-deviation row -> the
    # thoracic vertex and the boundary point become the same coordinate ->
    # a zero-length ray -> _angle_2d's degenerate guard -> bend 0, not a crash.
    back_profile = [0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    indices = signed_curvature_profile_indices(back_profile, chord_len=10.0, n=7)
    thoracic, lumbar = spine_zone_bend_angles(back_profile, 10.0, indices, boundary_fraction=0.1)
    assert thoracic == pytest.approx(0.0)


def test_spine_zone_bend_angles_none_when_profile_indices_none():
    assert spine_zone_bend_angles([1.0, 2.0, 3.0], 100.0, []) == (None, None)


def test_spine_zone_bend_angles_none_when_back_profile_too_short():
    indices = [(0.1, 0), (0.2, 1)]
    assert spine_zone_bend_angles([10.0, 11.0], 100.0, indices) == (None, None)


def test_spine_zone_bend_angles_none_when_chord_len_nonpositive():
    back_profile = [10.0] * 21
    indices = signed_curvature_profile_indices(back_profile, chord_len=20.0, n=7)
    assert spine_zone_bend_angles(back_profile, 0.0, indices) == (None, None)


def test_spine_zone_bend_angles_n_less_than_3_falls_back_to_whole_profile():
    back_profile = [10.0, 10.0, 25.0, 10.0, 10.0]
    indices = signed_curvature_profile_indices(back_profile, chord_len=20.0, n=2)
    assert indices is not None and len(indices) == 2
    thoracic, lumbar = spine_zone_bend_angles(back_profile, 20.0, indices)
    assert thoracic is not None and lumbar is not None
    assert thoracic == pytest.approx(lumbar)


def test_spine_zone_bend_angles_ignores_ordinary_body_taper_not_spine_curvature():
    """Regression test for a real bug: a torso naturally tapers from a wider
    shoulder width to a narrower waist and back out to a wider hip width --
    completely ordinary anatomy, present even for a perfectly straight
    spine. The angle geometry must anchor each zone's triangle to THIS
    tick's own shoulder/hip readings (deviation from the shoulder-hip
    chord, always exactly 0 at both ends) rather than the raw silhouette
    widths (which differ between the shoulder and hip ends for virtually
    every real body, and can shift independently of spine shape e.g.
    between standing and a squat) -- otherwise ordinary body-width taper
    alone reads as a large, spurious "bend" no matter how the back is
    actually held."""
    n_raw = 180
    shoulder_w, waist_w, hip_w = 45.0, 25.0, 50.0  # shoulder width != hip width
    back_profile = []
    for i in range(n_raw):
        frac = i / (n_raw - 1)
        if frac < 0.5:
            back_profile.append(shoulder_w - (shoulder_w - waist_w) * (frac / 0.5))
        else:
            back_profile.append(waist_w + (hip_w - waist_w) * ((frac - 0.5) / 0.5))

    indices = signed_curvature_profile_indices(back_profile, chord_len=180.0, n=10)
    thoracic, lumbar = spine_zone_bend_angles(back_profile, 180.0, indices)
    assert thoracic is not None and lumbar is not None
    # Ordinary taper alone must stay well clear of any plausible "large bend"
    # threshold (single-digit degrees), not the tens-of-degrees a raw-width
    # triangle would report for this same shape.
    assert abs(thoracic) < 2.0
    assert abs(lumbar) < 2.0


def test_spine_zone_bend_angles_detects_real_curvature_on_top_of_body_taper():
    # Same tapering body shape as above, but now with a genuine, sustained
    # outward bulge added in the thoracic zone -- must still be detected
    # clearly despite the shoulder/hip width mismatch that would otherwise
    # confound a raw-width-based triangle.
    n_raw = 180
    shoulder_w, waist_w, hip_w = 45.0, 25.0, 50.0
    back_profile = []
    for i in range(n_raw):
        frac = i / (n_raw - 1)
        if frac < 0.5:
            back_profile.append(shoulder_w - (shoulder_w - waist_w) * (frac / 0.5))
        else:
            back_profile.append(waist_w + (hip_w - waist_w) * ((frac - 0.5) / 0.5))
    for i in range(20, 35):
        back_profile[i] += 20.0  # a real, sustained thoracic bulge

    indices = signed_curvature_profile_indices(back_profile, chord_len=180.0, n=10)
    thoracic, lumbar = spine_zone_bend_angles(back_profile, 180.0, indices)
    assert thoracic is not None
    assert thoracic > 20.0  # clearly detected, not swamped by the taper
    assert thoracic > 0     # outward bulge -> positive sign


def test_spine_zone_bend_angles_stays_correct_up_to_default_zone_fraction():
    """Regression test for a real, measured cliff: growing zone_fraction
    past ~1/3 pulls in a bucket from the profile's middle region, where
    ordinary (non-spine) body-taper deviation from the shoulder-hip chord
    is largest -- for this exact synthetic body+bulge, zone_fraction=0.35
    already flips a genuine +24 degree thoracic bulge to a wrong -14
    degrees, and stays wrong through 0.45. This locks in that the DEFAULT
    _SPINE_ZONE_FRACTION stays on the correct (narrow) side of that cliff.
    If this ever needs to change, re-verify against this exact scenario
    first -- see _SPINE_ZONE_FRACTION's docstring in spine_contour.py."""
    n_raw = 180
    shoulder_w, waist_w, hip_w = 45.0, 25.0, 50.0
    back_profile = []
    for i in range(n_raw):
        frac = i / (n_raw - 1)
        if frac < 0.5:
            back_profile.append(shoulder_w - (shoulder_w - waist_w) * (frac / 0.5))
        else:
            back_profile.append(waist_w + (hip_w - waist_w) * ((frac - 0.5) / 0.5))
    for i in range(20, 35):
        back_profile[i] += 20.0

    indices = signed_curvature_profile_indices(back_profile, chord_len=180.0, n=10)
    thoracic, _ = spine_zone_bend_angles(back_profile, 180.0, indices, _SPINE_ZONE_FRACTION)
    assert thoracic > 20.0
    assert thoracic > 0


# --- spine_bend_anchor_indices ---------------------------------------------

def test_spine_bend_anchor_indices_matches_spine_zone_bend_angles_vertex_choice():
    n_raw = 21
    back_profile = [10.0] * n_raw
    back_profile[3] = 25.0
    back_profile[17] = -5.0
    indices = signed_curvature_profile_indices(back_profile, chord_len=20.0, n=7)
    anchors = spine_bend_anchor_indices(back_profile, indices)
    assert anchors == (0, 3, 10, 17, 20)


def test_spine_bend_anchor_indices_none_when_profile_indices_empty():
    assert spine_bend_anchor_indices([1.0, 2.0, 3.0], []) is None


def test_spine_bend_anchor_indices_none_when_back_profile_too_short():
    assert spine_bend_anchor_indices([1.0, 2.0], [(0.1, 0), (0.2, 1)]) is None


# --- spine_zone_for_index ---------------------------------------------------

def test_spine_zone_for_index_classifies_thoracic_middle_lumbar():
    n_raw = 30  # zone_fraction=1/3 -> zone_size = 10 -> thoracic 0-9, lumbar 20-29
    assert spine_zone_for_index(0, n_raw) == "thoracic"
    assert spine_zone_for_index(9, n_raw) == "thoracic"
    assert spine_zone_for_index(10, n_raw) == "middle"
    assert spine_zone_for_index(19, n_raw) == "middle"
    assert spine_zone_for_index(20, n_raw) == "lumbar"
    assert spine_zone_for_index(29, n_raw) == "lumbar"


def test_spine_zone_for_index_custom_zone_fraction():
    assert spine_zone_for_index(0, 10, zone_fraction=0.5) == "thoracic"
    assert spine_zone_for_index(9, 10, zone_fraction=0.5) == "lumbar"


# --- sparse_points_in_frame --------------------------------------------------

def test_sparse_points_in_frame_maps_selected_indices_only():
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    hip_px = (200.0, 300.0)
    shoulder_px = (200.0, 150.0)  # vertical, untilted chord
    result = crop_and_rotate_roi(frame, hip_px, shoulder_px)
    assert result is not None
    _, hip_point, shoulder_point, chord_len, M = result

    back_profile = [10.0, 12.0, 14.0, 16.0, 18.0]
    points = sparse_points_in_frame(M, hip_point, shoulder_point, back_profile,
                                     indices=[0, 2, 4], on_right_side=True)
    assert len(points) == 3
    # Vertical/untilted chord -> M is a pure translation, so mapping is exact:
    # orig_x = hip_px[0] + dist, orig_y = shoulder_px[1] + row_index.
    assert points[0] == (pytest.approx(210.0, abs=0.5), pytest.approx(150.0, abs=0.5))
    assert points[1] == (pytest.approx(214.0, abs=0.5), pytest.approx(152.0, abs=0.5))
    assert points[2] == (pytest.approx(218.0, abs=0.5), pytest.approx(154.0, abs=0.5))


def test_sparse_points_in_frame_empty_indices_gives_empty_list():
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    result = crop_and_rotate_roi(frame, (200.0, 300.0), (200.0, 150.0))
    assert result is not None
    _, hip_point, shoulder_point, chord_len, M = result
    assert sparse_points_in_frame(M, hip_point, shoulder_point, [1.0, 2.0], [], True) == []
