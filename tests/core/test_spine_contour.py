import pytest
import numpy as np
import cv2

from core.spine_contour import (facing_left, crop_and_rotate_roi, extract_back_contour,
                                  signed_curvature_ratio, back_contour_points_in_frame,
                                  downsample_points,
                                  _MIN_CHORD_PX, _ROI_MARGIN_FRAC, _ROI_END_PAD_FRAC)


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
