import pytest
import numpy as np
import cv2

from core.spine_contour import facing_left, crop_and_rotate_roi, extract_back_contour, signed_curvature_ratio, _MIN_CHORD_PX, _ROI_MARGIN_FRAC, _ROI_END_PAD_FRAC


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
    rotated, hip_point, shoulder_point, chord_len = result
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
    rotated, hip_point, shoulder_point, chord_len = result
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
    rotated, hip_point, shoulder_point, chord_len = result
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
    rotated, hip_point, shoulder_point, chord_len = result

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
    _, _, _, chord_len = result
    assert chord_len == pytest.approx(_MIN_CHORD_PX, abs=0.01)


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
