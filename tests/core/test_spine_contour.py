import pytest
import numpy as np
import cv2

from core.spine_contour import facing_left, crop_and_rotate_roi, _MIN_CHORD_PX, _ROI_MARGIN_FRAC, _ROI_END_PAD_FRAC


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
