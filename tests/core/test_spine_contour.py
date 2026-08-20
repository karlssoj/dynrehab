import pytest

from core.spine_contour import facing_left


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
