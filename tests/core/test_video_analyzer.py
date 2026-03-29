import numpy as np
from unittest.mock import MagicMock, patch, call
from core.video_analyzer import analyze_reference_video


def _make_mock_cap(frames):
    """frames: list of np arrays, or empty list for zero-frame video."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    read_returns = [(True, f) for f in frames] + [(False, None)]
    mock_cap.read.side_effect = read_returns
    return mock_cap


def test_returns_empty_when_video_cannot_open(mocker):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mocker.patch("core.video_analyzer.cv2.VideoCapture", return_value=mock_cap)
    assert analyze_reference_video("nonexistent.mp4") == {}


def test_returns_empty_when_no_frames(mocker):
    mock_cap = _make_mock_cap([])
    mocker.patch("core.video_analyzer.cv2.VideoCapture", return_value=mock_cap)

    mock_pose = MagicMock()
    mock_pose.__enter__ = MagicMock(return_value=mock_pose)
    mock_pose.__exit__ = MagicMock(return_value=False)
    mocker.patch("core.video_analyzer.Pose", return_value=mock_pose)

    assert analyze_reference_video("video.mp4") == {}


def test_returns_empty_when_no_landmarks_detected(mocker):
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_cap = _make_mock_cap([dummy_frame])
    mocker.patch("core.video_analyzer.cv2.VideoCapture", return_value=mock_cap)
    mocker.patch("core.video_analyzer.cv2.cvtColor", return_value=dummy_frame)

    mock_results = MagicMock()
    mock_results.pose_landmarks = None
    mock_pose = MagicMock()
    mock_pose.__enter__ = MagicMock(return_value=mock_pose)
    mock_pose.__exit__ = MagicMock(return_value=False)
    mock_pose.process.return_value = mock_results
    mocker.patch("core.video_analyzer.Pose", return_value=mock_pose)

    assert analyze_reference_video("video.mp4") == {}


def test_filters_joints_with_range_at_or_below_10(mocker):
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_cap = _make_mock_cap([dummy_frame, dummy_frame])
    mocker.patch("core.video_analyzer.cv2.VideoCapture", return_value=mock_cap)
    mocker.patch("core.video_analyzer.cv2.cvtColor", return_value=dummy_frame)

    mock_lm = MagicMock()
    mock_lm.x, mock_lm.y, mock_lm.z, mock_lm.visibility = 0.5, 0.5, 0.0, 0.9
    mock_results = MagicMock()
    mock_results.pose_landmarks.landmark = [mock_lm] * 33
    mock_pose = MagicMock()
    mock_pose.__enter__ = MagicMock(return_value=mock_pose)
    mock_pose.__exit__ = MagicMock(return_value=False)
    mock_pose.process.return_value = mock_results
    mocker.patch("core.video_analyzer.Pose", return_value=mock_pose)

    call_count = {"n": 0}
    def mock_angles(_kp):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Frame 1: elbow at 160°, knee at 170°
            return {"left_elbow_angle": 160.0, "left_knee_angle": 170.0}
        # Frame 2: elbow at 45° (range 115° → included), knee at 165° (range 5° → filtered)
        return {"left_elbow_angle": 45.0, "left_knee_angle": 165.0}

    mocker.patch("core.video_analyzer.calculate_angles", side_effect=mock_angles)

    result = analyze_reference_video("video.mp4")

    assert "left_elbow_angle" in result
    assert result["left_elbow_angle"]["min"] == 45.0
    assert result["left_elbow_angle"]["max"] == 160.0
    assert result["left_elbow_angle"]["range"] == 115.0

    assert "left_knee_angle" not in result  # range = 5° → filtered


def test_rounds_values_to_one_decimal(mocker):
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_cap = _make_mock_cap([dummy_frame, dummy_frame])
    mocker.patch("core.video_analyzer.cv2.VideoCapture", return_value=mock_cap)
    mocker.patch("core.video_analyzer.cv2.cvtColor", return_value=dummy_frame)

    mock_lm = MagicMock()
    mock_lm.x, mock_lm.y, mock_lm.z, mock_lm.visibility = 0.5, 0.5, 0.0, 0.9
    mock_results = MagicMock()
    mock_results.pose_landmarks.landmark = [mock_lm] * 33
    mock_pose = MagicMock()
    mock_pose.__enter__ = MagicMock(return_value=mock_pose)
    mock_pose.__exit__ = MagicMock(return_value=False)
    mock_pose.process.return_value = mock_results
    mocker.patch("core.video_analyzer.Pose", return_value=mock_pose)

    frame_angles = [{"left_elbow_angle": 160.3333}, {"left_elbow_angle": 44.6789}]
    mocker.patch("core.video_analyzer.calculate_angles", side_effect=frame_angles)

    result = analyze_reference_video("video.mp4")
    assert result["left_elbow_angle"]["min"] == 44.7
    assert result["left_elbow_angle"]["max"] == 160.3
    assert result["left_elbow_angle"]["range"] == 115.7
