import numpy as np
from unittest.mock import MagicMock

from core.pose_engine import PoseEngine, _should_sample_spine, _SPINE_SAMPLE_INTERVAL_SECS


def test_should_sample_spine_true_when_interval_elapsed():
    assert _should_sample_spine(last_sample_time=0.0, now=0.3, interval=0.25) is True


def test_should_sample_spine_false_when_interval_not_elapsed():
    assert _should_sample_spine(last_sample_time=0.0, now=0.1, interval=0.25) is False


def test_should_sample_spine_true_on_exact_boundary():
    assert _should_sample_spine(last_sample_time=1.0, now=1.25, interval=0.25) is True


def test_should_sample_spine_true_for_first_ever_call():
    # last_sample_time starts at 0.0 (PoseEngine.__init__ default); real
    # time.time() values are always far larger, so the very first call in a
    # live session always samples immediately.
    assert _should_sample_spine(last_sample_time=0.0, now=1_700_000_000.0, interval=0.25) is True


def test_should_sample_spine_uses_default_interval():
    assert _should_sample_spine(last_sample_time=0.0, now=_SPINE_SAMPLE_INTERVAL_SECS) is True
    assert _should_sample_spine(last_sample_time=0.0, now=_SPINE_SAMPLE_INTERVAL_SECS - 0.01) is False


class _RaisingSegBackend:
    """Fake backend whose keypoints pass the hip/shoulder confidence gate and
    produce a valid ROI, but whose get_segmentation_mask blows up — simulating
    a missing/corrupt YOLO segmentation model file or an internal
    ultralytics/torch error."""

    def process(self, frame, highlight_joints=frozenset()):
        keypoints = {
            "nose": (0.4, 0.1, 0.0, 0.9),
            "left_shoulder": (0.5, 0.2, 0.0, 0.9),
            "left_hip": (0.5, 0.6, 0.0, 0.9),
        }
        return keypoints, frame

    def get_segmentation_mask(self, roi_image):
        raise RuntimeError("segmentation model file missing or corrupt")

    def close(self):
        pass


def test_run_survives_get_segmentation_mask_exception(mocker):
    """A raising get_segmentation_mask must not propagate out of _run(): the
    rest of the loop (frame capture, angle calculation, subscriber callbacks)
    must keep working, with the spine sample simply skipped for that tick."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = [(True, frame), (False, None)]
    mocker.patch("core.pose_engine.cv2.VideoCapture", return_value=mock_cap)

    engine = PoseEngine(backend=_RaisingSegBackend())
    received = []
    engine.subscribe(lambda pose_frame, annotated: received.append(pose_frame))

    engine._running = True
    engine._run(source=0)  # webcam-style source; exits the loop after one frame

    assert len(received) == 1
    assert received[0].spine_curvature_ratio is None
