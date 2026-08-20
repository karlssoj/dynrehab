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


class _SegmentingBackend:
    """Fake backend that produces a valid, successful segmentation sample
    every tick (straight vertical mask, no bulge) -- used to verify the back
    contour actually gets drawn onto the annotated frame end-to-end."""

    def process(self, frame, highlight_joints=frozenset()):
        keypoints = {
            "nose": (0.4, 0.1, 0.0, 0.9),
            "left_shoulder": (0.5, 0.2, 0.0, 0.9),
            "left_hip": (0.5, 0.6, 0.0, 0.9),
        }
        return keypoints, frame

    def get_segmentation_mask(self, roi_image):
        h, w = roi_image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        center_x = w // 2
        mask[:, max(0, center_x - 20):center_x + 20] = 255
        return mask

    def close(self):
        pass


def test_run_draws_back_contour_on_annotated_frame_when_sample_succeeds(mocker):
    """When a spine sample succeeds, the back contour must be drawn onto the
    annotated frame handed to subscribers -- this is the actual on-screen
    visualization, not just the numeric spine_curvature_ratio field."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = [(True, frame), (False, None)]
    mocker.patch("core.pose_engine.cv2.VideoCapture", return_value=mock_cap)

    engine = PoseEngine(backend=_SegmentingBackend())
    received = []
    engine.subscribe(lambda pose_frame, annotated: received.append((pose_frame, annotated)))

    engine._running = True
    engine._run(source=0)

    assert len(received) == 1
    pose_frame, annotated = received[0]
    assert pose_frame.spine_curvature_ratio is not None
    assert annotated.max() > 0


def test_run_persists_back_contour_drawing_between_samples(mocker):
    """The drawn contour must persist on frames between the periodic
    samples (not flicker), since sampling only runs every ~250ms while
    drawing happens every frame."""
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = [(True, frame1), (True, frame2), (False, None)]
    mocker.patch("core.pose_engine.cv2.VideoCapture", return_value=mock_cap)
    # First tick samples (elapsed since epoch is huge), second tick does not
    # (interval not yet elapsed) -- forces the second frame's drawing to
    # come from the cached _last_spine_points, not a fresh sample.
    times = iter([1_700_000_000.0, 1_700_000_000.0, 1_700_000_000.05, 1_700_000_000.05,
                  1_700_000_000.05, 1_700_000_000.05, 1_700_000_000.05, 1_700_000_000.05])
    mocker.patch("core.pose_engine.time.time", side_effect=lambda: next(times))

    engine = PoseEngine(backend=_SegmentingBackend())
    received = []
    engine.subscribe(lambda pose_frame, annotated: received.append((pose_frame, annotated)))

    engine._running = True
    engine._run(source=0)

    assert len(received) == 2
    assert received[0][0].spine_curvature_ratio is not None
    assert received[1][0].spine_curvature_ratio is None  # not re-sampled this tick
    assert received[0][1].max() > 0
    assert received[1][1].max() > 0  # still drawn, from the cached points
