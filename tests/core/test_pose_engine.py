import pytest
import numpy as np
from unittest.mock import MagicMock

from core.pose_engine import (PoseEngine, _should_sample_spine, _SPINE_SAMPLE_INTERVAL_SECS,
                               _spine_scan_shift_frac, _hip_bend_deg)


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


# --- _spine_scan_shift_frac ---------------------------------------------------

def test_spine_scan_shift_frac_standing_below_low_threshold():
    assert _spine_scan_shift_frac(0.0, standing_frac=0.22, bent_frac=0.12,
                                   bend_low_deg=10.0, bend_high_deg=45.0) == pytest.approx(0.22)
    assert _spine_scan_shift_frac(5.0, standing_frac=0.22, bent_frac=0.12,
                                   bend_low_deg=10.0, bend_high_deg=45.0) == pytest.approx(0.22)


def test_spine_scan_shift_frac_bent_above_high_threshold():
    assert _spine_scan_shift_frac(45.0, standing_frac=0.22, bent_frac=0.12,
                                   bend_low_deg=10.0, bend_high_deg=45.0) == pytest.approx(0.12)
    assert _spine_scan_shift_frac(90.0, standing_frac=0.22, bent_frac=0.12,
                                   bend_low_deg=10.0, bend_high_deg=45.0) == pytest.approx(0.12)


def test_spine_scan_shift_frac_interpolates_linearly_between_thresholds():
    midpoint = _spine_scan_shift_frac(27.5, standing_frac=0.22, bent_frac=0.12,
                                       bend_low_deg=10.0, bend_high_deg=45.0)
    assert midpoint == pytest.approx(0.17, abs=1e-6)


def test_spine_scan_shift_frac_degenerate_threshold_range_falls_back_to_bent():
    assert _spine_scan_shift_frac(0.0, standing_frac=0.22, bent_frac=0.12,
                                   bend_low_deg=45.0, bend_high_deg=45.0) == pytest.approx(0.12)
    assert _spine_scan_shift_frac(0.0, standing_frac=0.22, bent_frac=0.12,
                                   bend_low_deg=45.0, bend_high_deg=10.0) == pytest.approx(0.12)


# --- _hip_bend_deg -------------------------------------------------------------

def test_hip_bend_deg_zero_when_standing_straight():
    # Shoulder directly above hip, knee directly below hip -- collinear.
    shoulder_px = (200.0, 100.0)
    hip_px = (200.0, 300.0)
    knee_px = (200.0, 500.0)
    assert _hip_bend_deg(shoulder_px, hip_px, knee_px) == pytest.approx(0.0, abs=1e-6)


def test_hip_bend_deg_increases_as_hip_flexes_forward():
    hip_px = (200.0, 300.0)
    knee_px = (200.0, 500.0)  # knee stays directly below hip
    shoulder_straight = (200.0, 100.0)
    shoulder_bent = (280.0, 120.0)  # shoulder swung forward
    straight_bend = _hip_bend_deg(shoulder_straight, hip_px, knee_px)
    bent_bend = _hip_bend_deg(shoulder_bent, hip_px, knee_px)
    assert bent_bend > straight_bend


def test_hip_bend_deg_zero_on_degenerate_geometry():
    hip_px = (200.0, 300.0)
    assert _hip_bend_deg(hip_px, hip_px, (200.0, 500.0)) == pytest.approx(0.0)


class _RaisingSegBackend:
    """Fake backend whose keypoints pass the hip/shoulder confidence gate and
    produce a valid ROI, but whose get_segmentation_mask blows up — simulating
    a missing/corrupt YOLO segmentation model file or an internal
    ultralytics/torch error."""

    def process(self, frame, highlight_joints=frozenset()):
        keypoints = {
            "nose": (0.4, 0.1, 0.0, 0.9),
            "left_shoulder": (0.5, 0.2, 0.0, 0.9),
            "right_shoulder": (0.51, 0.2, 0.0, 0.9),  # close to left -> profile stance
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
            "right_shoulder": (0.51, 0.2, 0.0, 0.9),  # close to left -> profile stance
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


def test_spine_sample_interval_defaults_to_constant(monkeypatch):
    monkeypatch.delenv("SPINE_SAMPLE_INTERVAL_SECS", raising=False)
    engine = PoseEngine(backend=_RaisingSegBackend())
    assert engine._spine_sample_interval == _SPINE_SAMPLE_INTERVAL_SECS


def test_spine_sample_interval_reads_from_env(monkeypatch):
    monkeypatch.setenv("SPINE_SAMPLE_INTERVAL_SECS", "0.05")
    engine = PoseEngine(backend=_RaisingSegBackend())
    assert engine._spine_sample_interval == pytest.approx(0.05)


def test_construction_prints_configured_spine_sample_interval(capsys, monkeypatch):
    monkeypatch.setenv("SPINE_SAMPLE_INTERVAL_SECS", "0.05")
    PoseEngine(backend=_RaisingSegBackend())
    out = capsys.readouterr().out
    assert "0.05" in out


def test_spine_sampling_exception_is_logged_once_not_every_tick(mocker, capsys):
    """A raising get_segmentation_mask must still print the actual error --
    silently, exception must not disappear entirely -- but only on the
    first occurrence, not every ~250ms tick, to avoid log spam."""
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = [(True, frame1), (True, frame2), (False, None)]
    mocker.patch("core.pose_engine.cv2.VideoCapture", return_value=mock_cap)
    # Force both frames to be sampling ticks (interval effectively zero).
    times = iter([1_700_000_000.0] * 10)
    mocker.patch("core.pose_engine.time.time", side_effect=lambda: next(times))

    engine = PoseEngine(backend=_RaisingSegBackend())
    engine._spine_sample_interval = 0.0
    engine.subscribe(lambda pose_frame, annotated: None)

    engine._running = True
    engine._run(source=0)

    out = capsys.readouterr().out
    assert out.count("segmentation model file missing or corrupt") == 1


class _FlakySegBackend:
    """Fake backend: first get_segmentation_mask call succeeds (valid mask),
    every later call returns None -- simulates losing the contour (e.g. the
    person moves, occlusion) after having found it once."""

    def __init__(self):
        self._call_count = 0

    def process(self, frame, highlight_joints=frozenset()):
        keypoints = {
            "nose": (0.4, 0.1, 0.0, 0.9),
            "left_shoulder": (0.5, 0.2, 0.0, 0.9),
            "right_shoulder": (0.51, 0.2, 0.0, 0.9),  # close to left -> profile stance
            "left_hip": (0.5, 0.6, 0.0, 0.9),
        }
        return keypoints, frame

    def get_segmentation_mask(self, roi_image):
        self._call_count += 1
        if self._call_count == 1:
            h, w = roi_image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            center_x = w // 2
            mask[:, max(0, center_x - 20):center_x + 20] = 255
            return mask
        return None

    def close(self):
        pass


def test_run_clears_cached_spine_points_when_a_later_sample_fails(mocker):
    """If a sampling tick fails to find the contour (e.g. the mask is lost),
    the previously drawn line must be cleared, not left hanging from the
    last successful sample."""
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = [(True, frame1), (True, frame2), (False, None)]
    mocker.patch("core.pose_engine.cv2.VideoCapture", return_value=mock_cap)
    times = iter([1_700_000_000.0] * 10)
    mocker.patch("core.pose_engine.time.time", side_effect=lambda: next(times))

    engine = PoseEngine(backend=_FlakySegBackend())
    engine._spine_sample_interval = 0.0  # force every tick to sample
    received = []
    engine.subscribe(lambda pose_frame, annotated: received.append((pose_frame, annotated)))

    engine._running = True
    engine._run(source=0)

    assert len(received) == 2
    assert received[0][0].spine_curvature_ratio is not None
    assert received[0][1].max() > 0  # first frame: line drawn

    assert received[1][0].spine_curvature_ratio is None
    assert received[1][1].max() == 0  # second frame: line cleared, nothing drawn
    assert engine._last_spine_points is None


def test_spine_contour_point_count_defaults_to_constant(monkeypatch):
    from core.pose_engine import _SPINE_CONTOUR_POINT_COUNT
    monkeypatch.delenv("SPINE_CONTOUR_POINT_COUNT", raising=False)
    engine = PoseEngine(backend=_RaisingSegBackend())
    assert engine._spine_contour_point_count == _SPINE_CONTOUR_POINT_COUNT


def test_spine_contour_point_count_reads_from_env(monkeypatch):
    monkeypatch.setenv("SPINE_CONTOUR_POINT_COUNT", "5")
    engine = PoseEngine(backend=_RaisingSegBackend())
    assert engine._spine_contour_point_count == 5


def test_run_draws_full_resolution_contour_not_downsampled(mocker):
    """The drawn contour line uses the FULL per-row resolution regardless
    of _spine_contour_point_count -- that setting only controls the angle
    measurement's vertex-search bucket count now, not how many points get
    drawn (an earlier version downsampled the drawn points to match it;
    this verifies the on-screen line no longer throws away resolution)."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = [(True, frame), (False, None)]
    mocker.patch("core.pose_engine.cv2.VideoCapture", return_value=mock_cap)

    engine = PoseEngine(backend=_SegmentingBackend())
    engine._spine_contour_point_count = 3
    received = []
    engine.subscribe(lambda pose_frame, annotated: received.append(pose_frame))

    engine._running = True
    engine._run(source=0)

    assert len(received) == 1
    assert engine._last_spine_points is not None
    assert len(engine._last_spine_points) > engine._spine_contour_point_count
    assert engine._last_spine_zone_labels is not None
    assert len(engine._last_spine_zone_labels) == len(engine._last_spine_points)


class _FacingCameraSegmentingBackend:
    """Same as _SegmentingBackend (a valid, successful segmentation sample
    every tick), but keypoints describe a person facing the camera (wide
    shoulder lateral span), not standing in profile."""

    def process(self, frame, highlight_joints=frozenset()):
        keypoints = {
            "nose": (0.5, 0.1, 0.0, 0.9),
            "left_shoulder": (0.3, 0.2, 0.0, 0.9),
            "right_shoulder": (0.7, 0.2, 0.0, 0.9),  # far from left -> facing camera
            "left_hip": (0.3, 0.6, 0.0, 0.9),
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


def test_run_skips_spine_sampling_when_not_in_profile(mocker):
    """Spine sampling must be skipped entirely when the person is facing
    the camera rather than standing in profile (wide shoulder_lateral_span)
    -- the whole geometric premise (hip-shoulder line as a sagittal-plane
    proxy) doesn't hold otherwise, and drawing a contour on a front-facing
    person would be meaningless."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = [(True, frame), (False, None)]
    mocker.patch("core.pose_engine.cv2.VideoCapture", return_value=mock_cap)

    engine = PoseEngine(backend=_FacingCameraSegmentingBackend())
    received = []
    engine.subscribe(lambda pose_frame, annotated: received.append((pose_frame, annotated)))

    engine._running = True
    engine._run(source=0)

    assert len(received) == 1
    pose_frame, annotated = received[0]
    assert pose_frame.spine_curvature_ratio is None
    assert annotated.max() == 0
    assert engine._last_spine_points is None


def test_spine_max_shoulder_lateral_ratio_defaults_to_constant(monkeypatch):
    from core.pose_engine import _SPINE_MAX_SHOULDER_LATERAL_RATIO
    monkeypatch.delenv("SPINE_MAX_SHOULDER_LATERAL_RATIO", raising=False)
    engine = PoseEngine(backend=_RaisingSegBackend())
    assert engine._spine_max_shoulder_lateral_ratio == _SPINE_MAX_SHOULDER_LATERAL_RATIO


def test_spine_max_shoulder_lateral_ratio_reads_from_env(monkeypatch):
    monkeypatch.setenv("SPINE_MAX_SHOULDER_LATERAL_RATIO", "0.6")
    engine = PoseEngine(backend=_RaisingSegBackend())
    assert engine._spine_max_shoulder_lateral_ratio == pytest.approx(0.6)


class _DistantFacingCameraSegmentingBackend:
    """Same as _FacingCameraSegmentingBackend, but the person is small in
    frame (far from the camera) -- the ABSOLUTE shoulder_lateral_span is
    small even though they're still facing the camera, because everything
    shrinks together with distance. A scale-invariant check (shoulder span
    relative to torso length) must still reject this; a fixed absolute
    threshold like the old 0.15 would incorrectly accept it."""

    def process(self, frame, highlight_joints=frozenset()):
        keypoints = {
            "nose": (0.5, 0.28, 0.0, 0.9),
            "left_shoulder": (0.45, 0.3, 0.0, 0.9),
            "right_shoulder": (0.55, 0.3, 0.0, 0.9),  # span=0.10, below the old absolute 0.15
            "left_hip": (0.45, 0.42, 0.0, 0.9),        # torso length=0.12 -> ratio ~0.83, clearly facing camera
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


def test_init_creates_exactly_two_spine_angle_filters(monkeypatch):
    # Filter count is fixed at 2 (one per zone), independent of
    # SPINE_CONTOUR_POINT_COUNT (which now only controls the vertex-search
    # bucket granularity and the on-screen dot count, not filter count).
    monkeypatch.setenv("SPINE_CONTOUR_POINT_COUNT", "7")
    engine = PoseEngine(backend=_RaisingSegBackend())
    assert engine._spine_thoracic_angle_filter is not None
    assert engine._spine_lumbar_angle_filter is not None
    assert engine._spine_thoracic_angle_filter is not engine._spine_lumbar_angle_filter


def test_run_sets_zone_bend_fields_on_successful_sample(mocker):
    """A successful spine sample must populate both new zone bend-angle
    fields, not just the legacy spine_curvature_ratio."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = [(True, frame), (False, None)]
    mocker.patch("core.pose_engine.cv2.VideoCapture", return_value=mock_cap)

    engine = PoseEngine(backend=_SegmentingBackend())
    received = []
    engine.subscribe(lambda pose_frame, annotated: received.append(pose_frame))

    engine._running = True
    engine._run(source=0)

    assert len(received) == 1
    assert received[0].spine_thoracic_bend_2d is not None
    assert received[0].spine_lumbar_bend_2d is not None


def test_run_leaves_zone_bend_fields_none_when_sample_fails(mocker):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = [(True, frame), (False, None)]
    mocker.patch("core.pose_engine.cv2.VideoCapture", return_value=mock_cap)

    engine = PoseEngine(backend=_RaisingSegBackend())
    received = []
    engine.subscribe(lambda pose_frame, annotated: received.append(pose_frame))

    engine._running = True
    engine._run(source=0)

    assert len(received) == 1
    assert received[0].spine_thoracic_bend_2d is None
    assert received[0].spine_lumbar_bend_2d is None


def test_run_resets_angle_filters_after_a_failed_sampling_tick(mocker):
    """After a tick that fails to produce fresh angles (person lost, mask
    lost, exception), both per-zone OneEuroFilters must be reset --
    verified via OneEuroFilter's documented behavior that the first filter()
    call after a reset returns the raw, un-lagged value regardless of any
    prior state."""
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = [(True, frame1), (True, frame2), (False, None)]
    mocker.patch("core.pose_engine.cv2.VideoCapture", return_value=mock_cap)
    times = iter([1_700_000_000.0] * 10)
    mocker.patch("core.pose_engine.time.time", side_effect=lambda: next(times))

    engine = PoseEngine(backend=_FlakySegBackend())
    engine._spine_sample_interval = 0.0  # force every tick to sample

    engine._running = True
    engine._run(source=0)

    assert engine._spine_thoracic_angle_filter._t_prev is None
    assert engine._spine_lumbar_angle_filter._t_prev is None


def test_run_skips_spine_sampling_for_distant_person_facing_camera(mocker):
    """A small-in-frame person facing the camera must still be rejected --
    proves the orientation check is scale-invariant (relative to torso
    length), not just comparing an absolute shoulder_lateral_span value
    that shrinks with distance regardless of facing direction."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = [(True, frame), (False, None)]
    mocker.patch("core.pose_engine.cv2.VideoCapture", return_value=mock_cap)

    engine = PoseEngine(backend=_DistantFacingCameraSegmentingBackend())
    received = []
    engine.subscribe(lambda pose_frame, annotated: received.append((pose_frame, annotated)))

    engine._running = True
    engine._run(source=0)

    assert len(received) == 1
    pose_frame, annotated = received[0]
    assert pose_frame.spine_curvature_ratio is None
    assert annotated.max() == 0


def test_run_shifts_scan_window_toward_shoulder_before_cropping(mocker):
    """The hip/shoulder points handed to crop_and_rotate_roi must be shifted
    uniformly toward the shoulder/neck side by _spine_scan_shift_frac of the
    chord length -- not the raw joint keypoints -- so the sampled window
    doesn't run exactly hip-joint-to-shoulder-joint (over-including the
    buttocks at the bottom, excluding the neck at the top)."""
    import math
    from core.spine_contour import crop_and_rotate_roi as real_crop_and_rotate_roi

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = [(True, frame), (False, None)]
    mocker.patch("core.pose_engine.cv2.VideoCapture", return_value=mock_cap)

    captured = {}

    def spy_crop(frame_arg, hip_px, shoulder_px):
        captured["hip_px"] = hip_px
        captured["shoulder_px"] = shoulder_px
        return real_crop_and_rotate_roi(frame_arg, hip_px, shoulder_px)

    mocker.patch("core.pose_engine.crop_and_rotate_roi", side_effect=spy_crop)

    engine = PoseEngine(backend=_SegmentingBackend())
    engine._spine_scan_shift_frac = 0.12
    # Pin the standing fraction to the same value too -- this test is about the
    # shift-toward-shoulder mechanics, not the standing/bent interpolation
    # (see _spine_scan_shift_frac's own dedicated tests), and _SegmentingBackend
    # doesn't provide a knee keypoint so hip_bend_2d guards to 0.0 (would
    # otherwise select the standing fraction here).
    engine._spine_scan_shift_frac_standing = 0.12
    engine._running = True
    engine._run(source=0)

    # From _SegmentingBackend: shoulder=(0.5, 0.2) -> px (320, 96);
    # hip=(0.5, 0.6) -> px (320, 288) on a 480x640 frame.
    raw_hip_px = (320.0, 288.0)
    raw_shoulder_px = (320.0, 96.0)
    dx = raw_shoulder_px[0] - raw_hip_px[0]
    dy = raw_shoulder_px[1] - raw_hip_px[1]
    expected_hip = (raw_hip_px[0] + dx * 0.12, raw_hip_px[1] + dy * 0.12)
    expected_shoulder = (raw_shoulder_px[0] + dx * 0.12, raw_shoulder_px[1] + dy * 0.12)

    assert captured["hip_px"] == pytest.approx(expected_hip)
    assert captured["shoulder_px"] == pytest.approx(expected_shoulder)
    # A pure translation must not change the chord length.
    raw_chord = math.hypot(dx, dy)
    shifted_dx = captured["shoulder_px"][0] - captured["hip_px"][0]
    shifted_dy = captured["shoulder_px"][1] - captured["hip_px"][1]
    assert math.hypot(shifted_dx, shifted_dy) == pytest.approx(raw_chord)



def test_spine_scan_shift_frac_defaults_to_constant(monkeypatch):
    from core.pose_engine import _SPINE_SCAN_SHIFT_FRAC
    monkeypatch.delenv("SPINE_SCAN_SHIFT_FRAC", raising=False)
    engine = PoseEngine(backend=_RaisingSegBackend())
    assert engine._spine_scan_shift_frac == _SPINE_SCAN_SHIFT_FRAC


def test_spine_scan_shift_frac_reads_from_env(monkeypatch):
    monkeypatch.setenv("SPINE_SCAN_SHIFT_FRAC", "0.2")
    engine = PoseEngine(backend=_RaisingSegBackend())
    assert engine._spine_scan_shift_frac == pytest.approx(0.2)


def test_spine_zone_display_fraction_is_independent_of_measurement_fraction(monkeypatch):
    """Widening the on-screen zone-coloring fraction must NOT widen the
    fraction used by the actual angle measurement -- see
    _SPINE_ZONE_FRACTION's docstring for the measured cliff this
    separation exists to prevent."""
    from core.pose_engine import _SPINE_ZONE_FRACTION, _SPINE_ZONE_DISPLAY_FRACTION
    monkeypatch.delenv("SPINE_ZONE_FRACTION", raising=False)
    monkeypatch.setenv("SPINE_ZONE_DISPLAY_FRACTION", "0.45")
    engine = PoseEngine(backend=_RaisingSegBackend())
    assert engine._spine_zone_fraction == pytest.approx(_SPINE_ZONE_FRACTION)
    assert engine._spine_zone_display_fraction == pytest.approx(0.45)
    assert engine._spine_zone_fraction != engine._spine_zone_display_fraction
