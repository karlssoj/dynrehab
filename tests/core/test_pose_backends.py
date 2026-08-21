import sys
import types

import numpy as np
import pytest
from unittest.mock import MagicMock


def _install_fake_ultralytics(monkeypatch, model_instance):
    fake_module = types.ModuleType("ultralytics")
    fake_module.YOLO = MagicMock(return_value=model_instance)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)


def _make_yolo_result(xyn_row, conf_row):
    """Build a fake ultralytics Results object with one detected person."""
    kp = MagicMock()
    kp.shape = (1,)
    xyn_mock = MagicMock()
    xyn_mock.cpu.return_value.numpy.return_value = np.array(xyn_row, dtype=float)
    kp.xyn = [xyn_mock]
    conf_mock = MagicMock()
    conf_mock.cpu.return_value.numpy.return_value = np.array(conf_row, dtype=float)
    kp.conf = [conf_mock]
    result = MagicMock()
    result.keypoints = kp
    return result


def _no_person_result():
    result = MagicMock()
    result.keypoints = None
    return result


def _frame():
    return np.zeros((100, 100, 3), dtype=np.uint8)


def _full_row(value_for_nose, other=0.5):
    """17 keypoints worth of (x, y) rows / confidences, only 'nose' (index 0) varies."""
    xyn = [list(value_for_nose)] + [[other, other]] * 16
    return xyn


def _make_backend(monkeypatch, model_instance, **kwargs):
    _install_fake_ultralytics(monkeypatch, model_instance)
    from core.pose_backends import YOLOBackend
    return YOLOBackend(**kwargs)


def test_first_detection_not_lagged(monkeypatch):
    model = MagicMock()
    xyn = _full_row((0.3, 0.4))
    conf = [0.9] * 17
    model.return_value = [_make_yolo_result(xyn, conf)]

    backend = _make_backend(monkeypatch, model)
    keypoints, _ = backend.process(_frame())

    x, y, z, c = keypoints["nose"]
    assert x == pytest.approx(0.3)
    assert y == pytest.approx(0.4)
    assert c == pytest.approx(0.9)


def test_confidence_flicker_holds_last_position_within_grace_period(monkeypatch):
    model = MagicMock()
    frame1 = _make_yolo_result(_full_row((0.3, 0.4)), [0.9] * 17)
    frame2 = _make_yolo_result(_full_row((0.9, 0.9)), [0.1] + [0.9] * 16)
    model.side_effect = [[frame1], [frame2]]

    times = iter([0.0, 0.05])
    monkeypatch.setattr("core.pose_backends.time.time", lambda: next(times))

    backend = _make_backend(monkeypatch, model, hold_max_seconds=0.25)
    backend.process(_frame())
    keypoints, _ = backend.process(_frame())

    assert "nose" in keypoints
    x, y, z, c = keypoints["nose"]
    assert x == pytest.approx(0.3)
    assert y == pytest.approx(0.4)
    assert c == pytest.approx(0.9)


def test_keypoint_dropped_after_grace_period_expires(monkeypatch):
    model = MagicMock()
    frame1 = _make_yolo_result(_full_row((0.3, 0.4)), [0.9] * 17)
    frame2 = _make_yolo_result(_full_row((0.9, 0.9)), [0.1] + [0.9] * 16)
    model.side_effect = [[frame1], [frame2]]

    times = iter([0.0, 0.5])
    monkeypatch.setattr("core.pose_backends.time.time", lambda: next(times))

    backend = _make_backend(monkeypatch, model, hold_max_seconds=0.25)
    backend.process(_frame())
    keypoints, _ = backend.process(_frame())

    assert "nose" not in keypoints


def test_reappearance_after_grace_expiry_is_not_lagged(monkeypatch):
    model = MagicMock()
    frame1 = _make_yolo_result(_full_row((0.3, 0.4)), [0.9] * 17)
    frame2 = _make_yolo_result(_full_row((0.9, 0.9)), [0.1] + [0.9] * 16)
    frame3 = _make_yolo_result(_full_row((0.8, 0.7)), [0.9] * 17)
    model.side_effect = [[frame1], [frame2], [frame3]]

    times = iter([0.0, 0.5, 0.55])
    monkeypatch.setattr("core.pose_backends.time.time", lambda: next(times))

    backend = _make_backend(monkeypatch, model, hold_max_seconds=0.25)
    backend.process(_frame())
    backend.process(_frame())
    keypoints, _ = backend.process(_frame())

    x, y, z, c = keypoints["nose"]
    assert x == pytest.approx(0.8)
    assert y == pytest.approx(0.7)


def test_no_person_detected_returns_empty_dict(monkeypatch):
    model = MagicMock()
    model.return_value = [_no_person_result()]

    backend = _make_backend(monkeypatch, model)
    keypoints, annotated = backend.process(_frame())

    assert keypoints == {}
    assert annotated is not None


def test_never_confident_keypoint_omitted(monkeypatch):
    model = MagicMock()
    xyn = _full_row((0.3, 0.4))
    conf = [0.1] + [0.9] * 16  # nose never above threshold
    model.return_value = [_make_yolo_result(xyn, conf)]

    backend = _make_backend(monkeypatch, model)
    keypoints, _ = backend.process(_frame())

    assert "nose" not in keypoints
    assert "left_eye" in keypoints


def test_default_get_segmentation_mask_returns_none():
    from core.pose_backends import PoseBackend

    class _DummyBackend(PoseBackend):
        def process(self, frame, highlight_joints=frozenset()):
            return {}, frame

        def close(self):
            pass

    backend = _DummyBackend()
    assert backend.get_segmentation_mask(np.zeros((10, 10, 3), dtype=np.uint8)) is None


def _make_seg_result(mask_arrays, classes):
    """Build a fake ultralytics segmentation Results object with N detected
    masks (mask_arrays: list of 2D np.ndarray) and their per-detection class
    ids (classes: list[int], same length/order as mask_arrays)."""
    masks = MagicMock()
    masks.data = MagicMock()
    masks.data.shape = (len(mask_arrays),)
    masks.data.cpu.return_value.numpy.return_value = np.stack(mask_arrays)
    masks.data.__len__ = MagicMock(return_value=len(mask_arrays))

    boxes = MagicMock()
    boxes.cls = MagicMock()
    boxes.cls.cpu.return_value.numpy.return_value = np.array(classes, dtype=float)

    result = MagicMock()
    result.masks = masks
    result.boxes = boxes
    return result


def test_yolo_get_segmentation_mask_returns_resized_binary_mask(monkeypatch):
    model = MagicMock()
    mask_source = np.zeros((160, 160), dtype=np.float32)
    mask_source[40:120, 40:120] = 1.0

    result = _make_seg_result([mask_source], classes=[0])  # class 0 = person
    model.return_value = [result]

    backend = _make_backend(monkeypatch, model)
    roi = np.zeros((80, 80, 3), dtype=np.uint8)
    mask = backend.get_segmentation_mask(roi)

    assert mask is not None
    assert mask.shape == (80, 80)
    assert mask.dtype == np.uint8
    assert mask.max() == 255


def test_yolo_get_segmentation_mask_returns_none_when_no_mask(monkeypatch):
    model = MagicMock()
    result = MagicMock()
    result.masks = None
    model.return_value = [result]

    backend = _make_backend(monkeypatch, model)
    roi = np.zeros((80, 80, 3), dtype=np.uint8)
    assert backend.get_segmentation_mask(roi) is None


def test_yolo_get_segmentation_mask_load_failure_returns_none_without_raising(monkeypatch):
    # First YOLO(...) call (in __init__) is the pose model and succeeds;
    # the second (lazy segmentation model load) raises.
    fake_module = types.ModuleType("ultralytics")
    pose_model = MagicMock()
    fake_module.YOLO = MagicMock(side_effect=[pose_model, RuntimeError("no weights")])
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)

    from core.pose_backends import YOLOBackend
    backend = YOLOBackend()

    roi = np.zeros((80, 80, 3), dtype=np.uint8)
    assert backend.get_segmentation_mask(roi) is None


def test_yolo_get_segmentation_mask_load_failure_is_latched_not_retried(monkeypatch):
    fake_module = types.ModuleType("ultralytics")
    pose_model = MagicMock()
    fake_module.YOLO = MagicMock(side_effect=[pose_model, RuntimeError("no weights")])
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)

    from core.pose_backends import YOLOBackend
    backend = YOLOBackend()

    roi = np.zeros((80, 80, 3), dtype=np.uint8)
    assert backend.get_segmentation_mask(roi) is None
    assert fake_module.YOLO.call_count == 2  # pose model (ctor) + failed seg model attempt

    # Second call must not attempt to construct YOLO again.
    assert backend.get_segmentation_mask(roi) is None
    assert fake_module.YOLO.call_count == 2


def test_yolo_get_segmentation_mask_filters_out_non_person_class(monkeypatch):
    model = MagicMock()
    mask_source = np.zeros((160, 160), dtype=np.float32)
    mask_source[40:120, 40:120] = 1.0

    # Only detection is class 57 ("chair") -- must not be treated as the
    # back profile even though it's the only/first mask.
    result = _make_seg_result([mask_source], classes=[57])
    model.return_value = [result]

    backend = _make_backend(monkeypatch, model)
    roi = np.zeros((80, 80, 3), dtype=np.uint8)
    assert backend.get_segmentation_mask(roi) is None


def test_yolo_get_segmentation_mask_prefers_person_over_other_class(monkeypatch):
    model = MagicMock()
    # Chair mask (class 57) is listed first / larger area, person mask
    # (class 0) is second / smaller area -- person must still win because
    # only person-class detections are eligible.
    chair_mask = np.ones((160, 160), dtype=np.float32)
    person_mask = np.zeros((160, 160), dtype=np.float32)
    person_mask[60:100, 60:100] = 1.0

    result = _make_seg_result([chair_mask, person_mask], classes=[57, 0])
    model.return_value = [result]

    backend = _make_backend(monkeypatch, model)
    roi = np.zeros((80, 80, 3), dtype=np.uint8)
    mask = backend.get_segmentation_mask(roi)

    assert mask is not None
    # person_mask covers a 40x40 region out of 160x160, scaled to an 80x80
    # roi -> a 20x20 region of 255s, not the full chair-sized mask.
    assert (mask == 255).sum() == pytest.approx(20 * 20, abs=4)


def test_draw_back_contour_points_draws_a_dot_at_each_point():
    from core.pose_backends import draw_back_contour_points
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    points = [(20.0, 20.0), (50.0, 50.0), (80.0, 20.0)]
    draw_back_contour_points(frame, points)
    for x, y in points:
        region = frame[int(y) - 3:int(y) + 3, int(x) - 3:int(x) + 3]
        assert region.max() > 0
    # No line drawn between points -- the midpoint between the first two
    # points should stay background.
    mid_x, mid_y = 35, 35
    assert frame[mid_y, mid_x].max() == 0


def test_draw_back_contour_points_draws_single_dot_for_one_point():
    from core.pose_backends import draw_back_contour_points
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    draw_back_contour_points(frame, [(50.0, 50.0)])
    assert frame[47:54, 47:54].max() > 0


def test_draw_back_contour_points_no_op_for_empty_list():
    from core.pose_backends import draw_back_contour_points
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    draw_back_contour_points(frame, [])
    assert frame.max() == 0
    assert frame.max() == 0
