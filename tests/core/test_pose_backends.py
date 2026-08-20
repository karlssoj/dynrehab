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
