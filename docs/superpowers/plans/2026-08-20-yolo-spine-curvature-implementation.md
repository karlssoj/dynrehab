# YOLO Spine Curvature Estimation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Estimate back curvature (excessive arch "svank" vs. excessive rounding "bula") from the body silhouette between the hip and shoulder keypoints, using YOLO11 segmentation, and expose it as a new `pose_data` field available every ~250ms regardless of movement state.

**Architecture:** A new pure-geometry module (`core/spine_contour.py`) crops/rotates the frame around the hip→shoulder line, gets a segmentation mask from the backend, extracts the back-side silhouette edge, and computes a signed, chord-length-normalized deviation. `core/pose_engine.py` calls this on an independent 4 Hz wall-clock timer (not gated on movement) and writes the result into a new `PoseFrame.spine_curvature_ratio` field, which flows to generated analysis code exactly like every other angle field — no sandbox changes.

**Tech Stack:** Python, OpenCV (`cv2`), NumPy, Ultralytics YOLO11 (`yolo11n-seg.pt`), pytest + pytest-mock.

**Spec:** `docs/superpowers/specs/2026-08-20-yolo-spine-curvature-design.md`

## Global Constraints

- YOLO11 backend only in this plan. `MediaPipeBackend` gets the base class's default (`None`) — no MediaPipe Tasks API migration here.
- No changes to the sandboxed generated-code safe-builtins list (`client_app/services/session_service.py`) — this feature is a plain `pose_data` field, not new sandbox capability.
- Segmentation runs on an **already-cropped** ROI image at `imgsz=320`, never on the full frame.
- Sampling is wall-clock-gated at `_SPINE_SAMPLE_INTERVAL_SECS = 0.25` (4 Hz), independent of movement/rep state.
- Every stage returns `None` on any failure (no person, low confidence, short chord, empty mask, undetectable contour, unknown facing direction) — no exceptions raised from this feature's code paths.
- `_ROI_MARGIN_FRAC`, `_ROI_END_PAD_FRAC`, `_MIN_CHORD_PX`, and the sign convention are stated in the spec as starting points to refine from real footage, not validated constants — implement them exactly as specified; do not attempt to "improve" the numbers during implementation.

---

### Task 1: `facing_left()` in `core/spine_contour.py`

**Files:**
- Create: `core/spine_contour.py`
- Test: `tests/core/test_spine_contour.py`

**Interfaces:**
- Produces: `facing_left(keypoints: dict) -> bool | None`, used by Task 4 and Task 7.

- [ ] **Step 1: Write the failing test**

Create `tests/core/test_spine_contour.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/core/test_spine_contour.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.spine_contour'`

- [ ] **Step 3: Write minimal implementation**

Create `core/spine_contour.py`:

```python
"""Back-contour geometry for estimating spine curvature from a segmentation
mask, given only hip/shoulder keypoints (no dedicated spine landmarks).

See docs/superpowers/specs/2026-08-20-yolo-spine-curvature-design.md.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def facing_left(keypoints: dict) -> bool | None:
    """True if the person's face points toward the image-left half (nose
    x-coordinate is less than shoulder x-coordinate). None if insufficient
    visibility to tell."""
    nose = keypoints.get("nose")
    shoulder = keypoints.get("left_shoulder") or keypoints.get("right_shoulder")
    if not nose or not shoulder or min(nose[3], shoulder[3]) < 0.3:
        return None
    return nose[0] < shoulder[0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/pytest tests/core/test_spine_contour.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add core/spine_contour.py tests/core/test_spine_contour.py
git commit -m "feat: add facing_left() for spine curvature sign convention"
```

---

### Task 2: `crop_and_rotate_roi()` in `core/spine_contour.py`

**Files:**
- Modify: `core/spine_contour.py` (append)
- Test: `tests/core/test_spine_contour.py` (append)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `crop_and_rotate_roi(frame: np.ndarray, hip_px: tuple[float, float], shoulder_px: tuple[float, float]) -> tuple[np.ndarray, tuple[float, float], tuple[float, float], float] | None`, returning `(rotated_image, hip_point, shoulder_point, chord_len)`. Used by Task 3 and Task 7.

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_spine_contour.py`:

```python
import numpy as np

from core.spine_contour import crop_and_rotate_roi, _MIN_CHORD_PX, _ROI_MARGIN_FRAC, _ROI_END_PAD_FRAC


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
```

Add `import cv2` and `import pytest` to the test file's top-level imports (alongside `numpy as np`).

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/core/test_spine_contour.py -v`
Expected: FAIL with `ImportError: cannot import name 'crop_and_rotate_roi'` (and `_MIN_CHORD_PX` etc.)

- [ ] **Step 3: Write minimal implementation**

Append to `core/spine_contour.py`:

```python
_ROI_MARGIN_FRAC = 0.3    # crop half-width, as a fraction of hip-shoulder chord length
_ROI_END_PAD_FRAC = 0.15  # extra length past hip and past shoulder, as a fraction of chord length
_MIN_CHORD_PX = 40        # skip if hip-shoulder pixel distance is smaller than this


def crop_and_rotate_roi(frame: np.ndarray, hip_px: tuple[float, float],
                         shoulder_px: tuple[float, float]
                         ) -> tuple[np.ndarray, tuple[float, float], tuple[float, float], float] | None:
    """Rotate/crop `frame` so the hip->shoulder segment becomes vertical and
    centered. Returns (rotated_image, hip_point, shoulder_point, chord_len) in
    the rotated image's coordinate space, or None if the chord is too short
    to be reliable."""
    hip_x, hip_y = hip_px
    shoulder_x, shoulder_y = shoulder_px
    dx = shoulder_x - hip_x
    dy = shoulder_y - hip_y
    chord_len = math.hypot(dx, dy)
    if chord_len < _MIN_CHORD_PX:
        return None

    angle_deg = math.degrees(math.atan2(dx, -dy))
    mid_x = (hip_x + shoulder_x) / 2.0
    mid_y = (hip_y + shoulder_y) / 2.0

    out_w = int(round(chord_len * (1 + 2 * _ROI_MARGIN_FRAC)))
    out_h = int(round(chord_len * (1 + 2 * _ROI_END_PAD_FRAC)))

    M = cv2.getRotationMatrix2D((mid_x, mid_y), angle_deg, 1.0)
    M[0, 2] += out_w / 2.0 - mid_x
    M[1, 2] += out_h / 2.0 - mid_y

    rotated = cv2.warpAffine(frame, M, (out_w, out_h))

    hip_point = (out_w / 2.0, out_h / 2.0 + chord_len / 2.0)
    shoulder_point = (out_w / 2.0, out_h / 2.0 - chord_len / 2.0)
    return rotated, hip_point, shoulder_point, chord_len
```

This rotation formula has already been verified empirically (a marker placed
at `hip_px`/`shoulder_px` in a synthetic frame lands within ~0.15px of the
computed `hip_point`/`shoulder_point` after the transform) — if Step 4 fails
on the alignment/marker tests, double-check the test's own coordinate setup
before suspecting the formula.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/pytest tests/core/test_spine_contour.py -v`
Expected: 8 passed (4 from Task 1 + 4 new)

- [ ] **Step 5: Commit**

```bash
git add core/spine_contour.py tests/core/test_spine_contour.py
git commit -m "feat: add crop_and_rotate_roi() for spine curvature ROI extraction"
```

---

### Task 3: `extract_back_contour()` in `core/spine_contour.py`

**Files:**
- Modify: `core/spine_contour.py` (append)
- Test: `tests/core/test_spine_contour.py` (append)

**Interfaces:**
- Consumes: `hip_point`/`shoulder_point` shape from Task 2 (a `(x, y)` float tuple pair, with `hip_point[0] == shoulder_point[0]` and `hip_point[1] > shoulder_point[1]`).
- Produces: `extract_back_contour(mask: np.ndarray, hip_point: tuple[float, float], shoulder_point: tuple[float, float]) -> tuple[list[float], list[float]] | None`, returning `(left_profile, right_profile)`. Used by Task 4 and Task 7.

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_spine_contour.py`:

```python
from core.spine_contour import extract_back_contour


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/core/test_spine_contour.py -v`
Expected: FAIL with `ImportError: cannot import name 'extract_back_contour'`

- [ ] **Step 3: Write minimal implementation**

Append to `core/spine_contour.py`:

```python
def extract_back_contour(mask: np.ndarray, hip_point: tuple[float, float],
                          shoulder_point: tuple[float, float]
                          ) -> tuple[list[float], list[float]] | None:
    """Scan each row between shoulder and hip for the mask's edge distance on
    each side of the vertical centerline (x = hip_point[0] ==
    shoulder_point[0]). Returns (left_profile, right_profile), or None if
    fewer than 3 rows have the centerline inside the mask."""
    center_x = int(round(hip_point[0]))
    y_start = int(round(shoulder_point[1]))
    y_end = int(round(hip_point[1]))
    h, w = mask.shape[:2]

    left_profile: list[float] = []
    right_profile: list[float] = []
    for y in range(max(0, y_start), min(h, y_end + 1)):
        row = mask[y]
        if center_x < 0 or center_x >= w or row[center_x] == 0:
            continue

        x = center_x
        while x > 0 and row[x] > 0:
            x -= 1
        left_profile.append(float(center_x - x))

        x = center_x
        while x < w - 1 and row[x] > 0:
            x += 1
        right_profile.append(float(x - center_x))

    if len(left_profile) < 3:
        return None
    return left_profile, right_profile
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/pytest tests/core/test_spine_contour.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add core/spine_contour.py tests/core/test_spine_contour.py
git commit -m "feat: add extract_back_contour() for spine curvature edge profiles"
```

---

### Task 4: `signed_curvature_ratio()` in `core/spine_contour.py`

**Files:**
- Modify: `core/spine_contour.py` (append)
- Test: `tests/core/test_spine_contour.py` (append)

**Interfaces:**
- Consumes: `(left_profile, right_profile)` shape from Task 3 (two same-length lists of float), `facing_left()`'s bool from Task 1.
- Produces: `signed_curvature_ratio(left_profile: list[float], right_profile: list[float], chord_len: float, facing_left: bool) -> float | None`. Used by Task 7.

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_spine_contour.py`:

```python
from core.spine_contour import signed_curvature_ratio


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/core/test_spine_contour.py -v`
Expected: FAIL with `ImportError: cannot import name 'signed_curvature_ratio'`

- [ ] **Step 3: Write minimal implementation**

Append to `core/spine_contour.py`:

```python
def signed_curvature_ratio(left_profile: list[float], right_profile: list[float],
                            chord_len: float, facing_left: bool) -> float | None:
    """Signed, chord-length-normalized peak deviation of the back-side
    silhouette from the straight line between its own two endpoints.
    Positive = outward bulge (excessive rounding / "bula"). Negative =
    inward cave (excessive arch / "svank"). None if the back-side profile
    is empty."""
    back_profile = right_profile if facing_left else left_profile
    n = len(back_profile)
    if n == 0:
        return None

    start, end = back_profile[0], back_profile[-1]

    def straight_at(i: int) -> float:
        if n <= 1:
            return start
        t = i / (n - 1)
        return start + t * (end - start)

    deviations = [back_profile[i] - straight_at(i) for i in range(n)]
    peak = max(deviations, key=abs)
    return peak / chord_len
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/pytest tests/core/test_spine_contour.py -v`
Expected: 16 passed

- [ ] **Step 5: Commit**

```bash
git add core/spine_contour.py tests/core/test_spine_contour.py
git commit -m "feat: add signed_curvature_ratio() for spine svank/bula sign"
```

---

### Task 5: `PoseBackend.get_segmentation_mask()` in `core/pose_backends.py`

**Files:**
- Modify: `core/pose_backends.py:49-51` (add default method to `PoseBackend`), `core/pose_backends.py:138-149` (add `_seg_model`/`_seg_model_path` fields to `YOLOBackend.__init__`), `core/pose_backends.py:196-199` (add `get_segmentation_mask` override before `close`)
- Test: `tests/core/test_pose_backends.py` (extend — reuse existing `_install_fake_ultralytics`/`_make_backend` helpers)

**Interfaces:**
- Consumes: nothing from Tasks 1-4 (independent of `spine_contour.py`).
- Produces: `PoseBackend.get_segmentation_mask(self, roi_image: np.ndarray) -> np.ndarray | None` (default `None`), overridden by `YOLOBackend`. Used by Task 7.

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_pose_backends.py` (the file already has `_install_fake_ultralytics`, `_make_backend`, `MagicMock`, `np` imported):

```python
def test_default_get_segmentation_mask_returns_none():
    from core.pose_backends import PoseBackend

    class _DummyBackend(PoseBackend):
        def process(self, frame, highlight_joints=frozenset()):
            return {}, frame

        def close(self):
            pass

    backend = _DummyBackend()
    assert backend.get_segmentation_mask(np.zeros((10, 10, 3), dtype=np.uint8)) is None


def test_yolo_get_segmentation_mask_returns_resized_binary_mask(monkeypatch):
    model = MagicMock()
    mask_source = np.zeros((160, 160), dtype=np.float32)
    mask_source[40:120, 40:120] = 1.0

    masks = MagicMock()
    masks.data = MagicMock()
    masks.data.shape = (1,)
    single_mask = MagicMock()
    single_mask.cpu.return_value.numpy.return_value = mask_source
    masks.data.__getitem__ = MagicMock(return_value=single_mask)

    result = MagicMock()
    result.masks = masks
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/core/test_pose_backends.py -v -k segmentation_mask`
Expected: FAIL — `AttributeError: 'PoseBackend' object has no attribute 'get_segmentation_mask'` (or equivalent for `_DummyBackend`/`YOLOBackend`)

- [ ] **Step 3: Write minimal implementation**

In `core/pose_backends.py`, add a default method to the `PoseBackend` ABC (after the existing `close` abstract method, `core/pose_backends.py:49-51`):

```python
    @abstractmethod
    def close(self) -> None:
        """Release backend resources."""

    def get_segmentation_mask(self, roi_image: np.ndarray) -> np.ndarray | None:
        """roi_image: an already-cropped BGR image (e.g. the rotated torso
        ROI from spine_contour.crop_and_rotate_roi). Returns a binary mask
        (uint8, 0/255) the same size as roi_image, or None if this backend
        doesn't support segmentation or no person/mask is found. Default:
        unsupported."""
        return None
```

In `YOLOBackend.__init__` (`core/pose_backends.py:138-149`), add two new fields (lazy-loaded model, mirroring the pattern already used for `_filter_state`):

```python
    def __init__(self, model_path: str = "yolo11n-pose.pt",
                 min_cutoff: float = _DEFAULT_MIN_CUTOFF,
                 beta: float = _DEFAULT_BETA,
                 d_cutoff: float = _DEFAULT_D_CUTOFF,
                 hold_max_seconds: float = _DEFAULT_HOLD_MAX_SECONDS,
                 seg_model_path: str = "yolo11n-seg.pt"):
        from ultralytics import YOLO
        self._model = YOLO(model_path)
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._hold_max_seconds = hold_max_seconds
        self._filter_state: dict[str, _KeypointState] = {}
        self._seg_model_path = seg_model_path
        self._seg_model = None
```

Add the override method to `YOLOBackend`, right before its `close` method (`core/pose_backends.py:198`):

```python
    def get_segmentation_mask(self, roi_image: np.ndarray) -> np.ndarray | None:
        if self._seg_model is None:
            from ultralytics import YOLO
            self._seg_model = YOLO(self._seg_model_path)
        results = self._seg_model(roi_image, verbose=False, imgsz=320)
        for r in results:
            if r.masks is None or r.masks.data.shape[0] == 0:
                return None
            mask = r.masks.data[0].cpu().numpy()
            mask_u8 = (mask > 0.5).astype("uint8") * 255
            return cv2.resize(mask_u8, (roi_image.shape[1], roi_image.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
        return None

    def close(self) -> None:
        pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/pytest tests/core/test_pose_backends.py -v`
Expected: all tests pass (existing + 3 new)

- [ ] **Step 5: Commit**

```bash
git add core/pose_backends.py tests/core/test_pose_backends.py
git commit -m "feat: add YOLOBackend.get_segmentation_mask() for spine curvature"
```

---

### Task 6: `PoseFrame.spine_curvature_ratio` field in `core/data_contract.py`

**Files:**
- Modify: `core/data_contract.py:44` (insert new field before `keypoints`)
- Test: `tests/core/test_data_contract.py` (extend)

**Interfaces:**
- Consumes: nothing.
- Produces: `PoseFrame.spine_curvature_ratio: float | None` (default `None`). Used by Task 7 and by generated analysis code via `PoseFrame.to_dict()`.

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_data_contract.py`:

```python
def test_pose_frame_spine_curvature_ratio_defaults_to_none():
    from core.data_contract import PoseFrame
    pf = PoseFrame(timestamp=0.0)
    assert pf.spine_curvature_ratio is None
    assert pf.to_dict()["spine_curvature_ratio"] is None


def test_pose_frame_spine_curvature_ratio_settable():
    from core.data_contract import PoseFrame
    pf = PoseFrame(timestamp=0.0)
    pf.spine_curvature_ratio = 0.12
    assert pf.to_dict()["spine_curvature_ratio"] == 0.12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/core/test_data_contract.py -v -k spine_curvature`
Expected: FAIL with `TypeError: PoseFrame.__init__() got an unexpected keyword argument` or `AttributeError: 'PoseFrame' object has no attribute 'spine_curvature_ratio'`

- [ ] **Step 3: Write minimal implementation**

In `core/data_contract.py`, add the field to the `PoseFrame` dataclass, right before the `keypoints` field (`core/data_contract.py:44-45`):

```python
    body_rotation_z: float = 0.0
    spine_curvature_ratio: float | None = None
    keypoints: dict = field(default_factory=dict)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/pytest tests/core/test_data_contract.py -v`
Expected: all tests pass (existing + 2 new)

- [ ] **Step 5: Commit**

```bash
git add core/data_contract.py tests/core/test_data_contract.py
git commit -m "feat: add PoseFrame.spine_curvature_ratio field"
```

---

### Task 7: Periodic sampling integration in `core/pose_engine.py`

**Files:**
- Modify: `core/pose_engine.py:8` (imports), `core/pose_engine.py:21` (module-level constant + pure function, inserted after the `LANDMARK_NAMES` dict), `core/pose_engine.py:32` (`__init__`), `core/pose_engine.py:86-91` (`_run()`)
- Test: `tests/core/test_pose_engine.py` (new)

**Interfaces:**
- Consumes: `crop_and_rotate_roi`, `extract_back_contour`, `signed_curvature_ratio`, `facing_left` from Tasks 1-4; `PoseBackend.get_segmentation_mask` from Task 5; `PoseFrame.spine_curvature_ratio` from Task 6.
- Produces: `_should_sample_spine(last_sample_time: float, now: float, interval: float = _SPINE_SAMPLE_INTERVAL_SECS) -> bool` (module-level, testable in isolation); `PoseEngine._last_spine_sample: float` instance attribute.

- [ ] **Step 1: Write the failing test**

Create `tests/core/test_pose_engine.py`:

```python
from core.pose_engine import _should_sample_spine, _SPINE_SAMPLE_INTERVAL_SECS


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/core/test_pose_engine.py -v`
Expected: FAIL with `ImportError: cannot import name '_should_sample_spine'`

- [ ] **Step 3: Write minimal implementation**

In `core/pose_engine.py`, add the import (line 8, alongside the existing `core.pose_backends` import):

```python
from core.pose_backends import POSE_CONNECTIONS, PoseBackend, MediaPipeBackend
from core.spine_contour import (crop_and_rotate_roi, extract_back_contour,
                                 signed_curvature_ratio, facing_left)
```

After the `LANDMARK_NAMES` dict (after line 21), add the module-level constant and pure function:

```python
_SPINE_SAMPLE_INTERVAL_SECS = 0.25  # 4 Hz — see spec's measured performance baseline


def _should_sample_spine(last_sample_time: float, now: float,
                          interval: float = _SPINE_SAMPLE_INTERVAL_SECS) -> bool:
    return now - last_sample_time >= interval
```

In `PoseEngine.__init__` (line 32, after `self._seek_start = False`):

```python
        self._seek_start = False
        self._last_spine_sample = 0.0
```

In `PoseEngine._run()`, extend the existing `if keypoints:` block (`core/pose_engine.py:87-91`) — the new block goes inside it, since spine sampling also requires non-empty `keypoints`:

```python
                pose_frame = PoseFrame(timestamp=time.time() - start_time)
                if keypoints:
                    angles = calculate_angles(keypoints)
                    for k, v in angles.items():
                        setattr(pose_frame, k, v)
                    pose_frame.keypoints = keypoints

                    if _should_sample_spine(self._last_spine_sample, frame_start):
                        self._last_spine_sample = frame_start
                        hip = keypoints.get("left_hip") or keypoints.get("right_hip")
                        shoulder = keypoints.get("left_shoulder") or keypoints.get("right_shoulder")
                        if hip and shoulder and min(hip[3], shoulder[3]) > 0.3:
                            h, w = frame.shape[:2]
                            hip_px = (hip[0] * w, hip[1] * h)
                            shoulder_px = (shoulder[0] * w, shoulder[1] * h)
                            roi_result = crop_and_rotate_roi(frame, hip_px, shoulder_px)
                            if roi_result:
                                roi_image, hip_point, shoulder_point, chord_len = roi_result
                                mask = self._backend.get_segmentation_mask(roi_image)
                                if mask is not None:
                                    profiles = extract_back_contour(mask, hip_point, shoulder_point)
                                    if profiles:
                                        face_left = facing_left(keypoints)
                                        if face_left is not None:
                                            left_profile, right_profile = profiles
                                            pose_frame.spine_curvature_ratio = signed_curvature_ratio(
                                                left_profile, right_profile, chord_len, face_left)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/pytest tests/core/test_pose_engine.py -v`
Expected: 5 passed

Then run the full suite to confirm no regressions from the new imports/constructor field:

Run: `.venv/Scripts/pytest tests/ -q`
Expected: same pass/fail count as before this task, plus the new tests — no newly-broken tests.

- [ ] **Step 5: Commit**

```bash
git add core/pose_engine.py tests/core/test_pose_engine.py
git commit -m "feat: sample spine curvature periodically in PoseEngine"
```

---

## Manual verification (after all tasks)

This plan's automated tests cover the geometry and wiring in isolation; they
do not (and cannot, without a camera) prove the feature produces sensible
numbers on a real body. After Task 7:

1. Run the client app against a real side-view webcam feed with a
   YOLO11-backed exercise (e.g. "Squat from side new").
2. Confirm no crashes and no FPS regression versus before this feature
   (spot-check: the skeleton overlay should feel as responsive as it did
   after the earlier jitter-fix work).
3. Temporarily log `pose_frame.spine_curvature_ratio` (e.g. `print()` in the
   `_run()` loop, removed before committing) while performing a known
   exaggerated sway-back and a known exaggerated rounded-back posture.
   Confirm the sign matches the spec's convention (Risk #1) — if inverted,
   flip the `facing_left`/`right_profile` mapping in `signed_curvature_ratio`
   (Task 4), not the sign of the returned ratio, to keep the "positive =
   outward" contract intact for future callers.
4. Re-measure the `imgsz=320` segmentation cost on the actual cropped ROI
   size produced by `crop_and_rotate_roi` (Risk #4) — adjust `imgsz` in
   `YOLOBackend.get_segmentation_mask` (Task 5) if the crop's real aspect
   ratio makes 320 an awkward or slow fit.
