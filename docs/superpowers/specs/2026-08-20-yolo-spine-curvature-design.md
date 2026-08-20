# Spine Curvature Estimation via Segmentation Contour (YOLO11) — Design

## Context

Physiotherapy exercises analyzed from a side-view camera need to assess spinal
posture — specifically whether the back stays reasonably neutral, or shows
excessive lumbar arch ("svank"/lordosis) or excessive rounding ("bula"/kyphosis)
during movement (e.g. a squat) or while standing between reps.

Neither pose backend exposes spine keypoints. YOLO11 provides only the 17 COCO
keypoints (no spine, no feet). A dedicated "spinepose" landmark model was tried
and found unreliable. `trunk_lean_2d` (the existing hip→shoulder straight-line
angle) only captures overall trunk *tilt*, not back *curvature* — a person can
have a perfectly reasonable trunk lean angle while their back is rounded or
overly arched, and `trunk_lean_2d` cannot see that.

This document designs a way to estimate back curvature from the **shape of the
body silhouette** between the hip and shoulder keypoints, using person
**segmentation** (not landmark detection) to sidestep the "find spine points
through clothing" problem entirely — the method only needs the visible
silhouette's shape, not discrete anatomical points.

## Scope

**In scope:** YOLO11 backend only. `MediaPipeBackend` gets a no-op stub for the
same interface, so the feature degrades to "unavailable" (not "crashes") on
MediaPipe-backed exercises.

**Explicitly out of scope for this design (follow-up work):**
- MediaPipe segmentation support (blocked on migrating `MediaPipeBackend` from
  the legacy `mp.solutions` API to the new Tasks API — a separate, unrelated
  pre-existing issue discovered during this design's research, tracked
  separately from this feature).
- Any change to per-exercise generated analysis code's sandbox or safe-builtins
  list — this feature produces a plain numeric `pose_data` field, same shape as
  every existing angle field, so generated code needs no new capability to use it.

## Performance baseline (measured, not assumed)

Benchmarked on `yolo11n-pose.pt` / `yolo11n-seg.pt` against an existing
reference video (`data/exercises/086ffb08-1ca9-4f05-9b23-6c5b765ba3aa/reference.mp4`),
using the `anaconda3/envs/mediapipe` conda environment — confirmed by direct
inspection to be the one with both `mediapipe` (with the legacy `solutions`
API the current `MediaPipeBackend` needs) and `ultralytics` installed, and
therefore the most likely candidate for the actual app runtime, though this
should be confirmed against the real launch command/shortcut used on the
target machine before relying on it further:

| Configuration | ms/frame | FPS |
|---|---|---|
| Pose only (current) | 56 | 17.9 |
| Pose + segmentation, every frame, full res | 113 | 8.9 |
| Segmentation alone, `imgsz=320` | 27 | 37.7 |

Running segmentation every frame roughly halves throughput — unacceptable on
the constrained QTrobot target hardware (this benchmark ran on a desktop CPU;
the robot's onboard compute is expected to be weaker still). At `imgsz=320` a
single segmentation pass costs ~27ms. The design below runs segmentation
**periodically** (time-gated, independent of movement/rep state — a person
standing still between reps still gets periodic posture samples), not every
frame, keeping the added average load small regardless of absolute per-call cost.

## Architecture

```
core/pose_engine.py (PoseEngine._run loop, every frame)
  │
  ├─ backend.process(frame) ──────────► keypoints (every frame, unchanged)
  ├─ calculate_angles(keypoints) ─────► angles dict (every frame, unchanged)
  │
  └─ every ~250ms (wall-clock, independent of movement state), if hip and
     shoulder keypoints are both confident:
        core/spine_contour.py:
          crop_and_rotate_roi(frame, hip_px, shoulder_px)
              ─► (rotated ROI image, hip_point, shoulder_point, chord_len)
        backend.get_segmentation_mask(rotated ROI image) ─► binary mask
              │
              ▼ (None if backend doesn't support it, or no mask found)
        core/spine_contour.py:
          extract_back_contour(mask, hip_point, shoulder_point)
              ─► (left_profile, right_profile)
          facing_left(keypoints) ─► bool
          signed_curvature_ratio(left_profile, right_profile, chord_len, facing_left)
              ─► float
              │
              ▼
        PoseFrame.spine_curvature_ratio = float | None
```

`PoseFrame.spine_curvature_ratio` flows to generated analysis code exactly
like every other field, via the existing `PoseFrame.to_dict()` →
`SessionService.process_frame(pose_data)` path (`client_app/ui/session_view.py:132`).
No sandbox changes needed.

## Components

### 1. `core/pose_backends.py` — `PoseBackend.get_segmentation_mask()`

Add a **non-abstract** method to the `PoseBackend` base class with a default
implementation returning `None` (graceful "unsupported" — `MediaPipeBackend`
inherits this default unchanged). The method takes an **already-cropped**
image, not the full frame — cropping to the torso region happens first, in
`core/spine_contour.py` (next section), so the segmentation model spends its
resolution budget on the torso instead of mostly-irrelevant background. This
is what makes `imgsz=320` viable without shrinking an already-small torso
region even further:

```python
def get_segmentation_mask(self, roi_image: np.ndarray) -> np.ndarray | None:
    """roi_image: an already-cropped BGR image (e.g. the rotated torso ROI
    from spine_contour.crop_and_rotate_roi). Returns a binary mask (uint8,
    0/255) the same size as roi_image, or None if no person/mask found.
    Default: unsupported."""
    return None
```

`YOLOBackend` overrides it:

```python
def get_segmentation_mask(self, roi_image: np.ndarray) -> np.ndarray | None:
    if self._seg_model is None:
        from ultralytics import YOLO
        self._seg_model = YOLO(self._seg_model_path)   # lazy-loaded, e.g. "yolo11n-seg.pt"
    results = self._seg_model(roi_image, verbose=False, imgsz=320)
    for r in results:
        if r.masks is None or r.masks.data.shape[0] == 0:
            return None
        mask = r.masks.data[0].cpu().numpy()             # (h', w') float 0..1, model's internal size
        mask_u8 = (mask > 0.5).astype("uint8") * 255
        return cv2.resize(mask_u8, (roi_image.shape[1], roi_image.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
    return None
```

`_seg_model` / `_seg_model_path` are new `YOLOBackend.__init__` fields
(`_seg_model = None`, `_seg_model_path: str = "yolo11n-seg.pt"`, constructor
param with that default — same lazy-load-on-first-use pattern the pose model
itself does not currently need, since it's already loaded eagerly in
`__init__`, but the seg model should load lazily so exercises/backends that
never call `get_segmentation_mask` never pay the extra ~6MB weight download or
memory cost).

The caller (`core/spine_contour.py`, see below) is responsible for producing
the crop; `YOLOBackend` just segments whatever image it's given.

### 2. `core/spine_contour.py` (new module)

Pure, backend-agnostic geometry — takes a frame + two keypoints + a mask
function, returns a signed ratio. No YOLO- or MediaPipe-specific code lives
here; it only calls the `get_segmentation_mask` callable it's given.

**Constants** (tune from real footage, same "informed starting point, not a
validated constant" status as the existing `_knee_forward_ratio` 0.35
threshold):

```python
_ROI_MARGIN_FRAC = 0.3   # crop half-width, as a fraction of hip-shoulder chord length
_ROI_END_PAD_FRAC = 0.15 # extra length past hip and past shoulder, as a fraction of chord length
_MIN_CHORD_PX = 40       # skip if hip-shoulder pixel distance is smaller than this (too small/far to be reliable)
```

**`crop_and_rotate_roi(frame, hip_px, shoulder_px) -> tuple[np.ndarray, tuple[float, float], tuple[float, float], float] | None`**

Returns `(rotated_image, hip_point, shoulder_point, chord_len)` — the two
points are explicit pixel coordinates *in the rotated output image*, not
re-derived later from `chord_len` and the padding constants. Keeping them
explicit avoids `extract_back_contour` needing to privately know
`_ROI_END_PAD_FRAC` just to agree with this function about where the hip and
shoulder ended up — the two functions would otherwise be coupled through a
shared constant instead of a parameter.

Algorithm:
1. `chord = shoulder_px - hip_px`; `chord_len = hypot(*chord)`. If `chord_len < _MIN_CHORD_PX`, return `None`.
2. `angle_deg` = angle needed to rotate `chord` to point straight up (negative-y direction in image coordinates: `math.degrees(math.atan2(chord.x, -chord.y))` — rotating by this angle around the midpoint maps the hip→shoulder vector onto the vertical axis).
3. `midpoint = (hip_px + shoulder_px) / 2`.
4. `M = cv2.getRotationMatrix2D(midpoint, angle_deg, 1.0)`.
5. Output canvas size: width = `chord_len * (1 + 2*_ROI_MARGIN_FRAC)`, height = `chord_len * (1 + 2*_ROI_END_PAD_FRAC)`.
6. Adjust `M`'s translation terms so the midpoint lands at the center of the output canvas (standard "rotate about a point into a fixed-size output" adjustment — add `(out_w/2 - midpoint.x, out_h/2 - midpoint.y)` to `M`'s translation column after rotation).
7. `rotated = cv2.warpAffine(frame, M, (out_w, out_h))`.
8. `hip_point = (out_w/2, out_h/2 + chord_len/2)`, `shoulder_point = (out_w/2, out_h/2 - chord_len/2)` — return `(rotated, hip_point, shoulder_point, chord_len)`.

**`extract_back_contour(mask, hip_point, shoulder_point) -> tuple[list[float], list[float]] | None`**

Given the mask from the rotated ROI (person's vertical axis is now the
canvas's vertical center line, at `hip_point[0]` == `shoulder_point[0]`):
1. For each row `y` from `shoulder_point[1]` to `hip_point[1]`, scan outward
   from the centerline (`x = hip_point[0]`) in **both** directions to find
   the mask edge (255→0 transition scanning outward from center — chosen
   over scanning inward from the crop edges for robustness against noise
   right at the crop boundary).
2. This produces two edge-distance profiles (left side, right side) as a
   function of `y`. Return **both** (`left_profile`, `right_profile`) —
   `signed_curvature_ratio` (next function) decides which one is the "back"
   using the facing-direction signal, so this function stays agnostic to
   facing direction.
3. If fewer than 3 rows have a detectable edge on either side (e.g. mask is
   empty or the crop missed the person), return `None`.

**`signed_curvature_ratio(left_profile, right_profile, chord_len, facing_left: bool) -> float | None`**

1. `back_profile = right_profile if facing_left else left_profile` — when the
   person faces left in the original frame (nose/ear x < shoulder x), their
   back is on the image-right side of the rotated vertical axis, and vice
   versa. (`facing_left` is computed by the caller from the nose/ear vs
   shoulder keypoints — see `facing_left()` below.)
2. `straight_at(y) = linear interpolation between back_profile[0] and
   back_profile[-1]` — the reference "neutral" line is the chord between the
   profile's own two endpoints (matching the established
   `_knee_forward_ratio` philosophy of measuring deviation from the
   hip↔shoulder chord), **not** the profile's mean. A mean-based reference
   would self-normalize away a real, uniform bulge (comparing the profile to
   its own average always nets close to zero) — the endpoint-chord reference
   avoids that.
3. `deviations = [back_profile[i] - straight_at(y_i) for i in range(len(back_profile))]`
4. `peak = max(deviations, key=abs)` (largest-magnitude deviation, signed).
5. Return `peak / chord_len` (normalized, scale-independent, same units
   convention as `_knee_forward_ratio`).
6. **Sign convention:** positive = back-side silhouette bulges *outward*
   relative to the hip-shoulder chord (toward the camera-visible back side) —
   interpreted as excessive rounding ("bula"/kyphosis). Negative = the
   back-side silhouette caves *inward* relative to the chord — interpreted as
   excessive arch ("svank"/lordosis). This sign mapping is the single
   riskiest unvalidated assumption in this design (see Risks) — confirm it
   against real footage of a known-svank and a known-bula posture before
   trusting the sign in generated feedback text.

**`facing_left(pose_data_or_keypoints: dict) -> bool | None`**

```python
def facing_left(keypoints: dict) -> bool | None:
    """True if the person's face points toward the image-left half (nose/ear
    x-coordinate is less than shoulder x-coordinate). None if insufficient
    visibility to tell."""
    nose = keypoints.get("nose")
    shoulder = keypoints.get("left_shoulder") or keypoints.get("right_shoulder")
    if not nose or not shoulder or min(nose[3], shoulder[3]) < 0.3:
        return None
    return nose[0] < shoulder[0]
```

### 3. `core/pose_engine.py` — periodic sampling

New module-level constant: `_SPINE_SAMPLE_INTERVAL_SECS = 0.25` (4 Hz — cheap
enough per the benchmark above to run continuously at this rate: ~27ms every
250ms is ~11% average overhead, negligible next to the existing 56ms/frame
pose cost, and independent of the free-running variable frame rate since it's
gated by wall-clock time, not frame count).

Extract the sampling decision into a small pure function (unit-testable
without threading/camera/model concerns, mirroring the
`_KeypointState`/`OneEuroFilter` pattern from the earlier jitter-fix work):

```python
def _should_sample_spine(last_sample_time: float, now: float,
                          interval: float = _SPINE_SAMPLE_INTERVAL_SECS) -> bool:
    return now - last_sample_time >= interval
```

In `PoseEngine._run()`, after `keypoints, annotated = self._backend.process(...)`
and building `pose_frame`:

```python
if keypoints and _should_sample_spine(self._last_spine_sample, frame_start):
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

`self._last_spine_sample = 0.0` added to `PoseEngine.__init__`.

`hip`/`shoulder` selection above uses the existing codebase convention of
`kpts.get("left_x") or kpts.get("right_x")` (prefer left, fall back to right
if left is entirely absent from the dict) — the same simple fallback already
used elsewhere (e.g. the hip-height read in the deployed squat module). This
is not a true confidence comparison between sides, just "use left unless it's
missing." The single-frame-independent leg-selection lesson from the
knee-forward-ratio fix does not apply here regardless — that fix mattered
because *one bad frame* could flip a per-rep verdict; this runs on an
independent 4 Hz clock with no per-rep aggregation, so an occasional
sample from the less-visible side just means that one sample is noisier,
not that a whole rep's verdict flips.

### 4. `core/data_contract.py` — new field

```python
spine_curvature_ratio: float | None = None
```

Added to `PoseFrame`, alongside the other computed fields. `None` means "not
sampled this frame" (the common case — most frames fall between samples).

## Data flow summary

Every frame: keypoints + angles computed as today, unchanged, unaffected speed.
Every ~250ms: additionally, a cropped/rotated segmentation pass populates
`spine_curvature_ratio` for that one frame; all other frames carry `None` for
that field. Generated analysis code (any exercise, once written to use it)
filters `None` the same way it already filters `trunk_lean_2d <= 0` or
`_knee_forward_ratio <= 0.05` — an established, existing pattern, not a new one.

## Error handling / graceful degradation

Every stage returns `None` on failure rather than raising: no person detected,
hip/shoulder not confident, chord too short, segmentation mask empty, contour
unextractable, facing direction unknown. The engine simply leaves
`spine_curvature_ratio` as `None` for that sample and tries again at the next
4 Hz tick — no retries, no exceptions, no visible failure mode to the user
beyond "this field is occasionally absent for a bit longer," identical in
spirit to how the rest of the pipeline already treats low-confidence frames.

## Testing plan

- **`tests/core/test_spine_contour.py`** (new, pure functions, no mocking):
  - `crop_and_rotate_roi`: given known hip/shoulder pixel coordinates, verify
    output canvas dimensions and that a synthetic point at the hip/shoulder
    location maps to the expected fixed output coordinates.
  - `crop_and_rotate_roi` returns `None` when chord length < `_MIN_CHORD_PX`.
  - `extract_back_contour`: construct a synthetic binary mask with a known
    bulge on one side (e.g. a rectangle with one edge displaced outward in
    the middle rows) and verify the returned profile reflects it; verify
    `None` on an all-background (empty) mask.
  - `signed_curvature_ratio`: straight-sided synthetic profile → ratio ≈ 0;
    profile bulged on the "back" side (per a fixed `facing_left`) → positive;
    profile caved on the "back" side → negative; verify the sign flips when
    `facing_left` flips with the same raw profiles (proves the sign
    convention is actually driven by facing direction, not hardcoded).
  - `facing_left`: known nose/shoulder positions → correct bool; low
    visibility → `None`.
- **`tests/core/test_pose_backends.py`** (extend): `YOLOBackend.get_segmentation_mask`
  using the same `sys.modules["ultralytics"]` stubbing pattern already
  established for the pose model — mock a `masks.data` result, verify a
  resized binary mask comes back; verify `None` on no detection.
  `MediaPipeBackend.get_segmentation_mask` (inherited default) returns `None`.
- **`tests/core/test_pose_engine.py`** (new — `pose_engine.py` currently has
  zero coverage): unit test `_should_sample_spine` directly (pure function,
  no threading) across boundary cases (`now - last == interval`,
  `now - last < interval`, `last == 0.0` i.e. first-ever call).

## Risks / open questions to validate against real footage

1. **Sign convention** (svank vs. bula) is a reasoned-but-unverified
   assumption — must be checked against footage of both known-bad postures
   before trusting the sign in user-facing feedback text.
2. **ROI margin constants** (`_ROI_MARGIN_FRAC`, `_ROI_END_PAD_FRAC`) are
   starting guesses, not measured — likely need adjustment once real
   silhouettes are inspected.
3. **Loose clothing** remains a fundamental limitation of any vision-based
   method, contour or keypoint — expectation-setting only, not solvable in code.
4. **`imgsz` for the crop**: the benchmark measured `imgsz=320` on a
   *full frame*, not yet on a pre-cropped torso ROI (which will be smaller and
   differently-proportioned) — re-measure once the crop is implemented, and
   adjust `imgsz` if the crop's aspect ratio makes 320 an awkward fit.
