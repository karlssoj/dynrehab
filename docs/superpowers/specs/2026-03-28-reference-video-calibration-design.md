# Reference Video Calibration + Prompt Bug Fixes

**Date:** 2026-03-28
**Branch:** feature/physiomotion-ai

---

## Problem

Code generation currently receives only the exercise name, camera view, and physiotherapist
text instructions. It has no real angle data to work from, so it guesses thresholds. This
produces bugs like:

- `_STRETCH_THRESHOLD = 10` with `elbow_angle <= 10` — anatomically unreachable, so reps
  are never counted and the exercise gets stuck in the stretch phase forever.
- `analyze_frame` fires feedback every frame (~30 FPS) with no self-throttling, leaving
  the TTS cooldown as the only guard.
- `on_rep_complete` uses `elif` for independent quality checks, so only one piece of
  feedback can fire per rep.

Additionally, the recorded reference video (filmed by the physiotherapist demonstrating
the correct movement) is stored but never used. It contains exactly the angle data the
LLM needs to set accurate thresholds.

---

## Goals

1. Fix the three prompt bugs so the LLM reliably generates correct code structure.
2. Before generating code, run pose estimation on the reference video to extract
   per-joint angle ranges from the therapist's actual demo.
3. Inject those measurements into the prompt so the LLM uses real thresholds, for any
   exercise type (knee, hip, shoulder, trunk, etc.).

---

## Architecture

```
generate_module(exercise_id, name, camera_view, instructions)
    │
    ├─ ExerciseService.get(exercise_id)
    │       └─ reference_video_path
    │
    ├─ [if video exists] video_analyzer.analyze_reference_video(path)
    │       └─ dict: joint → {min, max, range} for joints with range > 10°
    │
    ├─ build_prompt(name, camera_view, instructions, reference_data)
    │       └─ includes _REFERENCE_DATA_SECTION if reference_data is not None
    │
    └─ Claude API call → validate → store
```

No database schema changes. No UI changes. The generation trigger is unchanged.

---

## Part 1 — Prompt Fixes (`physio_app/services/llm_service.py`)

### 1a. Self-throttling in `analyze_frame`

Add to `_FUNCTION_SPEC`:

> `analyze_frame` is called at ~30 FPS. Use a module-level `_last_feedback_at = 0.0`
> variable and `pose_data["timestamp"]` to throttle: only emit feedback if at least
> 4 seconds have passed since the last emission.

Update `_FEW_SHOT` to show this pattern:

```python
_last_feedback_at = 0.0
_FEEDBACK_COOLDOWN = 4.0

def analyze_frame(pose_data):
    global _last_feedback_at
    now = pose_data.get("timestamp", 0.0)
    feedback = []
    if pose_data["left_shoulder_angle"] < 160:
        if now - _last_feedback_at >= _FEEDBACK_COOLDOWN:
            feedback.append({"message": "Keep your upper arm still", "joint": "left_shoulder"})
            _last_feedback_at = now
    return feedback
```

### 1b. Bidirectional threshold guidance for `detect_rep`

Add to `_FUNCTION_SPEC`:

> For exercises where a joint moves away from its resting position and then returns
> (e.g., bending then extending), the two phase transitions use **opposite** comparisons:
> - Phase 1 complete: angle crosses threshold in the movement direction (e.g., `<= BEND_THRESHOLD`)
> - Phase 2 complete: angle returns past a separate threshold in the opposite direction (e.g., `>= EXTEND_THRESHOLD`)
>
> Never use `<= small_value` to detect a return-to-extension — the arm cannot reach
> near-zero degrees during a normal stretch.

Add a second few-shot example showing a two-phase exercise with correct bidirectional
logic (e.g., a shoulder elevation: angle decreases to bend threshold, then increases
back to an extend threshold).

### 1c. Independent checks in `on_rep_complete`

Add to `_FUNCTION_SPEC`:

> Use separate `if` statements (not `elif`) for independent quality checks, so the
> patient can receive feedback on both insufficient range going down AND insufficient
> range coming back.

---

## Part 2 — Reference Video Calibration

### New: `core/video_analyzer.py`

Single public function:

```python
def analyze_reference_video(video_path: str) -> dict[str, dict]:
```

**Process:**
1. Open `video_path` with `cv2.VideoCapture`.
2. For each frame, run `mediapipe.solutions.pose.Pose` (same model already used by
   `PoseEngine`).
3. Convert landmarks to keypoints dict and call `core.angle_calculator.calculate_angles()`
   — the same function already used by `PoseEngine`. No angle math is duplicated.
4. Accumulate per-joint angle lists.
5. After all frames, compute `min`, `max`, `range` per joint.
6. Return only joints where `range > 10.0` — this filters out joints that barely moved
   and keeps the prompt concise regardless of exercise type.

Returns `{}` if the video file cannot be opened or has no valid pose detections.

**No new dependencies** — `cv2` and `mediapipe` are already used by `PoseEngine`.

### Modified: `build_prompt()` in `physio_app/services/llm_service.py`

Add an optional `reference_data: dict | None = None` parameter.

When present, inject a `_REFERENCE_DATA_SECTION` block after the pose data description:

```
Reference movement data (measured from therapist's demo video):
  left_elbow_angle:    min=44°  max=163°  range=119°
  right_elbow_angle:   min=47°  max=161°  range=114°
  left_shoulder_angle: min=28°  max=171°  range=143°

Use these measured values to calibrate your detection thresholds. Joints not listed
did not move significantly during the demo.
```

### Modified: `generate_module()` in `physio_app/services/llm_service.py`

```python
def generate_module(self, exercise_id, name, camera_view, instructions):
    ex = self.ex_svc.get(exercise_id)
    reference_data = None
    if ex and ex.reference_video_path:
        reference_data = analyze_reference_video(ex.reference_video_path)
    prompt = build_prompt(name, camera_view, instructions, reference_data)
    ...
```

---

## Error Handling

- If `reference_video_path` is empty or the file does not exist: skip analysis, generate
  without reference data (existing behaviour).
- If `analyze_reference_video` raises or returns `{}`: skip the reference section in the
  prompt, generate without it.
- Errors in video analysis are logged but do not block code generation.

---

## Testing

- `tests/core/test_video_analyzer.py`: unit tests using a short synthetic video or
  a fixture MP4 with known pose; assert correct min/max/range extraction and that joints
  with range ≤ 10° are filtered out.
- `tests/physio_app/test_llm_service.py`: extend existing tests — mock
  `analyze_reference_video` to return a known dict, assert the reference section appears
  in `build_prompt` output; assert it is absent when reference data is `None` or `{}`.
- Existing module validator tests are unaffected.

---

## Out of Scope

- Storing the extracted angle summary in the database (can be added later if regeneration
  should reuse it without re-processing).
- Detecting rep count from the reference video automatically.
- Sending video frames to the Claude vision API.
