# Reference Video Calibration + Prompt Bug Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three structural bugs in the LLM code-generation prompt and add a reference video calibration step that extracts real joint angle measurements from the therapist's demo video before generating analysis code.

**Architecture:** A new `core/video_analyzer.py` module runs MediaPipe pose estimation on `reference.mp4`, aggregates per-joint angle statistics, and filters to joints with range > 10°. `LLMService.generate_module()` calls this before building the prompt, and `build_prompt()` gains an optional `reference_data` parameter that injects a measured-values section into the prompt. Prompt bug fixes update `_FEW_SHOT` and `_FUNCTION_SPEC` so the LLM learns self-throttling, bidirectional thresholds, and independent `if` statements.

**Tech Stack:** Python, mediapipe, cv2 (both already present), anthropic SDK, pytest + pytest-mock

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `core/video_analyzer.py` | Read reference.mp4, run MediaPipe, return angle stats dict |
| Create | `tests/core/test_video_analyzer.py` | Unit tests for analyze_reference_video |
| Modify | `physio_app/services/llm_service.py` | Fix prompt constants; add reference_data to build_prompt; wire video analyzer into generate_module |
| Modify | `tests/physio_app/test_llm_service.py` | Tests for prompt fixes and reference data injection |

---

## Task 1: Fix prompt bugs in llm_service.py

**Files:**
- Modify: `physio_app/services/llm_service.py:7-103`
- Modify: `tests/physio_app/test_llm_service.py`

### Bug summary
1. `analyze_frame` has no self-throttling — fires every frame at 30 FPS.
2. Bidirectional threshold guidance is absent — LLM invents `<= 10` for return-to-extension, which is anatomically unreachable.
3. `on_rep_complete` uses `elif` for independent checks — only one feedback message can fire per rep.

- [ ] **Step 1: Write failing tests for the prompt fixes**

Add to `tests/physio_app/test_llm_service.py`:

```python
def test_build_prompt_contains_throttle_pattern():
    prompt = build_prompt("Squat", "side", "instructions")
    assert "_last_feedback_at" in prompt
    assert "_FEEDBACK_COOLDOWN" in prompt


def test_build_prompt_contains_bidirectional_threshold_guidance():
    prompt = build_prompt("Squat", "side", "instructions")
    assert "EXTEND_THRESHOLD" in prompt


def test_build_prompt_contains_independent_if_guidance():
    prompt = build_prompt("Squat", "side", "instructions")
    assert "elif" not in prompt or "not elif" in prompt.lower() or "separate if" in prompt.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/physio_app/test_llm_service.py::test_build_prompt_contains_throttle_pattern tests/physio_app/test_llm_service.py::test_build_prompt_contains_bidirectional_threshold_guidance tests/physio_app/test_llm_service.py::test_build_prompt_contains_independent_if_guidance -v
```

Expected: all three FAIL.

- [ ] **Step 3: Replace `_FUNCTION_SPEC` and `_FEW_SHOT` in llm_service.py**

Replace the entire `_FUNCTION_SPEC` constant (lines 27–44):

```python
_FUNCTION_SPEC = """\
Implement exactly these three functions:

def analyze_frame(pose_data: dict) -> list[dict]:
    # Called every frame (~30 FPS). Return feedback dicts or [].
    # Each dict: {"message": str, "joint": str | None}
    # "joint" is a landmark name to highlight (e.g. "left_knee") or None.
    # IMPORTANT: use a module-level _last_feedback_at = 0.0 variable and
    # pose_data["timestamp"] to throttle — only emit feedback if at least
    # _FEEDBACK_COOLDOWN seconds have passed since the last emission.
    # This prevents the same message from firing 30 times per second.

def detect_rep(pose_data: dict) -> bool:
    # Called every frame. Return True exactly once when a full rep is completed.
    # Use module-level state variables to track the rep phase.
    # For exercises where a joint moves away from resting and then returns,
    # the two phase transitions use OPPOSITE comparisons:
    #   Phase 1 complete: angle crosses threshold in movement direction (e.g. <= BEND_THRESHOLD)
    #   Phase 2 complete: angle returns past a separate EXTEND_THRESHOLD  (e.g. >= EXTEND_THRESHOLD)
    # NEVER use <= small_value to detect a return-to-extension.
    # The return phase ends when the angle rises back ABOVE a large threshold (~160°+).

def on_rep_complete(rep_data: dict) -> list[dict]:
    # Called once after detect_rep returns True.
    # rep_data: {"rep_number": int, "frames": list[dict], "duration_seconds": float}
    # Return feedback dicts or [].
    # Use separate if statements (NOT elif) for independent quality checks —
    # the patient may need feedback on both insufficient bend AND insufficient extension.
"""
```

Replace the entire `_FEW_SHOT` constant (lines 46–77):

```python
_FEW_SHOT = """\
Example 1 — Side-view bicep curl:

```python
_phase = "down"
_CURL_UP = 60        # angle at peak curl (arm bent)
_CURL_DOWN = 150     # angle at full extension — use >= to detect the return
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

def detect_rep(pose_data):
    global _phase
    avg = (pose_data["left_elbow_angle"] + pose_data["right_elbow_angle"]) / 2
    if _phase == "down" and avg <= _CURL_UP:       # angle DECREASES into the curl
        _phase = "up"
    elif _phase == "up" and avg >= _CURL_DOWN:     # angle INCREASES back to extension
        _phase = "down"
        return True
    return False

def on_rep_complete(rep_data):
    frames = rep_data["frames"]
    min_elbow = min(f["left_elbow_angle"] for f in frames)
    max_elbow = max(f["left_elbow_angle"] for f in frames)
    feedback = []
    if min_elbow > 70:
        feedback.append({"message": "Curl higher for full range of motion", "joint": None})
    if max_elbow < 140:                            # separate if, NOT elif
        feedback.append({"message": "Fully extend your arm at the bottom", "joint": None})
    if not feedback:
        feedback.append({"message": "Great curl!", "joint": None})
    return feedback
```

Example 2 — Side-view knee bend (angle decreases into bend, increases on return):

```python
_phase = "up"
_BEND_THRESHOLD = 100    # knee angle at deepest bend — phase switches when <= this
_EXTEND_THRESHOLD = 160  # knee angle at full extension — phase switches when >= this (NOT a small value)
_last_feedback_at = 0.0
_FEEDBACK_COOLDOWN = 4.0

def analyze_frame(pose_data):
    global _last_feedback_at
    now = pose_data.get("timestamp", 0.0)
    feedback = []
    if pose_data.get("trunk_lean_angle", 0) > 30:
        if now - _last_feedback_at >= _FEEDBACK_COOLDOWN:
            feedback.append({"message": "Keep your back straight", "joint": None})
            _last_feedback_at = now
    return feedback

def detect_rep(pose_data):
    global _phase
    angle = pose_data.get("left_knee_angle", 180)
    if _phase == "up" and angle <= _BEND_THRESHOLD:       # angle DECREASES into bend
        _phase = "down"
    elif _phase == "down" and angle >= _EXTEND_THRESHOLD: # angle INCREASES back up
        _phase = "up"
        return True
    return False

def on_rep_complete(rep_data):
    frames = rep_data["frames"]
    min_knee = min(f.get("left_knee_angle", 180) for f in frames)
    max_knee = max(f.get("left_knee_angle", 180) for f in frames)
    feedback = []
    if min_knee > _BEND_THRESHOLD:
        feedback.append({"message": f"Bend deeper — reached {min_knee:.0f}°, aim for {_BEND_THRESHOLD}°", "joint": None})
    if max_knee < _EXTEND_THRESHOLD:               # separate if, NOT elif
        feedback.append({"message": f"Straighten more — reached {max_knee:.0f}°, aim for {_EXTEND_THRESHOLD}°", "joint": None})
    if not feedback:
        feedback.append({"message": "Great rep!", "joint": None})
    return feedback
```
"""
```

- [ ] **Step 4: Run the three new tests plus existing tests**

```
pytest tests/physio_app/test_llm_service.py -v
```

Expected: all PASS (including existing tests for instructions, function names, angle fields).

- [ ] **Step 5: Commit**

```bash
git add physio_app/services/llm_service.py tests/physio_app/test_llm_service.py
git commit -m "fix: update llm prompt with throttle pattern, bidirectional threshold guidance, and independent if checks"
```

---

## Task 2: Create core/video_analyzer.py

**Files:**
- Create: `core/video_analyzer.py`
- Create: `tests/core/test_video_analyzer.py`

`analyze_reference_video` reads a video file frame-by-frame, runs MediaPipe Pose on each frame, calls `calculate_angles()` with the resulting keypoints, accumulates per-joint angle lists, and returns a summary dict filtered to joints with range > 10°.

- [ ] **Step 1: Write failing tests**

Create `tests/core/test_video_analyzer.py`:

```python
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
    mocker.patch("core.video_analyzer.mp.solutions.pose.Pose", return_value=mock_pose)

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
    mocker.patch("core.video_analyzer.mp.solutions.pose.Pose", return_value=mock_pose)

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
    mock_cap = _make_mock_cap([dummy_frame])
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
    mocker.patch("core.video_analyzer.mp.solutions.pose.Pose", return_value=mock_pose)
    mocker.patch("core.video_analyzer.calculate_angles", return_value={"left_elbow_angle": 44.6789})

    # Single frame means range is 0 → filtered. Add a second frame to get range > 10.
    # Redo with two frames so there is a range.
    mock_cap2 = _make_mock_cap([dummy_frame, dummy_frame])
    mocker.patch("core.video_analyzer.cv2.VideoCapture", return_value=mock_cap2)
    frame_angles = [{"left_elbow_angle": 160.3333}, {"left_elbow_angle": 44.6789}]
    mocker.patch("core.video_analyzer.calculate_angles", side_effect=frame_angles)

    result = analyze_reference_video("video.mp4")
    assert result["left_elbow_angle"]["min"] == 44.7
    assert result["left_elbow_angle"]["max"] == 160.3
    assert result["left_elbow_angle"]["range"] == 115.7
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/core/test_video_analyzer.py -v
```

Expected: all FAIL with `ModuleNotFoundError: No module named 'core.video_analyzer'`.

- [ ] **Step 3: Implement core/video_analyzer.py**

Create `core/video_analyzer.py`:

```python
import cv2
import mediapipe as mp
from core.angle_calculator import calculate_angles
from core.pose_engine import LANDMARK_NAMES

_MIN_RANGE = 10.0


def analyze_reference_video(video_path: str) -> dict[str, dict]:
    """Run pose estimation on a reference video and return per-joint angle statistics.

    Returns a dict of { joint_name: {"min": float, "max": float, "range": float} }
    for every joint where the range of motion exceeded 10 degrees.
    Returns {} if the video cannot be opened or contains no detected poses.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    angle_lists: dict[str, list[float]] = {}

    try:
        with mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        ) as pose:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = pose.process(rgb)
                if results.pose_landmarks:
                    lms = results.pose_landmarks.landmark
                    keypoints = {
                        LANDMARK_NAMES[i]: (lm.x, lm.y, lm.z, lm.visibility)
                        for i, lm in enumerate(lms)
                        if i in LANDMARK_NAMES
                    }
                    angles = calculate_angles(keypoints)
                    for joint, value in angles.items():
                        angle_lists.setdefault(joint, []).append(value)
    finally:
        cap.release()

    if not angle_lists:
        return {}

    result = {}
    for joint, values in angle_lists.items():
        mn = min(values)
        mx = max(values)
        rng = mx - mn
        if rng > _MIN_RANGE:
            result[joint] = {
                "min": round(mn, 1),
                "max": round(mx, 1),
                "range": round(rng, 1),
            }
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/core/test_video_analyzer.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add core/video_analyzer.py tests/core/test_video_analyzer.py
git commit -m "feat: add video_analyzer — extract joint angle stats from reference video"
```

---

## Task 3: Wire reference data into build_prompt

**Files:**
- Modify: `physio_app/services/llm_service.py` (add `_format_reference_data`, update `build_prompt`)
- Modify: `tests/physio_app/test_llm_service.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/physio_app/test_llm_service.py`:

```python
def test_build_prompt_includes_reference_data_when_provided():
    reference_data = {
        "left_elbow_angle": {"min": 44.0, "max": 163.0, "range": 119.0},
        "left_shoulder_angle": {"min": 28.0, "max": 171.0, "range": 143.0},
    }
    prompt = build_prompt("Bicep Curl", "side", "Curl slowly", reference_data=reference_data)
    assert "Reference movement data" in prompt
    assert "left_elbow_angle" in prompt
    assert "min=44" in prompt
    assert "max=163" in prompt
    assert "left_shoulder_angle" in prompt


def test_build_prompt_omits_reference_section_when_none():
    prompt = build_prompt("Squat", "side", "instructions", reference_data=None)
    assert "Reference movement data" not in prompt


def test_build_prompt_omits_reference_section_when_empty_dict():
    prompt = build_prompt("Squat", "side", "instructions", reference_data={})
    assert "Reference movement data" not in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/physio_app/test_llm_service.py::test_build_prompt_includes_reference_data_when_provided tests/physio_app/test_llm_service.py::test_build_prompt_omits_reference_section_when_none tests/physio_app/test_llm_service.py::test_build_prompt_omits_reference_section_when_empty_dict -v
```

Expected: all three FAIL.

- [ ] **Step 3: Add `_format_reference_data` and update `build_prompt` signature**

Add this function just before `build_prompt` in `physio_app/services/llm_service.py`:

```python
def _format_reference_data(reference_data: dict) -> str:
    lines = ["Reference movement data (measured from therapist's demo video):"]
    for joint, stats in sorted(reference_data.items()):
        lines.append(
            f"  {joint}: min={stats['min']:.0f}°  max={stats['max']:.0f}°  range={stats['range']:.0f}°"
        )
    lines += [
        "",
        "Use these measured values to calibrate your detection thresholds.",
        "Joints not listed did not move significantly during the demo.",
    ]
    return "\n".join(lines)
```

Update the `build_prompt` signature and body:

```python
def build_prompt(exercise_name: str, camera_view: str, instructions: str,
                 reference_data: dict | None = None) -> str:
    ref_section = ""
    if reference_data:
        ref_section = "\n" + _format_reference_data(reference_data) + "\n"
    return f"""\
You are generating a Python movement analysis module for a physiotherapy application.

Exercise: {exercise_name}
Camera view: {camera_view}

Physiotherapist instructions:
{instructions}

Available pose data fields:
{_POSE_DATA_DESCRIPTION}{ref_section}
{_FEW_SHOT}

Now generate the analysis module for '{exercise_name}' following the same structure.

{_FUNCTION_SPEC}

Rules:
- Only import math or statistics if needed. Do NOT import os, sys, subprocess, socket, or requests.
- Use module-level variables for state (rep phase tracking etc.).
- Return ONLY valid Python code. No markdown fences. No explanations.
"""
```

- [ ] **Step 4: Run all llm_service tests**

```
pytest tests/physio_app/test_llm_service.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add physio_app/services/llm_service.py tests/physio_app/test_llm_service.py
git commit -m "feat: inject reference video angle measurements into llm prompt"
```

---

## Task 4: Wire analyze_reference_video into generate_module

**Files:**
- Modify: `physio_app/services/llm_service.py` (update `generate_module`, add import)
- Modify: `tests/physio_app/test_llm_service.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/physio_app/test_llm_service.py`:

```python
def test_generate_module_calls_video_analyzer_when_video_exists(tmp_db, mocker, tmp_path):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text=(
            "def analyze_frame(pose_data): return []\n"
            "def detect_rep(pose_data): return False\n"
            "def on_rep_complete(rep_data): return []\n"
        ))]
    )
    mocker.patch("physio_app.services.llm_service.anthropic.Anthropic", return_value=mock_client)

    mock_analyze = mocker.patch(
        "physio_app.services.llm_service.analyze_reference_video",
        return_value={"left_elbow_angle": {"min": 44.0, "max": 163.0, "range": 119.0}},
    )

    # Create a dummy video file so the path check passes
    video_path = str(tmp_path / "reference.mp4")
    open(video_path, "wb").close()

    from physio_app.services.exercise_service import ExerciseService
    ex_svc = ExerciseService(tmp_db)
    ex = ex_svc.create("Bicep Curl", "side", "Keep elbow close")
    ex_svc.update(ex.id, reference_video_path=video_path)

    svc = LLMService(tmp_db, api_key="test-key")
    svc.generate_module(ex.id, ex.name, ex.camera_view, ex.instructions_text)

    mock_analyze.assert_called_once_with(video_path)

    # reference data should appear in the prompt that was sent to Claude
    sent_prompt = mock_client.messages.create.call_args[1]["messages"][0]["content"]
    assert "Reference movement data" in sent_prompt
    assert "left_elbow_angle" in sent_prompt


def test_generate_module_skips_video_analyzer_when_no_video(tmp_db, mocker):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text=(
            "def analyze_frame(pose_data): return []\n"
            "def detect_rep(pose_data): return False\n"
            "def on_rep_complete(rep_data): return []\n"
        ))]
    )
    mocker.patch("physio_app.services.llm_service.anthropic.Anthropic", return_value=mock_client)
    mock_analyze = mocker.patch("physio_app.services.llm_service.analyze_reference_video")

    from physio_app.services.exercise_service import ExerciseService
    ex_svc = ExerciseService(tmp_db)
    ex = ex_svc.create("Squat", "side", "Keep knees aligned")
    # no reference_video_path set

    svc = LLMService(tmp_db, api_key="test-key")
    svc.generate_module(ex.id, ex.name, ex.camera_view, ex.instructions_text)

    mock_analyze.assert_not_called()

    sent_prompt = mock_client.messages.create.call_args[1]["messages"][0]["content"]
    assert "Reference movement data" not in sent_prompt


def test_generate_module_proceeds_when_video_analyzer_returns_empty(tmp_db, mocker, tmp_path):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text=(
            "def analyze_frame(pose_data): return []\n"
            "def detect_rep(pose_data): return False\n"
            "def on_rep_complete(rep_data): return []\n"
        ))]
    )
    mocker.patch("physio_app.services.llm_service.anthropic.Anthropic", return_value=mock_client)
    mocker.patch("physio_app.services.llm_service.analyze_reference_video", return_value={})

    video_path = str(tmp_path / "reference.mp4")
    open(video_path, "wb").close()

    from physio_app.services.exercise_service import ExerciseService
    ex_svc = ExerciseService(tmp_db)
    ex = ex_svc.create("Squat", "side", "Keep knees aligned")
    ex_svc.update(ex.id, reference_video_path=video_path)

    svc = LLMService(tmp_db, api_key="test-key")
    result = svc.generate_module(ex.id, ex.name, ex.camera_view, ex.instructions_text)

    assert result["status"] == "validated"
    sent_prompt = mock_client.messages.create.call_args[1]["messages"][0]["content"]
    assert "Reference movement data" not in sent_prompt
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/physio_app/test_llm_service.py::test_generate_module_calls_video_analyzer_when_video_exists tests/physio_app/test_llm_service.py::test_generate_module_skips_video_analyzer_when_no_video tests/physio_app/test_llm_service.py::test_generate_module_proceeds_when_video_analyzer_returns_empty -v
```

Expected: all three FAIL.

- [ ] **Step 3: Add import and update generate_module**

Add import at the top of `physio_app/services/llm_service.py` (after existing imports):

```python
from core.video_analyzer import analyze_reference_video
```

Replace the `generate_module` method body:

```python
def generate_module(self, exercise_id: str, name: str, camera_view: str,
                    instructions: str) -> dict:
    """Call Claude, validate the result, store it. Returns the saved module dict."""
    reference_data = None
    ex = self.ex_svc.get(exercise_id)
    if ex and ex.reference_video_path:
        try:
            reference_data = analyze_reference_video(ex.reference_video_path) or None
        except Exception:
            reference_data = None

    prompt = build_prompt(name, camera_view, instructions, reference_data)
    response_text = ""
    status = "failed"
    try:
        message = self._client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = message.content[0].text.strip()
        validation = validate_module(response_text)
        status = "validated" if validation["valid"] else "failed"
    except Exception as e:
        response_text = str(e)

    self._log(exercise_id, prompt, response_text)
    return self.ex_svc.save_module(exercise_id, response_text, status)
```

- [ ] **Step 4: Run all tests**

```
pytest tests/ -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add physio_app/services/llm_service.py tests/physio_app/test_llm_service.py
git commit -m "feat: run pose estimation on reference video before code generation"
```

---

## Self-Review

**Spec coverage:**
- ✅ 1a. Self-throttling in `analyze_frame` — Task 1 (`_FUNCTION_SPEC` + `_FEW_SHOT`)
- ✅ 1b. Bidirectional threshold guidance — Task 1 (`_FUNCTION_SPEC` + second few-shot example)
- ✅ 1c. Independent `if` checks in `on_rep_complete` — Task 1 (`_FUNCTION_SPEC` + both examples)
- ✅ `core/video_analyzer.py` — Task 2
- ✅ `range > 10°` filter — Task 2 (`_MIN_RANGE = 10.0`)
- ✅ Reuse `calculate_angles` from `core.angle_calculator` — Task 2
- ✅ `build_prompt` reference_data parameter — Task 3
- ✅ Reference data section format — Task 3 (`_format_reference_data`)
- ✅ `generate_module` wiring — Task 4
- ✅ Error handling: no video → skip — Task 4 (`if ex and ex.reference_video_path`)
- ✅ Error handling: analyzer fails → proceed without — Task 4 (`try/except`)
- ✅ Error handling: analyzer returns `{}` → no reference section — Task 3 (`if reference_data:`)

**No placeholders:** All steps contain complete code.

**Type consistency:** `reference_data: dict | None` used consistently across Task 3 and Task 4. `analyze_reference_video` returns `dict[str, dict]` in Task 2, consumed as `dict | None` in Task 4. `build_prompt` signature updated in Task 3, used with keyword argument in Task 4.
