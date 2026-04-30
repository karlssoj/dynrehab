# QTRobot Modular Exercise App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `extras/qt_robot_app/` — a self-contained, modular exercise app for QTRobot that uses MediaPipe pose estimation, static angle calculation identical to `qt/`, and accepts copy-pasted `analysis_module.py` files from the physio portal without modification.

**Architecture:** Self-contained package (no dependency on `qt/`). Core files `angle_calculator.py` and `data_contract.py` are copied verbatim. Three robot-specific files replace the CustomTkinter UI stack: `pose_engine.py` subscribes to the ROS camera topic, `tts.py` calls the QTRobot speech service via a background thread queue, and `display.py` renders overlays using cv2. `session_runner.py` is copied from `qt/lib/` with one added field. `run.py` (one per exercise) drives a simple queue-based main loop.

**Tech Stack:** Python 3, MediaPipe, OpenCV, rospy, cv_bridge, qt_robot_interface, numpy.

---

## File Map

```
extras/qt_robot_app/
├── lib/
│   ├── __init__.py                  create — empty
│   ├── requirements.txt             create
│   ├── core/
│   │   ├── __init__.py              create — empty
│   │   ├── angle_calculator.py      create — verbatim copy from qt/lib/core/angle_calculator.py
│   │   ├── data_contract.py         create — verbatim copy from qt/lib/core/data_contract.py
│   │   └── pose_engine.py           create — adapted: ROS subscriber instead of VideoCapture
│   ├── tts.py                       create — speech queue + QTRobot ROS service
│   ├── display.py                   create — cv2 overlay renderer
│   └── session_runner.py            create — verbatim copy from qt/lib/session_runner.py + 1 line
├── squat_from_side/
│   ├── __init__.py                  create — empty
│   ├── exercise_config.py           create — robot config (after_window mode)
│   ├── analysis_module.py           create — verbatim copy from qt/squat_from_side/analysis_module.py
│   └── run.py                       create — ROS node entry point
└── README.md                        create
```

---

## Task 1: Project scaffold and static copies

**Files:**
- Create: `extras/qt_robot_app/lib/__init__.py`
- Create: `extras/qt_robot_app/lib/requirements.txt`
- Create: `extras/qt_robot_app/lib/core/__init__.py`
- Create: `extras/qt_robot_app/lib/core/angle_calculator.py`
- Create: `extras/qt_robot_app/lib/core/data_contract.py`
- Create: `extras/qt_robot_app/squat_from_side/__init__.py`
- Create: `extras/qt_robot_app/squat_from_side/analysis_module.py`
- Create: `extras/qt_robot_app/squat_from_side/exercise_config.py`

- [ ] **Step 1: Create empty `__init__.py` files**

```
extras/qt_robot_app/lib/__init__.py          — empty file
extras/qt_robot_app/lib/core/__init__.py     — empty file
extras/qt_robot_app/squat_from_side/__init__.py — empty file
```

- [ ] **Step 2: Create `lib/requirements.txt`**

```
mediapipe>=0.10
opencv-python>=4.8
numpy>=1.24
```

Note: `rospy` and `cv_bridge` are provided by the ROS installation — do not list them here.

- [ ] **Step 3: Copy `angle_calculator.py` verbatim**

Copy the entire contents of `qt/lib/core/angle_calculator.py` into `extras/qt_robot_app/lib/core/angle_calculator.py` without any changes.

The file starts with `import numpy as np` and ends with `"body_rotation_z": ls[2] - rs[2],`. It is 166 lines.

- [ ] **Step 4: Copy `data_contract.py` verbatim**

Copy the entire contents of `qt/lib/core/data_contract.py` into `extras/qt_robot_app/lib/core/data_contract.py` without any changes.

The file defines the `PoseFrame` dataclass with 37+ angle fields and a `to_dict()` method. It is 49 lines.

- [ ] **Step 5: Copy `analysis_module.py` verbatim**

Copy the entire contents of `qt/squat_from_side/analysis_module.py` into `extras/qt_robot_app/squat_from_side/analysis_module.py` without any changes.

- [ ] **Step 6: Create `squat_from_side/exercise_config.py`**

The robot version always uses `after_window` so the session runner handles the time-based end automatically. `after_exercise` triggers the session summary speech.

```python
CONFIG = {
    "name": "Squat from side",
    "camera_view": "side",
    "client_instructions": (
        "Stand sideways to the camera with your feet shoulder-width apart. "
        "Perform slow, controlled squats, aiming for a 90-degree knee bend "
        "at the bottom of each rep. Keep your heels flat on the floor and "
        "prevent your knees from travelling too far past your toes."
    ),
    "session_duration_secs": 60,
    "feedback_mode": ["after_window", "after_exercise"],
}
```

- [ ] **Step 7: Verify copies look correct**

```bash
python -c "import ast; ast.parse(open('extras/qt_robot_app/lib/core/angle_calculator.py').read()); print('OK')"
python -c "import ast; ast.parse(open('extras/qt_robot_app/lib/core/data_contract.py').read()); print('OK')"
python -c "import ast; ast.parse(open('extras/qt_robot_app/squat_from_side/analysis_module.py').read()); print('OK')"
```

Expected: all three print `OK`.

- [ ] **Step 8: Commit**

```bash
git add extras/qt_robot_app/lib/__init__.py
git add extras/qt_robot_app/lib/requirements.txt
git add extras/qt_robot_app/lib/core/__init__.py
git add extras/qt_robot_app/lib/core/angle_calculator.py
git add extras/qt_robot_app/lib/core/data_contract.py
git add extras/qt_robot_app/squat_from_side/__init__.py
git add extras/qt_robot_app/squat_from_side/exercise_config.py
git add extras/qt_robot_app/squat_from_side/analysis_module.py
git commit -m "feat: scaffold qt_robot_app with static copies"
```

---

## Task 2: `lib/tts.py` — ROS speech service wrapper

**Files:**
- Create: `extras/qt_robot_app/lib/tts.py`

Background: the QTRobot speech service (`/qt_robot/speech/say`) is a blocking ROS service call. Calling it directly inside a ROS subscriber callback would delay frame processing. Instead, `tts.py` runs a dedicated worker thread that drains a queue. `speak()` enqueues (non-blocking). `speak_sync()` enqueues and waits for the speech to finish before returning — used for instructions and session feedback where the app must wait before advancing.

- [ ] **Step 1: Write `lib/tts.py`**

```python
import threading
import queue as _queue_mod
import rospy
from qt_robot_interface.srv import speech_say

_q: _queue_mod.Queue = _queue_mod.Queue()
_say = None


def _worker():
    global _say
    while True:
        text = _q.get()
        if text is None:
            _q.task_done()
            break
        try:
            if _say is None:
                rospy.wait_for_service('/qt_robot/speech/say', timeout=5.0)
                _say = rospy.ServiceProxy('/qt_robot/speech/say', speech_say)
            _say(str(text))
        except Exception as e:
            rospy.logwarn(f"[tts] speech failed: {e}")
        finally:
            _q.task_done()


_worker_thread = threading.Thread(target=_worker, daemon=True)
_worker_thread.start()


def speak(text: str) -> None:
    """Queue text for speech. Returns immediately."""
    _q.put(str(text))


def speak_sync(text: str) -> None:
    """Queue text and block until the worker has spoken it."""
    _q.put(str(text))
    _q.join()


def stop() -> None:
    """Signal the worker thread to exit."""
    _q.put(None)
    _worker_thread.join(timeout=2.0)
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('extras/qt_robot_app/lib/tts.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add extras/qt_robot_app/lib/tts.py
git commit -m "feat: add tts speech queue for QTRobot"
```

---

## Task 3: `lib/core/pose_engine.py` — ROS camera subscriber

**Files:**
- Create: `extras/qt_robot_app/lib/core/pose_engine.py`

Background: the `qt/` version uses `cv2.VideoCapture` in a background thread. The robot version subscribes to `/camera/color/image_raw` (a ROS topic) and converts each message to a cv2 frame using `cv_bridge.CvBridge`. All MediaPipe processing, keypoint extraction, angle calculation, and overlay drawing are identical to `qt/lib/core/pose_engine.py`. The `seek_to_start()` method (used for video file replay in `qt/`) is omitted as the robot always uses live camera.

- [ ] **Step 1: Write `lib/core/pose_engine.py`**

```python
import threading
import time
from typing import Callable
import cv2
import mediapipe as mp
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from core.data_contract import PoseFrame
from core.angle_calculator import calculate_angles

LANDMARK_NAMES = {
    0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
    4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
    7: "left_ear", 8: "right_ear", 9: "mouth_left", 10: "mouth_right",
    11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist", 17: "left_pinky", 18: "right_pinky",
    19: "left_index", 20: "right_index", 21: "left_thumb", 22: "right_thumb",
    23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle", 29: "left_heel", 30: "right_heel",
    31: "left_foot_index", 32: "right_foot_index",
}

POSE_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_ankle", "left_foot_index"), ("right_ankle", "right_foot_index"),
    ("nose", "left_shoulder"), ("nose", "right_shoulder"),
]


class PoseEngine:
    def __init__(self):
        self._subscribers: list[Callable] = []
        self._running = False
        self._bridge = CvBridge()
        self._mp_pose = mp.solutions.pose
        self._lock = threading.Lock()
        self._highlight_joints: set[str] = set()
        self._pose = self._mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        )
        self._start_time = time.time()
        self._sub = None

    def subscribe(self, callback: Callable[[PoseFrame, np.ndarray], None]):
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable):
        with self._lock:
            self._subscribers = [s for s in self._subscribers if s != callback]

    def set_highlight_joints(self, joints: set[str]):
        with self._lock:
            self._highlight_joints = set(joints)

    def start(self):
        self._running = True
        self._sub = rospy.Subscriber(
            '/camera/color/image_raw', Image, self._on_image,
            queue_size=1, buff_size=2 ** 24,
        )

    def stop(self):
        self._running = False
        if self._sub is not None:
            self._sub.unregister()
            self._sub = None

    def _on_image(self, msg: Image):
        if not self._running:
            return
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logwarn(f"[pose_engine] cv_bridge error: {e}")
            return
        self._process(frame)

    def _process(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._pose.process(rgb)
        rgb.flags.writeable = True
        annotated = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        pose_frame = PoseFrame(timestamp=time.time() - self._start_time)

        if results.pose_landmarks:
            lms = results.pose_landmarks.landmark
            keypoints = {
                LANDMARK_NAMES[i]: (lm.x, lm.y, lm.z, lm.visibility)
                for i, lm in enumerate(lms)
                if i in LANDMARK_NAMES
            }
            angles = calculate_angles(keypoints)
            for k, v in angles.items():
                setattr(pose_frame, k, v)
            pose_frame.keypoints = keypoints
            annotated = self._draw_overlay(annotated, keypoints)

        with self._lock:
            subs = list(self._subscribers)

        for cb in subs:
            try:
                cb(pose_frame, annotated)
            except Exception as e:
                rospy.logwarn(f"[pose_engine] subscriber error: {e}")

    def _draw_overlay(self, frame: np.ndarray, keypoints: dict) -> np.ndarray:
        h, w = frame.shape[:2]
        with self._lock:
            highlight = set(self._highlight_joints)

        for a_name, b_name in POSE_CONNECTIONS:
            if a_name in keypoints and b_name in keypoints:
                a = keypoints[a_name]
                b = keypoints[b_name]
                if a[3] > 0.5 and b[3] > 0.5:
                    pt1 = (int(a[0] * w), int(a[1] * h))
                    pt2 = (int(b[0] * w), int(b[1] * h))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        for name, (x, y, z, vis) in keypoints.items():
            if vis > 0.5:
                pt = (int(x * w), int(y * h))
                color = (0, 0, 255) if name in highlight else (255, 255, 255)
                cv2.circle(frame, pt, 5, color, -1)

        return frame
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('extras/qt_robot_app/lib/core/pose_engine.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add extras/qt_robot_app/lib/core/pose_engine.py
git commit -m "feat: add ROS camera pose engine for qt_robot_app"
```

---

## Task 4: `lib/display.py` — cv2 overlay renderer

**Files:**
- Create: `extras/qt_robot_app/lib/display.py`

Background: renders four state-specific overlays matching the information shown in `qt/lib/app.py`, but using cv2 instead of CustomTkinter. Stores the most-recently-seen feedback lines so the feedback overlay persists across frames (the session runner only sets `feedback_lines` on the first frame of the feedback state).

- [ ] **Step 1: Write `lib/display.py`**

```python
import cv2
import numpy as np

_FONT = cv2.FONT_HERSHEY_SIMPLEX


class Display:
    def __init__(self, window_name: str = "QT Exercise", exercise_secs: int = 60):
        self._window = window_name
        self._exercise_secs = exercise_secs
        self._last_feedback: list[str] = []
        cv2.namedWindow(self._window, cv2.WINDOW_NORMAL)

    def update(self, frame: np.ndarray, state: dict) -> bool:
        """Render state overlays and show in window. Returns False when ESC is pressed."""
        display = frame.copy()
        h, w = display.shape[:2]
        s = state.get("state", "")

        new_feedback = state.get("feedback_lines")
        if new_feedback:
            self._last_feedback = list(new_feedback)

        if s == "calibration":
            self._render_calibration(display, h, w, state)
        elif s == "countdown":
            self._render_countdown(display, h, w, state)
        elif s == "exercise":
            self._render_exercise(display, h, w, state)
        elif s == "feedback":
            self._render_feedback(display, h, w)

        cv2.imshow(self._window, display)
        return cv2.waitKey(1) != 27

    def _render_calibration(self, display: np.ndarray, h: int, w: int, state: dict):
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)
        cv2.putText(display, "Positioning", (30, 46), _FONT, 1.0, (0, 220, 255), 2)
        msg = state.get("calibration_message", "")
        if msg:
            color = (0, 255, 0) if state.get("calibration_status") == "ready" else (0, 180, 255)
            cv2.putText(display, msg.rstrip("."), (30, h - 40), _FONT, 1.0, color, 2)

    def _render_countdown(self, display: np.ndarray, h: int, w: int, state: dict):
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)
        cnt = state.get("countdown_speak")
        remaining = state.get("time_remaining", 0.0)
        num = cnt if cnt is not None else max(0, int(remaining))
        if num > 0:
            text = str(num)
            ts = cv2.getTextSize(text, _FONT, 5.0, 8)[0]
            cv2.putText(display, text,
                        ((w - ts[0]) // 2, (h + ts[1]) // 2),
                        _FONT, 5.0, (0, 255, 255), 8)
        else:
            ts = cv2.getTextSize("GO!", _FONT, 4.0, 8)[0]
            cv2.putText(display, "GO!",
                        ((w - ts[0]) // 2, (h + ts[1]) // 2),
                        _FONT, 4.0, (0, 255, 0), 8)

    def _render_exercise(self, display: np.ndarray, h: int, w: int, state: dict):
        reps = state.get("total_reps", 0)
        time_left = state.get("time_remaining", 0.0)
        cv2.putText(display, f"Reps: {reps}", (10, 35), _FONT, 1.0, (0, 255, 0), 2)
        cv2.putText(display, f"Time: {max(0.0, time_left):.0f}s", (10, 65), _FONT, 0.7, (0, 255, 255), 2)
        elapsed = self._exercise_secs - time_left
        if self._exercise_secs > 0:
            bar_w = int(min(1.0, elapsed / self._exercise_secs) * w)
            cv2.rectangle(display, (0, h - 8), (bar_w, h), (0, 200, 255), -1)
        joint_values = state.get("joint_values", {})
        y = 35
        for label_text, val in joint_values.items():
            text = f"{label_text}: {val:.1f}" if isinstance(val, float) else f"{label_text}: {val}"
            tw = cv2.getTextSize(text, _FONT, 0.55, 1)[0][0]
            cv2.putText(display, text, (w - tw - 10, y), _FONT, 0.55, (200, 200, 200), 1)
            y += 24

    def _render_feedback(self, display: np.ndarray, h: int, w: int):
        feedback = self._last_feedback
        if not feedback:
            return
        line_h = 30
        block_h = len(feedback) * line_h + 20
        overlay = display.copy()
        cv2.rectangle(overlay, (0, h - block_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
        for i, line in enumerate(feedback):
            cv2.putText(display, line,
                        (10, h - block_h + 20 + i * line_h),
                        _FONT, 0.65, (255, 255, 100), 2)

    def close(self):
        cv2.destroyAllWindows()
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('extras/qt_robot_app/lib/display.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add extras/qt_robot_app/lib/display.py
git commit -m "feat: add cv2 display renderer for qt_robot_app"
```

---

## Task 5: `lib/session_runner.py` — expose `feedback_type`

**Files:**
- Create: `extras/qt_robot_app/lib/session_runner.py`

Background: this is a verbatim copy of `qt/lib/session_runner.py` with exactly one change: `result["feedback_type"] = self._feedback_type` is added inside `process_frame()`. This lets `run.py` distinguish per-rep feedback (should resume exercise) from round feedback (should end session). Without this, `run.py` cannot tell the two apart.

- [ ] **Step 1: Copy `qt/lib/session_runner.py` to `extras/qt_robot_app/lib/session_runner.py`**

Copy the entire file verbatim first (212 lines as of this writing).

- [ ] **Step 2: Add `feedback_type` to the result dict**

Find this block in `process_frame()`:

```python
        result = {
            "state": self._state,
            "round_number": self._round_number,
            "round_rep_count": self._round_rep_count,
            "total_reps": self.rep_count,
            "time_remaining": 0.0,
            "countdown_speak": None,
            "feedback_lines": None,
            "highlight_joints": set(),
        }
```

Add one field so it becomes:

```python
        result = {
            "state": self._state,
            "round_number": self._round_number,
            "round_rep_count": self._round_rep_count,
            "total_reps": self.rep_count,
            "time_remaining": 0.0,
            "countdown_speak": None,
            "feedback_lines": None,
            "highlight_joints": set(),
            "feedback_type": self._feedback_type,
        }
```

- [ ] **Step 3: Verify syntax**

```bash
python -c "import ast; ast.parse(open('extras/qt_robot_app/lib/session_runner.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Write the test**

Create `extras/qt_robot_app/tests/test_session_runner_feedback_type.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

from session_runner import SessionRunner

_BEND_MODULE_CODE = """
_phase = "ready"
_rep_count = 0

def get_instructions():
    return ["Stand sideways."]

def detect_rep(pose_data):
    global _phase, _rep_count
    bend = pose_data.get("left_knee_bend_2d", 0.0)
    if _phase == "ready" and bend > 25:
        _phase = "descending"
    elif _phase == "descending" and bend < 15:
        _phase = "ready"
        _rep_count += 1
        return True
    return False

def reset_round():
    global _phase, _rep_count
    _phase = "ready"
    _rep_count = 0

def generate_round_feedback(round_data):
    return ["Round done."]

def get_session_summary(session_data):
    return ["Session done."]

def get_relevant_joints():
    return [("Left Knee", "left_knee_bend_2d")]
"""

import types

def _make_module(code):
    m = types.ModuleType("test_mod")
    exec(compile(code, "<test>", "exec"), m.__dict__)
    return m


def test_feedback_type_window():
    mod = _make_module(_BEND_MODULE_CODE)
    config = {
        "camera_view": "side",
        "session_duration_secs": 1,
        "feedback_mode": ["after_window"],
    }
    runner = SessionRunner(config, mod)
    runner.start_countdown()

    import time
    # Fast-forward through countdown
    runner._state_wall_start -= 6.0
    runner._enter_exercise()
    # Fast-forward through exercise
    runner._state_wall_start -= 2.0

    result = runner.process_frame({"keypoints": {}})
    assert result["feedback_type"] == "window"


def test_feedback_type_rep():
    mod = _make_module(_BEND_MODULE_CODE)
    config = {
        "camera_view": "side",
        "session_duration_secs": 60,
        "feedback_mode": ["after_rep"],
    }

    AFTER_REP_CODE = _BEND_MODULE_CODE + """
def generate_rep_feedback(rep_data):
    return ["Rep done."]
"""
    mod = _make_module(AFTER_REP_CODE)
    runner = SessionRunner(config, mod)
    runner._enter_exercise()

    # Drive rep: descend
    for _ in range(3):
        runner.process_frame({"left_knee_bend_2d": 40.0, "keypoints": {}})
    # Complete rep
    result = runner.process_frame({"left_knee_bend_2d": 5.0, "keypoints": {}})

    assert result.get("feedback_lines") is not None
    assert result["feedback_type"] == "rep"
```

- [ ] **Step 5: Run the tests**

```bash
cd extras/qt_robot_app
python -m pytest tests/test_session_runner_feedback_type.py -v
```

Expected:
```
tests/test_session_runner_feedback_type.py::test_feedback_type_window PASSED
tests/test_session_runner_feedback_type.py::test_feedback_type_rep PASSED
2 passed
```

- [ ] **Step 6: Commit**

```bash
git add extras/qt_robot_app/lib/session_runner.py
git add extras/qt_robot_app/tests/test_session_runner_feedback_type.py
git commit -m "feat: add qt_robot_app session_runner with feedback_type field"
```

---

## Task 6: `squat_from_side/run.py` — ROS node entry point

**Files:**
- Create: `extras/qt_robot_app/squat_from_side/run.py`

Background: the main loop runs in the main thread (not in the ROS subscriber callback). The pose engine queues the latest frame; the loop drains that queue, calls the session runner, handles speech events, and drives the display. This allows blocking speech calls (`speak_sync`) for important transitions (instructions, session feedback) without stalling the ROS subscriber.

For per-rep feedback the robot app uses `after_window` by default (see `exercise_config.py`), so `feedback_type` will always be `"window"` in the reference exercise. The logic in `run.py` handles both `"window"` and `"rep"` so the file works correctly if a future exercise uses `"after_rep"`.

- [ ] **Step 1: Write `squat_from_side/run.py`**

```python
#!/usr/bin/env python3
import os
import queue
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import rospy

import exercise_config
import analysis_module
import tts
from core.pose_engine import PoseEngine
from session_runner import SessionRunner
from display import Display


def main():
    rospy.init_node("qt_exercise", anonymous=True)

    config = exercise_config.CONFIG
    runner = SessionRunner(config, analysis_module)
    display = Display(
        window_name=config.get("name", "QT Exercise"),
        exercise_secs=config.get("session_duration_secs", 60),
    )
    engine = PoseEngine()

    _frame_q: queue.Queue = queue.Queue(maxsize=1)

    def _on_frame(pose_frame, annotated_frame):
        if not _frame_q.full():
            _frame_q.put((pose_frame, annotated_frame))

    engine.subscribe(_on_frame)

    # Speak instructions synchronously before starting the loop
    for line in runner.get_instructions():
        tts.speak_sync(line)
    runner.start_calibration()

    relevant_joints = runner.get_relevant_joints()
    _last_countdown = None
    _last_calib_speak = None
    _last_rep_cue = None
    _done = False

    engine.start()

    while not rospy.is_shutdown() and not _done:
        try:
            pose_frame, annotated_frame = _frame_q.get(timeout=0.1)
        except queue.Empty:
            cv2.waitKey(1)
            continue

        state = runner.process_frame(pose_frame.to_dict())

        # Attach live joint values for the display
        pose_dict = pose_frame.to_dict()
        state["joint_values"] = {
            label: float(pose_dict.get(key, 0.0))
            for label, key in relevant_joints
        }

        # Calibration speech — deduplicated, async (short messages)
        calib_speak = state.get("calibration_speak")
        if calib_speak and calib_speak != _last_calib_speak:
            _last_calib_speak = calib_speak
            tts.speak(calib_speak)

        # Countdown speech — fires once per number change, async
        cnt = state.get("countdown_speak")
        if cnt is not None and cnt != _last_countdown:
            _last_countdown = cnt
            tts.speak(str(cnt) if cnt > 0 else "Go!")

        # Rep cue speech — deduplicated, async
        rep_cue = state.get("rep_cue")
        if rep_cue and rep_cue != _last_rep_cue:
            _last_rep_cue = rep_cue
            tts.speak(rep_cue)

        # Feedback speech — synchronous so the patient hears it before session advances
        feedback = state.get("feedback_lines")
        if feedback:
            for line in feedback:
                tts.speak_sync(line)
            if state.get("feedback_type") == "rep":
                # Per-rep feedback: resume exercise after speaking
                runner.end_feedback()
            else:
                # Round/window feedback: speak session summary then exit
                _done = True
                for line in runner.get_session_summary_speech():
                    tts.speak_sync(line)
                engine.stop()
                break

        if not display.update(annotated_frame, state):
            _done = True
            engine.stop()
            break

    display.close()
    tts.stop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('extras/qt_robot_app/squat_from_side/run.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add extras/qt_robot_app/squat_from_side/run.py
git commit -m "feat: add squat_from_side run.py for qt_robot_app"
```

---

## Task 7: `README.md`

**Files:**
- Create: `extras/qt_robot_app/README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# QTRobot Modular Exercise App

Modular physiotherapy exercise app for QTRobot. Uses MediaPipe pose estimation
and the same analysis module interface as the physio portal's desktop app (`qt/`).

## Prerequisites

- ROS (with `rospy`, `sensor_msgs`, `cv_bridge`, `qt_robot_interface`)
- Python 3 packages: `pip install mediapipe opencv-python numpy`

## Running an exercise

```bash
cd extras/qt_robot_app/squat_from_side
python run.py
```

Press **ESC** to exit early.

## Adding a new exercise

1. Create a new folder: `extras/qt_robot_app/<exercise_name>/`

2. Create `__init__.py` (empty) and `exercise_config.py`:

```python
CONFIG = {
    "name": "My Exercise",
    "camera_view": "side",          # "side" | "front" | "back"
    "client_instructions": "...",   # spoken at start
    "session_duration_secs": 60,
    "feedback_mode": ["after_window", "after_exercise"],
}
```

3. Copy `run.py` from `squat_from_side/run.py` — **only change the path in the first
   `sys.path.insert` line** if needed (it already points to `../lib`). No other changes needed.

4. Drop in the generated `analysis_module.py` from the physio portal — **no changes needed**.
   The module interface is identical: `detect_rep`, `generate_round_feedback`,
   `get_session_summary`, `get_relevant_joints`, `get_instructions`, `reset_round`.

## Feedback modes

| Mode | Behaviour |
|------|-----------|
| `after_window` | Round feedback spoken when session time expires |
| `after_rep` | Feedback spoken after each rep; exercise pauses then resumes |
| `during_exercise` | Short cue spoken after each rep (requires `generate_rep_cue`) |
| `after_exercise` | Session summary spoken at the end |

Use `["after_window", "after_exercise"]` in `exercise_config.py` for most exercises.
```

- [ ] **Step 2: Commit**

```bash
git add extras/qt_robot_app/README.md
git add extras/qt_robot_app/tests/
git commit -m "docs: add qt_robot_app README"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|-----------------|------|
| Self-contained package in `extras/qt_robot_app/` | Task 1 |
| `angle_calculator.py` verbatim copy | Task 1 |
| `data_contract.py` verbatim copy | Task 1 |
| ROS camera subscriber (pose_engine) | Task 3 |
| ROS speech service (tts) | Task 2 |
| cv2 overlay renderer (display) | Task 4 |
| Session state machine (session_runner) | Task 5 |
| Squat exercise config | Task 1 |
| Squat analysis_module verbatim copy | Task 1 |
| ROS node entry point (run.py) | Task 6 |
| README / how to add exercises | Task 7 |
| `feedback_type` field to distinguish rep vs window feedback | Task 5 |

**Placeholder scan:** No TBDs. All code blocks are complete and runnable.

**Type consistency:**
- `Display(window_name, exercise_secs)` — constructor in Task 4; called with named args in Task 6. ✓
- `SessionRunner(config, module)` — signature in Task 5 (copied from `qt/`); called in Task 6. ✓
- `state["feedback_type"]` — added in Task 5; read in Task 6. ✓
- `tts.speak()` / `tts.speak_sync()` / `tts.stop()` — defined in Task 2; called in Task 6. ✓
- `PoseEngine.start()` takes no args (ROS topic is hardcoded) — defined in Task 3; called in Task 6. ✓
