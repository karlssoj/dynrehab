# QTRobot Modular Exercise App Design

**Goal:** A self-contained, modular exercise app living in `extras/qt_robot_app/` that runs on QTRobot, uses MediaPipe pose estimation and static angle calculation from the existing `qt/` library, and accepts generated `analysis_module.py` files dropped into an exercise folder with zero modification.

**Architecture:** Self-contained package (Approach A). Core library files that are identical to `qt/lib/` are copied verbatim. Three robot-specific files replace the CustomTkinter UI and desktop voice: a ROS-subscribed pose engine, a ROS speech wrapper, and a cv2 display renderer. The session state machine logic is preserved unchanged.

**Tech Stack:** Python 3, MediaPipe, OpenCV, ROS (rospy + cv_bridge), QTRobot speech service.

---

## Directory Structure

```
extras/qt_robot_app/
├── lib/
│   ├── __init__.py
│   ├── requirements.txt
│   ├── core/
│   │   ├── __init__.py
│   │   ├── angle_calculator.py   # verbatim copy from qt/lib/core/
│   │   ├── data_contract.py      # verbatim copy from qt/lib/core/
│   │   └── pose_engine.py        # adapted: ROS camera subscriber
│   ├── session_runner.py         # adapted: no CustomTkinter, returns state dict
│   ├── display.py                # cv2 overlay renderer
│   └── tts.py                    # ROS speech service wrapper
├── squat_from_side/
│   ├── run.py                    # ROS node entry point
│   ├── exercise_config.py        # identical format to qt/
│   └── analysis_module.py        # identical to qt/squat_from_side/
└── README.md                     # how to add new exercises
```

---

## Component Specifications

### `lib/core/angle_calculator.py` and `lib/core/data_contract.py`

Verbatim copies of `qt/lib/core/angle_calculator.py` and `qt/lib/core/data_contract.py`. No changes. These provide the full set of 30+ computed angle fields and the `PoseFrame` dataclass.

### `lib/core/pose_engine.py`

Replaces `cv2.VideoCapture` with a ROS subscriber:

- Subscribes to `/camera/color/image_raw` (sensor_msgs/Image) via `rospy`
- Converts each incoming message to a cv2 frame using `cv_bridge.CvBridge`
- Passes the frame through MediaPipe Pose (identical settings: complexity=1, detection=0.5, tracking=0.5)
- Extracts 33 keypoints, calls `calculate_angles()`, creates a `PoseFrame`
- Calls all registered subscriber callbacks with `(pose_frame, annotated_frame)`
- Skeleton overlay drawing is identical to `qt/lib/core/pose_engine.py`

### `lib/tts.py`

Single `speak(text: str) -> None` function:

- Calls `/qt_robot/speech/say` as a blocking ROS service call (waits for completion)
- Replaces the `voice.py` subprocess + `message_queue.jsonl` + `voice_done.json` handshake entirely
- Session runner calls `tts.speak(line)` directly; no IPC needed

### `lib/display.py`

`Display` class wrapping a `cv2.imshow` window:

- `update(annotated_frame, state_dict)` — renders overlays on the annotated frame and shows it
- Overlays rendered:
  - **Top-left:** Rep count (large text)
  - **Top-right:** State label (`CALIBRATING` / `COUNTDOWN: N` / `EXERCISE` / `FEEDBACK`)
  - **Right edge:** Live joint angle values from `get_relevant_joints()` (label: value)
  - **Bottom:** Feedback lines (one per line, wrapping if long)
- `close()` — destroys cv2 window

### `lib/session_runner.py`

Same state machine as `qt/lib/session_runner.py`:

**States:** `calibration → countdown → exercise → feedback → summary`

**Changes from qt/ version:**
- No CustomTkinter imports or UI calls
- `process_frame(pose_data)` returns a `state_dict`:
  ```python
  {
      "state": str,           # current state name
      "rep_count": int,
      "countdown": int,       # seconds remaining (countdown state only)
      "feedback_lines": list, # lines to display/speak (feedback state only)
      "joint_values": dict,   # {label: value} for live display
      "done": bool,           # True when session complete
  }
  ```
- TTS calls: `tts.speak(text)` directly (blocking), no message queue; the `tts` module is passed into `SessionRunner.__init__()` so it can be called internally
- Calibration logic, rep detection, feedback mode routing, and session summary logic are identical to `qt/lib/session_runner.py`

All four feedback modes are supported: `after_rep`, `after_window`, `during_exercise`, `after_exercise`.

### `<exercise>/run.py`

ROS node entry point — minimal, ~20 lines:

```python
import rospy
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.core.pose_engine import PoseEngine
from lib.session_runner import SessionRunner
from lib.display import Display
from lib import tts
import exercise_config
import analysis_module

def main():
    rospy.init_node("qt_exercise")
    runner = SessionRunner(exercise_config.CONFIG, analysis_module)
    display = Display()
    engine = PoseEngine()

    def on_frame(pose_frame, annotated_frame):
        state = runner.process_frame(pose_frame.to_dict())
        display.update(annotated_frame, state)
        if state["done"]:
            engine.stop()
            rospy.signal_shutdown("done")

    engine.subscribe(on_frame)
    engine.start()
    rospy.spin()

if __name__ == "__main__":
    main()
```

### `<exercise>/exercise_config.py`

Identical format to `qt/`:

```python
CONFIG = {
    "name": "Squat from side",
    "camera_view": "side",          # "side" | "front" | "back"
    "client_instructions": "...",
    "session_duration_secs": 60,
    "feedback_mode": ["after_rep", "after_exercise"],
}
```

### `<exercise>/analysis_module.py`

Identical interface to `qt/`. Generated modules from the physio portal can be copied here without modification:

```python
def get_instructions() -> list
def detect_rep(pose_data: dict) -> bool
def reset_round()
def generate_rep_feedback(rep_data: dict) -> list    # required if "after_rep" in feedback_mode
def generate_round_feedback(round_data: dict) -> list  # required if "after_window" in feedback_mode
def get_session_summary(session_data: dict) -> list
def get_relevant_joints() -> list
```

---

## Adding a New Exercise

1. Create `extras/qt_robot_app/<exercise_name>/` directory
2. Copy `run.py` from `squat_from_side/` — change only the two import lines at the bottom
3. Write `exercise_config.py` with name, camera_view, instructions, duration, feedback_mode
4. Drop in the generated `analysis_module.py` from the physio portal — no changes needed
5. Run with `python run.py` from the exercise directory (with ROS environment sourced)

---

## Data Flow

```
ROS /camera/color/image_raw
    → cv_bridge → cv2 frame
    → MediaPipe → 33 keypoints
    → angle_calculator → 30+ angle fields
    → PoseFrame.to_dict() → pose_data dict
    → SessionRunner.process_frame() → state_dict
        → analysis_module.detect_rep() [exercise state]
        → tts.speak() [blocking ROS call on state transitions]
        → analysis_module.generate_rep_feedback() / generate_round_feedback()
    → Display.update(annotated_frame, state_dict) → cv2.imshow
```

---

## Requirements

```
mediapipe>=0.10
opencv-python>=4.8
numpy>=1.24
rospy          # provided by ROS installation
cv_bridge      # provided by ROS installation
```
