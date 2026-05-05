# QTRobot Modular Exercise App

Modular physiotherapy exercise app for QTRobot. Uses MediaPipe pose estimation
and the same analysis module interface as the physio portal's desktop app (`qt/`).

## Prerequisites

- ROS (with `rospy`, `sensor_msgs`, `cv_bridge`, `qt_robot_interface`)
- Python 3 packages: `pip install mediapipe opencv-python numpy`

## Running an exercise

```bash
cd platforms/qt/squat_from_side
python run.py
```

Press **ESC** to exit early.

## Adding a new exercise

1. Create a new folder: `platforms/qt/<exercise_name>/`

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

3. Copy `run.py` from `squat_from_side/run.py` — no changes needed. The path inserts
   use `Path(__file__).parent` so they resolve correctly from any exercise folder.

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
