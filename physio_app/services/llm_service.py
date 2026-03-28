import sqlite3
import uuid
import anthropic
from core.module_validator import validate_module
from physio_app.services.exercise_service import ExerciseService

_POSE_DATA_DESCRIPTION = """\
The `pose_data` dict has these fields (all angles in degrees, calculated in 3D using x,y,z):
  timestamp: float — seconds since session start
  left_knee_angle / right_knee_angle — hip-knee-ankle angle; ~170° standing, ~90° deep squat
  left_hip_angle / right_hip_angle — shoulder-hip-knee angle; ~180° standing
  left_ankle_angle / right_ankle_angle — knee-ankle-foot_index angle
  left_shoulder_angle / right_shoulder_angle — elbow-shoulder-hip angle
  left_elbow_angle / right_elbow_angle — shoulder-elbow-wrist angle
  left_wrist_angle / right_wrist_angle — elbow-wrist-index_finger angle
  trunk_lean_angle — trunk from vertical; 0°=upright, increases when leaning forward
  neck_angle — angle at shoulder-midpoint between trunk direction and nose
  pelvic_tilt — left-hip→right-hip line from horizontal; 0°=level
  left_hka_alignment / right_hka_alignment — frontal-plane hip-knee-ankle angle; ~180°=straight
  keypoints: dict[str, tuple[float,float,float,float]] — name→(x,y,z,visibility)
    x,y in [0,1] (y increases downward), z=depth, visibility in [0,1]
    landmark names: nose, left_shoulder, right_shoulder, left_elbow, right_elbow,
    left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee,
    left_ankle, right_ankle, left_foot_index, right_foot_index, left_heel, right_heel
"""

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


def build_prompt(exercise_name: str, camera_view: str, instructions: str) -> str:
    return f"""\
You are generating a Python movement analysis module for a physiotherapy application.

Exercise: {exercise_name}
Camera view: {camera_view}

Physiotherapist instructions:
{instructions}

Available pose data fields:
{_POSE_DATA_DESCRIPTION}

{_FEW_SHOT}

Now generate the analysis module for '{exercise_name}' following the same structure.

{_FUNCTION_SPEC}

Rules:
- Only import math or statistics if needed. Do NOT import os, sys, subprocess, socket, or requests.
- Use module-level variables for state (rep phase tracking etc.).
- Return ONLY valid Python code. No markdown fences. No explanations.
"""


class LLMService:
    def __init__(self, conn: sqlite3.Connection, api_key: str):
        self.conn = conn
        self.ex_svc = ExerciseService(conn)
        self._client = anthropic.Anthropic(api_key=api_key)

    def generate_module(self, exercise_id: str, name: str, camera_view: str,
                        instructions: str) -> dict:
        """Call Claude, validate the result, store it. Returns the saved module dict."""
        prompt = build_prompt(name, camera_view, instructions)
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

    def _log(self, exercise_id: str, prompt: str, response: str):
        self.conn.execute(
            "INSERT INTO llm_logs (id, exercise_id, prompt, response) VALUES (?, ?, ?, ?)",
            (str(uuid.uuid4()), exercise_id, prompt, response),
        )
        self.conn.commit()
