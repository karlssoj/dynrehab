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
    # Called every frame. Return feedback dicts or [].
    # Each dict: {"message": str, "joint": str | None}
    # "joint" is a landmark name to highlight (e.g. "left_knee") or None.
    # Only return feedback when a rule is currently violated.

def detect_rep(pose_data: dict) -> bool:
    # Called every frame. Return True exactly once when a full rep is completed.
    # Use module-level state variables to track the rep phase.

def on_rep_complete(rep_data: dict) -> list[dict]:
    # Called once after detect_rep returns True.
    # rep_data: {"rep_number": int, "frames": list[dict], "duration_seconds": float}
    # Return feedback dicts or [].
"""

_FEW_SHOT = """\
Example — Side-view bicep curl:

```python
_phase = "down"
_CURL_UP = 60
_CURL_DOWN = 150

def analyze_frame(pose_data):
    feedback = []
    if pose_data["left_shoulder_angle"] < 160:
        feedback.append({"message": "Keep your upper arm still", "joint": "left_shoulder"})
    return feedback

def detect_rep(pose_data):
    global _phase
    avg = (pose_data["left_elbow_angle"] + pose_data["right_elbow_angle"]) / 2
    if _phase == "down" and avg < _CURL_UP:
        _phase = "up"
    elif _phase == "up" and avg > _CURL_DOWN:
        _phase = "down"
        return True
    return False

def on_rep_complete(rep_data):
    frames = rep_data["frames"]
    min_elbow = min(f["left_elbow_angle"] for f in frames)
    if min_elbow > 70:
        return [{"message": "Try to curl higher for full range of motion", "joint": None}]
    return [{"message": "Great curl!", "joint": None}]
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
