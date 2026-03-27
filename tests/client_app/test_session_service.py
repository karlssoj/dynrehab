import time
from unittest.mock import MagicMock
from client_app.services.session_service import SessionService

VALID_MODULE_CODE = """
_phase = "down"

def analyze_frame(pose_data):
    if pose_data.get("left_knee_angle", 180) < 100:
        return [{"message": "Bend your knee more", "joint": "left_knee"}]
    return []

def detect_rep(pose_data):
    global _phase
    angle = pose_data.get("left_knee_angle", 180)
    if _phase == "down" and angle < 100:
        _phase = "up"
    elif _phase == "up" and angle > 160:
        _phase = "down"
        return True
    return False

def on_rep_complete(rep_data):
    frames = rep_data["frames"]
    min_angle = min(f.get("left_knee_angle", 180) for f in frames)
    if min_angle > 110:
        return [{"message": "Go deeper", "joint": None}]
    return [{"message": "Great squat!", "joint": None}]
"""


def test_load_module_succeeds():
    svc = SessionService(exercise_id="ex1", module_code=VALID_MODULE_CODE)
    assert svc._analyze_frame is not None
    assert svc._detect_rep is not None
    assert svc._on_rep_complete is not None


def test_process_frame_no_violation():
    svc = SessionService(exercise_id="ex1", module_code=VALID_MODULE_CODE)
    result = svc.process_frame({"left_knee_angle": 170.0, "timestamp": 0.0})
    assert result["feedback"] == []
    assert result["rep_completed"] is False


def test_process_frame_violation_feedback():
    svc = SessionService(exercise_id="ex1", module_code=VALID_MODULE_CODE)
    result = svc.process_frame({"left_knee_angle": 80.0, "timestamp": 0.0})
    assert len(result["feedback"]) == 1
    assert result["feedback"][0]["message"] == "Bend your knee more"


def test_rep_detection_and_count():
    svc = SessionService(exercise_id="ex1", module_code=VALID_MODULE_CODE)
    svc.process_frame({"left_knee_angle": 90.0, "timestamp": 0.1})   # phase → up
    result = svc.process_frame({"left_knee_angle": 170.0, "timestamp": 0.5})  # rep complete
    assert result["rep_completed"] is True
    assert svc.rep_count == 1


def test_on_rep_complete_feedback():
    svc = SessionService(exercise_id="ex1", module_code=VALID_MODULE_CODE)
    svc.process_frame({"left_knee_angle": 90.0, "timestamp": 0.1})
    result = svc.process_frame({"left_knee_angle": 170.0, "timestamp": 0.5})
    assert result["rep_completed"] is True
    assert len(result["post_rep_feedback"]) == 1


def test_session_summary():
    svc = SessionService(exercise_id="ex1", module_code=VALID_MODULE_CODE)
    # Complete 2 reps
    for _ in range(2):
        svc.process_frame({"left_knee_angle": 90.0, "timestamp": 0.0})
        svc.process_frame({"left_knee_angle": 170.0, "timestamp": 0.5})
    summary = svc.get_summary()
    assert summary["rep_count"] == 2
    assert "quality_pct" in summary
    assert "feedback_log" in summary
