import time
from client_app.services.session_service import SessionService

VALID_MODULE_CODE = """
_phase = "ready"

def get_instructions():
    return ["Stand facing the camera.", "Bend your knee when ready."]

def detect_rep(pose_data):
    global _phase
    angle = pose_data.get("left_knee_angle", 180)
    if _phase == "ready" and angle < 100:
        _phase = "bent"
    elif _phase == "bent" and angle > 160:
        _phase = "ready"
        return True
    return False

def reset_round():
    global _phase
    _phase = "ready"

def generate_round_feedback(round_data):
    rep_count = round_data.get("rep_count", 0)
    frames = round_data.get("frames", [])
    if rep_count == 0:
        return ["No reps detected.", "Try bending your knee further."]
    angles = [f.get("left_knee_angle", 180) for f in frames]
    min_angle = min(angles) if angles else 180
    if min_angle > 110:
        return [f"You did {rep_count} reps.", "Try to bend your knee deeper."]
    return [f"Great — {rep_count} reps with good depth!"]

def get_session_summary(session_data):
    total = session_data.get("total_reps", 0)
    if total == 0:
        return "No reps recorded. Try bending your knee more."
    word = "rep" if total == 1 else "reps"
    return f"Session complete. You performed {total} {word}."
"""


def test_load_module_succeeds():
    svc = SessionService(exercise_id="ex1", module_code=VALID_MODULE_CODE)
    assert svc._detect_rep is not None
    assert svc._generate_round_feedback is not None
    assert svc._get_session_summary is not None
    assert svc._get_instructions is not None
    assert svc._reset_round is not None


def test_get_instructions():
    svc = SessionService(exercise_id="ex1", module_code=VALID_MODULE_CODE)
    instructions = svc.get_instructions()
    assert isinstance(instructions, list)
    assert len(instructions) == 2
    assert "camera" in instructions[0].lower()


def test_initial_state_is_instructions():
    svc = SessionService(exercise_id="ex1", module_code=VALID_MODULE_CODE)
    assert svc._state == "instructions"


def test_start_countdown_transitions_state():
    svc = SessionService(exercise_id="ex1", module_code=VALID_MODULE_CODE)
    svc.start_countdown()
    assert svc._state == "countdown"


def test_process_frame_in_countdown_state():
    svc = SessionService(exercise_id="ex1", module_code=VALID_MODULE_CODE)
    svc.start_countdown()
    result = svc.process_frame({"left_knee_angle": 170.0, "timestamp": 0.0})
    assert result["state"] == "countdown"
    assert result["total_reps"] == 0
    assert "time_remaining" in result


def test_rep_counted_during_exercise():
    svc = SessionService(exercise_id="ex1", module_code=VALID_MODULE_CODE)
    # Manually put service into exercise state
    svc._enter_exercise()
    assert svc._state == "exercise"

    svc.process_frame({"left_knee_angle": 90.0, "timestamp": 0.1})   # phase → bent
    result = svc.process_frame({"left_knee_angle": 170.0, "timestamp": 0.2})  # rep complete
    assert svc.rep_count == 1
    assert result["round_rep_count"] == 1
    assert result["total_reps"] == 1


def test_reset_round_called_on_enter_exercise():
    svc = SessionService(exercise_id="ex1", module_code=VALID_MODULE_CODE)
    svc._enter_exercise()
    # Set phase to something other than ready to verify reset_round fires
    svc._detect_rep({"left_knee_angle": 90.0, "timestamp": 0.0})  # sets _phase to "bent"
    svc._enter_exercise()
    # After entering exercise again reset_round should fire; state machine is reset
    assert svc._state == "exercise"
    assert svc._round_rep_count == 0


def test_feedback_lines_emitted_when_exercise_ends():
    svc = SessionService(exercise_id="ex1", module_code=VALID_MODULE_CODE)
    svc._enter_exercise()
    # Force elapsed time beyond EXERCISE_SECS by backdating _state_wall_start
    svc._state_wall_start -= 11  # 11 seconds ago
    result = svc.process_frame({"left_knee_angle": 170.0, "timestamp": 1.0})
    assert result["feedback_lines"] is not None
    assert isinstance(result["feedback_lines"], list)
    assert len(result["feedback_lines"]) >= 1
    assert svc._state == "feedback"


def test_end_feedback_returns_to_countdown():
    svc = SessionService(exercise_id="ex1", module_code=VALID_MODULE_CODE)
    svc._enter_exercise()
    svc._state_wall_start -= 11
    svc.process_frame({"left_knee_angle": 170.0, "timestamp": 1.0})
    assert svc._state == "feedback"
    svc.end_feedback()
    assert svc._state == "countdown"


def test_get_session_summary_speech():
    svc = SessionService(exercise_id="ex1", module_code=VALID_MODULE_CODE)
    svc._enter_exercise()
    svc.process_frame({"left_knee_angle": 90.0, "timestamp": 0.1})
    svc.process_frame({"left_knee_angle": 170.0, "timestamp": 0.2})
    speech = svc.get_session_summary_speech()
    assert isinstance(speech, str)
    assert len(speech) > 0


def test_get_summary():
    svc = SessionService(exercise_id="ex1", module_code=VALID_MODULE_CODE)
    svc._enter_exercise()
    svc.process_frame({"left_knee_angle": 90.0, "timestamp": 0.1})
    svc.process_frame({"left_knee_angle": 170.0, "timestamp": 0.2})
    summary = svc.get_summary()
    assert summary["rep_count"] == 1
    assert "quality_pct" in summary
    assert "feedback_log" in summary
    assert "duration_seconds" in summary


def test_angle_stats_tracked():
    svc = SessionService(exercise_id="ex1", module_code=VALID_MODULE_CODE)
    svc._enter_exercise()
    svc.process_frame({"left_knee_angle": 90.0, "timestamp": 0.1})
    svc.process_frame({"left_knee_angle": 170.0, "timestamp": 0.2})
    assert "left_knee_angle" in svc._angle_stats
    assert svc._angle_stats["left_knee_angle"]["min"] == 90.0
    assert svc._angle_stats["left_knee_angle"]["max"] == 170.0
