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
    # Force elapsed time beyond _exercise_secs (default 10 s) by backdating _state_wall_start
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
    assert isinstance(speech, list)
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


MODULE_WITH_REP_CUE = """
_phase = "ready"

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

def generate_rep_cue(cue_data):
    if cue_data.get("trigger") == "timeout":
        return "Keep going!"
    return "Good squat!"

def generate_round_feedback(round_data):
    return ["Round done."]

def get_session_summary(session_data):
    cues = session_data.get("rep_cues", [])
    return [f"Session done. {len(cues)} cues given."]
"""


def test_during_exercise_emits_rep_cue_after_rep():
    svc = SessionService(
        exercise_id="ex1", module_code=MODULE_WITH_REP_CUE,
        exercise_secs=10, feedback_mode=["during_exercise"],
    )
    svc._enter_exercise()
    svc.process_frame({"left_knee_angle": 90.0, "timestamp": 0.1})
    result = svc.process_frame({"left_knee_angle": 170.0, "timestamp": 0.2})
    assert result.get("rep_cue") == "Good squat!"


def test_during_exercise_timeout_cue_fires_after_5s():
    svc = SessionService(
        exercise_id="ex1", module_code=MODULE_WITH_REP_CUE,
        exercise_secs=60, feedback_mode=["during_exercise"],
    )
    svc._enter_exercise()
    svc._last_cue_time -= 6
    result = svc.process_frame({"left_knee_angle": 170.0, "timestamp": 0.0})
    assert result.get("rep_cue") == "Keep going!"


def test_no_after_window_mode_exercise_continues_past_window():
    svc = SessionService(
        exercise_id="ex1", module_code=MODULE_WITH_REP_CUE,
        exercise_secs=10, feedback_mode=["during_exercise"],
    )
    svc._enter_exercise()
    svc._state_wall_start -= 11
    result = svc.process_frame({"left_knee_angle": 170.0, "timestamp": 0.0})
    assert result["state"] == "exercise"
    assert result.get("feedback_lines") is None


def test_after_window_mode_still_triggers_feedback():
    svc = SessionService(
        exercise_id="ex1", module_code=MODULE_WITH_REP_CUE,
        exercise_secs=10, feedback_mode=["after_window"],
    )
    svc._enter_exercise()
    svc._state_wall_start -= 11
    result = svc.process_frame({"left_knee_angle": 170.0, "timestamp": 0.0})
    assert result["state"] == "feedback"
    assert result["feedback_lines"] is not None


def test_get_session_summary_speech_returns_list_with_rep_cues():
    svc = SessionService(
        exercise_id="ex1", module_code=MODULE_WITH_REP_CUE,
        exercise_secs=60, feedback_mode=["during_exercise", "after_exercise"],
    )
    svc._enter_exercise()
    svc._rep_cues.append("Good squat!")
    result = svc.get_session_summary_speech()
    assert isinstance(result, list)
    assert len(result) > 0
    assert "1 cues given" in result[0]


def test_during_exercise_no_timeout_cue_before_5s():
    svc = SessionService(
        exercise_id="ex1", module_code=MODULE_WITH_REP_CUE,
        exercise_secs=60, feedback_mode=["during_exercise"],
    )
    svc._enter_exercise()
    svc._last_cue_time = time.time() - 2
    result = svc.process_frame({"left_knee_angle": 170.0, "timestamp": 0.0})
    assert result.get("rep_cue") is None


# ── calibration state ────────────────────────────────────────────────────────

def _make_kp(x, y, vis):
    return (x, y, 0.0, vis)


def _full_body_pose(facing="side"):
    """Pose data with all critical keypoints visible and body spanning 80% of frame."""
    shoulder_sep = 0.05 if facing == "side" else 0.25
    kpts = {
        "nose":            _make_kp(0.5, 0.05, 0.9),
        "left_shoulder":   _make_kp(0.5 - shoulder_sep / 2, 0.20, 0.9),
        "right_shoulder":  _make_kp(0.5 + shoulder_sep / 2, 0.20, 0.9),
        "left_hip":        _make_kp(0.5, 0.45, 0.9),
        "right_hip":       _make_kp(0.5, 0.45, 0.9),
        "left_knee":       _make_kp(0.5, 0.65, 0.9),
        "right_knee":      _make_kp(0.5, 0.65, 0.9),
        "left_ankle":      _make_kp(0.5, 0.85, 0.9),
        "right_ankle":     _make_kp(0.5, 0.87, 0.9),
        "left_wrist":      _make_kp(0.4, 0.40, 0.9),
        "right_wrist":     _make_kp(0.6, 0.40, 0.9),
    }
    return {"keypoints": kpts}


def test_start_calibration_sets_state():
    svc = SessionService(exercise_id="ex1", module_code=VALID_MODULE_CODE)
    svc.start_calibration()
    assert svc._state == "calibration"


def test_calibration_no_person_detected():
    svc = SessionService(exercise_id="ex1", module_code=VALID_MODULE_CODE)
    svc.start_calibration()
    result = svc.process_frame({"keypoints": {}})
    assert result["calibration_status"] == "no_person"
    assert "calibration_speak" in result


def test_calibration_too_far_when_body_small():
    svc = SessionService(exercise_id="ex1", module_code=VALID_MODULE_CODE)
    svc.start_calibration()
    # nose at top, ankle only 40% below — body height < 75%
    kpts = {
        "nose":           _make_kp(0.5, 0.20, 0.9),
        "left_shoulder":  _make_kp(0.48, 0.30, 0.9),
        "right_shoulder": _make_kp(0.52, 0.30, 0.9),
        "left_hip":       _make_kp(0.5, 0.42, 0.9),
        "right_hip":      _make_kp(0.5, 0.42, 0.9),
        "left_knee":      _make_kp(0.5, 0.50, 0.9),
        "right_knee":     _make_kp(0.5, 0.50, 0.9),
        "left_ankle":     _make_kp(0.5, 0.58, 0.9),
        "right_ankle":    _make_kp(0.5, 0.58, 0.9),
    }
    result = svc.process_frame({"keypoints": kpts})
    assert result["calibration_status"] == "too_far"


def test_calibration_wrong_orientation_for_side_view():
    svc = SessionService(exercise_id="ex1", module_code=VALID_MODULE_CODE, camera_view="side")
    svc.start_calibration()
    result = svc.process_frame(_full_body_pose(facing="front"))
    assert result["calibration_status"] == "wrong_orientation"
    assert "sideways" in result.get("calibration_speak", "").lower()


def test_calibration_ready_when_correctly_positioned_side():
    svc = SessionService(exercise_id="ex1", module_code=VALID_MODULE_CODE, camera_view="side")
    svc.start_calibration()
    result = svc.process_frame(_full_body_pose(facing="side"))
    assert result["calibration_status"] == "ready"


def test_calibration_advances_to_countdown_after_hold():
    svc = SessionService(exercise_id="ex1", module_code=VALID_MODULE_CODE, camera_view="side")
    svc.start_calibration()
    svc._calibration_ready_since = time.time() - 2.0  # already been ready > 1.5s
    result = svc.process_frame(_full_body_pose(facing="side"))
    assert result["state"] == "countdown"
    assert result.get("calibration_speak") == "Good! Starting now."


def test_calibration_speak_respects_cooldown():
    svc = SessionService(exercise_id="ex1", module_code=VALID_MODULE_CODE)
    svc.start_calibration()
    r1 = svc.process_frame({"keypoints": {}})
    assert "calibration_speak" in r1
    r2 = svc.process_frame({"keypoints": {}})
    # second frame within cooldown window — should not emit again
    assert "calibration_speak" not in r2


def test_calibration_front_view_orientation():
    svc = SessionService(exercise_id="ex1", module_code=VALID_MODULE_CODE, camera_view="front")
    svc.start_calibration()
    result = svc.process_frame(_full_body_pose(facing="side"))
    assert result["calibration_status"] == "wrong_orientation"


def test_calibration_too_close_when_ankles_missing_and_upper_body_large():
    svc = SessionService(exercise_id="ex1", module_code=VALID_MODULE_CODE)
    svc.start_calibration()
    # Upper body spans 60% of frame (nose y=0.05, hip y=0.65) but no ankles visible
    kpts = {
        "nose":           _make_kp(0.5, 0.05, 0.9),
        "left_shoulder":  _make_kp(0.48, 0.20, 0.9),
        "right_shoulder": _make_kp(0.52, 0.20, 0.9),
        "left_hip":       _make_kp(0.5, 0.65, 0.9),
        "right_hip":      _make_kp(0.5, 0.65, 0.9),
        "left_knee":      _make_kp(0.5, 0.80, 0.9),
        "right_knee":     _make_kp(0.5, 0.80, 0.9),
        # no ankles
    }
    result = svc.process_frame({"keypoints": kpts})
    assert result["calibration_status"] == "too_close"
    assert "back" in result.get("calibration_speak", "").lower()


def test_calibration_wrong_orientation_detected_despite_low_visibility_joints():
    """Orientation check fires even when far-side joints have low visibility.

    When facing camera (wrong for a side exercise), far-side knee/ankle keypoints
    can have lower MediaPipe confidence.  The fix moves the shoulder-based orientation
    check before the full-visibility check so the correct message is emitted.
    """
    svc = SessionService(exercise_id="ex1", module_code=VALID_MODULE_CODE, camera_view="side")
    svc.start_calibration()
    # Front-facing pose: large shoulder separation, but some lower-body joints at vis=0.25
    # (below _CALIB_VIS_HIGH=0.4 — would have incorrectly triggered "too_far" in the old code)
    kpts = {
        "nose":            _make_kp(0.5,  0.05, 0.9),
        "left_shoulder":   _make_kp(0.35, 0.20, 0.9),   # shoulder_sep = 0.30 > 0.13
        "right_shoulder":  _make_kp(0.65, 0.20, 0.9),
        "left_hip":        _make_kp(0.45, 0.45, 0.9),
        "right_hip":       _make_kp(0.55, 0.45, 0.9),
        "left_knee":       _make_kp(0.45, 0.65, 0.25),  # low visibility — occluded from front
        "right_knee":      _make_kp(0.55, 0.65, 0.9),
        "left_ankle":      _make_kp(0.45, 0.85, 0.25),  # low visibility
        "right_ankle":     _make_kp(0.55, 0.87, 0.9),
    }
    result = svc.process_frame({"keypoints": kpts})
    assert result["calibration_status"] == "wrong_orientation"
    assert result.get("calibration_message") == "Turn sideways to the camera."


def test_calibration_message_always_present_in_result():
    svc = SessionService(exercise_id="ex1", module_code=VALID_MODULE_CODE, camera_view="side")
    svc.start_calibration()
    result = svc.process_frame(_full_body_pose(facing="front"))
    assert "calibration_message" in result
    assert result["calibration_message"] == "Turn sideways to the camera."
