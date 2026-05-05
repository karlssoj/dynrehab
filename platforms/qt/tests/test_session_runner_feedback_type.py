import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

import types
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

_AFTER_REP_MODULE_CODE = _BEND_MODULE_CODE + """
def generate_rep_feedback(rep_data):
    return ["Rep done."]
"""


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
    runner._enter_exercise()
    # Fast-forward past exercise duration
    runner._state_wall_start -= 2.0

    result = runner.process_frame({"keypoints": {}})
    assert result["feedback_type"] == "window"


def test_feedback_type_rep():
    mod = _make_module(_AFTER_REP_MODULE_CODE)
    config = {
        "camera_view": "side",
        "session_duration_secs": 60,
        "feedback_mode": ["after_rep"],
    }
    runner = SessionRunner(config, mod)
    runner._enter_exercise()

    # Drive rep: descend
    for _ in range(3):
        runner.process_frame({"left_knee_bend_2d": 40.0, "keypoints": {}})
    # Complete rep
    result = runner.process_frame({"left_knee_bend_2d": 5.0, "keypoints": {}})

    assert result.get("feedback_lines") is not None
    assert result["feedback_type"] == "rep"


def test_feedback_type_after_end_feedback_reverts_to_window():
    """end_feedback() resumes exercise; subsequent window expiry should yield feedback_type=='window'."""
    mod = _make_module(_AFTER_REP_MODULE_CODE)
    config = {
        "camera_view": "side",
        "session_duration_secs": 60,
        "feedback_mode": ["after_rep", "after_window"],
    }
    runner = SessionRunner(config, mod)
    runner._enter_exercise()

    # Trigger a rep
    for _ in range(3):
        runner.process_frame({"left_knee_bend_2d": 40.0, "keypoints": {}})
    rep_result = runner.process_frame({"left_knee_bend_2d": 5.0, "keypoints": {}})
    assert rep_result["feedback_type"] == "rep"

    # Resume exercise after per-rep feedback
    runner.end_feedback()
    assert runner._state == "exercise"

    # Fast-forward past session duration to trigger window feedback
    runner._state_wall_start -= 70.0
    window_result = runner.process_frame({"left_knee_bend_2d": 5.0, "keypoints": {}})
    assert window_result["feedback_type"] == "window"


def test_feedback_type_present_in_non_feedback_states():
    """feedback_type field is always present in the result dict (initial dict path)."""
    mod = _make_module(_BEND_MODULE_CODE)
    config = {
        "camera_view": "side",
        "session_duration_secs": 60,
        "feedback_mode": ["after_window"],
    }
    runner = SessionRunner(config, mod)
    runner._enter_exercise()
    result = runner.process_frame({"left_knee_bend_2d": 10.0, "keypoints": {}})
    assert "feedback_type" in result
    assert result["state"] == "exercise"
