import time

from client_app.services.session_service import SessionService, COUNTDOWN_SECS

_MODULE_CODE = """
def detect_rep(pose_data):
    return True

def generate_round_feedback(round_data):
    return ["Good work!"]

def generate_rep_feedback(rep_data):
    return ["Rep done."]

def get_session_summary(session_data):
    return "Session done."

def reset_round():
    pass
"""


def _spine_pose(thoracic, lumbar):
    return {"timestamp": 0.0, "spine_thoracic_curvature": thoracic, "spine_lumbar_curvature": lumbar}


_NO_SPINE_POSE = {"timestamp": 0.0, "spine_thoracic_curvature": None, "spine_lumbar_curvature": None}


def _new_service(**kwargs):
    return SessionService(exercise_id="ex1", module_code=_MODULE_CODE, **kwargs)


def test_countdown_latches_last_valid_reading_and_ignores_later_none():
    svc = _new_service()
    svc.start_countdown()
    svc.process_frame(_NO_SPINE_POSE)
    svc.process_frame(_spine_pose(0.1, -0.2))
    svc.process_frame(_NO_SPINE_POSE)
    assert svc._pending_spine_baseline == {"thoracic": 0.1, "lumbar": -0.2}


def test_baseline_committed_when_countdown_reaches_zero():
    svc = _new_service()
    svc.start_countdown()
    svc.process_frame(_spine_pose(0.1, -0.2))
    svc._state_wall_start = time.time() - (COUNTDOWN_SECS + 0.6)
    result = svc.process_frame(_NO_SPINE_POSE)
    assert result["state"] == "exercise"
    assert svc._spine_baseline == {"thoracic": 0.1, "lumbar": -0.2}
    assert svc._pending_spine_baseline is None


def test_baseline_is_none_when_no_valid_reading_during_countdown():
    svc = _new_service()
    svc.start_countdown()
    svc.process_frame(_NO_SPINE_POSE)
    svc._state_wall_start = time.time() - (COUNTDOWN_SECS + 0.6)
    svc.process_frame(_NO_SPINE_POSE)
    assert svc._spine_baseline is None


def test_baseline_unchanged_while_exercise_state_is_active():
    svc = SessionService(exercise_id="ex1", module_code=_MODULE_CODE, feedback_mode=["after_exercise"])
    svc.start_countdown()
    svc.process_frame(_spine_pose(0.2, -0.3))
    svc._state_wall_start = time.time() - (COUNTDOWN_SECS + 0.6)
    svc.process_frame(_NO_SPINE_POSE)
    assert svc._state == "exercise"

    svc.process_frame(_spine_pose(0.9, -0.9))
    assert svc._spine_baseline == {"thoracic": 0.2, "lumbar": -0.3}


def test_after_rep_resume_preserves_round_baseline():
    svc = SessionService(exercise_id="ex1", module_code=_MODULE_CODE, feedback_mode=["after_rep"])
    svc.start_countdown()
    svc.process_frame(_spine_pose(0.4, -0.1))
    svc._state_wall_start = time.time() - (COUNTDOWN_SECS + 0.6)
    svc.process_frame(_NO_SPINE_POSE)
    assert svc._spine_baseline == {"thoracic": 0.4, "lumbar": -0.1}

    svc.process_frame(_NO_SPINE_POSE)  # triggers the rep -> pauses into rep feedback
    assert svc._state == "feedback"
    svc.end_feedback()
    assert svc._state == "exercise"
    assert svc._spine_baseline == {"thoracic": 0.4, "lumbar": -0.1}


def test_round_data_carries_spine_baseline():
    svc = SessionService(exercise_id="ex1", module_code=_MODULE_CODE, feedback_mode=["after_window"],
                          exercise_secs=10)
    svc.start_countdown()
    svc.process_frame(_spine_pose(0.15, -0.25))
    svc._state_wall_start = time.time() - (COUNTDOWN_SECS + 0.6)
    svc.process_frame(_NO_SPINE_POSE)
    assert svc._state == "exercise"

    svc._state_wall_start = time.time() - 11.0
    svc.process_frame(_NO_SPINE_POSE)

    assert svc._all_rounds[-1]["spine_baseline"] == {"thoracic": 0.15, "lumbar": -0.25}


def test_rep_data_carries_spine_baseline():
    svc = SessionService(exercise_id="ex1", module_code=_MODULE_CODE, feedback_mode=["after_rep"])
    svc.start_countdown()
    svc.process_frame(_spine_pose(0.6, -0.6))
    svc._state_wall_start = time.time() - (COUNTDOWN_SECS + 0.6)
    svc.process_frame(_NO_SPINE_POSE)
    svc.process_frame(_NO_SPINE_POSE)  # triggers the rep

    assert svc._rep_snapshots[-1]["spine_baseline"] == {"thoracic": 0.6, "lumbar": -0.6}


def test_session_summary_rounds_carry_spine_baseline():
    svc = SessionService(exercise_id="ex1", module_code=_MODULE_CODE,
                          feedback_mode=["after_window", "after_exercise"], exercise_secs=10)
    svc.start_countdown()
    svc.process_frame(_spine_pose(0.25, -0.15))
    svc._state_wall_start = time.time() - (COUNTDOWN_SECS + 0.6)
    svc.process_frame(_NO_SPINE_POSE)
    svc._state_wall_start = time.time() - 11.0
    svc.process_frame(_NO_SPINE_POSE)

    svc.get_session_summary_speech()
    assert svc._all_rounds[-1]["spine_baseline"] == {"thoracic": 0.25, "lumbar": -0.15}
