from client_app.services.session_service import SessionService, COUNTDOWN_SECS


_MINIMAL_MODULE = """
def detect_rep(pose_data):
    return False

def generate_round_feedback(round_data):
    return ["Good work."]

def get_session_summary(session_data):
    return "Session done."
"""


def test_session_service_uses_custom_exercise_secs():
    svc = SessionService(exercise_id="test", module_code=_MINIMAL_MODULE, exercise_secs=25)
    assert svc._exercise_secs == 25


def test_session_service_default_exercise_secs():
    svc = SessionService(exercise_id="test", module_code=_MINIMAL_MODULE)
    assert svc._exercise_secs == 10


def test_exercise_secs_not_importable_as_constant():
    import client_app.services.session_service as mod
    assert not hasattr(mod, "EXERCISE_SECS"), \
        "EXERCISE_SECS should be removed — use exercise_secs parameter instead"
