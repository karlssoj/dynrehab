from core.module_validator import validate_module

VALID_CODE = """
_phase = "ready"

def detect_rep(pose_data):
    return False

def generate_round_feedback(round_data):
    return ["Round complete."]

def get_session_summary(session_data):
    return "Session done."
"""


def test_valid_module_passes():
    result = validate_module(VALID_CODE)
    assert result["valid"] is True
    assert result["error"] is None


def test_missing_detect_rep():
    code = "def generate_round_feedback(d): return []\ndef get_session_summary(d): return ''"
    result = validate_module(code)
    assert result["valid"] is False
    assert "detect_rep" in result["error"]


def test_missing_generate_round_feedback():
    code = "def detect_rep(d): return False\ndef get_session_summary(d): return ''"
    result = validate_module(code)
    assert result["valid"] is False
    assert "generate_round_feedback" in result["error"]


def test_missing_get_session_summary():
    code = "def detect_rep(d): return False\ndef generate_round_feedback(d): return []"
    result = validate_module(code, feedback_mode=["after_exercise"])
    assert result["valid"] is False
    assert "get_session_summary" in result["error"]


def test_get_session_summary_not_required_without_after_exercise_mode():
    code = "def detect_rep(d): return False\ndef generate_round_feedback(d): return []"
    result = validate_module(code, feedback_mode=["after_window"])
    assert result["valid"] is True


def test_generate_rep_cue_required_in_during_exercise_mode():
    code = "def detect_rep(d): return False"
    result = validate_module(code, feedback_mode=["during_exercise"])
    assert result["valid"] is False
    assert "generate_rep_cue" in result["error"]


def test_mode_only_requires_its_own_function():
    code = "def detect_rep(d): return False\ndef generate_rep_cue(d): return 'Go!'"
    result = validate_module(code, feedback_mode=["during_exercise"])
    assert result["valid"] is True


def test_banned_import_os():
    code = VALID_CODE + "\nimport os"
    result = validate_module(code)
    assert result["valid"] is False
    assert "os" in result["error"]


def test_banned_import_subprocess():
    code = VALID_CODE + "\nimport subprocess"
    result = validate_module(code)
    assert result["valid"] is False
    assert "subprocess" in result["error"]


def test_banned_from_import():
    code = VALID_CODE + "\nfrom os import path"
    result = validate_module(code)
    assert result["valid"] is False


def test_syntax_error():
    code = "def detect_rep(: pass"
    result = validate_module(code)
    assert result["valid"] is False
    assert "syntax" in result["error"].lower()


def test_math_import_allowed():
    code = VALID_CODE + "\nimport math"
    result = validate_module(code)
    assert result["valid"] is True


def test_nested_function_not_accepted():
    code = """
def wrapper():
    def detect_rep(pose_data):
        return False
    def generate_round_feedback(round_data):
        return []
    def get_session_summary(session_data):
        return ""
"""
    result = validate_module(code)
    assert result["valid"] is False
    assert "detect_rep" in result["error"]


def test_optional_functions_not_required():
    # get_instructions and reset_round are optional — module should be valid without them
    code = """
def detect_rep(pose_data):
    return False

def generate_round_feedback(round_data):
    return ["Done."]

def get_session_summary(session_data):
    return "Summary."
"""
    result = validate_module(code)
    assert result["valid"] is True
