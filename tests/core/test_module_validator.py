from core.module_validator import validate_module

VALID_CODE = """
_phase = "down"

def analyze_frame(pose_data):
    return []

def detect_rep(pose_data):
    return False

def on_rep_complete(rep_data):
    return []
"""


def test_valid_module_passes():
    result = validate_module(VALID_CODE)
    assert result["valid"] is True
    assert result["error"] is None


def test_missing_analyze_frame():
    code = "def detect_rep(d): return False\ndef on_rep_complete(d): return []"
    result = validate_module(code)
    assert result["valid"] is False
    assert "analyze_frame" in result["error"]


def test_missing_detect_rep():
    code = "def analyze_frame(d): return []\ndef on_rep_complete(d): return []"
    result = validate_module(code)
    assert result["valid"] is False
    assert "detect_rep" in result["error"]


def test_missing_on_rep_complete():
    code = "def analyze_frame(d): return []\ndef detect_rep(d): return False"
    result = validate_module(code)
    assert result["valid"] is False
    assert "on_rep_complete" in result["error"]


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
    code = "def analyze_frame(: pass"
    result = validate_module(code)
    assert result["valid"] is False
    assert "syntax" in result["error"].lower()


def test_math_import_allowed():
    code = VALID_CODE + "\nimport math"
    result = validate_module(code)
    assert result["valid"] is True
