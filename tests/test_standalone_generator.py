import pytest
from physio_app.services.standalone_generator import _slugify, _render_config


def test_slugify_basic():
    assert _slugify("Squat") == "squat"


def test_slugify_spaces():
    assert _slugify("Squat Press") == "squat_press"


def test_slugify_special_chars():
    assert _slugify("Squat (Side)") == "squat_side"


def test_slugify_empty():
    assert _slugify("") == "exercise"


def test_slugify_leading_trailing():
    assert _slugify("  squat  ") == "squat"


class _FakeExercise:
    name = "Squat"
    camera_view = "side"
    client_instructions = "Stand with feet shoulder-width apart."
    display_values = "left_knee_bend_2d"
    session_duration_secs = 60


def test_render_config_contains_name():
    code = _render_config(_FakeExercise())
    assert '"name"' in code
    assert "Squat" in code


def test_render_config_contains_duration():
    code = _render_config(_FakeExercise())
    assert "60" in code


def test_render_config_is_valid_python():
    code = _render_config(_FakeExercise())
    ns = {}
    exec(code, ns)
    assert ns["CONFIG"]["name"] == "Squat"
    assert ns["CONFIG"]["session_duration_secs"] == 60


def test_render_config_handles_quotes():
    ex = _FakeExercise()
    ex.name = 'Quad\'s "Burn"'
    code = _render_config(ex)
    ns = {}
    exec(code, ns)
    assert ns["CONFIG"]["name"] == 'Quad\'s "Burn"'
