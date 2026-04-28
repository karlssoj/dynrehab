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
    feedback_mode = '["after_window"]'


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


import sqlite3
import sys
from pathlib import Path

# Give the test access to the DB schema helpers
sys.path.insert(0, str(Path(__file__).parent.parent))

from physio_app.db import init_db
from physio_app.services.standalone_generator import StandaloneGenerator


def _seed_db(tmp_path):
    """Create an in-memory DB with one exercise + one validated module."""
    db_path = str(tmp_path / "test.db")
    conn = init_db(db_path)
    conn.execute(
        "INSERT INTO exercises (id, name, camera_view, client_instructions, "
        "llm_instructions, boundary_values, display_values, session_duration_secs) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("ex-1", "Squat Press", "side", "Stand tall.", "", "", "left_knee_bend_2d", 45),
    )
    conn.execute(
        "INSERT INTO analysis_modules (id, exercise_id, version, code, status, is_active) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("mod-1", "ex-1", 1, "def detect_rep(p): return False\n"
         "def generate_round_feedback(d): return ['Good']\n"
         "def get_session_summary(d): return 'Done'\n",
         "validated", 1),
    )
    conn.commit()
    return conn


def test_generate_creates_exercise_dir(tmp_path, monkeypatch):
    import physio_app.services.standalone_generator as gen_mod
    monkeypatch.setattr(gen_mod, "_QT_DIR", tmp_path / "qt")
    monkeypatch.setattr(gen_mod, "_CORE_SRC", Path(__file__).parent.parent / "core")
    conn = _seed_db(tmp_path)
    StandaloneGenerator.generate("ex-1", conn)
    assert (tmp_path / "qt" / "squat_press").is_dir()


def test_generate_writes_run_py(tmp_path, monkeypatch):
    import physio_app.services.standalone_generator as gen_mod
    monkeypatch.setattr(gen_mod, "_QT_DIR", tmp_path / "qt")
    monkeypatch.setattr(gen_mod, "_CORE_SRC", Path(__file__).parent.parent / "core")
    conn = _seed_db(tmp_path)
    StandaloneGenerator.generate("ex-1", conn)
    run_py = tmp_path / "qt" / "squat_press" / "run.py"
    assert run_py.exists()
    content = run_py.read_text()
    assert "StandaloneApp" in content
    assert "exercise_config" in content


def test_generate_writes_exercise_config(tmp_path, monkeypatch):
    import physio_app.services.standalone_generator as gen_mod
    monkeypatch.setattr(gen_mod, "_QT_DIR", tmp_path / "qt")
    monkeypatch.setattr(gen_mod, "_CORE_SRC", Path(__file__).parent.parent / "core")
    conn = _seed_db(tmp_path)
    StandaloneGenerator.generate("ex-1", conn)
    config_py = tmp_path / "qt" / "squat_press" / "exercise_config.py"
    assert config_py.exists()
    ns = {}
    exec(config_py.read_text(), ns)
    assert ns["CONFIG"]["name"] == "Squat Press"
    assert ns["CONFIG"]["session_duration_secs"] == 45


def test_generate_writes_analysis_module(tmp_path, monkeypatch):
    import physio_app.services.standalone_generator as gen_mod
    monkeypatch.setattr(gen_mod, "_QT_DIR", tmp_path / "qt")
    monkeypatch.setattr(gen_mod, "_CORE_SRC", Path(__file__).parent.parent / "core")
    conn = _seed_db(tmp_path)
    StandaloneGenerator.generate("ex-1", conn)
    mod_py = tmp_path / "qt" / "squat_press" / "analysis_module.py"
    assert mod_py.exists()
    assert "detect_rep" in mod_py.read_text()


def test_generate_syncs_lib_core(tmp_path, monkeypatch):
    import physio_app.services.standalone_generator as gen_mod
    monkeypatch.setattr(gen_mod, "_QT_DIR", tmp_path / "qt")
    monkeypatch.setattr(gen_mod, "_CORE_SRC", Path(__file__).parent.parent / "core")
    conn = _seed_db(tmp_path)
    StandaloneGenerator.generate("ex-1", conn)
    assert (tmp_path / "qt" / "lib" / "core" / "pose_engine.py").exists()
    assert (tmp_path / "qt" / "lib" / "core" / "angle_calculator.py").exists()
    assert (tmp_path / "qt" / "lib" / "core" / "data_contract.py").exists()


def test_generate_skips_missing_exercise(tmp_path, monkeypatch):
    import physio_app.services.standalone_generator as gen_mod
    monkeypatch.setattr(gen_mod, "_QT_DIR", tmp_path / "qt")
    monkeypatch.setattr(gen_mod, "_CORE_SRC", Path(__file__).parent.parent / "core")
    conn = _seed_db(tmp_path)
    StandaloneGenerator.generate("no-such-id", conn)  # must not raise
    assert not (tmp_path / "qt").exists()


def test_render_config_includes_feedback_mode():
    from physio_app.services.standalone_generator import _render_config
    from physio_app.services.exercise_service import Exercise

    ex = Exercise(
        id="x", name="Squat", camera_view="side",
        client_instructions="Stand", llm_instructions="",
        boundary_values="", display_values="",
        session_duration_secs=10,
        feedback_mode='["during_exercise","after_window"]',
    )
    config_code = _render_config(ex)
    assert '"feedback_mode"' in config_code
    assert "during_exercise" in config_code
