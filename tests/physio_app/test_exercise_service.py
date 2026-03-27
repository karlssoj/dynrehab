import pytest
from physio_app.services.exercise_service import ExerciseService


def test_create_and_get(tmp_db):
    svc = ExerciseService(tmp_db)
    ex = svc.create("Squat", "side", "Keep knees aligned")
    assert ex.id is not None
    fetched = svc.get(ex.id)
    assert fetched.name == "Squat"
    assert fetched.camera_view == "side"
    assert fetched.instructions_text == "Keep knees aligned"


def test_list_all(tmp_db):
    svc = ExerciseService(tmp_db)
    svc.create("Squat", "side", "")
    svc.create("Lunge", "front", "")
    exercises = svc.list_all()
    assert len(exercises) == 2
    names = {e.name for e in exercises}
    assert names == {"Squat", "Lunge"}


def test_update(tmp_db):
    svc = ExerciseService(tmp_db)
    ex = svc.create("Squat", "side", "")
    updated = svc.update(ex.id, name="Deep Squat", instructions_text="Go deeper")
    assert updated.name == "Deep Squat"
    assert updated.instructions_text == "Go deeper"


def test_delete(tmp_db):
    svc = ExerciseService(tmp_db)
    ex = svc.create("Squat", "side", "")
    svc.delete(ex.id)
    assert svc.get(ex.id) is None


def test_save_module_versions(tmp_db):
    svc = ExerciseService(tmp_db)
    ex = svc.create("Squat", "side", "")
    m1 = svc.save_module(ex.id, "def analyze_frame(d): return []", "validated")
    m2 = svc.save_module(ex.id, "def analyze_frame(d): return [1]", "validated")
    assert m1["version"] == 1
    assert m2["version"] == 2
    # Only m2 should be active
    active = svc.get_active_module(ex.id)
    assert active["id"] == m2["id"]
    assert active["is_active"] == 1


def test_list_modules(tmp_db):
    svc = ExerciseService(tmp_db)
    ex = svc.create("Squat", "side", "")
    svc.save_module(ex.id, "code_v1", "validated")
    svc.save_module(ex.id, "code_v2", "validated")
    modules = svc.list_modules(ex.id)
    assert len(modules) == 2
    assert modules[0]["version"] == 2  # newest first
