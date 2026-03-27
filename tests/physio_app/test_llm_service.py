from unittest.mock import MagicMock, patch
from physio_app.services.llm_service import LLMService, build_prompt


def test_build_prompt_contains_instructions():
    prompt = build_prompt("Squat", "side", "Keep knees aligned with toes")
    assert "Keep knees aligned with toes" in prompt


def test_build_prompt_contains_function_names():
    prompt = build_prompt("Squat", "side", "instructions")
    assert "analyze_frame" in prompt
    assert "detect_rep" in prompt
    assert "on_rep_complete" in prompt


def test_build_prompt_contains_angle_fields():
    prompt = build_prompt("Squat", "side", "instructions")
    assert "left_knee_angle" in prompt
    assert "trunk_lean_angle" in prompt


def test_generate_module_success(tmp_db, mocker):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text=(
            "def analyze_frame(pose_data): return []\n"
            "def detect_rep(pose_data): return False\n"
            "def on_rep_complete(rep_data): return []\n"
        ))]
    )
    mocker.patch("physio_app.services.llm_service.anthropic.Anthropic", return_value=mock_client)

    from physio_app.services.exercise_service import ExerciseService
    ex_svc = ExerciseService(tmp_db)
    ex = ex_svc.create("Squat", "side", "Keep knees aligned")

    svc = LLMService(tmp_db, api_key="test-key")
    result = svc.generate_module(ex.id, ex.name, ex.camera_view, ex.instructions_text)
    assert result["status"] == "validated"
    assert "analyze_frame" in result["code"]


def test_generate_module_logs_api_call(tmp_db, mocker):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text=(
            "def analyze_frame(d): return []\n"
            "def detect_rep(d): return False\n"
            "def on_rep_complete(d): return []\n"
        ))]
    )
    mocker.patch("physio_app.services.llm_service.anthropic.Anthropic", return_value=mock_client)

    from physio_app.services.exercise_service import ExerciseService
    ex_svc = ExerciseService(tmp_db)
    ex = ex_svc.create("Squat", "side", "")

    svc = LLMService(tmp_db, api_key="test-key")
    svc.generate_module(ex.id, ex.name, ex.camera_view, ex.instructions_text)

    log = tmp_db.execute("SELECT * FROM llm_logs WHERE exercise_id = ?", (ex.id,)).fetchone()
    assert log is not None
    assert "analyze_frame" in log["response"]
