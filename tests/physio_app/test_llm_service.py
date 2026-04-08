from unittest.mock import MagicMock, patch
from physio_app.services.llm_service import LLMService, build_prompt

_VALID_MODULE_RESPONSE = (
    "def detect_rep(pose_data): return False\n"
    "def generate_round_feedback(round_data): return ['Round done.']\n"
    "def get_session_summary(session_data): return 'Session done.'\n"
)

_DEFAULT_KWARGS = dict(
    client_instructions="",
    llm_instructions="Keep knees aligned with toes",
    boundary_values="",
    display_values="",
    session_duration_secs=10,
)


def test_build_prompt_contains_instructions():
    prompt = build_prompt("Squat", "side", **_DEFAULT_KWARGS)
    assert "Keep knees aligned with toes" in prompt


def test_build_prompt_contains_function_names():
    prompt = build_prompt("Squat", "side", **{**_DEFAULT_KWARGS, "llm_instructions": "instructions"})
    assert "detect_rep" in prompt
    assert "generate_round_feedback" in prompt
    assert "get_session_summary" in prompt


def test_build_prompt_contains_angle_fields():
    prompt = build_prompt("Squat", "side", **{**_DEFAULT_KWARGS, "llm_instructions": "instructions"})
    assert "left_knee_angle" in prompt
    assert "trunk_lean_angle" in prompt


def test_generate_module_success(tmp_db, mocker):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text=_VALID_MODULE_RESPONSE)]
    )
    mocker.patch("physio_app.services.llm_service.anthropic.Anthropic", return_value=mock_client)

    from physio_app.services.exercise_service import ExerciseService
    ex_svc = ExerciseService(tmp_db)
    ex = ex_svc.create("Squat", "side", "Keep knees aligned")

    svc = LLMService(tmp_db, api_key="test-key")
    result = svc.generate_module(
        ex.id, ex.name, ex.camera_view,
        client_instructions="",
        llm_instructions="Keep knees aligned",
        boundary_values="",
        display_values="",
        session_duration_secs=10,
    )
    assert result["status"] == "validated"
    assert "detect_rep" in result["code"]


def test_generate_module_logs_api_call(tmp_db, mocker):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text=_VALID_MODULE_RESPONSE)]
    )
    mocker.patch("physio_app.services.llm_service.anthropic.Anthropic", return_value=mock_client)

    from physio_app.services.exercise_service import ExerciseService
    ex_svc = ExerciseService(tmp_db)
    ex = ex_svc.create("Squat", "side", "")

    svc = LLMService(tmp_db, api_key="test-key")
    svc.generate_module(
        ex.id, ex.name, ex.camera_view,
        client_instructions="",
        llm_instructions="",
        boundary_values="",
        display_values="",
        session_duration_secs=10,
    )

    log = tmp_db.execute("SELECT * FROM llm_logs WHERE exercise_id = ?", (ex.id,)).fetchone()
    assert log is not None
    assert "detect_rep" in log["response"]


def test_build_prompt_contains_round_feedback_guidance():
    prompt = build_prompt("Squat", "side", **{**_DEFAULT_KWARGS, "llm_instructions": "instructions"})
    assert "generate_round_feedback" in prompt
    assert "round_data" in prompt


def test_build_prompt_contains_independent_if_guidance():
    prompt = build_prompt("Squat", "side", **{**_DEFAULT_KWARGS, "llm_instructions": "instructions"})
    assert "separate if" in prompt.lower() or "not elif" in prompt.lower()


def test_build_prompt_includes_reference_data_when_provided():
    reference_data = {
        "left_elbow_angle": {"min": 44.0, "max": 163.0, "range": 119.0},
        "left_shoulder_angle": {"min": 28.0, "max": 171.0, "range": 143.0},
    }
    prompt = build_prompt(
        "Bicep Curl", "side",
        **{**_DEFAULT_KWARGS, "llm_instructions": "Curl slowly"},
        reference_data=reference_data,
    )
    assert "REFERENCE VIDEO ANALYSIS" in prompt
    assert "left_elbow_angle" in prompt
    assert "min=44" in prompt
    assert "max=163" in prompt
    assert "left_shoulder_angle" in prompt


def test_build_prompt_omits_reference_section_when_none():
    prompt = build_prompt(
        "Squat", "side",
        **{**_DEFAULT_KWARGS, "llm_instructions": "instructions"},
        reference_data=None,
    )
    assert "REFERENCE VIDEO ANALYSIS" not in prompt


def test_build_prompt_omits_reference_section_when_empty_dict():
    prompt = build_prompt(
        "Squat", "side",
        **{**_DEFAULT_KWARGS, "llm_instructions": "instructions"},
        reference_data={},
    )
    assert "REFERENCE VIDEO ANALYSIS" not in prompt


def test_generate_module_calls_video_analyzer_when_video_exists(tmp_db, mocker, tmp_path):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text=_VALID_MODULE_RESPONSE)]
    )
    mocker.patch("physio_app.services.llm_service.anthropic.Anthropic", return_value=mock_client)

    mock_analyze = mocker.patch(
        "physio_app.services.llm_service.analyze_reference_video",
        return_value={"left_elbow_angle": {"min": 44.0, "max": 163.0, "range": 119.0}},
    )

    video_path = str(tmp_path / "reference.mp4")
    open(video_path, "wb").close()

    from physio_app.services.exercise_service import ExerciseService
    ex_svc = ExerciseService(tmp_db)
    ex = ex_svc.create("Bicep Curl", "side", "Keep elbow close")
    ex_svc.update(ex.id, reference_video_path=video_path)

    svc = LLMService(tmp_db, api_key="test-key")
    svc.generate_module(
        ex.id, ex.name, ex.camera_view,
        client_instructions="",
        llm_instructions="Keep elbow close",
        boundary_values="",
        display_values="",
        session_duration_secs=10,
    )

    mock_analyze.assert_called_once_with(video_path)

    sent_prompt = mock_client.messages.create.call_args[1]["messages"][0]["content"]
    assert "REFERENCE VIDEO ANALYSIS" in sent_prompt
    assert "left_elbow_angle" in sent_prompt


def test_generate_module_skips_video_analyzer_when_no_video(tmp_db, mocker):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text=_VALID_MODULE_RESPONSE)]
    )
    mocker.patch("physio_app.services.llm_service.anthropic.Anthropic", return_value=mock_client)
    mock_analyze = mocker.patch("physio_app.services.llm_service.analyze_reference_video")

    from physio_app.services.exercise_service import ExerciseService
    ex_svc = ExerciseService(tmp_db)
    ex = ex_svc.create("Squat", "side", "Keep knees aligned")
    # no reference_video_path set

    svc = LLMService(tmp_db, api_key="test-key")
    svc.generate_module(
        ex.id, ex.name, ex.camera_view,
        client_instructions="",
        llm_instructions="Keep knees aligned",
        boundary_values="",
        display_values="",
        session_duration_secs=10,
    )

    mock_analyze.assert_not_called()

    sent_prompt = mock_client.messages.create.call_args[1]["messages"][0]["content"]
    assert "Reference movement data" not in sent_prompt


def test_generate_module_proceeds_when_video_analyzer_returns_empty(tmp_db, mocker, tmp_path):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text=_VALID_MODULE_RESPONSE)]
    )
    mocker.patch("physio_app.services.llm_service.anthropic.Anthropic", return_value=mock_client)
    mocker.patch("physio_app.services.llm_service.analyze_reference_video", return_value={})

    video_path = str(tmp_path / "reference.mp4")
    open(video_path, "wb").close()

    from physio_app.services.exercise_service import ExerciseService
    ex_svc = ExerciseService(tmp_db)
    ex = ex_svc.create("Squat", "side", "Keep knees aligned")
    ex_svc.update(ex.id, reference_video_path=video_path)

    svc = LLMService(tmp_db, api_key="test-key")
    result = svc.generate_module(
        ex.id, ex.name, ex.camera_view,
        client_instructions="",
        llm_instructions="Keep knees aligned",
        boundary_values="",
        display_values="",
        session_duration_secs=10,
    )

    assert result["status"] == "validated"
    sent_prompt = mock_client.messages.create.call_args[1]["messages"][0]["content"]
    assert "Reference movement data" not in sent_prompt
