from physio_app.services.llm_service import build_prompt


def test_prompt_contains_all_five_sections():
    prompt = build_prompt(
        exercise_name="Squat",
        camera_view="side",
        client_instructions="Stand side-on",
        llm_instructions="Check knee bend depth",
        boundary_values="Knee must reach 90 degrees",
        display_values="Show left knee angle",
        session_duration_secs=20,
    )
    assert "Stand side-on" in prompt
    assert "Check knee bend depth" in prompt
    assert "Knee must reach 90 degrees" in prompt
    assert "Show left knee angle" in prompt
    assert "20-second" in prompt
    assert "Client instructions" in prompt
    assert "Analysis instructions" in prompt
    assert "Boundary values" in prompt
    assert "Display values" in prompt


def test_prompt_uses_custom_duration_in_function_spec():
    prompt = build_prompt(
        exercise_name="Squat",
        camera_view="side",
        client_instructions="",
        llm_instructions="",
        boundary_values="",
        display_values="",
        session_duration_secs=15,
    )
    assert "15-second" in prompt
    assert "10-second" not in prompt


def test_prompt_includes_reference_data():
    ref = {"left_knee_angle": {"min": 80.0, "max": 170.0, "range": 90.0}}
    prompt = build_prompt(
        exercise_name="Squat",
        camera_view="side",
        client_instructions="",
        llm_instructions="",
        boundary_values="",
        display_values="",
        session_duration_secs=10,
        reference_data=ref,
    )
    assert "REFERENCE VIDEO ANALYSIS" in prompt


def test_during_exercise_mode_includes_generate_rep_cue():
    prompt = build_prompt(
        exercise_name="Squat", camera_view="side",
        client_instructions="", llm_instructions="",
        boundary_values="", display_values="",
        session_duration_secs=10,
        feedback_mode=["during_exercise"],
    )
    assert "generate_rep_cue" in prompt
    assert "generate_round_feedback" not in prompt


def test_after_window_mode_includes_generate_round_feedback():
    prompt = build_prompt(
        exercise_name="Squat", camera_view="side",
        client_instructions="", llm_instructions="",
        boundary_values="", display_values="",
        session_duration_secs=10,
        feedback_mode=["after_window"],
    )
    assert "generate_round_feedback" in prompt
    assert "generate_rep_cue" not in prompt


def test_after_exercise_mode_includes_get_session_summary():
    prompt = build_prompt(
        exercise_name="Squat", camera_view="side",
        client_instructions="", llm_instructions="",
        boundary_values="", display_values="",
        session_duration_secs=10,
        feedback_mode=["after_exercise"],
    )
    assert "get_session_summary" in prompt
    assert "generate_round_feedback" not in prompt


def test_default_mode_is_after_window():
    prompt = build_prompt(
        exercise_name="Squat", camera_view="side",
        client_instructions="", llm_instructions="",
        boundary_values="", display_values="",
        session_duration_secs=10,
    )
    assert "generate_round_feedback" in prompt
