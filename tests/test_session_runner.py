import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "qt" / "lib"))

from session_runner import SessionRunner


class _FakeModule:
    _rep_triggered = False

    @staticmethod
    def detect_rep(pose_data):
        return False

    @staticmethod
    def generate_round_feedback(round_data):
        return ["Good work!"]

    @staticmethod
    def get_session_summary(session_data):
        return "Session done."

    @staticmethod
    def get_instructions():
        return ["Stand tall.", "Keep your back straight."]

    @staticmethod
    def get_relevant_joints():
        return [("Left Knee", "left_knee_bend_2d")]

    @staticmethod
    def reset_round():
        pass


_EMPTY_POSE = {"timestamp": 0.0}

_CONFIG = {
    "name": "Squat",
    "camera_view": "side",
    "client_instructions": "Stand.",
    "display_values": "",
    "session_duration_secs": 10,
}


def test_initial_state_is_instructions():
    runner = SessionRunner(_CONFIG, _FakeModule())
    result = runner.process_frame(_EMPTY_POSE)
    assert result["state"] == "instructions"


def test_get_instructions_delegates_to_module():
    runner = SessionRunner(_CONFIG, _FakeModule())
    assert runner.get_instructions() == ["Stand tall.", "Keep your back straight."]


def test_get_relevant_joints_delegates_to_module():
    runner = SessionRunner(_CONFIG, _FakeModule())
    assert runner.get_relevant_joints() == [("Left Knee", "left_knee_bend_2d")]


def test_start_countdown_changes_state():
    runner = SessionRunner(_CONFIG, _FakeModule())
    runner.start_countdown()
    result = runner.process_frame(_EMPTY_POSE)
    assert result["state"] == "countdown"


def test_countdown_emits_countdown_speak_on_change():
    runner = SessionRunner(_CONFIG, _FakeModule())
    runner.start_countdown()
    result = runner.process_frame(_EMPTY_POSE)
    assert result["countdown_speak"] is not None


def test_exercise_state_after_countdown():
    runner = SessionRunner(_CONFIG, _FakeModule())
    runner.start_countdown()
    runner._state_wall_start = time.time() - 6.0
    result = runner.process_frame(_EMPTY_POSE)
    assert result["state"] == "exercise"


def test_rep_count_increments():
    class _RepModule(_FakeModule):
        @staticmethod
        def detect_rep(pose_data):
            return True

    runner = SessionRunner(_CONFIG, _RepModule())
    runner._state = "exercise"
    runner._state_wall_start = time.time()
    runner.process_frame(_EMPTY_POSE)
    assert runner.rep_count == 1


def test_feedback_state_after_exercise_window():
    runner = SessionRunner(_CONFIG, _FakeModule())
    runner._state = "exercise"
    runner._round_number = 1
    start = time.time() - 11.0
    runner._state_wall_start = start
    result = runner.process_frame(_EMPTY_POSE)
    assert result["state"] == "feedback"
    assert result["feedback_lines"] == ["Good work!"]


def test_get_summary_returns_rep_count():
    runner = SessionRunner(_CONFIG, _FakeModule())
    runner.rep_count = 7
    summary = runner.get_summary()
    assert summary["rep_count"] == 7


class _RepCueModule(_FakeModule):
    @staticmethod
    def detect_rep(pose_data):
        return True

    @staticmethod
    def generate_rep_cue(cue_data):
        if cue_data.get("trigger") == "timeout":
            return "Keep going!"
        return "Good rep!"


def test_during_exercise_emits_rep_cue():
    config = {**_CONFIG, "feedback_mode": ["during_exercise"]}
    runner = SessionRunner(config, _RepCueModule())
    runner._state = "exercise"
    runner._state_wall_start = time.time()
    runner._last_cue_time = time.time()
    result = runner.process_frame(_EMPTY_POSE)
    assert result.get("rep_cue") == "Good rep!"


def test_no_window_mode_exercise_continues_past_window():
    config = {**_CONFIG, "feedback_mode": ["during_exercise"]}
    runner = SessionRunner(config, _FakeModule())
    runner._state = "exercise"
    runner._state_wall_start = time.time() - 11.0
    result = runner.process_frame(_EMPTY_POSE)
    assert result["state"] == "exercise"


def test_get_session_summary_speech_returns_list():
    runner = SessionRunner(_CONFIG, _FakeModule())
    result = runner.get_session_summary_speech()
    assert isinstance(result, list)
