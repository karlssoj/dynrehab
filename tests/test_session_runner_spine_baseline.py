import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "qt" / "lib"))

from session_runner import SessionRunner, COUNTDOWN_SECS


class _FakeModule:
    @staticmethod
    def detect_rep(pose_data):
        return False

    @staticmethod
    def generate_round_feedback(round_data):
        _FakeModule.last_round_data = round_data
        return ["Good work!"]

    @staticmethod
    def generate_rep_feedback(rep_data):
        _FakeModule.last_rep_data = rep_data
        return ["Rep done."]

    @staticmethod
    def get_session_summary(session_data):
        _FakeModule.last_session_data = session_data
        return "Session done."

    @staticmethod
    def get_instructions():
        return []

    @staticmethod
    def get_relevant_joints():
        return []

    @staticmethod
    def reset_round():
        pass


class _RepModule(_FakeModule):
    @staticmethod
    def detect_rep(pose_data):
        return True


_CONFIG = {
    "name": "Squat",
    "camera_view": "side",
    "client_instructions": "Stand.",
    "display_values": "",
    "session_duration_secs": 10,
}


def _spine_pose(thoracic, lumbar):
    return {"timestamp": 0.0, "spine_thoracic_curvature": thoracic, "spine_lumbar_curvature": lumbar}


_NO_SPINE_POSE = {"timestamp": 0.0, "spine_thoracic_curvature": None, "spine_lumbar_curvature": None}


def test_countdown_latches_last_valid_reading_and_ignores_later_none():
    runner = SessionRunner(_CONFIG, _FakeModule())
    runner.start_countdown()
    runner.process_frame(_NO_SPINE_POSE)
    runner.process_frame(_spine_pose(0.1, -0.2))
    runner.process_frame(_NO_SPINE_POSE)  # a later None frame must not clobber it
    assert runner._pending_spine_baseline == {"thoracic": 0.1, "lumbar": -0.2}


def test_countdown_latches_the_most_recent_valid_reading():
    runner = SessionRunner(_CONFIG, _FakeModule())
    runner.start_countdown()
    runner.process_frame(_spine_pose(0.1, -0.2))
    runner.process_frame(_spine_pose(0.3, -0.4))
    assert runner._pending_spine_baseline == {"thoracic": 0.3, "lumbar": -0.4}


def test_baseline_committed_when_countdown_reaches_zero():
    runner = SessionRunner(_CONFIG, _FakeModule())
    runner.start_countdown()
    runner.process_frame(_spine_pose(0.1, -0.2))
    runner._state_wall_start = time.time() - (COUNTDOWN_SECS + 0.6)
    result = runner.process_frame(_NO_SPINE_POSE)
    assert result["state"] == "exercise"
    assert runner._spine_baseline == {"thoracic": 0.1, "lumbar": -0.2}
    assert runner._pending_spine_baseline is None


def test_baseline_is_none_when_no_valid_reading_during_countdown():
    runner = SessionRunner(_CONFIG, _FakeModule())
    runner.start_countdown()
    runner.process_frame(_NO_SPINE_POSE)
    runner._state_wall_start = time.time() - (COUNTDOWN_SECS + 0.6)
    runner.process_frame(_NO_SPINE_POSE)
    assert runner._spine_baseline is None


def test_baseline_does_not_get_stale_value_from_previous_round():
    runner = SessionRunner(_CONFIG, _FakeModule())
    # Round 1: valid baseline captured.
    runner.start_countdown()
    runner.process_frame(_spine_pose(0.5, -0.5))
    runner._state_wall_start = time.time() - (COUNTDOWN_SECS + 0.6)
    runner.process_frame(_NO_SPINE_POSE)
    assert runner._spine_baseline == {"thoracic": 0.5, "lumbar": -0.5}

    # Round 2: countdown runs again with no valid reading this time -- the
    # round-2 baseline must be None, not a leftover from round 1.
    runner.start_countdown()
    runner.process_frame(_NO_SPINE_POSE)
    runner._state_wall_start = time.time() - (COUNTDOWN_SECS + 0.6)
    runner.process_frame(_NO_SPINE_POSE)
    assert runner._spine_baseline is None


def test_baseline_unchanged_while_exercise_state_is_active():
    runner = SessionRunner(_CONFIG, _FakeModule())
    runner.start_countdown()
    runner.process_frame(_spine_pose(0.2, -0.3))
    runner._state_wall_start = time.time() - (COUNTDOWN_SECS + 0.6)
    runner.process_frame(_NO_SPINE_POSE)
    assert runner._state == "exercise"

    # More frames streaming in during "exercise" must not touch the baseline.
    runner.process_frame(_spine_pose(0.9, -0.9))
    assert runner._spine_baseline == {"thoracic": 0.2, "lumbar": -0.3}


def test_after_rep_resume_preserves_round_baseline():
    config = {**_CONFIG, "feedback_mode": ["after_rep"]}
    runner = SessionRunner(config, _RepModule())
    runner.start_countdown()
    runner.process_frame(_spine_pose(0.4, -0.1))
    runner._state_wall_start = time.time() - (COUNTDOWN_SECS + 0.6)
    runner.process_frame(_NO_SPINE_POSE)
    assert runner._spine_baseline == {"thoracic": 0.4, "lumbar": -0.1}

    # Trigger a rep -> pauses into rep feedback -> resume via end_feedback()
    # ("rep" branch) must NOT start a new countdown, so the baseline survives.
    runner.process_frame(_NO_SPINE_POSE)
    assert runner._state == "feedback"
    runner.end_feedback()
    assert runner._state == "exercise"
    assert runner._spine_baseline == {"thoracic": 0.4, "lumbar": -0.1}


def test_round_data_carries_spine_baseline():
    runner = SessionRunner(_CONFIG, _FakeModule())
    runner.start_countdown()
    runner.process_frame(_spine_pose(0.15, -0.25))
    runner._state_wall_start = time.time() - (COUNTDOWN_SECS + 0.6)
    runner.process_frame(_NO_SPINE_POSE)
    assert runner._state == "exercise"
    runner._round_number = 1
    runner._state_wall_start = time.time() - 11.0
    runner.process_frame(_NO_SPINE_POSE)

    assert _FakeModule.last_round_data["spine_baseline"] == {"thoracic": 0.15, "lumbar": -0.25}
    assert runner._all_rounds[-1]["spine_baseline"] == {"thoracic": 0.15, "lumbar": -0.25}


def test_round_data_spine_baseline_is_none_when_not_captured():
    runner = SessionRunner(_CONFIG, _FakeModule())
    runner._state = "exercise"
    runner._round_number = 1
    runner._state_wall_start = time.time() - 11.0
    runner.process_frame(_NO_SPINE_POSE)
    assert _FakeModule.last_round_data["spine_baseline"] is None


def test_rep_data_carries_spine_baseline():
    config = {**_CONFIG, "feedback_mode": ["after_rep"]}
    runner = SessionRunner(config, _RepModule())
    runner.start_countdown()
    runner.process_frame(_spine_pose(0.6, -0.6))
    runner._state_wall_start = time.time() - (COUNTDOWN_SECS + 0.6)
    runner.process_frame(_NO_SPINE_POSE)
    runner.process_frame(_NO_SPINE_POSE)  # triggers the rep -> _enter_rep_feedback

    assert _FakeModule.last_rep_data["spine_baseline"] == {"thoracic": 0.6, "lumbar": -0.6}
    assert runner._all_rounds[-1]["spine_baseline"] == {"thoracic": 0.6, "lumbar": -0.6}


def test_session_summary_rounds_carry_spine_baseline():
    runner = SessionRunner(_CONFIG, _FakeModule())
    runner.start_countdown()
    runner.process_frame(_spine_pose(0.25, -0.15))
    runner._state_wall_start = time.time() - (COUNTDOWN_SECS + 0.6)
    runner.process_frame(_NO_SPINE_POSE)
    runner._round_number = 1
    runner._state_wall_start = time.time() - 11.0
    runner.process_frame(_NO_SPINE_POSE)

    runner.get_session_summary_speech()
    rounds = _FakeModule.last_session_data["rounds"]
    assert rounds[-1]["spine_baseline"] == {"thoracic": 0.25, "lumbar": -0.15}
