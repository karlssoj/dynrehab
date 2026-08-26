#!/usr/bin/env python3
import queue
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import rospy

import exercise_config
import analysis_module
import tts
from core.pose_engine import PoseEngine
from core.pose_backends import create_backend
from session_runner import SessionRunner
from display import Display


def main():
    rospy.init_node("qt_exercise", anonymous=True)

    config = exercise_config.CONFIG
    runner = SessionRunner(config, analysis_module)
    display = Display(
        window_name=config.get("name", "QT Exercise"),
        exercise_secs=config.get("session_duration_secs", 60),
    )
    engine = PoseEngine(backend=create_backend(config))

    _frame_q: queue.Queue = queue.Queue(maxsize=1)

    def _on_frame(pose_frame, annotated_frame):
        try:
            _frame_q.put_nowait((pose_frame, annotated_frame))
        except queue.Full:
            pass

    engine.subscribe(_on_frame)

    for line in runner.get_instructions():
        tts.speak_sync(line)
    runner.start_calibration()

    relevant_joints = runner.get_relevant_joints()
    _last_countdown = None
    _last_calib_speak = None
    _last_rep_cue = None
    _done = False

    engine.start()

    while not rospy.is_shutdown() and not _done:
        try:
            pose_frame, annotated_frame = _frame_q.get(timeout=0.1)
        except queue.Empty:
            cv2.waitKey(1)
            continue

        pose_dict = pose_frame.to_dict()
        state = runner.process_frame(pose_dict)

        state["joint_values"] = {
            label: float(pose_dict.get(key, 0.0))
            for label, key in relevant_joints
        }

        calib_speak = state.get("calibration_speak")
        if calib_speak and calib_speak != _last_calib_speak:
            _last_calib_speak = calib_speak
            tts.speak(calib_speak)

        cnt = state.get("countdown_speak")
        if cnt is not None and cnt != _last_countdown:
            _last_countdown = cnt
            tts.speak(str(cnt) if cnt > 0 else "Go!")

        rep_cue = state.get("rep_cue")
        if rep_cue and rep_cue != _last_rep_cue:
            _last_rep_cue = rep_cue
            tts.speak(rep_cue)

        feedback = state.get("feedback_lines")
        if feedback:
            for line in feedback:
                tts.speak_sync(line)
            if state.get("feedback_type") == "rep":
                runner.end_feedback()
            else:
                for line in runner.get_session_summary_speech():
                    tts.speak_sync(line)
                _done = True
                engine.stop()
                break

        if not display.update(annotated_frame, state):
            _done = True
            engine.stop()
            break

    engine.stop()
    display.close()
    tts.stop()


if __name__ == "__main__":
    main()
