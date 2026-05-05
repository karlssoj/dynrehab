import re
import shutil
from pathlib import Path

from physio_app.services.exercise_service import ExerciseService

_REPO_ROOT = Path(__file__).parent.parent.parent
_QT_DIR = _REPO_ROOT / "qt"
_ROBOT_DIR = _REPO_ROOT / "extras" / "qt_robot_app"
_CORE_SRC = _REPO_ROOT / "core"

_VOICE_PY = '''\
"""
voice.py — TTS module for standalone exercise apps.

Launched automatically by run.py. Reads message_queue.jsonl (one JSON
entry per line) and speaks each message in order via Windows PowerShell
System.Speech. Writes voice_done.json after each speech so the exercise
app knows when to advance.

On QTRobot, replace this file with a module that uses the robot\'s TTS API.
Contract: read message_queue.jsonl, speak each entry\'s "text" in order,
write voice_done.json with the entry\'s "timestamp" when done.
"""
import json
import subprocess
import time
from pathlib import Path

_QUEUE_FILE = Path(__file__).parent / "message_queue.jsonl"
_DONE_FILE = Path(__file__).parent / "voice_done.json"
_POLL_INTERVAL = 0.1


def _speak(text: str) -> None:
    safe = text.replace("\'", "\'\'")
    result = subprocess.run(
        [
            "powershell",
            "-NonInteractive", "-NoProfile", "-WindowStyle", "Hidden",
            "-Command",
            f"Add-Type -AssemblyName System.Speech; "
            f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"Start-Sleep -Milliseconds 300; "
            f"$s.Speak(\'{safe}\')",
        ],
        timeout=120,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[voice] speak failed (rc={result.returncode}): {result.stderr[:300]}")


def main():
    last_pos = 0
    try:
        if _QUEUE_FILE.exists():
            last_pos = _QUEUE_FILE.stat().st_size
    except Exception:
        pass

    print("[voice] ready — watching message_queue.jsonl")

    while True:
        try:
            if _QUEUE_FILE.exists():
                with open(_QUEUE_FILE, encoding="utf-8") as f:
                    f.seek(last_pos)
                    for raw in f:
                        try:
                            entry = raw.strip()
                            if not entry:
                                continue
                            data = json.loads(entry)
                            ts = data.get("timestamp")
                            text = data.get("text", "")
                            if text and ts:
                                print(f"[voice] {data.get(\'type\', \'?\')}: {text}")
                                _speak(text)
                                _DONE_FILE.write_text(
                                    json.dumps({"timestamp": ts}), encoding="utf-8"
                                )
                        except Exception as e:
                            print(f"[voice] entry error: {e}")
                    last_pos = f.tell()
        except Exception as e:
            print(f"[voice] error: {e}")
        time.sleep(_POLL_INTERVAL)


if __name__ == "__main__":
    main()
'''

_RUN_PY = '''\
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
sys.path.insert(0, str(Path(__file__).parent))

import exercise_config
import analysis_module
from app import StandaloneApp

if __name__ == "__main__":
    video_source = sys.argv[1] if len(sys.argv) > 1 else 0
    voice = subprocess.Popen([sys.executable, str(Path(__file__).parent / "voice.py")])
    try:
        StandaloneApp(exercise_config.CONFIG, analysis_module,
                      feedback_dir=Path(__file__).parent,
                      video_source=video_source).mainloop()
    finally:
        voice.terminate()
'''


_ROBOT_RUN_PY = '''\
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
    engine = PoseEngine()

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
'''


def _slugify(name: str) -> str:
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return slug or "exercise"


def _render_config(ex) -> str:
    import json as _json
    try:
        feedback_mode = _json.loads(ex.feedback_mode) if ex.feedback_mode else ["after_window"]
    except (ValueError, TypeError):
        feedback_mode = ["after_window"]
    lines = [
        "CONFIG = {",
        f"    \"name\": {repr(ex.name)},",
        f"    \"camera_view\": {repr(ex.camera_view)},",
        f"    \"client_instructions\": {repr(ex.client_instructions)},",
        f"    \"display_values\": {repr(ex.display_values)},",
        f"    \"session_duration_secs\": {ex.session_duration_secs},",
        f"    \"feedback_mode\": {_json.dumps(feedback_mode)},",
        "}",
        "",
    ]
    return "\n".join(lines)


def _render_robot_config(ex) -> str:
    import json as _json
    try:
        feedback_mode = _json.loads(ex.feedback_mode) if ex.feedback_mode else ["after_window"]
    except (ValueError, TypeError):
        feedback_mode = ["after_window"]
    if "after_exercise" not in feedback_mode:
        feedback_mode = list(feedback_mode) + ["after_exercise"]
    lines = [
        "CONFIG = {",
        f"    \"name\": {repr(ex.name)},",
        f"    \"camera_view\": {repr(ex.camera_view)},",
        f"    \"client_instructions\": {repr(ex.client_instructions)},",
        f"    \"session_duration_secs\": {ex.session_duration_secs},",
        f"    \"feedback_mode\": {_json.dumps(feedback_mode)},",
        "}",
        "",
    ]
    return "\n".join(lines)


def _sync_lib_core():
    core_dst = _QT_DIR / "lib" / "core"
    core_dst.mkdir(parents=True, exist_ok=True)
    for fname in ("pose_engine.py", "angle_calculator.py", "data_contract.py"):
        src = _CORE_SRC / fname
        if src.exists():
            shutil.copy2(src, core_dst / fname)
        else:
            print(f"[standalone] warning: core source {fname} not found at {src}")
    init = core_dst / "__init__.py"
    if not init.exists():
        init.write_text("", encoding="utf-8")


class StandaloneGenerator:
    @staticmethod
    def generate(exercise_id: str, conn) -> None:
        svc = ExerciseService(conn)
        ex = svc.get(exercise_id)
        if ex is None:
            print(f"[standalone] exercise {exercise_id} not found, skipping")
            return
        module = svc.get_active_module(exercise_id)
        if module is None:
            print(f"[standalone] no active module for {exercise_id}, skipping")
            return

        slug = _slugify(ex.name)
        exercise_dir = _QT_DIR / slug
        if exercise_dir.exists():
            print(f"[standalone] warning: qt/{slug}/ already exists, overwriting")
        exercise_dir.mkdir(parents=True, exist_ok=True)

        (exercise_dir / "exercise_config.py").write_text(
            _render_config(ex), encoding="utf-8"
        )
        (exercise_dir / "analysis_module.py").write_text(
            module["code"], encoding="utf-8"
        )
        (exercise_dir / "run.py").write_text(_RUN_PY, encoding="utf-8")
        (exercise_dir / "voice.py").write_text(_VOICE_PY, encoding="utf-8")

        _sync_lib_core()
        print(f"[standalone] generated qt/{slug}/")

        robot_dir = _ROBOT_DIR / slug
        if robot_dir.exists():
            print(f"[standalone] warning: extras/qt_robot_app/{slug}/ already exists, overwriting")
        robot_dir.mkdir(parents=True, exist_ok=True)
        (robot_dir / "__init__.py").write_text("", encoding="utf-8")
        (robot_dir / "exercise_config.py").write_text(
            _render_robot_config(ex), encoding="utf-8"
        )
        (robot_dir / "analysis_module.py").write_text(
            module["code"], encoding="utf-8"
        )
        (robot_dir / "run.py").write_text(_ROBOT_RUN_PY, encoding="utf-8")
        print(f"[standalone] generated extras/qt_robot_app/{slug}/")
