"""
voice.py — TTS module for QTRobot exercise apps.

Launched automatically by run.py. Reads message_queue.jsonl (one JSON
entry per line) and speaks each message in order via the QTRobot speech
service (/qt_robot/speech/say). Writes voice_done.json after each speech
so the exercise app knows when to advance.

ROS is initialized in a background thread so that a slow or missing ROS
master never blocks the poll loop — voice_done.json is always written.
"""
import json
import threading
import time
from pathlib import Path

_QUEUE_FILE = Path(__file__).parent / "message_queue.jsonl"
_DONE_FILE = Path(__file__).parent / "voice_done.json"
_POLL_INTERVAL = 0.1

_say = None
_ros_ready = False


def _init_ros():
    global _say, _ros_ready
    try:
        import rospy
        from qt_robot_interface.srv import speech_say as _speech_say_srv
        rospy.init_node('qt_exercise_voice', anonymous=True)
        rospy.wait_for_service('/qt_robot/speech/say', timeout=10.0)
        _say = rospy.ServiceProxy('/qt_robot/speech/say', _speech_say_srv)
        _ros_ready = True
        print("[voice] ROS speech service connected")
    except Exception as e:
        print(f"[voice] ROS init failed: {e} — running without speech")


def _speak(text: str) -> None:
    if not _ros_ready or _say is None:
        return
    try:
        _say(str(text))
    except Exception as e:
        print(f"[voice] speak failed: {e}")


def main():
    threading.Thread(target=_init_ros, daemon=True).start()

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
                                print(f"[voice] {data.get('type', '?')}: {text}")
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
