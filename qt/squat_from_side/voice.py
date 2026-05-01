"""
voice.py — TTS module for QTRobot exercise apps.

Launched automatically by run.py. Reads message_queue.jsonl (one JSON
entry per line) and speaks each message in order via the QTRobot speech
service (/qt_robot/speech/say). Writes voice_done.json after each speech
so the exercise app knows when to advance.
"""
import json
import time
from pathlib import Path
import rospy
from qt_robot_interface.srv import speech_say

_QUEUE_FILE = Path(__file__).parent / "message_queue.jsonl"
_DONE_FILE = Path(__file__).parent / "voice_done.json"
_POLL_INTERVAL = 0.1

_say = None


def _speak(text: str) -> None:
    global _say
    try:
        if _say is None:
            rospy.wait_for_service('/qt_robot/speech/say', timeout=5.0)
            _say = rospy.ServiceProxy('/qt_robot/speech/say', speech_say)
        _say(str(text))
    except Exception as e:
        rospy.logwarn(f"[voice] speak failed: {e}")
        _say = None


def main():
    rospy.init_node('qt_exercise_voice', anonymous=True)

    last_pos = 0
    try:
        if _QUEUE_FILE.exists():
            last_pos = _QUEUE_FILE.stat().st_size
    except Exception:
        pass

    rospy.loginfo("[voice] ready — watching message_queue.jsonl")

    while not rospy.is_shutdown():
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
                                rospy.loginfo(f"[voice] {data.get('type', '?')}: {text}")
                                _speak(text)
                                _DONE_FILE.write_text(
                                    json.dumps({"timestamp": ts}), encoding="utf-8"
                                )
                        except Exception as e:
                            rospy.logwarn(f"[voice] entry error: {e}")
                    last_pos = f.tell()
        except Exception as e:
            rospy.logwarn(f"[voice] error: {e}")
        time.sleep(_POLL_INTERVAL)


if __name__ == "__main__":
    main()
