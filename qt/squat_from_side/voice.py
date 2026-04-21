"""
voice.py — local TTS module for standalone exercise apps.

Run alongside run.py in a separate terminal:
    python voice.py

Watches message.json and speaks each new message via pyttsx3.
On QTRobot, replace this file with a module that uses the robot's TTS API.
"""
import json
import time
from pathlib import Path

_MESSAGE_FILE = Path(__file__).parent / "message.json"
_POLL_INTERVAL = 0.1


def main():
    try:
        import pyttsx3
    except ImportError:
        print("[voice] pyttsx3 not installed. Run: pip install pyttsx3")
        return

    engine = pyttsx3.init()
    last_timestamp = None
    print("[voice] ready — watching message.json")

    while True:
        try:
            if _MESSAGE_FILE.exists():
                data = json.loads(_MESSAGE_FILE.read_text(encoding="utf-8"))
                ts = data.get("timestamp")
                if ts != last_timestamp:
                    last_timestamp = ts
                    text = data.get("text", "")
                    if text:
                        print(f"[voice] {data.get('type', '?')}: {text}")
                        engine.say(text)
                        engine.runAndWait()
        except Exception as e:
            print(f"[voice] error: {e}")
        time.sleep(_POLL_INTERVAL)


if __name__ == "__main__":
    main()
