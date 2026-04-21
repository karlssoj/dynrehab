"""
voice.py — local TTS module for standalone exercise apps.

Launched automatically by run.py. Watches message.json and speaks each new
message via pyttsx3. Writes voice_done.json when speech completes so the
exercise app knows when to advance.

On QTRobot, replace this file with a module that uses the robot's TTS API.
The contract: read message.json, speak text, write voice_done.json with the
same timestamp when done.
"""
import json
import time
from pathlib import Path

_MESSAGE_FILE = Path(__file__).parent / "message.json"
_DONE_FILE = Path(__file__).parent / "voice_done.json"
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
                        _DONE_FILE.write_text(
                            json.dumps({"timestamp": ts}), encoding="utf-8"
                        )
        except Exception as e:
            print(f"[voice] error: {e}")
        time.sleep(_POLL_INTERVAL)


if __name__ == "__main__":
    main()
