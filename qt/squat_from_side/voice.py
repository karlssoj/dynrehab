"""
voice.py — TTS module for standalone exercise apps.

Launched automatically by run.py. Watches message.json and speaks each new
message via pyttsx3. Writes voice_done.json when each speech completes so
the exercise app knows when to advance.

On QTRobot, replace this file with a module that uses the robot's TTS API.
Contract: read message.json, speak text, write voice_done.json with the
same timestamp when done.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

_MESSAGE_FILE = Path(__file__).parent / "message.json"
_DONE_FILE = Path(__file__).parent / "voice_done.json"
_POLL_INTERVAL = 0.1

# Speak via a fresh subprocess so pyttsx3 SAPI5 COM state never accumulates.
# The sleep gives the Windows audio device time to wake before speech starts.
_SPEAK_CMD = (
    "import pyttsx3,time; e=pyttsx3.init(); time.sleep(0.3); e.say(text); e.runAndWait()"
)


def _speak(text: str) -> None:
    subprocess.run(
        [sys.executable, "-c", f"text={repr(text)}; {_SPEAK_CMD}"],
        timeout=120,
    )


def main():
    try:
        import pyttsx3  # noqa: F401 — verify importable before use
    except ImportError:
        print("[voice] pyttsx3 not installed. Run: pip install pyttsx3")
        return

    # Seed from message.json so we never replay the last message from a previous run
    last_timestamp = None
    try:
        if _MESSAGE_FILE.exists():
            data = json.loads(_MESSAGE_FILE.read_text(encoding="utf-8"))
            last_timestamp = data.get("timestamp")
    except Exception:
        pass

    print("[voice] ready — watching message.json")

    while True:
        try:
            if _MESSAGE_FILE.exists():
                data = json.loads(_MESSAGE_FILE.read_text(encoding="utf-8"))
                ts = data.get("timestamp")
                if ts and ts != last_timestamp:
                    last_timestamp = ts
                    text = data.get("text", "")
                    if text:
                        print(f"[voice] {data.get('type', '?')}: {text}")
                        _speak(text)
                        _DONE_FILE.write_text(
                            json.dumps({"timestamp": ts}), encoding="utf-8"
                        )
        except Exception as e:
            print(f"[voice] error: {e}")
        time.sleep(_POLL_INTERVAL)


if __name__ == "__main__":
    main()
