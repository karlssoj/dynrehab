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
import queue
import threading
import time
from pathlib import Path

_MESSAGE_FILE = Path(__file__).parent / "message.json"
_DONE_FILE = Path(__file__).parent / "voice_done.json"
_POLL_INTERVAL = 0.1


def _speaker(speech_queue):
    # pyttsx3 SAPI5 uses COM — must be initialised in the thread that will call it
    try:
        import pyttsx3
        engine = pyttsx3.init()
    except Exception as e:
        print(f"[voice] TTS init failed: {e}")
        return
    while True:
        ts, text = speech_queue.get()
        try:
            engine.say(text)
            engine.runAndWait()
            _DONE_FILE.write_text(json.dumps({"timestamp": ts}), encoding="utf-8")
        except Exception as e:
            print(f"[voice] speak error: {e}")


def main():
    try:
        import pyttsx3  # noqa: F401 — verify importable before spawning thread
    except ImportError:
        print("[voice] pyttsx3 not installed. Run: pip install pyttsx3")
        return

    # Skip any message already spoken in a previous run so we don't replay stale audio
    last_timestamp = None
    try:
        for path in (_DONE_FILE, _MESSAGE_FILE):
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                ts = data.get("timestamp")
                if ts:
                    last_timestamp = ts
                    break
    except Exception:
        pass

    speech_queue = queue.Queue()
    threading.Thread(target=_speaker, args=(speech_queue,), daemon=True).start()

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
                        speech_queue.put((ts, text))
        except Exception as e:
            print(f"[voice] error: {e}")
        time.sleep(_POLL_INTERVAL)


if __name__ == "__main__":
    main()
