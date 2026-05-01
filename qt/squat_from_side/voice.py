"""
voice.py — TTS module for standalone exercise apps.

Launched automatically by run.py. Reads message_queue.jsonl (one JSON
entry per line) and speaks each message in order via Windows PowerShell
System.Speech. Writes voice_done.json after each speech so the exercise
app knows when to advance.

On QTRobot, use extras/qt_robot_app/ instead — it has its own TTS that
calls the robot's speech service directly.
"""
import json
import subprocess
import time
from pathlib import Path

_QUEUE_FILE = Path(__file__).parent / "message_queue.jsonl"
_DONE_FILE = Path(__file__).parent / "voice_done.json"
_POLL_INTERVAL = 0.1


def _speak(text: str) -> None:
    safe = text.replace("'", "''")
    result = subprocess.run(
        [
            "powershell",
            "-NonInteractive", "-NoProfile", "-WindowStyle", "Hidden",
            "-Command",
            f"Add-Type -AssemblyName System.Speech; "
            f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"Start-Sleep -Milliseconds 300; "
            f"$s.Speak('{safe}')",
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
