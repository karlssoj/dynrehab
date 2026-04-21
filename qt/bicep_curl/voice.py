"""
voice.py — TTS module for standalone exercise apps.

Launched automatically by run.py. Reads message_queue.jsonl (one JSON
entry per line) and speaks each message in order via Windows PowerShell
System.Speech. Writes voice_done.json after each speech so the exercise
app knows when to advance.

On QTRobot, replace this file with a module that uses the robot's TTS API.
Contract: read message_queue.jsonl, speak each entry's "text" in order,
write voice_done.json with the entry's "timestamp" when done.
"""
import json
import subprocess
import time
from pathlib import Path

_QUEUE_FILE = Path(__file__).parent / "message_queue.jsonl"
_DONE_FILE = Path(__file__).parent / "voice_done.json"
_POLL_INTERVAL = 0.1


def _speak(text: str) -> None:
    # XML-escape text for SSML; leading <break> lets the audio device wake
    # before the first word so nothing gets clipped.
    safe = (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
    ssml = f'<speak><break time="300ms"/>{safe}</speak>'
    result = subprocess.run(
        [
            "powershell",
            "-NonInteractive", "-NoProfile", "-WindowStyle", "Hidden",
            "-Command",
            "Add-Type -AssemblyName System.Speech; "
            "(New-Object System.Speech.Synthesis.SpeechSynthesizer)"
            f".SpeakSsml('{ssml}')",
        ],
        timeout=120,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[voice] speak failed (rc={result.returncode}): {result.stderr[:300]}")


def main():
    # Seek past any existing queue entries so old messages aren't replayed
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
