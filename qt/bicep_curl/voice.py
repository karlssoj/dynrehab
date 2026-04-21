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
_SPEAK_TIMEOUT = 120.0


def _start_tts():
    """Start a persistent PowerShell session with System.Speech loaded."""
    ps = subprocess.Popen(
        ["powershell", "-NonInteractive", "-NoProfile", "-Command", "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    ps.stdin.write("Add-Type -AssemblyName System.Speech\n")
    ps.stdin.write("$s = New-Object System.Speech.Synthesis.SpeechSynthesizer\n")
    ps.stdin.write("Write-Host '__READY__'\n")
    ps.stdin.flush()
    deadline = time.time() + 15
    while time.time() < deadline:
        line = ps.stdout.readline()
        if "__READY__" in line:
            return ps
        if not line:
            break
    raise RuntimeError("PowerShell TTS session failed to start")


def _speak(ps, text: str) -> None:
    safe = text.replace("'", "''")
    ps.stdin.write(f"$s.Speak('{safe}')\n")
    ps.stdin.write("Write-Host '__DONE__'\n")
    ps.stdin.flush()
    deadline = time.time() + _SPEAK_TIMEOUT
    while time.time() < deadline:
        line = ps.stdout.readline()
        if not line or "__DONE__" in line:
            return
    print("[voice] speak timeout")


def main():
    try:
        ps = _start_tts()
        print("[voice] TTS ready")
    except Exception as e:
        print(f"[voice] TTS init failed: {e}")
        return

    # Seek past existing queue so previous session's messages aren't replayed
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
                                _speak(ps, text)
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
