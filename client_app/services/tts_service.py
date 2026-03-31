import os
import queue
import subprocess
import threading
import time


class TTSService:
    def __init__(self, cooldown_seconds: float = 2.0):
        self.cooldown_seconds = cooldown_seconds
        self._last_spoken_at: float = 0.0
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        self._current_proc: subprocess.Popen | None = None
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def is_speaking(self) -> bool:
        """Returns True if TTS is currently speaking or has messages still queued."""
        return (not self._queue.empty() or
                (self._current_proc is not None and self._current_proc.poll() is None))

    def speak(self, message: str):
        with self._lock:
            now = time.time()
            if now - self._last_spoken_at < self.cooldown_seconds:
                return
            self._last_spoken_at = now
        self._queue.put(message)

    def speak_immediate(self, message: str):
        """Speak regardless of cooldown — use for one-off announcements like session end."""
        with self._lock:
            self._last_spoken_at = time.time()
        self._queue.put(message)

    def _worker(self):
        while True:
            if self._current_proc is None or self._current_proc.poll() is not None:
                # Idle — block until next message
                message = self._queue.get()
            else:
                # Speaking — check if a newer message arrived
                try:
                    message = self._queue.get_nowait()
                    self._current_proc.terminate()
                    self._current_proc.wait()
                except queue.Empty:
                    time.sleep(0.05)
                    continue

            # Drain any further queued messages — only speak the latest
            while not self._queue.empty():
                try:
                    message = self._queue.get_nowait()
                except queue.Empty:
                    break

            env = {**os.environ, "SPEECH_MSG": message}
            self._current_proc = subprocess.Popen(
                [
                    "powershell", "-WindowStyle", "Hidden", "-Command",
                    "Add-Type -AssemblyName System.Speech; "
                    "(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak($env:SPEECH_MSG)",
                ],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
