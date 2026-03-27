import threading
import time
import pyttsx3


class TTSService:
    def __init__(self, cooldown_seconds: float = 2.0):
        self.cooldown_seconds = cooldown_seconds
        self._last_spoken_at: float = 0.0
        self._lock = threading.Lock()
        self._engine = pyttsx3.init()

    def speak(self, message: str):
        with self._lock:
            now = time.time()
            if now - self._last_spoken_at < self.cooldown_seconds:
                return
            self._last_spoken_at = now

        threading.Thread(target=self._say, args=(message,), daemon=True).start()

    def _say(self, message: str):
        self._engine.say(message)
        self._engine.runAndWait()
