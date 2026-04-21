import queue
import threading
import time


class TTSService:
    def __init__(self, cooldown_seconds: float = 2.0):
        self.cooldown_seconds = cooldown_seconds
        self._last_spoken_at: float = 0.0
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        self._speaking = False
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def is_speaking(self) -> bool:
        return self._speaking or not self._queue.empty()

    def speak(self, message: str):
        with self._lock:
            now = time.time()
            if now - self._last_spoken_at < self.cooldown_seconds:
                return
            self._last_spoken_at = now
        self._queue.put(message)

    def speak_immediate(self, message: str):
        with self._lock:
            self._last_spoken_at = time.time()
        self._queue.put(message)

    def stop(self):
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def _worker(self):
        import pyttsx3
        engine = pyttsx3.init()
        while True:
            message = self._queue.get()
            self._speaking = True
            try:
                engine.say(message)
                engine.runAndWait()
            except Exception:
                pass
            finally:
                self._speaking = False
