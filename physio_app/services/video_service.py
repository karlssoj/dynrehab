import cv2
import threading
from pathlib import Path


class VideoRecorder:
    """Records webcam video to a file using a background thread."""

    def __init__(self, output_path: str, fps: float = 24.0,
                 width: int = 640, height: int = 480):
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self._cap: cv2.VideoCapture = None
        self._writer: cv2.VideoWriter = None
        self._recording = False
        self._thread: threading.Thread = None
        self._preview_callback = None

    def set_preview_callback(self, callback):
        """Set a callback that receives each BGR frame for UI preview."""
        self._preview_callback = callback

    def start(self, camera_index: int = 0):
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        self._cap = cv2.VideoCapture(camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self.output_path, fourcc, self.fps,
                                       (self.width, self.height))
        self._recording = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._recording = False
        if self._thread:
            self._thread.join(timeout=3.0)

    def _run(self):
        try:
            while self._recording and self._cap.isOpened():
                ret, frame = self._cap.read()
                if not ret:
                    break
                self._writer.write(frame)
                if self._preview_callback:
                    try:
                        self._preview_callback(frame)
                    except Exception:
                        pass
        finally:
            if self._cap:
                self._cap.release()
            if self._writer:
                self._writer.release()
