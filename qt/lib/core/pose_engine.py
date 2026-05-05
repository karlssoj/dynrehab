import threading
import time
from typing import Callable
import cv2
import numpy as np
from core.data_contract import PoseFrame
from core.angle_calculator import calculate_angles
from core.pose_backends import POSE_CONNECTIONS, PoseBackend, MediaPipeBackend

# Kept for backwards compatibility — callers that imported LANDMARK_NAMES from here still work.
LANDMARK_NAMES = {
    0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
    4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
    7: "left_ear", 8: "right_ear", 9: "mouth_left", 10: "mouth_right",
    11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist", 17: "left_pinky", 18: "right_pinky",
    19: "left_index", 20: "right_index", 21: "left_thumb", 22: "right_thumb",
    23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle", 29: "left_heel", 30: "right_heel",
    31: "left_foot_index", 32: "right_foot_index",
}


class PoseEngine:
    def __init__(self, backend: PoseBackend = None):
        self._backend = backend if backend is not None else MediaPipeBackend()
        self._subscribers: list[Callable] = []
        self._running = False
        self._thread: threading.Thread = None
        self._highlight_joints: set[str] = set()
        self._lock = threading.Lock()
        self._seek_start = False

    def seek_to_start(self):
        with self._lock:
            self._seek_start = True

    def subscribe(self, callback: Callable[[PoseFrame, np.ndarray], None]):
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable):
        with self._lock:
            self._subscribers = [s for s in self._subscribers if s != callback]

    def set_highlight_joints(self, joints: set[str]):
        with self._lock:
            self._highlight_joints = set(joints)

    def start(self, source=0):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, args=(source,), daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _run(self, source):
        cap = cv2.VideoCapture(source)
        is_file = isinstance(source, str)
        raw_fps = cap.get(cv2.CAP_PROP_FPS) if is_file else 0.0
        frame_delay = 1.0 / raw_fps if raw_fps > 0 else 0.0
        try:
            start_time = time.time()
            while self._running and cap.isOpened():
                with self._lock:
                    if self._seek_start:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self._seek_start = False
                frame_start = time.time()
                ret, frame = cap.read()
                if not ret:
                    if is_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    self._running = False
                    break

                with self._lock:
                    highlight = frozenset(self._highlight_joints)

                keypoints, annotated = self._backend.process(frame, highlight)

                pose_frame = PoseFrame(timestamp=time.time() - start_time)
                if keypoints:
                    angles = calculate_angles(keypoints)
                    for k, v in angles.items():
                        setattr(pose_frame, k, v)
                    pose_frame.keypoints = keypoints

                with self._lock:
                    subs = list(self._subscribers)

                for cb in subs:
                    try:
                        cb(pose_frame, annotated)
                    except Exception:
                        pass

                if frame_delay > 0:
                    elapsed = time.time() - frame_start
                    remaining = frame_delay - elapsed
                    if remaining > 0:
                        time.sleep(remaining)
        finally:
            cap.release()
            self._backend.close()
