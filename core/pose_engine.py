import math
import os
import threading
import time
from typing import Callable
import cv2
import numpy as np
from core.data_contract import PoseFrame
from core.angle_calculator import calculate_angles
from core.pose_backends import POSE_CONNECTIONS, PoseBackend, MediaPipeBackend, draw_back_contour
from core.spine_contour import (crop_and_rotate_roi, extract_back_contour,
                                 signed_curvature_ratio, facing_left,
                                 back_contour_points_in_frame, downsample_points)

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

_SPINE_SAMPLE_INTERVAL_SECS = 0.25  # 4 Hz — see spec's measured performance baseline
# Override via the SPINE_SAMPLE_INTERVAL_SECS env var (e.g. in .env) — a
# smaller value samples more often (updates the drawn contour faster) at
# the cost of more CPU load per second; a larger value reduces load.

_SPINE_CONTOUR_POINT_COUNT = 10  # how many dots to draw along the back contour
# Override via the SPINE_CONTOUR_POINT_COUNT env var (e.g. in .env).

_SPINE_MAX_SHOULDER_LATERAL_RATIO = 0.4  # informed starting point, not validated —
# shoulder_lateral_span alone is NOT scale-invariant: it shrinks with distance
# from the camera regardless of facing direction, so a person far away (or
# partly out of frame) could pass an absolute threshold while still facing
# the camera. Normalizing by hip-shoulder distance (torso length in the same
# frame) makes the check invariant to how large the person is in the image —
# true profile gives a ratio near 0 regardless of distance; facing the
# camera gives a ratio close to (shoulder width / torso length), roughly
# 0.8-1.0 for typical body proportions. Override via
# SPINE_MAX_SHOULDER_LATERAL_RATIO.


def _should_sample_spine(last_sample_time: float, now: float,
                          interval: float = _SPINE_SAMPLE_INTERVAL_SECS) -> bool:
    return now - last_sample_time >= interval


class PoseEngine:
    def __init__(self, backend: PoseBackend = None):
        self._backend = backend if backend is not None else MediaPipeBackend()
        self._subscribers: list[Callable] = []
        self._running = False
        self._thread: threading.Thread = None
        self._highlight_joints: set[str] = set()
        self._lock = threading.Lock()
        self._seek_start = False
        self._last_spine_sample = 0.0
        self._last_spine_points: list[tuple[float, float]] | None = None
        self._spine_sample_interval = float(
            os.getenv("SPINE_SAMPLE_INTERVAL_SECS", _SPINE_SAMPLE_INTERVAL_SECS))
        self._spine_contour_point_count = int(
            os.getenv("SPINE_CONTOUR_POINT_COUNT", _SPINE_CONTOUR_POINT_COUNT))
        self._spine_max_shoulder_lateral_ratio = float(
            os.getenv("SPINE_MAX_SHOULDER_LATERAL_RATIO", _SPINE_MAX_SHOULDER_LATERAL_RATIO))
        self._spine_error_logged = False
        print(f"[pose_engine] spine sample interval: {self._spine_sample_interval}s "
              f"({'from SPINE_SAMPLE_INTERVAL_SECS env var' if 'SPINE_SAMPLE_INTERVAL_SECS' in os.environ else 'default'})")

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

                    if _should_sample_spine(self._last_spine_sample, frame_start,
                                             self._spine_sample_interval):
                        self._last_spine_sample = frame_start
                        spine_points = None  # cleared unless this attempt succeeds below
                        try:
                            hip = keypoints.get("left_hip") or keypoints.get("right_hip")
                            shoulder = keypoints.get("left_shoulder") or keypoints.get("right_shoulder")
                            in_profile = False
                            if hip and shoulder:
                                torso_span = math.hypot(shoulder[0] - hip[0], shoulder[1] - hip[1])
                                if torso_span > 1e-6:
                                    lateral_ratio = pose_frame.shoulder_lateral_span / torso_span
                                    in_profile = lateral_ratio <= self._spine_max_shoulder_lateral_ratio
                            if hip and shoulder and min(hip[3], shoulder[3]) >= 0.3 and in_profile:
                                h, w = frame.shape[:2]
                                hip_px = (hip[0] * w, hip[1] * h)
                                shoulder_px = (shoulder[0] * w, shoulder[1] * h)
                                roi_result = crop_and_rotate_roi(frame, hip_px, shoulder_px)
                                if roi_result:
                                    roi_image, hip_point, shoulder_point, chord_len, M = roi_result
                                    mask = self._backend.get_segmentation_mask(roi_image)
                                    if mask is not None:
                                        profiles = extract_back_contour(mask, hip_point, shoulder_point)
                                        if profiles:
                                            face_left = facing_left(keypoints)
                                            if face_left is not None:
                                                left_profile, right_profile = profiles
                                                back_profile = right_profile if face_left else left_profile
                                                full_points = back_contour_points_in_frame(
                                                    M, hip_point, shoulder_point, back_profile,
                                                    on_right_side=face_left)
                                                spine_points = downsample_points(
                                                    full_points, self._spine_contour_point_count)
                                                pose_frame.spine_curvature_ratio = signed_curvature_ratio(
                                                    left_profile, right_profile, chord_len, face_left)
                        except Exception as e:
                            # A failure anywhere in the spine-sampling pipeline (e.g. a
                            # missing/corrupt segmentation model file, an internal
                            # ultralytics/torch error) must degrade to "no sample this
                            # tick" rather than crashing the capture loop — the rest of
                            # _run() (frame capture, angle calculation, subscriber
                            # callbacks) must keep working regardless. Logged once (not
                            # every ~250ms tick) so a persistent failure is still visible.
                            if not self._spine_error_logged:
                                self._spine_error_logged = True
                                print(f"[pose_engine] spine sampling failed (will keep "
                                      f"retrying silently, this is logged once only): {e}")
                        # Replace the cached points on every sampling attempt, success or
                        # not -- a failed attempt (lost person, bad mask, exception) must
                        # clear a stale line rather than let it linger from the last
                        # successful sample.
                        self._last_spine_points = spine_points

                    if self._last_spine_points:
                        draw_back_contour(annotated, self._last_spine_points)

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
