from __future__ import annotations

import threading
import time
from typing import Callable
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from core.data_contract import PoseFrame
from core.angle_calculator import calculate_angles
from core.pose_backends import POSE_CONNECTIONS, PoseBackend, MediaPipeBackend

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
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._highlight_joints: set[str] = set()
        self._start_time = time.time()
        self._sub = None

    def subscribe(self, callback: Callable[[PoseFrame, np.ndarray], None]):
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable):
        with self._lock:
            self._subscribers = [s for s in self._subscribers if s != callback]

    def set_highlight_joints(self, joints: set[str]):
        with self._lock:
            self._highlight_joints = set(joints)

    def start(self):
        self._running = True
        self._sub = rospy.Subscriber(
            '/camera/color/image_raw', Image, self._on_image,
            queue_size=1, buff_size=2 ** 24,
        )

    def stop(self):
        self._running = False
        if self._sub is not None:
            self._sub.unregister()
            self._sub = None

    def _on_image(self, msg: Image):
        if not self._running:
            return
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logwarn(f"[pose_engine] cv_bridge error: {e}")
            return

        with self._lock:
            highlight = frozenset(self._highlight_joints)

        keypoints, annotated = self._backend.process(frame, highlight)

        pose_frame = PoseFrame(timestamp=time.time() - self._start_time)
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
            except Exception as e:
                rospy.logwarn(f"[pose_engine] subscriber error: {e}")
