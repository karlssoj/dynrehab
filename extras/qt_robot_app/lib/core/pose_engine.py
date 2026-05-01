import threading
import time
from typing import Callable
import cv2
import mediapipe as mp
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from core.data_contract import PoseFrame
from core.angle_calculator import calculate_angles

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

POSE_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_ankle", "left_foot_index"), ("right_ankle", "right_foot_index"),
    ("nose", "left_shoulder"), ("nose", "right_shoulder"),
]


class PoseEngine:
    def __init__(self):
        self._subscribers: list[Callable] = []
        self._running = False
        self._bridge = CvBridge()
        self._mp_pose = mp.solutions.pose
        self._lock = threading.Lock()
        self._highlight_joints: set[str] = set()
        self._pose = self._mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        )
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
        self._process(frame)

    def _process(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._pose.process(rgb)
        rgb.flags.writeable = True
        annotated = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        pose_frame = PoseFrame(timestamp=time.time() - self._start_time)

        if results.pose_landmarks:
            lms = results.pose_landmarks.landmark
            keypoints = {
                LANDMARK_NAMES[i]: (lm.x, lm.y, lm.z, lm.visibility)
                for i, lm in enumerate(lms)
                if i in LANDMARK_NAMES
            }
            angles = calculate_angles(keypoints)
            for k, v in angles.items():
                setattr(pose_frame, k, v)
            pose_frame.keypoints = keypoints
            annotated = self._draw_overlay(annotated, keypoints)

        with self._lock:
            subs = list(self._subscribers)

        for cb in subs:
            try:
                cb(pose_frame, annotated)
            except Exception as e:
                rospy.logwarn(f"[pose_engine] subscriber error: {e}")

    def _draw_overlay(self, frame: np.ndarray, keypoints: dict) -> np.ndarray:
        h, w = frame.shape[:2]
        with self._lock:
            highlight = set(self._highlight_joints)

        for a_name, b_name in POSE_CONNECTIONS:
            if a_name in keypoints and b_name in keypoints:
                a = keypoints[a_name]
                b = keypoints[b_name]
                if a[3] > 0.5 and b[3] > 0.5:
                    pt1 = (int(a[0] * w), int(a[1] * h))
                    pt2 = (int(b[0] * w), int(b[1] * h))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        for name, (x, y, z, vis) in keypoints.items():
            if vis > 0.5:
                pt = (int(x * w), int(y * h))
                color = (0, 0, 255) if name in highlight else (255, 255, 255)
                cv2.circle(frame, pt, 5, color, -1)

        return frame
