from __future__ import annotations

from abc import ABC, abstractmethod
import cv2
import numpy as np

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

_CONF_THRESHOLD = 0.3


def _draw_skeleton(frame: np.ndarray, keypoints: dict,
                   highlight_joints: frozenset = frozenset()) -> np.ndarray:
    h, w = frame.shape[:2]
    for a_name, b_name in POSE_CONNECTIONS:
        if a_name in keypoints and b_name in keypoints:
            a, b = keypoints[a_name], keypoints[b_name]
            if a[3] > 0.5 and b[3] > 0.5:
                cv2.line(frame, (int(a[0] * w), int(a[1] * h)),
                         (int(b[0] * w), int(b[1] * h)), (0, 255, 0), 2)
    for name, (x, y, z, vis) in keypoints.items():
        if vis > 0.5:
            color = (0, 0, 255) if name in highlight_joints else (255, 255, 255)
            cv2.circle(frame, (int(x * w), int(y * h)), 5, color, -1)
    return frame


class PoseBackend(ABC):
    @abstractmethod
    def process(self, frame: np.ndarray,
                highlight_joints: frozenset = frozenset()) -> tuple[dict, np.ndarray]:
        """Process a BGR frame.

        Returns (keypoints, annotated_frame).
        keypoints maps landmark name → (x_norm, y_norm, z, visibility).
        Returns empty dict if no person detected.
        """

    @abstractmethod
    def close(self) -> None:
        """Release backend resources."""


class MediaPipeBackend(PoseBackend):
    _LANDMARK_NAMES = {
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

    def __init__(self, model_complexity: int = 1,
                 detection_confidence: float = 0.5,
                 tracking_confidence: float = 0.5):
        import mediapipe as mp
        self._pose = mp.solutions.pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def process(self, frame: np.ndarray,
                highlight_joints: frozenset = frozenset()) -> tuple[dict, np.ndarray]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._pose.process(rgb)
        rgb.flags.writeable = True
        annotated = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if not results.pose_landmarks:
            return {}, annotated

        lms = results.pose_landmarks.landmark
        keypoints = {
            self._LANDMARK_NAMES[i]: (lm.x, lm.y, lm.z, lm.visibility)
            for i, lm in enumerate(lms)
            if i in self._LANDMARK_NAMES
        }
        return keypoints, _draw_skeleton(annotated, keypoints, highlight_joints)

    def close(self) -> None:
        self._pose.close()


class YOLOBackend(PoseBackend):
    _YOLO_TO_NAME = {
        0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
        5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
        9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
        13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle",
    }

    def __init__(self, model_path: str = "yolo11n-pose.pt"):
        from ultralytics import YOLO
        self._model = YOLO(model_path)

    def process(self, frame: np.ndarray,
                highlight_joints: frozenset = frozenset()) -> tuple[dict, np.ndarray]:
        results = self._model(frame, verbose=False)
        annotated = frame.copy()

        for r in results:
            if r.keypoints is None or r.keypoints.shape[0] == 0:
                break
            # xyn gives normalized [0,1] coordinates; take first detected person
            xyn = r.keypoints.xyn[0].cpu().numpy()   # (17, 2)
            conf = (r.keypoints.conf[0].cpu().numpy()
                    if r.keypoints.conf is not None else np.ones(17))

            keypoints = {
                name: (float(xyn[idx, 0]), float(xyn[idx, 1]), 0.0, float(conf[idx]))
                for idx, name in self._YOLO_TO_NAME.items()
                if idx < len(xyn) and conf[idx] >= _CONF_THRESHOLD
            }
            return keypoints, _draw_skeleton(annotated, keypoints, highlight_joints)

        return {}, annotated

    def close(self) -> None:
        pass


def create_backend(config: dict) -> PoseBackend:
    """Instantiate the pose backend named in config['pose_backend'] (default: 'mediapipe')."""
    name = config.get("pose_backend", "mediapipe").lower()
    if name == "yolo":
        return YOLOBackend(config.get("yolo_model", "yolo11n-pose.pt"))
    return MediaPipeBackend()
