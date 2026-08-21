from abc import ABC, abstractmethod
import time
import cv2
import numpy as np

from core.one_euro_filter import OneEuroFilter

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


def draw_back_contour_points(frame: np.ndarray, points: list[tuple[float, float]],
                              color: tuple = (0, 255, 255), radius: int = 4) -> np.ndarray:
    """Draw a filled dot at each point (already in this frame's pixel
    coordinates, e.g. from spine_contour.back_contour_points_in_frame /
    downsample_points) onto `frame` in place, with no connecting line
    between them. No-op if `points` is empty."""
    for x, y in points:
        cv2.circle(frame, (int(round(x)), int(round(y))), radius, color, -1)
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

    def get_segmentation_mask(self, roi_image: np.ndarray) -> np.ndarray | None:
        """roi_image: an already-cropped BGR image (e.g. the rotated torso
        ROI from spine_contour.crop_and_rotate_roi). Returns a binary mask
        (uint8, 0/255) the same size as roi_image, or None if this backend
        doesn't support segmentation or no person/mask is found. Default:
        unsupported."""
        return None


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


class _KeypointState:
    """Per-keypoint smoothing + hold-last-known-position state for YOLOBackend."""

    __slots__ = ("filter_x", "filter_y", "last_pos", "last_conf", "last_seen_time")

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float):
        self.filter_x = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.filter_y = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.last_pos: tuple[float, float] | None = None
        self.last_conf: float = 0.0
        self.last_seen_time: float = 0.0

    def reset(self) -> None:
        self.filter_x.reset()
        self.filter_y.reset()
        self.last_pos = None
        self.last_conf = 0.0
        self.last_seen_time = 0.0


class YOLOBackend(PoseBackend):
    _YOLO_TO_NAME = {
        0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
        5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
        9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
        13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle",
    }

    # One Euro Filter tuning for slow-to-moderate exercise movements (squats):
    # low min_cutoff favors smoothing at rest, small nonzero beta lets the
    # filter track fast motion (e.g. standing up quickly) without much lag.
    _DEFAULT_MIN_CUTOFF = 1.0
    _DEFAULT_BETA = 0.05
    _DEFAULT_D_CUTOFF = 1.0
    # How long to keep emitting the last known position after confidence
    # drops below _CONF_THRESHOLD, before treating the keypoint as lost.
    _DEFAULT_HOLD_MAX_SECONDS = 0.25

    def __init__(self, model_path: str = "yolo11n-pose.pt",
                 min_cutoff: float = _DEFAULT_MIN_CUTOFF,
                 beta: float = _DEFAULT_BETA,
                 d_cutoff: float = _DEFAULT_D_CUTOFF,
                 hold_max_seconds: float = _DEFAULT_HOLD_MAX_SECONDS,
                 seg_model_path: str = "yolo11n-seg.pt"):
        from ultralytics import YOLO
        self._model = YOLO(model_path)
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._hold_max_seconds = hold_max_seconds
        self._filter_state: dict[str, _KeypointState] = {}
        self._seg_model_path = seg_model_path
        self._seg_model = None

    def process(self, frame: np.ndarray,
                highlight_joints: frozenset = frozenset()) -> tuple[dict, np.ndarray]:
        results = self._model(frame, verbose=False)
        annotated = frame.copy()
        now = time.time()

        for r in results:
            if r.keypoints is None or r.keypoints.shape[0] == 0:
                break
            # xyn gives normalized [0,1] coordinates; take first detected person
            xyn = r.keypoints.xyn[0].cpu().numpy()   # (17, 2)
            conf = (r.keypoints.conf[0].cpu().numpy()
                    if r.keypoints.conf is not None else np.ones(17))

            keypoints = {}
            for idx, name in self._YOLO_TO_NAME.items():
                if idx >= len(xyn):
                    continue
                raw_conf = float(conf[idx])
                state = self._filter_state.get(name)

                if raw_conf >= _CONF_THRESHOLD:
                    if state is None:
                        state = _KeypointState(self._min_cutoff, self._beta, self._d_cutoff)
                        self._filter_state[name] = state
                    x = state.filter_x.filter(float(xyn[idx, 0]), now)
                    y = state.filter_y.filter(float(xyn[idx, 1]), now)
                    state.last_pos = (x, y)
                    state.last_conf = raw_conf
                    state.last_seen_time = now
                    keypoints[name] = (x, y, 0.0, raw_conf)
                elif state is not None and state.last_pos is not None:
                    if now - state.last_seen_time <= self._hold_max_seconds:
                        x, y = state.last_pos
                        keypoints[name] = (x, y, 0.0, state.last_conf)
                    else:
                        # Grace period expired: treat as truly lost and reset
                        # so a later reappearance isn't smoothed/lagged
                        # against a now-stale position.
                        state.reset()
                        del self._filter_state[name]
                # else: never detected yet and still below threshold -> omit.

            return keypoints, _draw_skeleton(annotated, keypoints, highlight_joints)

        return {}, annotated

    def get_segmentation_mask(self, roi_image: np.ndarray) -> np.ndarray | None:
        if self._seg_model is False:
            return None
        if self._seg_model is None:
            try:
                from ultralytics import YOLO
                self._seg_model = YOLO(self._seg_model_path)
            except Exception as e:
                print(f"[pose_backends] segmentation model load failed, disabling spine sampling: {e}")
                self._seg_model = False
                return None
        results = self._seg_model(roi_image, verbose=False, imgsz=320)
        for r in results:
            if r.masks is None or r.masks.data.shape[0] == 0:
                continue
            classes = r.boxes.cls.cpu().numpy() if r.boxes is not None else None
            masks_np = r.masks.data.cpu().numpy()
            if classes is not None:
                person_indices = [i for i, c in enumerate(classes) if int(c) == 0]
            else:
                person_indices = list(range(len(masks_np)))
            if not person_indices:
                return None
            best_idx = max(person_indices, key=lambda i: masks_np[i].sum())
            mask = masks_np[best_idx]
            mask_u8 = (mask > 0.5).astype("uint8") * 255
            return cv2.resize(mask_u8, (roi_image.shape[1], roi_image.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
        return None

    def close(self) -> None:
        pass


def create_backend(config: dict) -> PoseBackend:
    """Instantiate the pose backend named in config['pose_backend'] (default: 'mediapipe')."""
    name = config.get("pose_backend", "mediapipe").lower()
    if name in ("yolo", "yolo11"):
        return YOLOBackend(config.get("yolo_model", "yolo11n-pose.pt"))
    return MediaPipeBackend()
