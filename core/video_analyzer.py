import cv2
from mediapipe.python.solutions.pose import Pose
from core.angle_calculator import calculate_angles
from core.pose_engine import LANDMARK_NAMES

_MIN_RANGE = 10.0


def analyze_reference_video(video_path: str) -> dict[str, dict]:
    """Run pose estimation on a reference video and return per-joint angle statistics.

    Returns a dict of { joint_name: {"min": float, "max": float, "range": float} }
    for every joint where the range of motion exceeded 10 degrees.
    Returns {} if the video cannot be opened or contains no detected poses.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    angle_lists: dict[str, list[float]] = {}

    try:
        with Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        ) as pose:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = pose.process(rgb)
                if results.pose_landmarks:
                    lms = results.pose_landmarks.landmark
                    keypoints = {
                        LANDMARK_NAMES[i]: (lm.x, lm.y, lm.z, lm.visibility)
                        for i, lm in enumerate(lms)
                        if i in LANDMARK_NAMES
                    }
                    angles = calculate_angles(keypoints)
                    for joint, value in angles.items():
                        angle_lists.setdefault(joint, []).append(value)
    finally:
        cap.release()

    if not angle_lists:
        return {}

    result = {}
    for joint, values in angle_lists.items():
        mn = min(values)
        mx = max(values)
        rng = mx - mn
        if rng > _MIN_RANGE:
            result[joint] = {
                "min": round(mn, 1),
                "max": round(mx, 1),
                "range": round(rng, 1),
            }
    return result
