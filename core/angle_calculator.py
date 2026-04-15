import numpy as np


def _angle_3d(a: tuple, b: tuple, c: tuple) -> float:
    """Angle at point b formed by vectors b→a and b→c using x,y,z coordinates."""
    va = np.array(a[:3], dtype=float)
    vb = np.array(b[:3], dtype=float)
    vc = np.array(c[:3], dtype=float)
    ba = va - vb
    bc = vc - vb
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-10:
        return 0.0
    cosine = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _angle_frontal_plane(a: tuple, b: tuple, c: tuple) -> float:
    """Angle at b using x,y only (frontal plane projection). Used for HKA alignment."""
    va = np.array([a[0], a[1]], dtype=float)
    vb = np.array([b[0], b[1]], dtype=float)
    vc = np.array([c[0], c[1]], dtype=float)
    ba = va - vb
    bc = vc - vb
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-10:
        return 0.0
    cosine = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _vector_to_vertical_angle(a: tuple, b: tuple) -> float:
    """Angle of vector a→b relative to vertical axis (0,-1,0) in image coords (y increases down)."""
    v = np.array([b[0] - a[0], b[1] - a[1], b[2] - a[2]], dtype=float)
    vertical = np.array([0.0, -1.0, 0.0])
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return 0.0
    cosine = np.dot(v, vertical) / norm
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))



def _vector_to_horizontal_angle(a: tuple, b: tuple) -> float:
    """Angle of vector a→b relative to horizontal axis in the image plane."""
    v = np.array([b[0] - a[0], b[1] - a[1]], dtype=float)
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return 0.0
    horizontal = np.array([1.0, 0.0])
    cosine = np.dot(v, horizontal) / norm
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _trunk_lean_2d(a: tuple, b: tuple) -> float:
    """2D trunk lean using x,y only (immune to z-depth noise). 0°=upright, increases leaning."""
    v = np.array([b[0] - a[0], b[1] - a[1]], dtype=float)
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return 0.0
    vertical = np.array([0.0, -1.0])
    cosine = np.dot(v, vertical) / norm
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _signed_knee_valgus(hip: tuple, knee: tuple, ankle: tuple, side: str) -> float:
    """Signed lateral knee deviation from the hip-ankle midpoint x position.
    Positive = valgus (knee inward/medial), negative = varus (knee outward/lateral).
    Designed for front-facing camera.
    For front camera: patient's LEFT leg appears on the RIGHT of the image (higher x),
    so left-leg inward movement = lower x → negate raw for consistent sign.
    Value is multiplied by 100 so typical deviations are in the range ±1 to ±10
    rather than ±0.01 to ±0.10 (raw normalised units), making the number displayable.
    Threshold guidance: ±3 = clinically meaningful, ±5 = clearly visible.
    """
    midpoint_x = (hip[0] + ankle[0]) / 2.0
    raw = knee[0] - midpoint_x
    return float((-raw if side == "left" else raw) * 100.0)


def calculate_angles(keypoints: dict) -> dict:
    """Calculate all joint angles from a keypoints dict (name → (x,y,z,vis)).

    All angles are computed from x,y coordinates only — z is ignored throughout
    because MediaPipe depth estimates are too noisy for reliable angle calculation.
    """
    def lm(name):
        return keypoints.get(name, (0.0, 0.0, 0.0, 0.0))

    ls = lm("left_shoulder")
    rs = lm("right_shoulder")
    lh = lm("left_hip")
    rh = lm("right_hip")
    shoulder_mid = tuple((ls[i] + rs[i]) / 2 for i in range(4))
    hip_mid = tuple((lh[i] + rh[i]) / 2 for i in range(4))

    return {
        # Raw angles — ~180° when joint is straight, decreases as it bends
        "left_knee_angle":      _angle_frontal_plane(lm("left_hip"),    lm("left_knee"),    lm("left_ankle")),
        "right_knee_angle":     _angle_frontal_plane(lm("right_hip"),   lm("right_knee"),   lm("right_ankle")),
        "left_hip_angle":       _angle_frontal_plane(lm("left_shoulder"),  lm("left_hip"),  lm("left_knee")),
        "right_hip_angle":      _angle_frontal_plane(lm("right_shoulder"), lm("right_hip"), lm("right_knee")),
        "left_ankle_angle":     _angle_frontal_plane(lm("left_knee"),   lm("left_ankle"),   lm("left_foot_index")),
        "right_ankle_angle":    _angle_frontal_plane(lm("right_knee"),  lm("right_ankle"),  lm("right_foot_index")),
        "left_shoulder_angle":  _angle_frontal_plane(lm("left_elbow"),  lm("left_shoulder"),  lm("left_hip")),
        "right_shoulder_angle": _angle_frontal_plane(lm("right_elbow"), lm("right_shoulder"), lm("right_hip")),
        "left_elbow_angle":     _angle_frontal_plane(lm("left_shoulder"),  lm("left_elbow"),  lm("left_wrist")),
        "right_elbow_angle":    _angle_frontal_plane(lm("right_shoulder"), lm("right_elbow"), lm("right_wrist")),
        "left_wrist_angle":     _angle_frontal_plane(lm("left_elbow"),  lm("left_wrist"),  lm("left_index")),
        "right_wrist_angle":    _angle_frontal_plane(lm("right_elbow"), lm("right_wrist"), lm("right_index")),
        "trunk_lean_angle":     _trunk_lean_2d(hip_mid, shoulder_mid),
        "neck_angle":           _angle_frontal_plane(hip_mid, shoulder_mid, lm("nose")),
        "pelvic_tilt":          _vector_to_horizontal_angle(lm("left_hip"), lm("right_hip")),
        "left_hka_alignment":   _angle_frontal_plane(lm("left_hip"),  lm("left_knee"),  lm("left_ankle")),
        "right_hka_alignment":  _angle_frontal_plane(lm("right_hip"), lm("right_knee"), lm("right_ankle")),
        "left_arm_elevation":   _angle_frontal_plane(lm("left_hip"),  lm("left_shoulder"),  lm("left_wrist")),
        "right_arm_elevation":  _angle_frontal_plane(lm("right_hip"), lm("right_shoulder"), lm("right_wrist")),
        # Bend values — 0° = straight, increases as joint bends (= 180 − raw_angle)
        "left_elbow_bend_2d":  180.0 - _angle_frontal_plane(lm("left_shoulder"),  lm("left_elbow"),  lm("left_wrist")),
        "right_elbow_bend_2d": 180.0 - _angle_frontal_plane(lm("right_shoulder"), lm("right_elbow"), lm("right_wrist")),
        "left_knee_bend_2d":   180.0 - _angle_frontal_plane(lm("left_hip"),   lm("left_knee"),   lm("left_ankle")),
        "right_knee_bend_2d":  180.0 - _angle_frontal_plane(lm("right_hip"),  lm("right_knee"),  lm("right_ankle")),
        "left_hip_bend_2d":    180.0 - _angle_frontal_plane(lm("left_shoulder"),  lm("left_hip"),  lm("left_knee")),
        "right_hip_bend_2d":   180.0 - _angle_frontal_plane(lm("right_shoulder"), lm("right_hip"), lm("right_knee")),
        "trunk_lean_2d":       _trunk_lean_2d(hip_mid, shoulder_mid),
        "left_knee_valgus":    _signed_knee_valgus(lm("left_hip"),  lm("left_knee"),  lm("left_ankle"),  "left"),
        "right_knee_valgus":   _signed_knee_valgus(lm("right_hip"), lm("right_knee"), lm("right_ankle"), "right"),
        # Segment-from-vertical angles — primary for side view, also useful front view
        # Convention: 0° = segment vertical, increases as segment tilts away from vertical
        "left_shin_angle":     _trunk_lean_2d(lm("left_ankle"),  lm("left_knee")),
        "right_shin_angle":    _trunk_lean_2d(lm("right_ankle"), lm("right_knee")),
        "left_thigh_angle":    _trunk_lean_2d(lm("left_knee"),   lm("left_hip")),
        "right_thigh_angle":   _trunk_lean_2d(lm("right_knee"),  lm("right_hip")),
        # Bilateral tilt angles — primary for front view
        # Convention: 0° = both landmarks level, increases as one side is higher/lower
        "shoulder_tilt":       _vector_to_horizontal_angle(lm("left_shoulder"), lm("right_shoulder")),
    }
