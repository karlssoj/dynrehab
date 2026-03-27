from dataclasses import dataclass, field, asdict


@dataclass
class PoseFrame:
    timestamp: float
    left_knee_angle: float = 0.0
    right_knee_angle: float = 0.0
    left_hip_angle: float = 0.0
    right_hip_angle: float = 0.0
    left_ankle_angle: float = 0.0
    right_ankle_angle: float = 0.0
    left_shoulder_angle: float = 0.0
    right_shoulder_angle: float = 0.0
    left_elbow_angle: float = 0.0
    right_elbow_angle: float = 0.0
    left_wrist_angle: float = 0.0
    right_wrist_angle: float = 0.0
    trunk_lean_angle: float = 0.0
    neck_angle: float = 0.0
    pelvic_tilt: float = 0.0
    left_hka_alignment: float = 0.0
    right_hka_alignment: float = 0.0
    keypoints: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
