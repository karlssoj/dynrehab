from dataclasses import dataclass, field


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
        return {
            "timestamp": self.timestamp,
            "left_knee_angle": self.left_knee_angle,
            "right_knee_angle": self.right_knee_angle,
            "left_hip_angle": self.left_hip_angle,
            "right_hip_angle": self.right_hip_angle,
            "left_ankle_angle": self.left_ankle_angle,
            "right_ankle_angle": self.right_ankle_angle,
            "left_shoulder_angle": self.left_shoulder_angle,
            "right_shoulder_angle": self.right_shoulder_angle,
            "left_elbow_angle": self.left_elbow_angle,
            "right_elbow_angle": self.right_elbow_angle,
            "left_wrist_angle": self.left_wrist_angle,
            "right_wrist_angle": self.right_wrist_angle,
            "trunk_lean_angle": self.trunk_lean_angle,
            "neck_angle": self.neck_angle,
            "pelvic_tilt": self.pelvic_tilt,
            "left_hka_alignment": self.left_hka_alignment,
            "right_hka_alignment": self.right_hka_alignment,
            "keypoints": self.keypoints,
        }
