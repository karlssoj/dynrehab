from core.data_contract import PoseFrame


def test_pose_frame_to_dict_contains_all_angle_fields():
    frame = PoseFrame(timestamp=1.0)
    d = frame.to_dict()
    expected_keys = [
        "timestamp", "left_knee_angle", "right_knee_angle",
        "left_hip_angle", "right_hip_angle", "left_ankle_angle", "right_ankle_angle",
        "left_shoulder_angle", "right_shoulder_angle", "left_elbow_angle",
        "right_elbow_angle", "left_wrist_angle", "right_wrist_angle",
        "trunk_lean_angle", "neck_angle", "pelvic_tilt",
        "left_hka_alignment", "right_hka_alignment", "keypoints",
    ]
    for key in expected_keys:
        assert key in d, f"Missing key: {key}"


def test_pose_frame_defaults_to_zero():
    frame = PoseFrame(timestamp=0.0)
    assert frame.left_knee_angle == 0.0
    assert frame.keypoints == {}


def test_pose_frame_to_dict_roundtrip():
    frame = PoseFrame(timestamp=1.5, left_knee_angle=92.3)
    d = frame.to_dict()
    assert d["timestamp"] == 1.5
    assert d["left_knee_angle"] == 92.3
