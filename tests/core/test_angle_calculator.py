import math
from core.angle_calculator import _angle_3d, _angle_frontal_plane, _vector_to_vertical_angle, calculate_angles


def test_angle_3d_right_angle():
    a = (1.0, 0.0, 0.0, 1.0)
    b = (0.0, 0.0, 0.0, 1.0)
    c = (0.0, 1.0, 0.0, 1.0)
    assert abs(_angle_3d(a, b, c) - 90.0) < 0.01


def test_angle_3d_straight_line():
    a = (0.0, 0.0, 0.0, 1.0)
    b = (0.5, 0.0, 0.0, 1.0)
    c = (1.0, 0.0, 0.0, 1.0)
    assert abs(_angle_3d(a, b, c) - 180.0) < 0.01


def test_angle_3d_45_degrees():
    a = (1.0, 0.0, 0.0, 1.0)
    b = (0.0, 0.0, 0.0, 1.0)
    c = (1.0, 1.0, 0.0, 1.0)
    assert abs(_angle_3d(a, b, c) - 45.0) < 0.1


def test_angle_3d_uses_z_coordinate():
    # a has z=1, b at origin, c along y-axis
    # dot([1,0,1],[0,1,0]) = 0, so angle = 90 degrees regardless of z
    a_3d = (1.0, 0.0, 1.0, 1.0)
    b = (0.0, 0.0, 0.0, 1.0)
    c = (0.0, 1.0, 0.0, 1.0)
    angle = _angle_3d(a_3d, b, c)
    assert abs(angle - 90.0) < 0.01


def test_angle_frontal_plane_ignores_z():
    a1 = (1.0, 0.0, 0.0, 1.0)
    a2 = (1.0, 0.0, 5.0, 1.0)
    b = (0.0, 0.0, 0.0, 1.0)
    c = (0.0, 1.0, 0.0, 1.0)
    assert abs(_angle_frontal_plane(a1, b, c) - _angle_frontal_plane(a2, b, c)) < 0.001


def test_vector_to_vertical_upright():
    # Vector pointing straight up (y decreases upward in MediaPipe)
    hip = (0.5, 0.8, 0.0, 1.0)
    shoulder = (0.5, 0.4, 0.0, 1.0)
    assert abs(_vector_to_vertical_angle(hip, shoulder) - 0.0) < 0.01


def test_vector_to_vertical_leaning():
    hip = (0.5, 0.8, 0.0, 1.0)
    shoulder = (0.8, 0.4, 0.0, 1.0)  # shifted right = leaning
    angle = _vector_to_vertical_angle(hip, shoulder)
    assert 0.0 < angle < 90.0


def test_calculate_angles_returns_all_keys():
    kp = {name: (0.5, 0.5, 0.0, 1.0) for name in [
        "left_hip", "left_knee", "left_ankle", "left_foot_index",
        "right_hip", "right_knee", "right_ankle", "right_foot_index",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_index", "right_index",
        "nose",
    ]}
    result = calculate_angles(kp)
    expected = [
        "left_knee_angle", "right_knee_angle", "left_hip_angle", "right_hip_angle",
        "left_ankle_angle", "right_ankle_angle", "left_shoulder_angle", "right_shoulder_angle",
        "left_elbow_angle", "right_elbow_angle", "left_wrist_angle", "right_wrist_angle",
        "trunk_lean_angle", "neck_angle", "pelvic_tilt",
        "left_hka_alignment", "right_hka_alignment",
    ]
    for key in expected:
        assert key in result
