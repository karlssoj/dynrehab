import math
from core.angle_calculator import _angle_3d, _angle_frontal_plane, _vector_to_vertical_angle, _trunk_lean_2d, _signed_knee_valgus, calculate_angles


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
    # 3D angle = 60°, but 2D (x,y only) angle = 90° — proves z is included
    a = (1.0, 0.0, 1.0, 1.0)
    b = (0.0, 0.0, 0.0, 1.0)
    c = (0.0, 1.0, 1.0, 1.0)
    angle = _angle_3d(a, b, c)
    assert abs(angle - 60.0) < 0.1, f"Expected 60°, got {angle:.2f}° (3D must differ from 2D 90°)"


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


def test_trunk_lean_2d_upright():
    """Trunk pointing straight up → ~0° lean."""
    hip = (0.5, 0.8, 0.0, 1.0)
    shoulder = (0.5, 0.4, 0.0, 1.0)  # directly above hip → upright
    assert abs(_trunk_lean_2d(hip, shoulder)) < 0.01


def test_trunk_lean_2d_leaning():
    """Trunk shifted horizontally → lean > 0."""
    hip = (0.5, 0.8, 0.0, 1.0)
    shoulder = (0.7, 0.4, 0.0, 1.0)  # shifted right while up → leaning
    angle = _trunk_lean_2d(hip, shoulder)
    assert 0.0 < angle < 90.0


def test_signed_knee_valgus_left_neutral():
    """Left knee on the hip-ankle line → near zero."""
    hip = (0.6, 0.3, 0.0, 1.0)
    ankle = (0.5, 0.9, 0.0, 1.0)   # midpoint_x = 0.55
    knee = (0.55, 0.6, 0.0, 1.0)   # knee exactly at midpoint
    assert abs(_signed_knee_valgus(hip, knee, ankle, "left")) < 0.1


def test_signed_knee_valgus_left_valgus():
    """Left knee medial (lower x for left leg) → positive."""
    hip = (0.6, 0.3, 0.0, 1.0)
    ankle = (0.5, 0.9, 0.0, 1.0)   # midpoint_x = 0.55
    knee = (0.45, 0.6, 0.0, 1.0)   # knee at 0.45 < 0.55 → inward → valgus
    assert _signed_knee_valgus(hip, knee, ankle, "left") > 0.0


def test_signed_knee_valgus_left_varus():
    """Left knee lateral (higher x for left leg) → negative."""
    hip = (0.6, 0.3, 0.0, 1.0)
    ankle = (0.5, 0.9, 0.0, 1.0)   # midpoint_x = 0.55
    knee = (0.65, 0.6, 0.0, 1.0)   # knee at 0.65 > 0.55 → outward → varus
    assert _signed_knee_valgus(hip, knee, ankle, "left") < 0.0


def test_signed_knee_valgus_right_valgus():
    """Right knee medial (higher x for right leg) → positive."""
    hip = (0.4, 0.3, 0.0, 1.0)
    ankle = (0.45, 0.9, 0.0, 1.0)  # midpoint_x = 0.425
    knee = (0.5, 0.6, 0.0, 1.0)    # knee at 0.5 > 0.425 → inward → valgus
    assert _signed_knee_valgus(hip, knee, ankle, "right") > 0.0


def test_signed_knee_valgus_right_varus():
    """Right knee lateral (lower x for right leg) → negative."""
    hip = (0.4, 0.3, 0.0, 1.0)
    ankle = (0.45, 0.9, 0.0, 1.0)  # midpoint_x = 0.425
    knee = (0.35, 0.6, 0.0, 1.0)   # knee at 0.35 < 0.425 → outward → varus
    assert _signed_knee_valgus(hip, knee, ankle, "right") < 0.0


def test_calculate_angles_includes_new_keys():
    """All 9 new pre-computed keys must be present in calculate_angles output."""
    kp = {name: (0.5, 0.5, 0.0, 1.0) for name in [
        "left_hip", "left_knee", "left_ankle", "left_foot_index",
        "right_hip", "right_knee", "right_ankle", "right_foot_index",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_index", "right_index",
        "nose",
    ]}
    result = calculate_angles(kp)
    for key in [
        "left_elbow_bend_2d", "right_elbow_bend_2d",
        "left_knee_bend_2d", "right_knee_bend_2d",
        "left_hip_bend_2d", "right_hip_bend_2d",
        "trunk_lean_2d",
        "left_knee_valgus", "right_knee_valgus",
    ]:
        assert key in result, f"Missing key: {key}"


def test_elbow_bend_2d_straight_arm():
    """Straight arm (shoulder-elbow-wrist collinear vertically) → bend near 0."""
    kp = {
        "left_shoulder": (0.4, 0.3, 0.0, 1.0),
        "left_elbow":    (0.4, 0.5, 0.0, 1.0),
        "left_wrist":    (0.4, 0.7, 0.0, 1.0),
        **{name: (0.5, 0.5, 0.0, 1.0) for name in [
            "right_shoulder", "right_elbow", "right_wrist",
            "left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle", "left_foot_index", "right_foot_index",
            "left_index", "right_index", "nose",
        ]}
    }
    result = calculate_angles(kp)
    assert result["left_elbow_bend_2d"] < 5.0, \
        f"Straight arm should give ~0° bend, got {result['left_elbow_bend_2d']:.1f}°"


def test_elbow_bend_2d_bent_arm():
    """Bent arm (wrist moved forward and up from elbow) → bend > 90°."""
    kp = {
        "left_shoulder": (0.5, 0.3, 0.0, 1.0),
        "left_elbow":    (0.5, 0.5, 0.0, 1.0),
        "left_wrist":    (0.3, 0.35, 0.0, 1.0),  # wrist moved forward + up = curled
        **{name: (0.5, 0.5, 0.0, 1.0) for name in [
            "right_shoulder", "right_elbow", "right_wrist",
            "left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle", "left_foot_index", "right_foot_index",
            "left_index", "right_index", "nose",
        ]}
    }
    result = calculate_angles(kp)
    assert result["left_elbow_bend_2d"] > 90.0, \
        f"Bent arm should give >90° bend, got {result['left_elbow_bend_2d']:.1f}°"


def _full_kp(**overrides):
    """Return a full keypoints dict with all landmarks at (0.5, 0.5) unless overridden."""
    base = {name: (0.5, 0.5, 0.0, 1.0) for name in [
        "left_hip", "left_knee", "left_ankle", "left_foot_index",
        "right_hip", "right_knee", "right_ankle", "right_foot_index",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_index", "right_index", "nose",
    ]}
    base.update(overrides)
    return base


def test_calculate_angles_includes_new_segment_keys():
    """Five new segment/tilt fields must be present in calculate_angles output."""
    result = calculate_angles(_full_kp())
    for key in ["left_shin_angle", "right_shin_angle",
                "left_thigh_angle", "right_thigh_angle", "shoulder_tilt"]:
        assert key in result, f"Missing key: {key}"


def test_shin_angle_vertical():
    """Shin perfectly vertical → ~0°."""
    kp = _full_kp(
        left_ankle=(0.4, 0.9, 0.0, 1.0),
        left_knee= (0.4, 0.5, 0.0, 1.0),  # directly above ankle
    )
    result = calculate_angles(kp)
    assert result["left_shin_angle"] < 2.0, \
        f"Vertical shin should give ~0°, got {result['left_shin_angle']:.1f}°"


def test_shin_angle_tilted():
    """Shin tilted forward (ankle behind knee) → angle > 0°."""
    kp = _full_kp(
        left_ankle=(0.5, 0.9, 0.0, 1.0),
        left_knee= (0.4, 0.5, 0.0, 1.0),  # knee shifted forward of ankle
    )
    result = calculate_angles(kp)
    assert result["left_shin_angle"] > 5.0, \
        f"Tilted shin should give >5°, got {result['left_shin_angle']:.1f}°"


def test_thigh_angle_standing():
    """Thigh vertical (standing) → ~0°."""
    kp = _full_kp(
        left_knee=(0.4, 0.7, 0.0, 1.0),
        left_hip= (0.4, 0.3, 0.0, 1.0),  # directly above knee
    )
    result = calculate_angles(kp)
    assert result["left_thigh_angle"] < 2.0, \
        f"Standing thigh should give ~0°, got {result['left_thigh_angle']:.1f}°"


def test_thigh_angle_parallel_squat():
    """Thigh horizontal (parallel squat) → ~90°."""
    kp = _full_kp(
        left_knee=(0.4, 0.5, 0.0, 1.0),
        left_hip= (0.7, 0.5, 0.0, 1.0),  # hip at same height as knee → horizontal thigh
    )
    result = calculate_angles(kp)
    assert abs(result["left_thigh_angle"] - 90.0) < 2.0, \
        f"Parallel-squat thigh should give ~90°, got {result['left_thigh_angle']:.1f}°"


def test_shoulder_tilt_level():
    """Level shoulders → ~0°."""
    kp = _full_kp(
        left_shoulder= (0.3, 0.3, 0.0, 1.0),
        right_shoulder=(0.7, 0.3, 0.0, 1.0),  # same y → level
    )
    result = calculate_angles(kp)
    assert result["shoulder_tilt"] < 2.0, \
        f"Level shoulders should give ~0°, got {result['shoulder_tilt']:.1f}°"


def test_shoulder_tilt_dropped():
    """One shoulder lower than the other → tilt > 0°."""
    kp = _full_kp(
        left_shoulder= (0.3, 0.3, 0.0, 1.0),
        right_shoulder=(0.7, 0.5, 0.0, 1.0),  # right shoulder lower (higher y)
    )
    result = calculate_angles(kp)
    assert result["shoulder_tilt"] > 10.0, \
        f"Dropped shoulder should give >10°, got {result['shoulder_tilt']:.1f}°"
