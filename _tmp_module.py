# Knäböj framifrån (front) @ 2026-03-31 17:57:33
import math
import statistics

# ── thresholds ────────────────────────────────────────────────────────────────
_SQUAT_START       = 20    # knee_bend > this → squat begins
_SQUAT_PEAK        = 40    # knee_bend >= this → counts as a real squat attempt
_SQUAT_IDEAL       = 80    # knee_bend >= this → good depth (approx 90° knee angle)
_SQUAT_RETURN      = 15    # knee_bend <= this → standing again
_HYSTERESIS        = 10    # knee must recover this much from peak before "returning"
_PELVIC_TILT_LIMIT = 8.0   # pelvic_tilt degrees — beyond this is asymmetry
_VALGUS_THRESHOLD  = 0.03  # lateral deviation > this (normalised x) = knee caving inward
_TRUNK_LEAN_LIMIT  = 20.0  # trunk lateral lean beyond this is a problem
_VIS_THRESHOLD     = 0.35  # minimum MediaPipe visibility to trust a joint

# ── module-level rep detection state ─────────────────────────────────────────
_phase        = "ready"   # "ready" | "squatting" | "rising"
_max_knee     = 0.0
_peak_reached = False
_rep_frames   = []


def _angle_2d(ax, ay, bx, by, cx, cy):
    vax, vay = ax - bx, ay - by
    vcx, vcy = cx - bx, cy - by
    mag = math.hypot(vax, vay) * math.hypot(vcx, vcy)
    if mag < 1e-10:
        return 180.0
    return math.degrees(math.acos(max(-1.0, min(1.0, (vax * vcx + vay * vcy) / mag))))


def _knee_bend_2d(pose_data, side):
    kpts = pose_data.get("keypoints", {})
    h = kpts.get(f"{side}_hip")
    k = kpts.get(f"{side}_knee")
    a = kpts.get(f"{side}_ankle")
    if not (h and k and a) or min(h[3], k[3], a[3]) < _VIS_THRESHOLD:
        return 0.0
    return 180.0 - _angle_2d(h[0], h[1], k[0], k[1], a[0], a[1])


def _avg_knee_bend_2d(pose_data):
    left  = _knee_bend_2d(pose_data, "left")
    right = _knee_bend_2d(pose_data, "right")
    if left > 1.0 and right > 1.0:
        return (left + right) / 2.0
    if left > 1.0:
        return left
    if right > 1.0:
        return right
    return 0.0


def _knee_lateral_deviation(pose_data, side):
    # Returns positive if knee is MEDIAL (valgus/inward), negative if LATERAL (varus/outward).
    # Front-facing camera: patient's LEFT appears on RIGHT of image (high x).
    kpts = pose_data.get("keypoints", {})
    h = kpts.get(f"{side}_hip")
    k = kpts.get(f"{side}_knee")
    a = kpts.get(f"{side}_ankle")
    if not (h and k and a) or min(h[3], k[3], a[3]) < _VIS_THRESHOLD:
        return 0.0
    midpoint_x = (h[0] + a[0]) / 2.0
    raw = k[0] - midpoint_x
    return -raw if side == "left" else raw


def _pelvic_tilt_2d(pose_data):
    # Angle of hip line from horizontal, computed from keypoints x,y only.
    # Returns 0 when hips are level, increases when one hip is higher.
    # Works regardless of whether the camera image is mirrored.
    kpts = pose_data.get("keypoints", {})
    lh = kpts.get("left_hip")
    rh = kpts.get("right_hip")
    if not (lh and rh) or min(lh[3], rh[3]) < _VIS_THRESHOLD:
        return 0.0
    dx = abs(lh[0] - rh[0])
    dy = abs(lh[1] - rh[1])
    if dx < 1e-10:
        return 90.0
    return math.degrees(math.atan2(dy, dx))


def _lateral_trunk_lean_2d(pose_data):
    """
    Lateral trunk lean from front view.
    Uses mid-shoulder to mid-hip vector; measures deviation from vertical.
    Returns degrees; 0 = perfectly upright, positive = leaning to either side.
    """
    kpts = pose_data.get("keypoints", {})
    ls = kpts.get("left_shoulder")
    rs = kpts.get("right_shoulder")
    lh = kpts.get("left_hip")
    rh = kpts.get("right_hip")
    if not (ls and rs and lh and rh):
        return 0.0
    if min(ls[3], rs[3], lh[3], rh[3]) < _VIS_THRESHOLD:
        return 0.0
    mid_sh_x = (ls[0] + rs[0]) / 2.0
    mid_sh_y = (ls[1] + rs[1]) / 2.0
    mid_hi_x = (lh[0] + rh[0]) / 2.0
    mid_hi_y = (lh[1] + rh[1]) / 2.0
    dx = mid_sh_x - mid_hi_x
    dy = mid_sh_y - mid_hi_y   # negative when shoulder above hip
    dist = math.hypot(dx, dy)
    if dist < 1e-10 or dy >= 0:
        return 0.0
    # angle from vertical: 0 = straight up
    return math.degrees(math.asin(max(-1.0, min(1.0, abs(dx) / dist))))


def _joints_visible(pose_data):
    kpts = pose_data.get("keypoints", {})
    required = [
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
    ]
    for name in required:
        kp = kpts.get(name)
        if kp is None or kp[3] < _VIS_THRESHOLD:
            return False
    return True


def get_instructions():
    return [
        "Stand facing the camera so your whole body is visible.",
        "Slowly squat down until your knees are at ninety degrees, then push back up strongly.",
    ]


def detect_rep(pose_data):
    global _phase, _max_knee, _peak_reached, _rep_frames

    if not _joints_visible(pose_data):
        return False

    knee_bend = _avg_knee_bend_2d(pose_data)

    if _phase == "ready":
        if knee_bend > _SQUAT_START:
            _phase        = "squatting"
            _max_knee     = knee_bend
            _peak_reached = False
            _rep_frames   = [pose_data]

    elif _phase == "squatting":
        _rep_frames.append(pose_data)
        if knee_bend > _max_knee:
            _max_knee = knee_bend
        if not _peak_reached and knee_bend >= _SQUAT_PEAK:
            _peak_reached = True
        if knee_bend < _max_knee - _HYSTERESIS:
            _phase = "rising"

    elif _phase == "rising":
        _rep_frames.append(pose_data)
        if knee_bend <= _SQUAT_RETURN:
            _phase = "ready"
            if _peak_reached:
                _rep_frames = []
                return True
            _rep_frames = []

    return False


def reset_round():
    global _phase, _max_knee, _peak_reached, _rep_frames
    _phase        = "ready"
    _max_knee     = 0.0
    _peak_reached = False
    _rep_frames   = []


def generate_round_feedback(round_data):
    rep_count = round_data.get("rep_count", 0)
    frames    = round_data.get("frames", [])

    if not frames:
        return ["I couldn't see you clearly — make sure your whole body is visible facing the camera."]

    knee_vals        = [_avg_knee_bend_2d(f) for f in frames]
    pelvic_vals      = [_pelvic_tilt_2d(f) for f in frames]
    trunk_vals       = [_lateral_trunk_lean_2d(f) for f in frames]
    left_valgus_vals = [_knee_lateral_deviation(f, "left") for f in frames]
    right_valgus_vals= [_knee_lateral_deviation(f, "right") for f in frames]

    peak_knee        = max(knee_vals) if knee_vals else 0.0
    knee_range       = peak_knee - min(knee_vals) if knee_vals else 0.0
    max_pelvic       = max(pelvic_vals) if pelvic_vals else 0.0
    max_trunk        = max(trunk_vals) if trunk_vals else 0.0
    max_left_valgus  = max(left_valgus_vals) if left_valgus_vals else 0.0
    max_right_valgus = max(right_valgus_vals) if right_valgus_vals else 0.0

    lines = []

    # ── Rep count / depth opener ──────────────────────────────────────────────
    if rep_count == 0:
        lines.append("No complete squats were detected this round.")
        if knee_range < 10.0:
            lines.append(
                "It looks like you barely moved — try bending your knees and lowering your hips "
                "until your thighs are parallel to the floor."
            )
        elif peak_knee < _SQUAT_PEAK:
            lines.append(
                "You started to squat but didn't go deep enough to count — "
                "aim to lower yourself until your knees are roughly at a right angle."
            )
        else:
            lines.append(
                "You squatted down but didn't return fully upright between reps — "
                "make sure you straighten up completely before the next squat."
            )
    else:
        rep_word = "rep" if rep_count == 1 else "reps"
        lines.append(f"Good work — you completed {rep_count} {rep_word} this round.")
        if peak_knee >= _SQUAT_IDEAL:
            lines.append("You reached a great squat depth — keep it up.")
        else:
            lines.append(
                "Try to squat a little deeper — aim to get your thighs parallel to the floor."
            )

    # ── Quality checks — always run regardless of rep count ──────────────────
    if max_pelvic > _PELVIC_TILT_LIMIT:
        lines.append(
            "Your hips shifted to one side — try to keep your hip line "
            "level and even throughout the movement."
        )

    if max_left_valgus > _VALGUS_THRESHOLD or max_right_valgus > _VALGUS_THRESHOLD:
        lines.append(
            "Watch your knees — they drifted inward during the squat. "
            "Push your knees outward in line with your toes as you lower yourself."
        )

    if max_trunk > _TRUNK_LEAN_LIMIT:
        lines.append(
            "Your upper body leaned to one side — keep your torso upright and centred "
            "throughout the squat."
        )

    return lines


def get_relevant_joints():
    return [
        ("L knee",     "left_knee_angle"),
        ("R knee",     "right_knee_angle"),
        ("Pelvic tilt","pelvic_tilt"),
        ("L HKA",      "left_hka_alignment"),
    ]


def get_session_summary(session_data):
    total_reps = session_data.get("total_reps", 0)
    stats      = session_data.get("angle_stats", {})

    left_knee_min  = stats.get("left_knee_angle",  {}).get("min", 180.0)
    right_knee_min = stats.get("right_knee_angle", {}).get("min", 180.0)
    best_knee_raw  = min(left_knee_min, right_knee_min)
    best_knee_bend = 180.0 - best_knee_raw if best_knee_raw > 10.0 else 0.0

    # pelvic_tilt from angle_stats is unreliable for front-facing camera
    # (it uses a raw angle that reads ~180 when level); skip it in session summary

    parts = []

    if total_reps == 0:
        parts.append("No complete squats were recorded this session.")
        parts.append(
            "Next time, focus on bending your knees and lowering your hips until "
            "your thighs are parallel to the floor, then push back up to standing."
        )
        return " ".join(parts)

    rep_word = "rep" if total_reps == 1 else "reps"
    parts.append(f"Session complete — you performed {total_reps} {rep_word} in total.")

    if best_knee_bend >= _SQUAT_IDEAL:
        parts.append("You consistently reached a good squat depth — excellent technique.")
    elif best_knee_bend >= _SQUAT_PEAK:
        parts.append(
            "Your squat depth was reasonable, but try to lower yourself a bit further "
            "so your thighs are fully parallel to the floor."
        )
    elif best_knee_bend > 10.0:
        parts.append(
            "Your squats were quite shallow — work on bending your knees more deeply "
            "to get the full benefit of the exercise."
        )


    left_hka_min  = stats.get("left_hka_alignment",  {}).get("min", 180.0)
    right_hka_min = stats.get("right_hka_alignment", {}).get("min", 180.0)
    # Use HKA only as a proxy for any lateral deviation in session summary
    # (directional check requires per-frame keypoints not available here)
    if (left_hka_min < 168.0 and left_hka_min > 10.0) or \
       (right_hka_min < 168.0 and right_hka_min > 10.0):
        parts.append(
            "Work on keeping your knees tracking over your toes — "
            "avoid letting them cave inward as you lower into the squat."
        )

    return " ".join(parts)