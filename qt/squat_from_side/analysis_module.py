import math
import statistics

# ── thresholds ────────────────────────────────────────────────────────────────
_SQUAT_START       = 20    # knee_bend > this → squat begins (loose, for detection only)
_SQUAT_PEAK        = 40    # knee_bend >= this → counts as a real rep attempt
_SQUAT_RETURN      = 15    # knee_bend <= this → standing again
_HYSTERESIS        = 10    # bend must drop this much from peak before "returning"

# Quality targets (from physiotherapist boundary values)
_DEPTH_TARGET      = 90    # knee bend degrees for good squat depth (~90 deg)
_DEPTH_MIN         = 60    # minimum bend to consider meaningful depth
_TRUNK_LEAN_MIN    = 20    # minimum acceptable torso lean (degrees from vertical)
_TRUNK_LEAN_MAX    = 45    # maximum acceptable torso lean (degrees from vertical)
_SHIN_ANGLE_MAX    = 30    # shin angle from vertical — above this = knees too far forward
_HEEL_RISE_THRESH  = 0.02  # y-coordinate rise of heel keypoint (normalised) suggesting heel lift

_VIS_THRESHOLD     = 0.35  # minimum MediaPipe visibility to trust a joint

# ── module-level rep detection state ─────────────────────────────────────────
_phase        = "ready"   # "ready" | "squatting" | "returning"
_max_bend     = 0.0
_peak_reached = False
_rep_frames   = []


def _pick_side(keypoints):
    lv = keypoints.get("left_hip",  (0, 0, 0, 0))[3]
    rv = keypoints.get("right_hip", (0, 0, 0, 0))[3]
    return "left" if lv >= rv else "right"


def _joints_visible(keypoints, side):
    for part in ("hip", "knee", "ankle"):
        kp = keypoints.get(f"{side}_{part}")
        if kp is None or kp[3] < _VIS_THRESHOLD:
            return False
    return True


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
        return pose_data.get(f"{side}_knee_bend_2d", 0.0)
    raw = _angle_2d(h[0], h[1], k[0], k[1], a[0], a[1])
    return 180.0 - raw


def _trunk_lean_2d(pose_data, side):
    kpts = pose_data.get("keypoints", {})
    sh = kpts.get(f"{side}_shoulder")
    h  = kpts.get(f"{side}_hip")
    if not (sh and h) or min(sh[3], h[3]) < _VIS_THRESHOLD:
        return pose_data.get("trunk_lean_angle", 0.0)
    dx = sh[0] - h[0]
    dy = sh[1] - h[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-10 or dy >= 0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, -dy / dist))))


def _shin_angle(pose_data, side):
    return pose_data.get(f"{side}_shin_angle", 0.0)


def _heel_y(pose_data, side):
    kpts = pose_data.get("keypoints", {})
    heel = kpts.get(f"{side}_heel")
    if heel is None or heel[3] < _VIS_THRESHOLD:
        return None
    return heel[1]


def get_instructions():
    return [
        "Stand sideways to the camera so your full body is visible.",
        "Perform controlled squats for about 10 seconds, aiming to bend your knees to 90 degrees at the bottom of each rep.",
    ]


def detect_rep(pose_data):
    global _phase, _max_bend, _peak_reached, _rep_frames

    kpts = pose_data.get("keypoints", {})
    side = _pick_side(kpts)

    if not _joints_visible(kpts, side):
        return False

    bend = _knee_bend_2d(pose_data, side)

    if _phase == "ready":
        if bend > _SQUAT_START:
            _phase        = "squatting"
            _max_bend     = bend
            _peak_reached = False
            _rep_frames   = [pose_data]

    elif _phase == "squatting":
        _rep_frames.append(pose_data)
        if bend > _max_bend:
            _max_bend = bend
        if not _peak_reached and bend >= _SQUAT_PEAK:
            _peak_reached = True
        if bend < _max_bend - _HYSTERESIS:
            _phase = "returning"

    elif _phase == "returning":
        _rep_frames.append(pose_data)
        if bend <= _SQUAT_RETURN:
            _phase = "ready"
            if _peak_reached:
                _rep_frames = []
                return True
            _rep_frames = []

    return False


def reset_round():
    global _phase, _max_bend, _peak_reached, _rep_frames
    _phase        = "ready"
    _max_bend     = 0.0
    _peak_reached = False
    _rep_frames   = []


def generate_round_feedback(round_data):
    rep_count = round_data.get("rep_count", 0)
    frames    = round_data.get("frames", [])

    if not frames:
        return ["I couldn't see you clearly. Make sure your full body is visible side-on to the camera."]

    kpts = frames[0].get("keypoints", {})
    side = _pick_side(kpts)

    # Collect per-frame measurements
    bend_vals   = [_knee_bend_2d(f, side) for f in frames]
    trunk_vals  = [_trunk_lean_2d(f, side) for f in frames]
    shin_vals   = [_shin_angle(f, side) for f in frames if _shin_angle(f, side) > 1.0]
    heel_ys     = [_heel_y(f, side) for f in frames if _heel_y(f, side) is not None]

    peak_bend   = max(bend_vals) if bend_vals else 0.0
    bend_range  = max(bend_vals) - min(bend_vals) if bend_vals else 0.0
    peak_trunk  = max(trunk_vals) if trunk_vals else 0.0
    avg_trunk   = statistics.mean(trunk_vals) if trunk_vals else 0.0
    avg_shin    = statistics.mean(shin_vals) if shin_vals else 0.0

    # Heel rise: did the heel y-coordinate move upward (decrease in image y) substantially?
    heel_rise_detected = False
    if len(heel_ys) > 10:
        baseline_heel = statistics.mean(heel_ys[:5])
        min_heel_y    = min(heel_ys)
        # In image coords y increases downward; heel rising = y decreasing
        if baseline_heel - min_heel_y > _HEEL_RISE_THRESH:
            heel_rise_detected = True

    lines = []

    # ── No movement at all ───────────────────────────────────────────────────
    if bend_range < 8.0:
        lines.append("No squat movement was detected this round.")
        lines.append(
            "Make sure you are standing sideways to the camera and bending your knees into a squat."
        )
        return lines

    # ── No complete reps but movement present ────────────────────────────────
    if rep_count == 0:
        lines.append("No complete reps were counted this round.")
        if peak_bend < _SQUAT_PEAK:
            lines.append(
                f"You bent your knees partway down — try to squat deeper, "
                "bending your knees further until your thighs are closer to horizontal."
            )
        else:
            lines.append(
                "You squatted down but didn't fully straighten back up between reps. "
                "Stand back up tall after each squat so the rep counts."
            )
    else:
        rep_word = "rep" if rep_count == 1 else "reps"

        # Open with rep count + one positive quality observation
        if peak_bend >= _DEPTH_TARGET:
            lines.append(
                f"Good work — you completed {rep_count} {rep_word} and your squat depth was excellent, "
                "reaching close to a 90-degree knee bend."
            )
        elif peak_bend >= _DEPTH_MIN:
            lines.append(
                f"You completed {rep_count} {rep_word}. "
                "Your squat depth was reasonable — try to bend a little deeper to reach parallel."
            )
        else:
            lines.append(
                f"You completed {rep_count} {rep_word}, but your squat depth was quite shallow. "
                "Aim to bend your knees until your thighs are parallel to the floor."
            )

    # ── Depth coaching (independent of rep count) ────────────────────────────
    if rep_count > 0 and peak_bend >= _DEPTH_TARGET:
        pass  # already praised above
    elif rep_count > 0 and peak_bend < _DEPTH_TARGET:
        lines.append(
            "Try to squat a little deeper — aim for your thighs to reach horizontal at the bottom."
        )

    # ── Trunk lean ───────────────────────────────────────────────────────────
    if avg_trunk < _TRUNK_LEAN_MIN:
        lines.append(
            "Your torso was very upright throughout — a slight forward lean of around 20 to 45 degrees "
            "is natural and expected during squats."
        )
    elif avg_trunk > _TRUNK_LEAN_MAX:
        lines.append(
            "You were leaning forward quite a lot during the squat — "
            "try to keep your chest up and facing forward to avoid excessive forward lean."
        )
    else:
        lines.append(
            "Your torso lean looked good — you maintained a controlled, natural forward angle throughout."
        )

    # ── Shin / knees over toes ───────────────────────────────────────────────
    if avg_shin > _SHIN_ANGLE_MAX:
        lines.append(
            "Your knees were travelling quite far forward over your toes. "
            "Try to keep your shins more vertical by shifting your weight back through your heels."
        )

    # ── Heel rise ────────────────────────────────────────────────────────────
    if heel_rise_detected:
        lines.append(
            "It looks like your heels came off the ground during the squat — "
            "try to keep your feet flat throughout the movement to improve ankle mobility and stability."
        )

    # Ensure at least a positive comment if all checks passed and nothing extra was added
    if rep_count > 0 and len(lines) == 1:
        if avg_shin <= _SHIN_ANGLE_MAX:
            lines.append(
                "Your shin angle looked good — your knees stayed nicely over your feet without travelling too far forward."
            )

    return lines


def get_relevant_joints():
    return [
        ("R knee bend", "right_knee_bend_2d"),
        ("L knee bend", "left_knee_bend_2d"),
        ("Torso lean",  "trunk_lean_angle"),
    ]


def get_session_summary(session_data):
    total_reps = session_data.get("total_reps", 0)
    stats      = session_data.get("angle_stats", {})

    r_bend_max = stats.get("right_knee_bend_2d", {}).get("max", 0.0)
    l_bend_max = stats.get("left_knee_bend_2d",  {}).get("max", 0.0)
    best_bend  = max(r_bend_max, l_bend_max)

    trunk_max  = stats.get("trunk_lean_angle", {}).get("max", 0.0)
    trunk_min  = stats.get("trunk_lean_angle", {}).get("min", 0.0)

    r_shin_max = stats.get("right_shin_angle", {}).get("max", 0.0)
    l_shin_max = stats.get("left_shin_angle",  {}).get("max", 0.0)
    best_shin  = max(r_shin_max, l_shin_max)

    parts = []

    if total_reps == 0:
        parts.append("No complete squat reps were recorded this session.")
        parts.append(
            "Stand sideways to the camera, bend your knees into a full squat, "
            "then straighten back up completely between each rep."
        )
        return " ".join(parts)

    rep_word = "rep" if total_reps == 1 else "reps"
    parts.append(f"Session complete — you performed {total_reps} {rep_word} in total.")

    # Depth summary
    if best_bend >= _DEPTH_TARGET:
        parts.append(
            "Your squat depth was excellent — you reached close to a 90-degree knee bend, "
            "which is great for strengthening your legs through the full range."
        )
    elif best_bend >= _DEPTH_MIN:
        parts.append(
            "You reached a reasonable squat depth. "
            "Work on bending your knees a little further each time — aim for thighs parallel to the floor."
        )
    else:
        parts.append(
            "Your squat depth was quite shallow throughout the session. "
            "Focus on sitting lower into each squat to get the full benefit of the exercise."
        )

    # Trunk lean summary
    if _TRUNK_LEAN_MIN <= trunk_max <= _TRUNK_LEAN_MAX:
        parts.append(
            "Your torso lean was well controlled — you maintained a natural forward angle without hunching over."
        )
    elif trunk_max > _TRUNK_LEAN_MAX:
        parts.append(
            "You tended to lean forward quite a bit — focus on keeping your chest up "
            "and your torso more upright during the squat."
        )

    # Shin angle summary
    if best_shin > _SHIN_ANGLE_MAX and best_shin > 10.0:
        parts.append(
            "Try to keep your shins more vertical by sitting your hips back slightly — "
            "this will stop your knees from drifting too far over your toes."
        )

    return " ".join(parts)