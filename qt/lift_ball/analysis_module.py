import math
import statistics

# ── visibility threshold ───────────────────────────────────────────────────────
_VIS_THRESHOLD = 0.35

# ── rep detection thresholds (loose — for counting any attempt) ───────────────
_KNEE_BEND_START   = 20    # knee_bend > this → squat phase started
_KNEE_BEND_RETURN  = 10    # knee_bend <= this → returned to standing
_ARM_LIFT_START    = 40    # arm_elevation > this → lifting phase started
_ARM_LIFT_PEAK     = 60    # arm_elevation >= this → rep counted (ball above shoulder)
_ARM_RETURN        = 30    # arm_elevation <= this → arm returned down

# ── quality boundary values (for feedback only) ───────────────────────────────
_KNEE_BEND_GOOD    = 45    # knee bend at pickup → decent bend
_KNEE_BEND_IDEAL   = 84    # 0 + 0.6*130 → deep bend (ideal quality)
_TRUNK_LEAN_MAX    = 21    # 0 + 0.3*69 → trunk staying upright during pickup
_ARM_ELEV_GOOD     = 69    # 8 + 0.6*100 → ball clearly above shoulder level

# ── module-level rep detection state ─────────────────────────────────────────
_phase         = "ready"   # "ready" | "squatting" | "lifting" | "lowering"
_max_knee_bend = 0.0
_max_arm_elev  = 0.0
_peak_reached  = False
_rep_frames    = []


# ── helpers ───────────────────────────────────────────────────────────────────

def _angle_2d(ax, ay, bx, by, cx, cy):
    vax, vay = ax - bx, ay - by
    vcx, vcy = cx - bx, cy - by
    mag = math.hypot(vax, vay) * math.hypot(vcx, vcy)
    if mag < 1e-10:
        return 180.0
    return math.degrees(math.acos(max(-1.0, min(1.0, (vax*vcx + vay*vcy) / mag))))


def _visible(kpts, *names):
    for n in names:
        kp = kpts.get(n)
        if kp is None or kp[3] < _VIS_THRESHOLD:
            return False
    return True


def _pick_side(kpts):
    """Return which side of the body faces the camera (more visible side)."""
    lv = kpts.get("left_shoulder",  (0, 0, 0, 0))[3]
    rv = kpts.get("right_shoulder", (0, 0, 0, 0))[3]
    return "left" if lv >= rv else "right"


def _knee_bend(pose_data, side):
    """Return knee bend for given side (0=straight, higher=more bent)."""
    val = pose_data.get(f"{side}_knee_bend_2d", None)
    if val is not None:
        return val
    return 0.0


def _hip_bend(pose_data, side):
    """Return hip bend for given side (0=upright, higher=more flexed)."""
    val = pose_data.get(f"{side}_hip_bend_2d", None)
    if val is not None:
        return val
    return 0.0


def _arm_elevation(pose_data, side):
    """Return arm elevation for given side (0=at side, 90=horizontal, 180=overhead)."""
    val = pose_data.get(f"{side}_arm_elevation", None)
    if val is not None:
        return val
    return 0.0


def _trunk_lean(pose_data, side):
    """Trunk forward lean from vertical (0=upright)."""
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


def _body_visible(kpts, side):
    required = [f"{side}_shoulder", f"{side}_hip", f"{side}_knee", f"{side}_ankle"]
    return _visible(kpts, *required)


# ── public API ────────────────────────────────────────────────────────────────

def get_instructions():
    return [
        "Stand side-on to the camera so your full body is visible.",
        "Grab the ball from the floor on your left, bend your knees to pick it up, then rotate and lift it above your shoulder on the other side.",
    ]


def detect_rep(pose_data):
    global _phase, _max_knee_bend, _max_arm_elev, _peak_reached, _rep_frames

    kpts = pose_data.get("keypoints", {})
    side = _pick_side(kpts)

    if not _body_visible(kpts, side):
        return False

    knee_bend = _knee_bend(pose_data, side)
    arm_elev  = _arm_elevation(pose_data, side)

    if _phase == "ready":
        # Detect squat-down to pick up the ball
        if knee_bend > _KNEE_BEND_START:
            _phase         = "squatting"
            _max_knee_bend = knee_bend
            _max_arm_elev  = arm_elev
            _peak_reached  = False
            _rep_frames    = [pose_data]

    elif _phase == "squatting":
        _rep_frames.append(pose_data)
        if knee_bend > _max_knee_bend:
            _max_knee_bend = knee_bend
        # Transition: knees starting to straighten and arm rising → lifting phase
        if knee_bend < _max_knee_bend - 10 and arm_elev > _ARM_LIFT_START:
            _phase = "lifting"

    elif _phase == "lifting":
        _rep_frames.append(pose_data)
        if arm_elev > _max_arm_elev:
            _max_arm_elev = arm_elev
        if not _peak_reached and arm_elev >= _ARM_LIFT_PEAK:
            _peak_reached = True
        # Once arm has peaked and starts coming down, move to lowering
        if _peak_reached and arm_elev < _max_arm_elev - 15:
            _phase = "lowering"

    elif _phase == "lowering":
        _rep_frames.append(pose_data)
        if arm_elev <= _ARM_RETURN and knee_bend <= _KNEE_BEND_RETURN:
            _phase = "ready"
            if _peak_reached:
                _rep_frames = []
                return True
            _rep_frames = []

    return False


def reset_round():
    global _phase, _max_knee_bend, _max_arm_elev, _peak_reached, _rep_frames
    _phase         = "ready"
    _max_knee_bend = 0.0
    _max_arm_elev  = 0.0
    _peak_reached  = False
    _rep_frames    = []


def generate_round_feedback(round_data):
    rep_count = round_data.get("rep_count", 0)
    frames    = round_data.get("frames", [])

    if not frames:
        return ["I couldn't see you clearly — make sure your full body is visible side-on to the camera."]

    kpts = frames[0].get("keypoints", {})
    side = _pick_side(kpts)

    # Collect metrics across all frames
    knee_bends   = [_knee_bend(f, side) for f in frames]
    arm_elevs    = [_arm_elevation(f, side) for f in frames]
    trunk_leans  = [_trunk_lean(f, side) for f in frames]
    hip_bends    = [_hip_bend(f, side) for f in frames]

    peak_knee   = max(knee_bends)  if knee_bends  else 0.0
    peak_arm    = max(arm_elevs)   if arm_elevs   else 0.0
    peak_trunk  = max(trunk_leans) if trunk_leans else 0.0
    peak_hip    = max(hip_bends)   if hip_bends   else 0.0

    # Check for any meaningful movement
    any_movement = peak_knee > 10 or peak_arm > 20

    lines = []

    if rep_count == 0:
        if not any_movement:
            lines.append("No movement was detected this round — make sure you're fully visible and perform the full lift from floor to above shoulder.")
            return lines

        lines.append("I detected some movement but no complete reps were counted this round.")

        if peak_knee < _KNEE_BEND_START:
            lines.append("Try bending your knees more when picking the ball up from the floor.")
        elif peak_arm < _ARM_LIFT_PEAK:
            lines.append(
                f"You lifted the ball but didn't quite reach above shoulder level — "
                f"drive the ball higher and fully extend your arms overhead."
            )
        else:
            lines.append(
                "You reached a good height but the movement wasn't completed smoothly — "
                "try to bend your knees on the way down and extend fully on the way up."
            )
        return lines

    rep_word = "rep" if rep_count == 1 else "reps"
    opener_done = False

    # Positive observation: arm elevation
    if peak_arm >= _ARM_ELEV_GOOD:
        lines.append(f"Good work — you completed {rep_count} {rep_word} and lifted the ball well above shoulder level.")
        opener_done = True
    else:
        lines.append(f"Good effort — you completed {rep_count} {rep_word}.")

    # Knee bend quality at pickup
    if peak_knee >= _KNEE_BEND_IDEAL:
        lines.append("Excellent knee bend when picking the ball up — you really sat into the squat.")
    elif peak_knee >= _KNEE_BEND_GOOD:
        lines.append("Good knee bend at pickup — try to sink a little lower to take more load through your legs.")
    else:
        lines.append("Bend your knees more when picking the ball up — aim for a deep squat so your legs do the lifting, not your back.")

    # Arm elevation quality at the top
    if not opener_done:
        if peak_arm >= _ARM_ELEV_GOOD:
            lines.append("You lifted the ball well above shoulder level — great range.")
        else:
            lines.append("Try to lift the ball higher at the top — extend your arms fully so the ball clears shoulder level.")

    # Trunk lean quality — should stay upright
    if peak_trunk <= _TRUNK_LEAN_MAX:
        lines.append("Your torso stayed nicely upright throughout — good posture control.")
    else:
        lines.append("Keep your torso more upright — avoid leaning forward when picking up and lifting the ball.")

    return lines


def get_relevant_joints():
    return [
        ("Knee bend",  "left_knee_bend_2d"),
        ("Arm elev",   "left_arm_elevation"),
        ("Trunk lean", "trunk_lean_angle"),
        ("Hip bend",   "left_hip_bend_2d"),
    ]


def get_session_summary(session_data):
    total_reps = session_data.get("total_reps", 0)
    stats      = session_data.get("angle_stats", {})

    l_knee_max = stats.get("left_knee_bend_2d",   {}).get("max", 0.0)
    r_knee_max = stats.get("right_knee_bend_2d",  {}).get("max", 0.0)
    l_arm_max  = stats.get("left_arm_elevation",  {}).get("max", 0.0)
    r_arm_max  = stats.get("right_arm_elevation", {}).get("max", 0.0)
    trunk_max  = stats.get("trunk_lean_angle",    {}).get("max", 0.0)

    best_knee = max(l_knee_max, r_knee_max)
    best_arm  = max(l_arm_max,  r_arm_max)

    # Filter out detection artifacts
    if best_knee < 10.0:
        best_knee = 0.0
    if best_arm < 10.0:
        best_arm = 0.0

    parts = []

    if total_reps == 0:
        parts.append("No complete lifts were recorded this session.")
        parts.append(
            "Remember to bend your knees deeply to pick the ball up from the floor, "
            "then rotate and drive it above shoulder level on the other side."
        )
        return " ".join(parts)

    rep_word = "rep" if total_reps == 1 else "reps"
    parts.append(f"Well done — you completed {total_reps} {rep_word} across the session.")

    if best_arm >= _ARM_ELEV_GOOD:
        parts.append("You consistently lifted the ball above shoulder level — excellent range of motion.")
    elif best_arm > 20.0:
        parts.append(
            "Work on lifting the ball a little higher at the top — extend your arms fully so the ball clears shoulder level."
        )

    if best_knee >= _KNEE_BEND_IDEAL:
        parts.append("Your knee bend at pickup was deep and controlled — great technique using your legs.")
    elif best_knee >= _KNEE_BEND_GOOD:
        parts.append("Good knee bend at pickup — try to sink a little deeper into the squat for maximum leg engagement.")
    elif best_knee > 10.0:
        parts.append(
            "Focus on bending your knees more when picking the ball up — a deeper squat protects your back and builds leg strength."
        )

    if trunk_max > _TRUNK_LEAN_MAX and trunk_max > 10.0:
        parts.append("Try to keep your torso more upright throughout the movement — avoid leaning forward when lifting.")

    return " ".join(parts)