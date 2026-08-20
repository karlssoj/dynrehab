from __future__ import annotations

import math
import statistics

# ── thresholds ────────────────────────────────────────────────────────────────
# Primary detection: right arm elevation (most active joint per reference video)
# right_arm_elevation range: min=29°, max=129°, range=100°
# 'movement started' = 29 + 30% * 100 = 59°
# 'full rep'         = 29 + 60% * 100 = 89°
_ARM_RAISE_START   = 59    # arm elevation > this → movement begins
_ARM_RAISE_PEAK    = 89    # arm elevation >= this → counts as a real rep
_ARM_LOWER_RETURN  = 40    # arm elevation <= this → returned to start position
_ARM_HYSTERESIS    = 20    # arm must drop this much from peak before "lowering"

# Right knee bend (secondary, for coordination / squat component)
# right_knee_bend_2d: min=92°, max=158°, range=66° — but these are inverted (bend values above 90)
# This looks like the knee was already deeply bent in the demo (e.g. lunge/kneeling)
# The reference video shows right_knee_bend_2d min=92, max=158 — very high values
# suggesting this is a lunge or kneeling exercise where knee stays quite bent
# We'll use right_knee_bend_2d for secondary tracking
_KNEE_BEND_START   = 30    # knee bend > this → knee is bending
_KNEE_BEND_FULL    = 75    # clinical minimum for acceptable knee bend

# Left knee: use clinical defaults (range only 54° in demo)
_LEFT_KNEE_START   = 25
_LEFT_KNEE_FULL    = 80

# Shoulder tilt range was very large (103°) — arms going from asymmetric to overhead
# Arm elevation ideal target
_ARM_IDEAL         = 100   # arm elevation >= this → excellent range

# Right elbow (clinical defaults — demo range only 50°)
_ELBOW_START       = 30
_ELBOW_FULL        = 80

_VIS_THRESHOLD     = 0.35  # minimum MediaPipe visibility to trust a joint

# ── module-level rep detection state ─────────────────────────────────────────
_phase          = "ready"   # "ready" | "moving"
_max_arm_elev   = 0.0
_start_arm_elev = 0.0
_rep_frames     = []


def _pick_side(keypoints):
    lv = keypoints.get("left_shoulder",  (0, 0, 0, 0))[3]
    rv = keypoints.get("right_shoulder", (0, 0, 0, 0))[3]
    return "left" if lv >= rv else "right"


def _joints_visible(keypoints, side):
    for part in ("shoulder", "hip"):
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
    return math.degrees(math.acos(max(-1.0, min(1.0, (vax*vcx + vay*vcy) / mag))))


def _trunk_lean_2d(pose_data, side):
    kpts = pose_data.get("keypoints", {})
    sh = kpts.get(f"{side}_shoulder")
    h  = kpts.get(f"{side}_hip")
    if not (sh and h) or min(sh[3], h[3]) < _VIS_THRESHOLD:
        return pose_data.get("trunk_lean_2d", pose_data.get("trunk_lean_angle", 0.0))
    dx = sh[0] - h[0]
    dy = sh[1] - h[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-10 or dy >= 0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, -dy / dist))))


def _best_knee_bend(pose_data):
    l = pose_data.get("left_knee_bend_2d", 0.0) or 0.0
    r = pose_data.get("right_knee_bend_2d", 0.0) or 0.0
    return max(l, r)


def _depth_frames(frames):
    return [f for f in frames if _best_knee_bend(f) > 20.0]


def _avg_shin_angle(frames):
    vals = [max(f.get("left_shin_angle", 0.0) or 0.0, f.get("right_shin_angle", 0.0) or 0.0)
            for f in frames]
    vals = [v for v in vals if v > 1.0]
    return statistics.mean(vals) if vals else 0.0


def _heel_rise(pose_data, side="auto"):
    kpts = pose_data.get("keypoints", {})
    if side == "auto":
        lh = kpts.get("left_heel");  rh = kpts.get("right_heel")
        lf = kpts.get("left_foot_index"); rf = kpts.get("right_foot_index")
        l_vis = min(lh[3] if lh else 0, lf[3] if lf else 0)
        r_vis = min(rh[3] if rh else 0, rf[3] if rf else 0)
        side = "left" if l_vis >= r_vis else "right"
    heel       = kpts.get(f"{side}_heel")
    foot_index = kpts.get(f"{side}_foot_index")
    if not heel or not foot_index:
        return 0.0
    if min(heel[3], foot_index[3]) < 0.20:
        return 0.0
    return foot_index[1] - heel[1]


def get_instructions():
    return [
        "Stand side-on to the camera so your full body is visible.",
        "Perform the exercise, raising your arms and moving through the full range of motion.",
    ]


def detect_rep(pose_data):
    global _phase, _max_arm_elev, _start_arm_elev, _rep_frames

    # Side-view orientation guard
    if pose_data.get("shoulder_lateral_span", 1.0) > 0.15:
        _phase = "ready"
        return False

    kpts = pose_data.get("keypoints", {})
    side = _pick_side(kpts)

    if not _joints_visible(kpts, side):
        return False

    # Primary signal: arm elevation on visible side
    arm_key   = f"{side}_arm_elevation"
    arm_elev  = pose_data.get(arm_key, 0.0) or 0.0

    if _phase == "ready":
        if arm_elev > _ARM_RAISE_START:
            _phase          = "moving"
            _start_arm_elev = arm_elev
            _max_arm_elev   = arm_elev
            _rep_frames     = [pose_data]

    elif _phase == "moving":
        _rep_frames.append(pose_data)
        if arm_elev > _max_arm_elev:
            _max_arm_elev = arm_elev
        # Return phase: arm has dropped significantly from peak
        if arm_elev <= _max_arm_elev - _ARM_HYSTERESIS and arm_elev <= _ARM_LOWER_RETURN:
            _phase = "ready"
            completed = _rep_frames[:]
            _rep_frames = []
            return True

    return False


def reset_round():
    global _phase, _max_arm_elev, _start_arm_elev, _rep_frames
    _phase          = "ready"
    _max_arm_elev   = 0.0
    _start_arm_elev = 0.0
    _rep_frames     = []


def classify_movement(frames):
    if not frames:
        return None

    kpts0 = frames[0].get("keypoints", {})
    side  = _pick_side(kpts0)
    arm_key = f"{side}_arm_elevation"

    arm_vals = [f.get(arm_key, 0.0) or 0.0 for f in frames]
    arm_vals = [v for v in arm_vals if v > 1.0]
    if not arm_vals:
        return "Start the exercise by raising your arms up."

    arm_range = max(arm_vals) - min(arm_vals)
    if arm_range < 10.0:
        return "Start the exercise by raising your arms up through the full range of motion."

    return None


def generate_rep_cue(cue_data):
    trigger = cue_data.get("trigger", "rep")
    frames  = cue_data.get("frames", [])

    if trigger == "timeout":
        return "Keep going — raise your arms through the full range."

    if not frames:
        return "Good rep!"

    kpts0 = frames[0].get("keypoints", {})
    side  = _pick_side(kpts0)
    arm_key = f"{side}_arm_elevation"

    # Check arm elevation depth
    arm_vals = [f.get(arm_key, 0.0) or 0.0 for f in frames]
    peak_arm = max(arm_vals) if arm_vals else 0.0

    # Check heel rise (highest priority injury risk)
    bent_frames = _depth_frames(frames)
    if bent_frames:
        rises = [_heel_rise(f) for f in bent_frames]
        if rises and max(rises) > 0.02:
            return "Keep your heels flat on the floor throughout the movement."

    # Check shin angle / knees forward
    avg_shin = _avg_shin_angle(bent_frames if bent_frames else frames)
    if avg_shin > 30:
        return "Push your hips back — your knees are travelling too far forward."

    # Check arm elevation range
    if peak_arm < _ARM_RAISE_PEAK:
        return "Raise your arms a little higher to complete the full movement."

    if peak_arm >= _ARM_IDEAL:
        return "Excellent range — keep up that full arm swing!"

    return "Good rep — stay controlled through the movement."


def generate_round_feedback(round_data):
    rep_count = round_data.get("rep_count", 0)
    frames    = round_data.get("frames", [])

    if not frames:
        return ["I couldn't see you clearly. Make sure your full body is visible side-on to the camera."]

    kpts0 = frames[0].get("keypoints", {})
    side  = _pick_side(kpts0)
    arm_key = f"{side}_arm_elevation"

    arm_vals   = [f.get(arm_key, 0.0) or 0.0 for f in frames]
    peak_arm   = max(arm_vals) if arm_vals else 0.0

    knee_vals  = [_best_knee_bend(f) for f in frames]
    peak_knee  = max(knee_vals) if knee_vals else 0.0

    bent_frames = _depth_frames(frames)

    lines = []

    if rep_count == 0:
        lines.append("No complete reps were recorded this round.")
        if peak_arm < _ARM_RAISE_START:
            lines.append(
                "It looks like your arms didn't move enough — "
                "focus on raising them through the full range of motion."
            )
        elif peak_arm < _ARM_RAISE_PEAK:
            lines.append(
                "You started the movement but didn't reach the full range — "
                "try to raise your arms higher on each repetition."
            )
        else:
            lines.append(
                "You moved well but didn't complete a full rep cycle — "
                "make sure to return your arms back to the starting position."
            )
        return lines

    rep_word = "rep" if rep_count == 1 else "reps"
    if peak_arm >= _ARM_IDEAL:
        lines.append(f"Well done — you completed {rep_count} {rep_word} with excellent arm range.")
    else:
        lines.append(f"Good effort — you completed {rep_count} {rep_word} this round.")

    # Arm elevation quality
    if peak_arm >= _ARM_IDEAL:
        lines.append("Your arm elevation was excellent — great full range of motion.")
    elif peak_arm >= _ARM_RAISE_PEAK:
        lines.append(
            "Good arm height, but try to push a little further "
            "to get the full range out of each repetition."
        )
    else:
        lines.append(
            "Try to raise your arms higher on each rep — "
            "aim to lift them to shoulder height or above for the full benefit."
        )

    # Heel rise check
    if bent_frames:
        rises = [_heel_rise(f) for f in bent_frames]
        if rises and max(rises) > 0.02:
            lines.append(
                "Your heels were lifting off the ground during the movement — "
                "focus on keeping them flat throughout the exercise."
            )

    # Knees too far forward
    avg_shin = _avg_shin_angle(bent_frames if bent_frames else frames)
    if avg_shin > 30:
        lines.append(
            "Your knees were travelling quite far forward — "
            "try to push your hips back slightly to keep your shins more vertical."
        )

    # Ankle lateral span (stance check)
    ankle_spans = [f.get("ankle_lateral_span", 0.0) for f in frames]
    avg_ankle_span = statistics.mean(ankle_spans) if ankle_spans else 0.0
    if avg_ankle_span > 0.20:
        lines.append(
            "Make sure both feet are side by side — "
            "one foot appears to be in front of the other."
        )

    # Knee depth if relevant
    if peak_knee > 5.0:
        if peak_knee >= _KNEE_BEND_FULL:
            lines.append("Good knee bend depth — you achieved a solid range of motion.")
        elif peak_knee >= _KNEE_BEND_START:
            lines.append(
                "Try to bend your knee a little more on each rep "
                "to work through the full range of motion."
            )

    return lines


def get_relevant_joints():
    return [
        ("R arm elev",  "right_arm_elevation"),
        ("L arm elev",  "left_arm_elevation"),
        ("R knee bend", "right_knee_bend_2d"),
        ("Trunk lean",  "trunk_lean_2d"),
    ]


def get_session_summary(session_data):
    total_reps = session_data.get("total_reps", 0)
    stats      = session_data.get("angle_stats", {})

    r_arm = stats.get("right_arm_elevation", {}).get("max", 0.0)
    l_arm = stats.get("left_arm_elevation",  {}).get("max", 0.0)
    best_arm = max(r_arm, l_arm)

    parts = []

    if total_reps == 0:
        parts.append("No complete reps were recorded this session.")
        if best_arm < _ARM_RAISE_PEAK:
            parts.append(
                "Focus on raising your arms higher — "
                "work through the full range of motion on each repetition."
            )
        return " ".join(parts)

    rep_word = "rep" if total_reps == 1 else "reps"
    parts.append(f"Session complete — you performed {total_reps} {rep_word} across all rounds.")

    if best_arm >= _ARM_IDEAL:
        parts.append("Your arm elevation was excellent throughout the session — great technique.")
    elif best_arm >= _ARM_RAISE_PEAK:
        parts.append(
            "Your arm height was good — next time try to push a little further "
            "for an even fuller range of motion."
        )
    elif best_arm > 10.0:
        parts.append(
            "Work on raising your arms higher during the exercise — "
            "aim for shoulder height or above for the full benefit."
        )

    return " ".join(parts)