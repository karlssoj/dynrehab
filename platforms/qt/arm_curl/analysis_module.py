from __future__ import annotations

import math
import statistics

# ── thresholds ────────────────────────────────────────────────────────────────
_CURL_START      = 30    # elbow_bend_2d > this → curl begins
_CURL_PEAK       = 70    # elbow_bend_2d >= this → counts as a real rep (clinical minimum)
_CURL_RETURN     = 20    # elbow_bend_2d <= this → arm returned to straight
_CURL_IDEAL      = 110   # elbow_bend_2d >= this → excellent curl depth
_HYSTERESIS      = 15    # bend must drop this much from peak before counting as returning
_VIS_THRESHOLD   = 0.35  # minimum MediaPipe visibility to trust a joint

# shoulder elevation thresholds (arm_elevation: 0=at side, 90=horizontal, 180=overhead)
_ELEV_STABLE_MAX = 40    # arm_elevation should stay below this during a strict curl

# ── module-level rep detection state ─────────────────────────────────────────
_phase        = "ready"   # "ready" | "curling" | "returning"
_max_bend     = 0.0
_peak_reached = False
_rep_frames   = []


def _pick_side(pose_data):
    kpts = pose_data.get("keypoints", {})
    sides = {}
    for side in ("left", "right"):
        e = kpts.get(f"{side}_elbow")
        w = kpts.get(f"{side}_wrist")
        s = kpts.get(f"{side}_shoulder")
        if e and w and s:
            sides[side] = min(e[3], w[3], s[3])
    if not sides:
        return "right"
    return max(sides, key=sides.get)


def _joints_visible(pose_data, side):
    kpts = pose_data.get("keypoints", {})
    for part in ("shoulder", "elbow", "wrist"):
        kp = kpts.get(f"{side}_{part}")
        if kp is None or kp[3] < _VIS_THRESHOLD:
            return False
    return True


def _elbow_bend(pose_data, side):
    key = f"{side}_elbow_bend_2d"
    val = pose_data.get(key)
    if val is not None:
        return float(val)
    return 0.0


def _best_elbow_bend(pose_data):
    l = pose_data.get("left_elbow_bend_2d") or 0.0
    r = pose_data.get("right_elbow_bend_2d") or 0.0
    return max(float(l), float(r))


def _arm_elevation(pose_data, side):
    key = f"{side}_arm_elevation"
    val = pose_data.get(key)
    if val is not None:
        return float(val)
    return 0.0


def get_instructions():
    return [
        "Stand facing the camera with your arms at your sides.",
        "Curl both arms up toward your shoulders, then lower them back down slowly.",
    ]


def detect_rep(pose_data):
    global _phase, _max_bend, _peak_reached, _rep_frames

    side = _pick_side(pose_data)

    if not _joints_visible(pose_data, side):
        return False

    bend = _elbow_bend(pose_data, side)

    if _phase == "ready":
        if bend > _CURL_START:
            _phase        = "curling"
            _max_bend     = bend
            _peak_reached = False
            _rep_frames   = [pose_data]

    elif _phase == "curling":
        _rep_frames.append(pose_data)
        if bend > _max_bend:
            _max_bend = bend
        if not _peak_reached and bend >= _CURL_PEAK:
            _peak_reached = True
        if bend < _max_bend - _HYSTERESIS:
            _phase = "returning"

    elif _phase == "returning":
        _rep_frames.append(pose_data)
        if bend <= _CURL_RETURN:
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


def classify_movement(frames):
    if not frames:
        return None

    bends = [_best_elbow_bend(f) for f in frames if _best_elbow_bend(f) > 1.0]
    if not bends:
        return "Start the exercise by curling your arms up toward your shoulders."

    bend_range = max(bends) - min(bends)
    if bend_range < 15.0:
        return "Curl your arms up and lower them back down to complete a repetition."

    peak_bend = max(bends)
    if peak_bend < _CURL_PEAK:
        return "Curl your arms higher — bring your hands closer to your shoulders."

    return None


def generate_rep_cue(cue_data):
    trigger = cue_data.get("trigger", "rep")
    frames  = cue_data.get("frames", [])

    if trigger == "timeout":
        if not frames:
            return "Keep going — curl your arms up toward your shoulders."
        bends = [_best_elbow_bend(f) for f in frames if _best_elbow_bend(f) > 1.0]
        if not bends or max(bends) < _CURL_START:
            return "Start curling — bring your hands up toward your shoulders."
        if max(bends) < _CURL_PEAK:
            return "Curl higher — try to bring your hands all the way up to shoulder height."
        return "Good — keep curling with a steady rhythm."

    if not frames:
        return "Good rep!"

    side = _pick_side(frames[-1])

    # Collect bend and elevation values
    bend_vals = [_elbow_bend(f, side) for f in frames]
    bend_vals = [v for v in bend_vals if v > 1.0]
    peak_bend = max(bend_vals) if bend_vals else 0.0

    elev_vals = [_arm_elevation(f, side) for f in frames]
    elev_vals = [v for v in elev_vals if v > 1.0]
    peak_elev = max(elev_vals) if elev_vals else 0.0

    # Priority 1: shoulder swinging (injury risk / compensation)
    if peak_elev > _ELEV_STABLE_MAX:
        return "Keep your elbows tucked at your sides — avoid swinging your shoulders."

    # Priority 2: insufficient range of motion
    if peak_bend < _CURL_PEAK:
        return "Curl higher — bring your hands closer to your shoulders."

    # Priority 3: encouragement / depth praise
    if peak_bend >= _CURL_IDEAL:
        return "Excellent curl — great range of motion!"

    return "Good rep — keep it up!"


def generate_round_feedback(round_data):
    rep_count = round_data.get("rep_count", 0)
    frames    = round_data.get("frames", [])

    if not frames:
        return ["I couldn't detect you. Make sure you are facing the camera with your arms visible."]

    # Pick best side from last frame
    side = _pick_side(frames[-1])

    bend_vals = [_elbow_bend(f, side) for f in frames]
    bend_vals_nonzero = [v for v in bend_vals if v > 1.0]

    elev_vals = [_arm_elevation(f, side) for f in frames]
    elev_vals_nonzero = [v for v in elev_vals if v > 1.0]

    peak_bend = max(bend_vals_nonzero) if bend_vals_nonzero else 0.0
    peak_elev = max(elev_vals_nonzero) if elev_vals_nonzero else 0.0

    lines = []

    if rep_count == 0:
        lines.append("No complete reps were recorded this round.")
        if not bend_vals_nonzero or max(bend_vals_nonzero) < _CURL_START:
            lines.append(
                "It looks like you weren't moving much — "
                "try curling your arms up toward your shoulders and lowering them back down."
            )
        elif peak_bend < _CURL_PEAK:
            lines.append(
                "You started the curl but didn't reach enough height for it to count. "
                "Try to bring your hands all the way up toward your shoulders."
            )
        else:
            lines.append(
                "You curled up but didn't fully lower your arms back down between reps. "
                "Straighten your arms fully at the bottom before starting the next curl."
            )
        return lines

    rep_word = "rep" if rep_count == 1 else "reps"
    if peak_bend >= _CURL_IDEAL:
        lines.append(
            f"Well done — you completed {rep_count} {rep_word} with an excellent range of motion."
        )
    else:
        lines.append(
            f"Good effort — you completed {rep_count} {rep_word} this round."
        )

    # Depth / range of motion
    if peak_bend < _CURL_PEAK:
        lines.append(
            "Try to curl your arms higher — aim to bring your hands close to your shoulders "
            "for the full benefit of the exercise."
        )
    elif peak_bend < _CURL_IDEAL:
        lines.append(
            "You could curl a little higher — try to bring your hands right up to shoulder level."
        )
    else:
        lines.append(
            "Your curl depth was excellent — you brought your hands right up to shoulder height."
        )

    # Elbow stability / shoulder swing
    if peak_elev > _ELEV_STABLE_MAX:
        lines.append(
            "Try to keep your elbows tucked against your sides throughout the movement — "
            "avoid swinging your shoulders to lift the weight."
        )
    else:
        lines.append(
            "Good elbow control — your elbows stayed nicely tucked at your sides."
        )

    # Check return to full extension (bottom of rep)
    low_bend_vals = [v for v in bend_vals_nonzero if v < _CURL_RETURN + 10]
    if low_bend_vals:
        lines.append(
            "Good — you lowered your arms back down fully between reps."
        )
    else:
        lines.append(
            "Make sure to fully straighten your arms at the bottom of each rep "
            "to work through the full range of motion."
        )

    return lines


def get_relevant_joints():
    return [
        ("L elbow bend", "left_elbow_bend_2d"),
        ("R elbow bend", "right_elbow_bend_2d"),
        ("L arm elev",   "left_arm_elevation"),
        ("R arm elev",   "right_arm_elevation"),
    ]


def get_session_summary(session_data):
    total_reps = session_data.get("total_reps", 0)
    stats      = session_data.get("angle_stats", {})

    l_bend = stats.get("left_elbow_bend_2d",  {}).get("max", 0.0)
    r_bend = stats.get("right_elbow_bend_2d", {}).get("max", 0.0)
    best_bend = max(l_bend, r_bend)

    l_elev = stats.get("left_arm_elevation",  {}).get("max", 0.0)
    r_elev = stats.get("right_arm_elevation", {}).get("max", 0.0)
    best_elev = max(l_elev, r_elev)

    parts = []

    if total_reps == 0:
        parts.append("No complete arm curl reps were recorded this session.")
        if best_bend < _CURL_PEAK:
            parts.append(
                "Focus on curling your arms higher — try to bring your hands up to shoulder height."
            )
        return " ".join(parts)

    rep_word = "rep" if total_reps == 1 else "reps"
    parts.append(f"Session complete — you performed {total_reps} {rep_word} in total.")

    if best_bend >= _CURL_IDEAL:
        parts.append("Your curl depth was excellent throughout the session.")
    elif best_bend >= _CURL_PEAK:
        parts.append(
            "Your curl depth was adequate — keep working on bringing your hands closer to your shoulders."
        )
    else:
        parts.append(
            "Work on curling higher — aim to bring your hands all the way up to shoulder height each rep."
        )

    if best_elev > _ELEV_STABLE_MAX:
        parts.append(
            "Try to keep your elbows at your sides rather than letting your shoulders swing forward."
        )

    return " ".join(parts)