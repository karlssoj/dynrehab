from __future__ import annotations

import math
import statistics

# ---------------------------------------------------------------------------
# Module-level state for detect_rep
# ---------------------------------------------------------------------------
_phase = "ready"
_start_hip_y = 0.0
_peak_hip_y = 0.0

# Calibrated thresholds (bend convention, 0=straight)
_KNEE_MOVE_START = 34.0   # ~min(9) + 30% of range(83) from reference video
_KNEE_RETURN_TO_READY = 15.0
_KNEE_DEPTH_TARGET = 60.0  # explicit physio target for this exercise
_TRUNK_LEAN_MIN = 10.0
_TRUNK_LEAN_MAX = 30.0


def get_instructions() -> list:
    return [
        "Stand with either side of your body towards the camera.",
        "Perform slow squats, bending your knees to about 60 degrees while keeping your back naturally straight."
    ]


# ---------------------------------------------------------------------------
# Helpers (copied per spec)
# ---------------------------------------------------------------------------
def _best_knee_bend(pose_data: dict) -> float:
    l = pose_data.get("left_knee_bend_2d", 0.0) or 0.0
    r = pose_data.get("right_knee_bend_2d", 0.0) or 0.0
    return max(l, r)


def _depth_frames(frames: list) -> list:
    return [f for f in frames if _best_knee_bend(f) > 20.0]


def _leg_side_for_forward_ratio(frames: list):
    best_side, best_vis = None, 0.0
    for side in ("left", "right"):
        vis_vals = []
        for f in frames:
            kpts = f.get("keypoints", {})
            k = kpts.get(f"{side}_knee")
            a = kpts.get(f"{side}_ankle")
            if k and a:
                vis_vals.append(min(k[3], a[3]))
        if vis_vals:
            avg_vis = sum(vis_vals) / len(vis_vals)
            if avg_vis > best_vis:
                best_side, best_vis = side, avg_vis
    return best_side if best_vis > 0.3 else None


def _knee_forward_ratio(pose_data: dict, side: str) -> float:
    kpts = pose_data.get("keypoints", {})
    knee = kpts.get(f"{side}_knee")
    ankle = kpts.get(f"{side}_ankle")
    if not knee or not ankle or min(knee[3], ankle[3]) < 0.3:
        return 0.0
    shin_len = math.hypot(knee[0] - ankle[0], knee[1] - ankle[1])
    if shin_len < 0.02:
        return 0.0
    return abs(knee[0] - ankle[0]) / shin_len


def _second_highest(values: list) -> float:
    first = second = -1.0
    for v in values:
        if v > first:
            first, second = v, first
        elif v > second:
            second = v
    if second >= 0:
        return second
    return first if first >= 0 else 0.0


def _spine_zone_changes(frames: list, baseline) -> dict:
    if not baseline:
        return {"thoracic": None, "lumbar": None}
    result = {}
    for zone in ("thoracic", "lumbar"):
        base = baseline.get(zone)
        if base is None:
            result[zone] = None
            continue
        samples = [(f.get("timestamp", 0.0), f.get(f"spine_{zone}_curvature"))
                   for f in frames if f.get(f"spine_{zone}_curvature") is not None]
        if not samples:
            result[zone] = None
            continue
        deltas = [(t, v - base) for t, v in samples]
        t0 = deltas[0][0]
        t1, biggest = max(deltas, key=lambda td: abs(td[1]))
        span = max(t1 - t0, 0.5)
        result[zone] = {"delta": biggest, "rate": biggest / span}
    return result


# ---------------------------------------------------------------------------
# Rep detection
# ---------------------------------------------------------------------------
def detect_rep(pose_data: dict) -> bool:
    global _phase, _start_hip_y, _peak_hip_y

    if pose_data.get("shoulder_lateral_span", 1.0) > 0.15:
        _phase = "ready"
        return False

    knee_bend = _best_knee_bend(pose_data)
    kpts = pose_data.get("keypoints", {})
    h = kpts.get("left_hip") or kpts.get("right_hip")
    hip_y = h[1] if h and h[3] > 0.3 else 0.0

    if _phase == "ready":
        if knee_bend > _KNEE_MOVE_START:
            _phase = "moving"
            _start_hip_y = hip_y
            _peak_hip_y = hip_y
        return False

    if _phase == "moving":
        if hip_y > _peak_hip_y:
            _peak_hip_y = hip_y
        if knee_bend < _KNEE_RETURN_TO_READY:
            _phase = "ready"
            if _start_hip_y > 0.0 and _peak_hip_y > 0.0:
                return _peak_hip_y > _start_hip_y + 0.03
            return True
        return False

    return False


def reset_round():
    global _phase, _start_hip_y, _peak_hip_y
    _phase = "ready"
    _start_hip_y = 0.0
    _peak_hip_y = 0.0


def classify_movement(frames: list):
    bends = [_best_knee_bend(f) for f in frames if _best_knee_bend(f) > 1.0]
    bend_range = max(bends) - min(bends) if bends else 0.0
    if bend_range < 10.0:
        return "Start the exercise by bending both knees to squat down."

    bent = [f for f in frames if _best_knee_bend(f) > 20.0]
    stand = [f for f in frames if _best_knee_bend(f) < 5.0]
    if bent and stand:
        def _hy(f):
            kpts = f.get("keypoints", {})
            pts = [kpts[s][1] for s in ("left_hip", "right_hip")
                   if kpts.get(s) and kpts[s][3] > 0.3]
            return sum(pts) / len(pts) if pts else 0.0
        bent_hy = [_hy(f) for f in bent if _hy(f) > 0]
        stand_hy = [_hy(f) for f in stand if _hy(f) > 0]
        if bent_hy and stand_hy:
            if sum(bent_hy) / len(bent_hy) < sum(stand_hy) / len(stand_hy) + 0.03:
                return "Squat down by lowering your hips equally on both sides."
    return None


# ---------------------------------------------------------------------------
# Per-rep feedback
# ---------------------------------------------------------------------------
def generate_rep_feedback(rep_data: dict) -> list:
    feedback = []
    rep_number = rep_data.get("rep_number", 0)
    frames = rep_data.get("frames", [])

    feedback.append(f"Rep {rep_number} done.")

    if not frames:
        return feedback

    # Stance check
    ankle_spans = [f.get("ankle_lateral_span", 0.0) for f in frames]
    avg_ankle_span = statistics.mean(ankle_spans) if ankle_spans else 0.0
    if avg_ankle_span > 0.20:
        feedback.append("Make sure both feet are side by side — one foot appears to be in front of the other.")

    # --- Depth (always reported) ---
    knee_bends = [v for v in (_best_knee_bend(f) for f in frames) if v > 1.0]
    peak_knee = max(knee_bends) if knee_bends else 0.0
    if peak_knee >= _KNEE_DEPTH_TARGET:
        feedback.append("Good squat depth — you reached the target knee bend.")
    else:
        feedback.append("Try to squat deeper — aim to bend your knees at least 60 degrees.")

    # --- Trunk lean (always reported) ---
    bent_frames = _depth_frames(frames)
    check_frames = bent_frames if bent_frames else frames
    trunk_leans = [v for v in (f.get("trunk_lean_2d", 0.0) for f in check_frames) if v > 1.0]
    if trunk_leans:
        avg_lean = statistics.mean(trunk_leans)
        if avg_lean > _TRUNK_LEAN_MAX:
            feedback.append("You're leaning too far forward — keep your chest a bit more upright.")
        elif avg_lean < _TRUNK_LEAN_MIN:
            feedback.append("You're too upright — allow your torso to lean forward a little more as you squat.")
        else:
            feedback.append("Your upper body lean looks well balanced during the squat.")
    else:
        feedback.append("Your upper body lean looks well balanced during the squat.")

    # --- Knees over toes ---
    side = _leg_side_for_forward_ratio(check_frames)
    if side:
        ratios = [r for r in (_knee_forward_ratio(f, side) for f in check_frames) if r > 0.05]
        if ratios:
            robust_peak = _second_highest(ratios)
            if robust_peak > 0.35:
                feedback.append("Your knees are travelling too far over your toes — try to keep them more behind your toes.")

    # --- Spine curvature (always reported) ---
    baseline = rep_data.get("spine_baseline")
    if not baseline:
        feedback.append("Spine curvature could not be measured for this rep.")
    else:
        changes = _spine_zone_changes(frames, baseline)
        reported = False
        info_t = changes.get("thoracic")
        if info_t is not None:
            if info_t["delta"] > 0.04:
                feedback.append("Your upper back is rounding excessively — try to keep a more neutral spine.")
                reported = True
            elif abs(info_t["rate"]) > 0.15:
                feedback.append("Try to keep your upper back from moving so quickly during the squat.")
                reported = True
        info_l = changes.get("lumbar")
        if info_l is not None:
            if info_l["delta"] < -0.04:
                feedback.append("Your lower back is arching too much — try to keep a more neutral spine.")
                reported = True
            elif abs(info_l["rate"]) > 0.15:
                feedback.append("Try to keep your lower back from moving so quickly during the squat.")
                reported = True
        if not reported:
            feedback.append("Your back maintained a natural curve during the squat.")

    return feedback


# ---------------------------------------------------------------------------
# Session summary
# ---------------------------------------------------------------------------
def get_session_summary(session_data: dict) -> list:
    lines = []
    total_reps = session_data.get("total_reps", 0)

    if total_reps == 0:
        lines.append("No reps were completed this session.")
        lines.append("Try to perform a full squat by bending your knees to at least 60 degrees while keeping your back in a neutral, natural position.")
        return lines

    lines.append(f"You completed {total_reps} squat reps this session.")

    rounds = session_data.get("rounds", [])
    round_peaks = []
    for r in rounds:
        vals = [v for v in (_best_knee_bend(f) for f in r.get("frames", [])) if v > 5.0]
        round_peaks.append(max(vals) if vals else 0.0)

    if len(round_peaks) >= 2:
        if round_peaks[-1] > round_peaks[0] + 10:
            lines.append("Your squat depth improved across the session.")
        elif round_peaks[0] > round_peaks[-1] + 10:
            lines.append("Your squat depth was better in the earlier rounds than at the end.")
        elif all(p >= _KNEE_DEPTH_TARGET for p in round_peaks):
            lines.append("You maintained good squat depth throughout the session.")
    elif round_peaks and round_peaks[0] >= _KNEE_DEPTH_TARGET:
        lines.append("You reached good squat depth during your round.")

    round_feedback = session_data.get("round_feedback", [])

    def _contains(keyword):
        for rf in round_feedback:
            joined = " ".join(rf)
            if keyword in joined:
                return True
        return False

    forward_lean_flag = _contains("leaning too far forward")
    upright_flag = _contains("too upright")
    knees_flag = _contains("over your toes")
    kyphosis_flag = _contains("rounding excessively")
    arch_flag = _contains("arching too much")

    if forward_lean_flag:
        lines.append("Leaning too far forward with your upper body was a recurring issue — focus on keeping your chest lifted.")
    if upright_flag:
        lines.append("Staying too upright during the squat was a recurring issue — allow a bit more forward lean from your hips.")
    if knees_flag:
        lines.append("Your knees traveled too far over your toes in some reps — focus on pushing your hips back.")
    if kyphosis_flag:
        lines.append("Excessive upper back rounding showed up during the session — try to keep your chest open.")
    if arch_flag:
        lines.append("Excessive lower back arching appeared during the session — engage your core to keep a neutral spine.")

    if not (forward_lean_flag or upright_flag or knees_flag or kyphosis_flag or arch_flag):
        lines.append("Your form stayed safe and consistent throughout the session.")

    return lines


def get_relevant_joints() -> list:
    return [
        ("Knee Flex", "left_knee_bend_2d"),
        ("Trunk Lean", "trunk_lean_2d"),
        ("Hip Flex", "left_hip_bend_2d"),
    ]