from __future__ import annotations

import math
import statistics

_VIS_THRESHOLD = 0.3

_phase = "ready"
_start_hip_y = 0.0
_peak_hip_y = 0.0


def get_instructions() -> list[str]:
    return [
        "Stand with either side facing the camera.",
        "Perform slow squats, bending your knees to about 60 degrees."
    ]


def _best_knee_bend(pose_data: dict) -> float:
    l = pose_data.get("left_knee_bend_2d", 0.0) or 0.0
    r = pose_data.get("right_knee_bend_2d", 0.0) or 0.0
    return max(l, r)


def _depth_frames(frames: list) -> list:
    return [f for f in frames if _best_knee_bend(f) > 20.0]


def _leg_side_for_forward_ratio(frames: list) -> str | None:
    """Pick the more-visible leg once for the whole rep (not per frame —
    per-frame re-selection lets the measurement flicker between legs when
    left/right visibility is close)."""
    best_side, best_vis = None, 0.0
    for side in ("left", "right"):
        vis_vals = []
        for f in frames:
            kpts = f.get("keypoints", {})
            k = kpts.get(f"{side}_knee"); a = kpts.get(f"{side}_ankle")
            if k and a:
                vis_vals.append(min(k[3], a[3]))
        if vis_vals:
            avg_vis = sum(vis_vals) / len(vis_vals)
            if avg_vis > best_vis:
                best_side, best_vis = side, avg_vis
    return best_side if best_vis > 0.3 else None


def _knee_forward_ratio(pose_data: dict, side: str) -> float:
    """abs(knee_x - ankle_x) / shin_length for a fixed side. ~0.7 = knee over toes."""
    kpts = pose_data.get("keypoints", {})
    knee = kpts.get(f"{side}_knee"); ankle = kpts.get(f"{side}_ankle")
    if not knee or not ankle or min(knee[3], ankle[3]) < 0.3:
        return 0.0
    shin_len = math.hypot(knee[0] - ankle[0], knee[1] - ankle[1])
    if shin_len < 0.02:
        return 0.0
    return abs(knee[0] - ankle[0]) / shin_len


def _second_highest(values: list) -> float:
    """Second-largest value (duplicates counted individually); falls back to
    the single value when there's only one, or 0.0 when empty. Single-pass
    comparison instead of sorted() — sorted() is not in the sandbox's builtins
    whitelist and raises NameError at runtime."""
    first = second = -1.0
    for v in values:
        if v > first:
            first, second = v, first
        elif v > second:
            second = v
    if second >= 0:
        return second
    return first if first >= 0 else 0.0


def reset_round():
    global _phase, _start_hip_y, _peak_hip_y
    _phase = "ready"
    _start_hip_y = 0.0
    _peak_hip_y = 0.0


def detect_rep(pose_data: dict) -> bool:
    global _phase, _start_hip_y, _peak_hip_y

    if pose_data.get("shoulder_lateral_span", 1.0) > 0.15:
        _phase = "ready"
        return False

    knee_bend = _best_knee_bend(pose_data)

    kpts = pose_data.get("keypoints", {})
    h = kpts.get("left_hip") or kpts.get("right_hip")
    hip_y = h[1] if h and h[3] > 0.3 else 0.0

    START_THRESH = 20.0
    END_THRESH = 12.0

    if _phase == "ready":
        if knee_bend > START_THRESH:
            _phase = "moving"
            _start_hip_y = hip_y
            _peak_hip_y = hip_y
        return False

    elif _phase == "moving":
        if hip_y > _peak_hip_y:
            _peak_hip_y = hip_y
        if knee_bend < END_THRESH:
            _phase = "ready"
            if _start_hip_y == 0.0:
                # no reliable hip reading, allow the rep
                return True
            if _peak_hip_y > _start_hip_y + 0.03:
                return True
            return False
        return False

    return False


def classify_movement(frames: list) -> str | None:
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


def generate_rep_feedback(rep_data: dict) -> list[str]:
    frames = rep_data.get("frames", [])
    rep_number = rep_data.get("rep_number", 0)

    feedback = []
    feedback.append(f"Rep {rep_number} done.")

    if not frames:
        return feedback

    bent_frames = _depth_frames(frames)
    check_frames = bent_frames if bent_frames else frames

    # Depth check
    peak_knee = max((_best_knee_bend(f) for f in frames), default=0.0)
    if peak_knee < 60:
        feedback.append("Try to squat a bit deeper — aim for at least a 60 degree knee bend.")

    # Upper body lean check
    trunk_vals = [f.get("trunk_lean_2d", 0.0) for f in check_frames if f.get("trunk_lean_2d", 0.0) > 0]
    if trunk_vals:
        peak_trunk = max(trunk_vals)
        if peak_trunk > 45:
            feedback.append("You're leaning too far forward with your upper body — try to keep your chest a bit more upright.")
        if peak_trunk < 10:
            feedback.append("Your upper body stayed too straight or leaned backward — allow a slight natural forward lean as you squat.")

    # Knees over toes check — leg picked once per rep from aggregate visibility,
    # aggregated as the second-highest per-frame ratio (not a single max()) so
    # a single noisy frame can't decide the whole rep's verdict: genuine
    # excessive forward travel persists across several frames, a sensor
    # glitch doesn't.
    side = _leg_side_for_forward_ratio(check_frames)
    if side:
        ratios = [r for r in (_knee_forward_ratio(f, side) for f in check_frames) if r > 0.05]
        if ratios:
            robust_peak = _second_highest(ratios)
            if robust_peak > 0.35:
                feedback.append("Your knees are travelling too far over your toes — try shifting your weight back into your heels.")

    # Stance check
    ankle_spans = [f.get("ankle_lateral_span", 0.0) for f in frames]
    avg_ankle_span = statistics.mean(ankle_spans) if ankle_spans else 0.0
    if avg_ankle_span > 0.20:
        feedback.append("Make sure both feet are side by side — one foot appears to be in front of the other.")

    if len(feedback) == 1:
        feedback.append("Nice squat — good depth and posture.")

    return feedback


def get_session_summary(session_data: dict) -> list[str]:
    total_reps = session_data.get("total_reps", 0)
    lines = []

    if total_reps == 0:
        lines.append("No reps were completed this session.")
        lines.append("Next time, stand side-on to the camera and perform slow squats, bending your knees to at least a 60 degree angle.")
        return lines

    lines.append(f"You completed {total_reps} squats this session.")

    rounds = session_data.get("rounds", [])
    round_peaks = []
    for r in rounds:
        vals = [_best_knee_bend(f) for f in r.get("frames", []) if _best_knee_bend(f) > 5]
        round_peaks.append(max(vals) if vals else 0)

    if len(round_peaks) >= 2:
        if round_peaks[-1] > round_peaks[0] + 10:
            lines.append(f"Your squat depth improved from round 1 to round {len(round_peaks)}.")
        elif round_peaks[0] > round_peaks[-1] + 10:
            lines.append(f"Your squat depth was better in round 1 than in round {len(round_peaks)}.")
        elif all(p >= 60 for p in round_peaks):
            lines.append("You maintained good squat depth across all rounds.")

    round_feedback = session_data.get("round_feedback", [])
    if round_feedback:
        first_issues = " ".join(round_feedback[0]) if round_feedback[0] else ""
        last_issues = " ".join(round_feedback[-1]) if round_feedback[-1] else ""

        if "forward" in first_issues.lower() and "forward" not in last_issues.lower():
            lines.append("Your forward lean improved by the later rounds.")
        elif "forward" in first_issues.lower() and "forward" in last_issues.lower():
            lines.append("Leaning too far forward was a persistent issue throughout the session — keep working on keeping your chest upright.")

        if "toes" in first_issues.lower() and "toes" not in last_issues.lower():
            lines.append("Your knee position over your toes improved as the session went on.")
        elif "toes" in first_issues.lower() and "toes" in last_issues.lower():
            lines.append("Your knees traveled too far over your toes in multiple rounds — try sitting back into your hips more.")

    if round_peaks and max(round_peaks) >= 60:
        lines.append("You reached a good squat depth at least once during the session.")
    else:
        lines.append("Focus on bending your knees further to reach a deeper squat next time.")

    return lines


def get_relevant_joints() -> list:
    return [
        ("Knee Flex", "left_knee_bend_2d"),
        ("Trunk Lean", "trunk_lean_2d"),
        ("Hip Flex", "left_hip_bend_2d"),
    ]
