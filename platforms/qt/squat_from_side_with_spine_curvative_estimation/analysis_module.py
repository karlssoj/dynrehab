from __future__ import annotations

import math
import statistics

# ---------------------------------------------------------------------------
# Module-level state for rep detection
# ---------------------------------------------------------------------------
_phase = "ready"
_start_hip_y = 0.0
_peak_hip_y = 0.0


def get_instructions() -> list:
    return [
        "Stand with either side facing the camera, feet flat on the floor.",
        "Perform slow squats, bending your knees to about 60 degrees while keeping your back naturally aligned."
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _best_knee_bend(pose_data: dict) -> float:
    l = pose_data.get("left_knee_bend_2d", 0.0) or 0.0
    r = pose_data.get("right_knee_bend_2d", 0.0) or 0.0
    return max(l, r)


def _depth_frames(frames: list) -> list:
    return [f for f in frames if _best_knee_bend(f) > 20.0]


def _leg_side_for_forward_ratio(frames: list):
    """Pick the more-visible leg once for the whole rep."""
    best_side, best_vis = None, 0.0
    for side in ("left", "right"):
        vis_vals = []
        for f in frames:
            kpts = f.get("keypoints", {})
            k = kpts.get(side + "_knee")
            a = kpts.get(side + "_ankle")
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
    knee = kpts.get(side + "_knee")
    ankle = kpts.get(side + "_ankle")
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


# ---------------------------------------------------------------------------
# Rep detection
# ---------------------------------------------------------------------------
def reset_round():
    global _phase, _start_hip_y, _peak_hip_y
    _phase = "ready"
    _start_hip_y = 0.0
    _peak_hip_y = 0.0


def detect_rep(pose_data: dict) -> bool:
    global _phase, _start_hip_y, _peak_hip_y

    # Orientation guard — must be in side profile.
    if pose_data.get("shoulder_lateral_span", 1.0) > 0.15:
        _phase = "ready"
        return False

    knee_bend = _best_knee_bend(pose_data)

    kpts = pose_data.get("keypoints", {})
    h = kpts.get("left_hip") or kpts.get("right_hip")
    hip_y = h[1] if h and h[3] > 0.3 else 0.0

    START_THRESH = 34.0   # ~ bend_min(9) + 30% of range(83) from reference demo
    RETURN_THRESH = 15.0  # back near standing

    if _phase == "ready":
        if knee_bend > START_THRESH:
            _phase = "moving"
            _start_hip_y = hip_y
            _peak_hip_y = hip_y
        return False
    else:
        if hip_y > _peak_hip_y:
            _peak_hip_y = hip_y
        if knee_bend < RETURN_THRESH:
            _phase = "ready"
            if _start_hip_y > 0.0 and _peak_hip_y > 0.0:
                if _peak_hip_y > _start_hip_y + 0.03:
                    return True
                return False
            return True
        return False


def classify_movement(frames: list):
    bends = [_best_knee_bend(f) for f in frames if _best_knee_bend(f) > 1.0]
    bend_range = (max(bends) - min(bends)) if bends else 0.0
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
                return "Squat down by lowering your hips evenly on both sides."
    return None


# ---------------------------------------------------------------------------
# Per-rep feedback
# ---------------------------------------------------------------------------
def generate_rep_feedback(rep_data: dict) -> list:
    frames = rep_data.get("frames", [])
    rep_number = rep_data.get("rep_number", 0)
    feedback = []
    feedback.append("Rep " + str(rep_number) + " done.")

    if not frames:
        return feedback

    # Stance check
    ankle_spans = [f.get("ankle_lateral_span", 0.0) for f in frames]
    avg_ankle_span = statistics.mean(ankle_spans) if ankle_spans else 0.0
    if avg_ankle_span > 0.20:
        feedback.append("Make sure both feet are side by side — one foot appears to be in front of the other.")

    bent_frames = _depth_frames(frames)
    check_frames = bent_frames if bent_frames else frames

    # 1. Depth
    bends = [_best_knee_bend(f) for f in frames if _best_knee_bend(f) > 1.0]
    peak_knee = max(bends) if bends else 0.0
    if peak_knee < 60.0:
        feedback.append("Try to squat down further to reach at least a 60 degree knee bend.")
    else:
        feedback.append("Good squat depth — you reached at least 60 degrees of knee bend.")

    # 2. Trunk lean
    leans = [f.get("trunk_lean_2d", 0.0) or 0.0 for f in check_frames]
    leans = [v for v in leans if v > 0.5]
    if leans:
        peak_lean = max(leans)
        if peak_lean < 10.0:
            feedback.append("You are staying too upright — allow your upper body to lean forward a little more.")
        elif peak_lean > 30.0:
            feedback.append("You are leaning too far forward — keep your chest a bit more upright.")
        else:
            feedback.append("Your upper body lean looks good.")
    else:
        feedback.append("Your upper body lean looks good.")

    # 3. Knees over toes
    side = _leg_side_for_forward_ratio(check_frames)
    if side:
        ratios = [r for r in (_knee_forward_ratio(f, side) for f in check_frames) if r > 0.05]
        if ratios:
            robust_peak = _second_highest(ratios)
            if robust_peak > 0.35:
                feedback.append("Your knees are travelling too far over your toes — push your hips back as you squat.")

    # 4. Spine — thoracic (upper back rounding / kyphosis)
    thoracic_vals = [f.get("spine_thoracic_bend_2d") for f in frames
                      if f.get("spine_thoracic_bend_2d") is not None]
    if thoracic_vals:
        max_thoracic = max(thoracic_vals)
        if max_thoracic > 30.0:
            feedback.append("Your upper back is rounding too much — try to keep a more neutral spine.")
        else:
            feedback.append("Your upper back position looks good.")
    else:
        feedback.append("Could not clearly measure your upper back curvature this rep.")

    # 5. Spine — lumbar (lower back arch)
    lumbar_vals = [f.get("spine_lumbar_bend_2d") for f in frames
                    if f.get("spine_lumbar_bend_2d") is not None]
    if lumbar_vals:
        min_lumbar = min(lumbar_vals)
        if min_lumbar < -40.0:
            feedback.append("Your lower back is arching too much — engage your core to keep it more neutral.")
        else:
            feedback.append("Your lower back position looks good.")
    else:
        feedback.append("Could not clearly measure your lower back curvature this rep.")

    return feedback


# ---------------------------------------------------------------------------
# Session summary
# ---------------------------------------------------------------------------
def get_session_summary(session_data: dict) -> list:
    lines = []
    total_reps = session_data.get("total_reps", 0)
    rounds = session_data.get("rounds", [])
    round_feedback = session_data.get("round_feedback", [])

    if total_reps == 0:
        lines.append("No reps were completed this session.")
        lines.append("Next time, stand side-on to the camera and perform a full squat, bending your knees to about 60 degrees while keeping your back naturally aligned.")
        return lines

    lines.append("You completed " + str(total_reps) + " reps this session.")

    # Depth progression across rounds
    round_peaks = []
    for r in rounds:
        vals = [_best_knee_bend(f) for f in r.get("frames", []) if _best_knee_bend(f) > 5]
        round_peaks.append(max(vals) if vals else 0)

    if len(round_peaks) >= 2:
        if round_peaks[-1] > round_peaks[0] + 10:
            lines.append("Your squat depth improved across the session.")
        elif round_peaks[0] > round_peaks[-1] + 10:
            lines.append("Your squat depth was better in the earlier rounds.")
        elif all(p >= 60 for p in round_peaks):
            lines.append("You maintained good squat depth throughout the session.")

    def _issue_present(fragment, fb_list):
        for line in fb_list:
            if fragment in line:
                return True
        return False

    if round_feedback:
        first_fb = round_feedback[0]
        last_fb = round_feedback[-1]
        if _issue_present("rounding", first_fb) and not _issue_present("rounding", last_fb):
            lines.append("Your upper back rounding improved by the later rounds.")
        elif _issue_present("rounding", first_fb) and _issue_present("rounding", last_fb):
            lines.append("Work on reducing upper back rounding — it showed up across the session.")

        if _issue_present("arching", first_fb) and not _issue_present("arching", last_fb):
            lines.append("Your lower back arching improved by the later rounds.")
        elif _issue_present("arching", first_fb) and _issue_present("arching", last_fb):
            lines.append("Work on reducing lower back arching — it showed up across the session.")

    lines.append("Good effort working on your squat form and posture this session.")

    return lines


def get_relevant_joints() -> list:
    return [
        ("Knee Bend", "left_knee_bend_2d"),
        ("Trunk Lean", "trunk_lean_2d"),
        ("Thoracic", "spine_thoracic_bend_2d"),
        ("Lumbar", "spine_lumbar_bend_2d"),
    ]