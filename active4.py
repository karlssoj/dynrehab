import math
import statistics

# ---------------------------------------------------------------------------
# Module-level state for detect_rep
# ---------------------------------------------------------------------------
_phase = "ready"
_start_hip_y = 0.0
_peak_hip_y = 0.0

_VIS_THRESHOLD = 0.3

# Loose thresholds for rep-attempt detection (NOT quality thresholds)
_KNEE_BEND_START = 20.0
_KNEE_BEND_RETURN = 10.0


def get_instructions() -> list:
    return [
        "Stand with either side of your body facing the camera.",
        "Perform slow squats, bending your knees to about sixty degrees while keeping your back naturally aligned."
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
    """Pick the more-visible leg once for the whole rep."""
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
    """abs(knee_x - ankle_x) / shin_length for a fixed side. ~0.7 = knee over toes."""
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


# ---------------------------------------------------------------------------
# detect_rep
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
        if knee_bend > _KNEE_BEND_START:
            _phase = "moving"
            _start_hip_y = hip_y
            _peak_hip_y = hip_y
        return False

    if _phase == "moving":
        if hip_y > _peak_hip_y:
            _peak_hip_y = hip_y
        if knee_bend < _KNEE_BEND_RETURN:
            _phase = "ready"
            if _start_hip_y <= 0.0:
                # no reliable hip reading — allow the rep
                return True
            if _peak_hip_y > _start_hip_y + 0.03:
                return True
            return False
        return False

    return False


def reset_round():
    global _phase, _start_hip_y, _peak_hip_y
    _phase = "ready"
    _start_hip_y = 0.0
    _peak_hip_y = 0.0


# ---------------------------------------------------------------------------
# classify_movement
# ---------------------------------------------------------------------------
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
                return "Squat down by lowering your hips evenly, bending both knees together."
    return None


# ---------------------------------------------------------------------------
# generate_rep_feedback
# ---------------------------------------------------------------------------
def generate_rep_feedback(rep_data: dict) -> list:
    frames = rep_data.get("frames", [])
    rep_number = rep_data.get("rep_number", 0)

    feedback = [f"Rep {rep_number} done."]

    if not frames:
        return feedback

    bent_frames = _depth_frames(frames)
    check_frames = bent_frames if bent_frames else frames

    # --- Depth: always reported ---
    max_knee_bend = max((_best_knee_bend(f) for f in frames), default=0.0)
    if max_knee_bend >= 60.0:
        feedback.append("Good squat depth — you reached the target knee bend.")
    else:
        feedback.append("Try to squat a little deeper to reach the target knee bend.")

    # --- Trunk lean: always reported ---
    trunk_vals = [f.get("trunk_lean_2d", 0.0) for f in check_frames if f.get("trunk_lean_2d", 0.0) > 0]
    avg_trunk = statistics.mean(trunk_vals) if trunk_vals else 0.0
    if trunk_vals:
        if 10.0 <= avg_trunk <= 30.0:
            feedback.append("Your upper body lean was well controlled during the squat.")
        elif avg_trunk < 10.0:
            feedback.append("You stayed too upright — allow your upper body to lean forward a little more as you squat.")
        else:
            feedback.append("You leaned too far forward — try to keep your chest a bit more upright.")
    else:
        feedback.append("Not enough data to assess your upper body lean this rep.")

    # --- Thoracic spine bend (upper back): always reported ---
    thoracic_vals = [f.get("spine_thoracic_bend_2d") for f in frames if f.get("spine_thoracic_bend_2d") is not None]
    if thoracic_vals:
        peak_thoracic = max(thoracic_vals)
        if peak_thoracic > 25.0:
            feedback.append("Your upper back rounded too much — avoid excessive rounding through your shoulders.")
        elif peak_thoracic < 10.0:
            feedback.append("Try to keep a slight natural curve in your upper back rather than staying too flat.")
        else:
            feedback.append("Your upper back maintained a natural curve.")
    else:
        feedback.append("Not enough data to assess your upper back curve this rep.")

    # --- Lumbar spine bend (lower back): always reported ---
    lumbar_vals = [f.get("spine_lumbar_bend_2d") for f in frames if f.get("spine_lumbar_bend_2d") is not None]
    if lumbar_vals:
        peak_lumbar = min(lumbar_vals)
        if peak_lumbar < -25.0:
            feedback.append("Your lower back arched too much — engage your core to avoid excessive arching.")
        elif peak_lumbar > -10.0:
            feedback.append("Try to maintain a bit more of your lower back's natural inward curve.")
        else:
            feedback.append("Your lower back maintained a natural curve.")
    else:
        feedback.append("Not enough data to assess your lower back curve this rep.")

    # --- Knees over toes (only flagged when a problem exists) ---
    side = _leg_side_for_forward_ratio(check_frames)
    if side:
        ratios = [r for r in (_knee_forward_ratio(f, side) for f in check_frames) if r > 0.05]
        if ratios:
            robust_peak = _second_highest(ratios)
            if robust_peak > 0.35:
                feedback.append("Your knees are travelling too far over your toes — push your hips back a little.")

    # --- Stance check (only flagged when a problem exists) ---
    ankle_spans = [f.get("ankle_lateral_span", 0.0) for f in frames]
    avg_ankle_span = statistics.mean(ankle_spans) if ankle_spans else 0.0
    if avg_ankle_span > 0.20:
        feedback.append("Make sure both feet are side by side — one foot appears to be in front of the other.")

    return feedback


# ---------------------------------------------------------------------------
# get_session_summary
# ---------------------------------------------------------------------------
def get_session_summary(session_data: dict) -> list:
    total_reps = session_data.get("total_reps", 0)
    lines = []

    if total_reps == 0:
        lines.append("No reps were completed this session.")
        lines.append("Next time, stand side-on to the camera and perform slow squats, bending your knees to about sixty degrees.")
        lines.append("Focus on keeping your back naturally aligned — avoid rounding your upper back or over-arching your lower back.")
        return lines

    rounds = session_data.get("rounds", [])
    round_feedback = session_data.get("round_feedback", [])

    lines.append(f"You completed {total_reps} reps this session.")

    # Depth progression across rounds using knee bend (primary detection signal)
    round_peaks = []
    for r in rounds:
        vals = [_best_knee_bend(f) for f in r.get("frames", []) if _best_knee_bend(f) > 5]
        round_peaks.append(max(vals) if vals else 0.0)

    if len(round_peaks) >= 2:
        if round_peaks[-1] > round_peaks[0] + 10:
            lines.append(f"Your squat depth improved from round 1 to round {len(round_peaks)}.")
        elif round_peaks[0] > round_peaks[-1] + 10:
            lines.append(f"Your squat depth was better in round 1 than by round {len(round_peaks)}.")
        elif all(p >= 60.0 for p in round_peaks):
            lines.append("You maintained good squat depth across all rounds.")

    if round_peaks and max(round_peaks) >= 60.0:
        lines.append("You reached the target knee bend at least once during the session.")
    elif round_peaks:
        lines.append("You did not consistently reach the target knee bend depth — aim to squat a bit deeper.")

    # Use round_feedback text as secondary evidence for persistent/improving issues
    def _has_issue(fb_list, keyword):
        for s in fb_list:
            if keyword in s:
                return True
        return False

    if round_feedback:
        first_fb = round_feedback[0]
        last_fb = round_feedback[-1]

        if _has_issue(first_fb, "rounded too much") and not _has_issue(last_fb, "rounded too much"):
            lines.append("Your upper back rounding improved by the later rounds.")
        elif all(_has_issue(fb, "rounded too much") for fb in round_feedback):
            lines.append("Excessive upper back rounding was a persistent issue throughout the session.")

        if _has_issue(first_fb, "arched too much") and not _has_issue(last_fb, "arched too much"):
            lines.append("Your lower back arching improved by the later rounds.")
        elif all(_has_issue(fb, "arched too much") for fb in round_feedback):
            lines.append("Excessive lower back arching was a persistent issue throughout the session.")

        if all(_has_issue(fb, "leaned too far forward") for fb in round_feedback):
            lines.append("Leaning too far forward with your upper body was a persistent issue throughout the session.")

    lines.append("Keep working on a controlled squat depth with a naturally aligned back.")

    return lines


def get_relevant_joints() -> list:
    return [
        ("Knee Flex", "left_knee_bend_2d"),
        ("Trunk Lean", "trunk_lean_2d"),
        ("Hip Flex", "left_hip_bend_2d"),
        ("Thoracic", "spine_thoracic_bend_2d"),
        ("Lumbar", "spine_lumbar_bend_2d"),
    ]