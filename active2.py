import math
import statistics

_phase = "ready"
_start_hip_y = 0.0
_peak_hip_y = 0.0


def get_instructions() -> list:
    return [
        "Stand with either side facing the camera, feet flat on the floor.",
        "Perform slow squats, bending your knees to about 60 degrees while keeping your back naturally aligned."
    ]


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
    """Second-largest value; falls back to single value or 0.0 when empty."""
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
        if knee_bend > 34.0:
            _phase = "moving"
            _start_hip_y = hip_y
            _peak_hip_y = hip_y
        return False
    else:
        if hip_y > _peak_hip_y:
            _peak_hip_y = hip_y
        if knee_bend < 20.0:
            _phase = "ready"
            if _start_hip_y > 0.0 and _peak_hip_y > 0.0:
                if _peak_hip_y > _start_hip_y + 0.03:
                    return True
                return False
            return True
        return False


def reset_round():
    global _phase, _start_hip_y, _peak_hip_y
    _phase = "ready"
    _start_hip_y = 0.0
    _peak_hip_y = 0.0


def classify_movement(frames: list):
    if not frames:
        return None
    bends = [_best_knee_bend(f) for f in frames if _best_knee_bend(f) > 1.0]
    bend_range = (max(bends) - min(bends)) if bends else 0.0
    if bend_range < 10.0:
        return "Start the exercise by bending your knees to squat down."

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
                return "Squat down by lowering your hips, not just bending your knees."
    return None


def generate_rep_feedback(rep_data: dict) -> list:
    frames = rep_data.get("frames", [])
    rep_number = rep_data.get("rep_number", 0)
    feedback = []
    feedback.append(f"Rep {rep_number} done.")

    if not frames:
        return feedback

    ankle_spans = [f.get("ankle_lateral_span", 0.0) for f in frames]
    avg_ankle_span = statistics.mean(ankle_spans) if ankle_spans else 0.0
    if avg_ankle_span > 0.20:
        feedback.append("Make sure both feet are side by side — one foot appears to be in front of the other.")

    bent_frames = _depth_frames(frames)
    check_frames = bent_frames if bent_frames else frames

    # Depth — always reported
    bends = [_best_knee_bend(f) for f in frames if _best_knee_bend(f) > 1.0]
    peak_bend = max(bends) if bends else 0.0
    if peak_bend >= 60.0:
        feedback.append("Good squat depth — you reached at least a 60 degree knee bend.")
    else:
        feedback.append("Try to squat deeper — aim for at least a 60 degree knee bend.")

    # Trunk lean — always reported
    leans = [f.get("trunk_lean_2d", 0.0) or 0.0 for f in check_frames]
    peak_lean = max(leans) if leans else 0.0
    if peak_lean < 10.0:
        feedback.append("You're staying too upright — let your upper body lean forward slightly for balance.")
    elif peak_lean > 30.0:
        feedback.append("You're leaning too far forward with your upper body — keep your chest a bit more upright.")
    else:
        feedback.append("Your upper body lean looks well balanced.")

    # Knees over toes
    side = _leg_side_for_forward_ratio(check_frames)
    if side:
        ratios = [r for r in (_knee_forward_ratio(f, side) for f in check_frames) if r > 0.05]
        if ratios:
            robust_peak = _second_highest(ratios)
            if robust_peak > 0.35:
                feedback.append("Your knees are travelling too far over your toes — push your hips back as you squat.")

    # Spine curvature vs baseline — always reported when baseline available
    baseline = rep_data.get("spine_baseline")
    if baseline:
        changes = _spine_zone_changes(frames, baseline)
        thoracic = changes.get("thoracic")
        lumbar = changes.get("lumbar")
        reported_back = False
        if thoracic is not None and thoracic["delta"] > 0.15:
            feedback.append("Your upper back is rounding too much — try to keep a more natural curve through your shoulders.")
            reported_back = True
        if lumbar is not None and lumbar["delta"] < -0.15:
            feedback.append("Your lower back is arching too much — engage your core to keep it more neutral.")
            reported_back = True
        if not reported_back:
            feedback.append("Your back maintained a natural curve during the squat.")

    return feedback


def get_session_summary(session_data: dict) -> list:
    lines = []
    total_reps = session_data.get("total_reps", 0)
    rounds = session_data.get("rounds", [])
    round_feedback = session_data.get("round_feedback", [])

    if total_reps == 0:
        lines.append("No reps were completed this session.")
        lines.append("Next time, stand side-on to the camera and perform slow squats, "
                      "bending your knees to at least 60 degrees while keeping a natural back posture.")
        return lines

    lines.append(f"You completed {total_reps} squat repetitions this session.")

    round_peaks = []
    for r in rounds:
        vals = [max(f.get("left_knee_bend_2d", 0.0) or 0.0, f.get("right_knee_bend_2d", 0.0) or 0.0)
                for f in r.get("frames", [])]
        vals = [v for v in vals if v > 5.0]
        round_peaks.append(max(vals) if vals else 0.0)

    if len(round_peaks) >= 2:
        if round_peaks[-1] > round_peaks[0] + 10:
            lines.append(f"Your squat depth improved from round 1 to round {len(round_peaks)}.")
        elif round_peaks[0] > round_peaks[-1] + 10:
            lines.append(f"Your squat depth was better in round 1 than in round {len(round_peaks)}.")
        elif all(p >= 60.0 for p in round_peaks):
            lines.append("You maintained good squat depth across all rounds.")
    elif round_peaks and round_peaks[0] >= 60.0:
        lines.append("You reached a good squat depth.")

    def _has(fb_list, keyword):
        for s in fb_list:
            if keyword in s:
                return True
        return False

    if round_feedback:
        first = round_feedback[0]
        last = round_feedback[-1]
        if _has(first, "leaning too far forward") and not _has(last, "leaning too far forward"):
            lines.append("Your forward lean improved by the later rounds.")
        if _has(first, "too upright") and not _has(last, "too upright"):
            lines.append("Your upper body positioning improved as the session went on.")

        persistent_issues = []
        if all(_has(fb, "rounding") for fb in round_feedback):
            persistent_issues.append("rounding your upper back")
        if all(_has(fb, "arching") for fb in round_feedback):
            persistent_issues.append("arching your lower back")
        if all(_has(fb, "over your toes") for fb in round_feedback):
            persistent_issues.append("letting your knees travel too far forward")
        if persistent_issues:
            lines.append("You consistently showed a tendency toward " + ", ".join(persistent_issues) +
                          " — focus on this next session.")

    deltas = []
    for r in rounds:
        baseline = r.get("spine_baseline")
        if baseline:
            changes = _spine_zone_changes(r.get("frames", []), baseline)
            for zone in ("thoracic", "lumbar"):
                info = changes.get(zone)
                if info is not None:
                    deltas.append(info["delta"])
    if len(deltas) >= 2:
        spread = max(deltas) - min(deltas)
        if spread > 0.3:
            lines.append("Your back position varied quite a bit between reps — work on keeping a more consistent posture.")

    if len(lines) == 1:
        lines.append("Good effort completing the squats — keep working on consistent depth and posture.")

    return lines


def get_relevant_joints() -> list:
    return [
        ("Knee Flex", "left_knee_bend_2d"),
        ("Trunk Lean", "trunk_lean_2d"),
        ("Hip Flex", "left_hip_bend_2d"),
    ]