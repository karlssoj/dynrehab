from __future__ import annotations

import math
import statistics

# ---------------------------------------------------------------------------
# Module-level state for detect_rep
# ---------------------------------------------------------------------------
_phase = "ready"          # "ready" | "moving"
_rep_count = 0

_VIS_THRESHOLD = 0.35

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _best_knee_bend(pose_data: dict) -> float:
    l = pose_data.get("left_knee_bend_2d", 0.0) or 0.0
    r = pose_data.get("right_knee_bend_2d", 0.0) or 0.0
    return max(l, r)


def _depth_frames(frames: list) -> list:
    return [f for f in frames if _best_knee_bend(f) > 20.0]


def _avg_shin_angle(frames: list) -> float:
    vals = [max(f.get("left_shin_angle", 0.0) or 0.0, f.get("right_shin_angle", 0.0) or 0.0)
            for f in frames]
    vals = [v for v in vals if v > 1.0]
    return statistics.mean(vals) if vals else 0.0


def _heel_rise(pose_data: dict, side: str = "auto") -> float:
    kpts = pose_data.get("keypoints", {})
    if side == "auto":
        lh = kpts.get("left_heel")
        rh = kpts.get("right_heel")
        lf = kpts.get("left_foot_index")
        rf = kpts.get("right_foot_index")
        l_vis = min(lh[3] if lh else 0, lf[3] if lf else 0)
        r_vis = min(rh[3] if rh else 0, rf[3] if rf else 0)
        side = "left" if l_vis >= r_vis else "right"
    heel = kpts.get(f"{side}_heel")
    foot_index = kpts.get(f"{side}_foot_index")
    if not heel or not foot_index:
        return 0.0
    if min(heel[3], foot_index[3]) < 0.20:
        return 0.0
    return foot_index[1] - heel[1]


def _trunk_lean(frames: list) -> float:
    """Return mean trunk lean over the supplied frames (degrees from vertical)."""
    vals = [f.get("trunk_lean_2d", 0.0) or 0.0 for f in frames]
    vals = [v for v in vals if v >= 0.0]
    return statistics.mean(vals) if vals else 0.0


def _max_knee_bend(frames: list) -> float:
    """Return the maximum knee bend (0=straight) seen across the supplied frames."""
    bends = [_best_knee_bend(f) for f in frames]
    return max(bends) if bends else 0.0


def _shoulder_drift(frames: list) -> float:
    """
    Estimate horizontal shoulder drift from vertical during the squat.
    Returns the range of the shoulder x-coordinate across the frames (in
    normalised image units).  Large values indicate the torso is lurching
    forward or backward rather than moving vertically.
    """
    kpts_list = [f.get("keypoints", {}) for f in frames]
    xs = []
    for kpts in kpts_list:
        ls = kpts.get("left_shoulder")
        rs = kpts.get("right_shoulder")
        # prefer the shoulder with better visibility
        if ls and rs:
            chosen = ls if ls[3] >= rs[3] else rs
        elif ls:
            chosen = ls
        elif rs:
            chosen = rs
        else:
            continue
        if chosen[3] >= _VIS_THRESHOLD:
            xs.append(chosen[0])
    if len(xs) < 2:
        return 0.0
    return max(xs) - min(xs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_instructions() -> list[str]:
    return [
        "Stand sideways to the camera with your feet shoulder-width apart.",
        "Perform a controlled squat, bending your knees to about ninety degrees, then return to standing.",
    ]


def reset_round():
    global _phase, _rep_count
    _phase = "ready"
    _rep_count = 0


def detect_rep(pose_data: dict) -> bool:
    """
    Two-phase state machine:
      ready   → moving  when knee bend exceeds ~30 degrees
      moving  → ready   when knee bend returns to under ~15 degrees  → count rep
    """
    global _phase, _rep_count

    bend = _best_knee_bend(pose_data)

    if _phase == "ready":
        if bend > 30.0:
            _phase = "moving"
    elif _phase == "moving":
        if bend < 15.0:
            _phase = "ready"
            _rep_count += 1
            return True

    return False


def generate_rep_feedback(rep_data: dict) -> list[str]:
    rep_number = rep_data.get("rep_number", 1)
    frames = rep_data.get("frames", [])

    lines = [f"Rep {rep_number} done."]

    if not frames:
        lines.append("Keep going — maintain a controlled pace.")
        return lines

    bent_frames = _depth_frames(frames)

    # --- 1. Heel rise (highest injury-risk priority) ---
    if bent_frames:
        rises = [_heel_rise(f) for f in bent_frames]
        if rises and max(rises) > 0.02:
            lines.append(
                "Your heels lifted off the floor — focus on keeping them flat as you squat down."
            )

    # --- 2. Knees travelling too far forward ---
    avg_shin = _avg_shin_angle(bent_frames if bent_frames else frames)
    if avg_shin > 30.0:
        lines.append(
            "Your knees are travelling too far over your toes — push your hips back and keep your shins more vertical."
        )

    # --- 3. Squat depth ---
    max_bend = _max_knee_bend(frames)
    if max_bend < 60.0:
        lines.append(
            "Try to squat a little deeper — aim to get your thighs closer to parallel with the floor."
        )
    elif max_bend >= 90.0:
        lines.append("Great depth — you reached a full ninety-degree squat.")

    # --- 4. Torso lean ---
    mean_lean = _trunk_lean(frames)
    if mean_lean > 45.0:
        lines.append(
            "You are leaning too far forward with your torso — try to keep your chest more upright throughout the movement."
        )
    elif mean_lean < 5.0 and max_bend > 40.0:
        # Suspiciously upright — may indicate compensatory movement but is not harmful; skip.
        pass

    # --- 5. Shoulder vertical path ---
    drift = _shoulder_drift(frames)
    if drift > 0.15:
        lines.append(
            "Your shoulders are drifting forward — try to move more vertically and keep your weight centred."
        )

    # --- Encouragement if no issues found ---
    if len(lines) == 1:
        lines.append("Excellent form — keep up the controlled movement.")

    return lines


def get_session_summary(session_data: dict) -> list[str]:
    total_reps = session_data.get("total_reps", 0)
    rounds = session_data.get("rounds", [])
    round_feedback = session_data.get("round_feedback", [])

    lines = []

    # --- Zero reps edge case ---
    if total_reps == 0:
        lines.append("No reps were completed this session.")
        lines.append(
            "Next time, stand sideways to the camera, bend your knees as if sitting back into a chair, "
            "and aim to reach a ninety-degree knee bend before returning to standing."
        )
        return lines

    lines.append(f"You completed {total_reps} squat{'s' if total_reps != 1 else ''} in total — well done.")

    # --- Depth progression across rounds ---
    round_peaks = []
    for r in rounds:
        r_frames = r.get("frames", [])
        bends = [_best_knee_bend(f) for f in r_frames if _best_knee_bend(f) > 5]
        round_peaks.append(max(bends) if bends else 0.0)

    _DEPTH_THRESHOLD = 90.0  # bend degrees for a full squat

    if len(round_peaks) >= 2:
        first = round_peaks[0]
        last = round_peaks[-1]
        if last > first + 10.0:
            lines.append("Your squat depth improved as the session progressed — great work pushing deeper.")
        elif first > last + 10.0:
            lines.append(
                f"Your depth was best in round 1 and decreased toward the end — this may be fatigue, "
                "so focus on maintaining form as you tire."
            )
        elif all(p >= _DEPTH_THRESHOLD for p in round_peaks):
            lines.append("You maintained excellent squat depth in every round.")
        elif all(p < _DEPTH_THRESHOLD for p in round_peaks):
            lines.append(
                "Your squat depth was a little shallow across all rounds — work on sitting back further to reach the full ninety-degree bend."
            )

    # --- Persistent form issues detected from round feedback ---
    heel_issue_rounds = []
    knee_forward_rounds = []
    lean_issue_rounds = []

    for i, fb_list in enumerate(round_feedback):
        round_label = f"round {i + 1}"
        if not isinstance(fb_list, list):
            continue
        combined = " ".join(fb_list).lower()
        if "heel" in combined:
            heel_issue_rounds.append(round_label)
        if "toes" in combined or "shin" in combined or "knees are travelling" in combined:
            knee_forward_rounds.append(round_label)
        if "leaning" in combined or "forward" in combined and "torso" in combined:
            lean_issue_rounds.append(round_label)

    if heel_issue_rounds:
        if len(heel_issue_rounds) == len(round_feedback) and len(round_feedback) > 0:
            lines.append(
                "Heel rise was noted in every round — work on ankle mobility stretches between sessions to help keep your heels flat."
            )
        else:
            rounds_str = " and ".join(heel_issue_rounds)
            lines.append(
                f"Heel lifting was detected in {rounds_str} — keep practising ankle flexibility to address this."
            )

    if knee_forward_rounds:
        if len(knee_forward_rounds) == len(round_feedback) and len(round_feedback) > 0:
            lines.append(
                "Your knees consistently travelled too far over your toes — focus on pushing your hips back at the start of each squat."
            )
        else:
            rounds_str = " and ".join(knee_forward_rounds)
            lines.append(
                f"Knee forward travel was an issue in {rounds_str} — practise sitting back into the squat."
            )

    if lean_issue_rounds:
        if len(lean_issue_rounds) == len(round_feedback) and len(round_feedback) > 0:
            lines.append(
                "Excessive forward lean appeared throughout the session — working on hip and ankle flexibility should help you stay more upright."
            )
        else:
            rounds_str = " and ".join(lean_issue_rounds)
            lines.append(
                f"Torso lean was too far forward in {rounds_str} — try to keep your chest lifted as you descend."
            )

    # --- Positive close if no major persistent issues ---
    if not heel_issue_rounds and not knee_forward_rounds and not lean_issue_rounds:
        lines.append("Your form was consistent across the session — keep up the excellent work.")

    return lines


def get_relevant_joints() -> list:
    return [
        ("L Knee Bend", "left_knee_bend_2d"),
        ("R Knee Bend", "right_knee_bend_2d"),
        ("Torso Lean", "trunk_lean_2d"),
    ]