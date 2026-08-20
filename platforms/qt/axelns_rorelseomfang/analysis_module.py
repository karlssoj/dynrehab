from __future__ import annotations

import math
import statistics

# ---------------------------------------------------------------------------
# Module-level state for detect_rep
# ---------------------------------------------------------------------------
_PHASE = "ready"          # "ready" | "moving"
_SIDE = "left"            # which arm to track (determined at start)
_VIS_THRESHOLD = 0.35

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_arm_elevation(pose_data: dict) -> float:
    """Return the arm elevation angle for the more visible arm (side view)."""
    l_vis = _landmark_vis(pose_data, "left_shoulder")
    r_vis = _landmark_vis(pose_data, "right_shoulder")
    if l_vis >= r_vis:
        val = pose_data.get("left_arm_elevation", 0.0) or 0.0
    else:
        val = pose_data.get("right_arm_elevation", 0.0) or 0.0
    return val


def _landmark_vis(pose_data: dict, name: str) -> float:
    kpts = pose_data.get("keypoints", {})
    lm = kpts.get(name)
    if lm is None:
        return 0.0
    return lm[3]


def _best_arm_elevation(pose_data: dict) -> float:
    """Return arm elevation for the side with best visibility."""
    l = pose_data.get("left_arm_elevation", 0.0) or 0.0
    r = pose_data.get("right_arm_elevation", 0.0) or 0.0
    l_vis = _landmark_vis(pose_data, "left_shoulder")
    r_vis = _landmark_vis(pose_data, "right_shoulder")
    if l_vis >= r_vis:
        return l
    return r


def _best_elbow_bend(pose_data: dict) -> float:
    l = pose_data.get("left_elbow_bend_2d", 0.0) or 0.0
    r = pose_data.get("right_elbow_bend_2d", 0.0) or 0.0
    l_vis = _landmark_vis(pose_data, "left_elbow")
    r_vis = _landmark_vis(pose_data, "right_elbow")
    if l_vis >= r_vis:
        return l
    return r


def _trunk_lean(pose_data: dict) -> float:
    return pose_data.get("trunk_lean_2d", 0.0) or 0.0


# ---------------------------------------------------------------------------
# Required functions
# ---------------------------------------------------------------------------

def get_instructions() -> list:
    return [
        "Stand with one side facing the camera.",
        "Perform slow, controlled arm movements forwards and upwards for flexion, "
        "then backwards for extension, keeping your arm straight throughout."
    ]


def reset_round():
    global _PHASE
    _PHASE = "ready"


def detect_rep(pose_data: dict) -> bool:
    """
    Two-phase state machine:
      ready   -> moving  : arm elevation rises above START_THRESHOLD
      moving  -> ready   : arm elevation returns below RETURN_THRESHOLD  -> count rep
    Thresholds are intentionally loose so any genuine attempt is counted.
    """
    global _PHASE

    START_THRESHOLD = 20.0    # degrees of arm elevation to begin a rep
    RETURN_THRESHOLD = 15.0   # degrees to consider the arm back at rest

    elevation = _best_arm_elevation(pose_data)

    if _PHASE == "ready":
        if elevation > START_THRESHOLD:
            _PHASE = "moving"
            return False
    elif _PHASE == "moving":
        if elevation <= RETURN_THRESHOLD:
            _PHASE = "ready"
            return True
    return False


def generate_rep_feedback(rep_data: dict) -> list:
    frames = rep_data.get("frames", [])
    rep_number = rep_data.get("rep_number", 1)

    feedback = []

    opening = f"Rep {rep_number} done."
    feedback.append(opening)

    if not frames:
        feedback.append("Keep going — maintain good form.")
        return feedback

    # ------------------------------------------------------------------
    # 1. Measure peak arm elevation (flexion and total range)
    # ------------------------------------------------------------------
    elevations = [f.get("left_arm_elevation", 0.0) or 0.0 for f in frames]
    elevations_r = [f.get("right_arm_elevation", 0.0) or 0.0 for f in frames]
    # Use whichever side has better average elevation (more visible)
    if statistics.mean(elevations) >= statistics.mean(elevations_r):
        elev_vals = elevations
    else:
        elev_vals = elevations_r

    peak_elevation = max(elev_vals) if elev_vals else 0.0

    # ------------------------------------------------------------------
    # 2. Elbow bend check (should stay close to straight)
    # ------------------------------------------------------------------
    ELBOW_BEND_LIMIT = 10.0  # degrees

    elbow_bends = []
    for f in frames:
        l_vis = _landmark_vis(f, "left_elbow")
        r_vis = _landmark_vis(f, "right_elbow")
        if l_vis >= r_vis:
            val = f.get("left_elbow_bend_2d", 0.0) or 0.0
        else:
            val = f.get("right_elbow_bend_2d", 0.0) or 0.0
        if val is not None:
            elbow_bends.append(val)

    max_elbow_bend = max(elbow_bends) if elbow_bends else 0.0

    if max_elbow_bend > ELBOW_BEND_LIMIT:
        feedback.append(
            "Try to keep your arm straighter — your elbow was bending during the movement."
        )

    # ------------------------------------------------------------------
    # 3. Trunk compensation check
    # ------------------------------------------------------------------
    TRUNK_LEAN_THRESHOLD = 10.0  # degrees

    # During flexion (elevation going up) check for backward lean
    # During extension check for forward lean
    # We approximate by looking at max trunk lean across all frames
    trunk_leans = [abs(f.get("trunk_lean_2d", 0.0) or 0.0) for f in frames]
    max_trunk_lean = max(trunk_leans) if trunk_leans else 0.0

    if max_trunk_lean > TRUNK_LEAN_THRESHOLD:
        feedback.append(
            "Keep your upper body upright — avoid leaning your trunk to compensate for the arm movement."
        )

    # ------------------------------------------------------------------
    # 4. Range of motion encouragement / guidance
    # ------------------------------------------------------------------
    if peak_elevation < 60.0:
        feedback.append(
            "Try to raise your arm a little higher when possible — aim for a fuller range of motion."
        )
    elif peak_elevation >= 150.0:
        feedback.append(
            "Excellent range of motion — your arm reached well overhead."
        )
    elif peak_elevation >= 90.0:
        feedback.append(
            "Good range of motion — your arm reached at least shoulder height."
        )
    else:
        feedback.append(
            "Good effort — continue working on lifting a little higher each rep."
        )

    return feedback


def get_session_summary(session_data: dict) -> list:
    total_reps = session_data.get("total_reps", 0)
    rounds = session_data.get("rounds", [])
    round_feedback = session_data.get("round_feedback", [])

    lines = []

    if total_reps == 0:
        lines.append("No reps were completed this session.")
        lines.append(
            "Next time, stand with your side fully facing the camera and let your arm "
            "hang relaxed at your side before you start."
        )
        lines.append(
            "Then lift your arm slowly forwards and upwards, keeping the elbow straight, "
            "and lower it back down to complete one rep."
        )
        return lines

    lines.append(
        f"Session complete — you finished {total_reps} rep{'s' if total_reps != 1 else ''} in total."
    )

    # ------------------------------------------------------------------
    # Per-round peak elevation (flexion range)
    # ------------------------------------------------------------------
    round_peaks = []
    for r in rounds:
        r_frames = r.get("frames", [])
        vals_l = [f.get("left_arm_elevation", 0.0) or 0.0 for f in r_frames]
        vals_r = [f.get("right_arm_elevation", 0.0) or 0.0 for f in r_frames]
        vals = [max(l, r_) for l, r_ in zip(vals_l, vals_r)] if vals_l else []
        vals = [v for v in vals if v > 5.0]
        round_peaks.append(max(vals) if vals else 0.0)

    if len(round_peaks) >= 2:
        first = round_peaks[0]
        last = round_peaks[-1]
        if last > first + 15.0:
            lines.append(
                f"Your range of motion improved across the session — "
                f"your arm reached higher by round {len(round_peaks)} compared to round 1."
            )
        elif first > last + 15.0:
            lines.append(
                "Your range of motion was greatest in the early rounds — "
                "it is normal to feel some fatigue toward the end, but try to maintain range throughout."
            )
        else:
            if all(p >= 90.0 for p in round_peaks):
                lines.append(
                    "You maintained a good range of motion in every round, reaching at least shoulder height."
                )
            else:
                lines.append(
                    "Your range of motion was similar across rounds — "
                    "keep working on lifting a little higher with each session."
                )

    # ------------------------------------------------------------------
    # Elbow straightness across session
    # ------------------------------------------------------------------
    all_elbow_bends = []
    for r in rounds:
        for f in r.get("frames", []):
            l_vis = _landmark_vis(f, "left_elbow")
            r_vis = _landmark_vis(f, "right_elbow")
            if l_vis >= r_vis:
                val = f.get("left_elbow_bend_2d", 0.0) or 0.0
            else:
                val = f.get("right_elbow_bend_2d", 0.0) or 0.0
            all_elbow_bends.append(val)

    elbow_issues_per_round = []
    for r in rounds:
        bends = []
        for f in r.get("frames", []):
            l_vis = _landmark_vis(f, "left_elbow")
            r_vis = _landmark_vis(f, "right_elbow")
            if l_vis >= r_vis:
                val = f.get("left_elbow_bend_2d", 0.0) or 0.0
            else:
                val = f.get("right_elbow_bend_2d", 0.0) or 0.0
            bends.append(val)
        max_bend = max(bends) if bends else 0.0
        elbow_issues_per_round.append(max_bend > 10.0)

    if all(elbow_issues_per_round) and len(elbow_issues_per_round) > 0:
        lines.append(
            "Focus on keeping your elbow fully straight throughout the movement — "
            "bending was noted in every round."
        )
    elif any(elbow_issues_per_round):
        problem_rounds = [i + 1 for i, v in enumerate(elbow_issues_per_round) if v]
        if len(problem_rounds) == 1:
            lines.append(
                f"In round {problem_rounds[0]}, your elbow bent during the movement — "
                "aim to keep it locked straight."
            )
        else:
            rounds_str = ", ".join(str(x) for x in problem_rounds)
            lines.append(
                f"Your elbow bent in rounds {rounds_str} — work on keeping it straight throughout."
            )
    else:
        lines.append(
            "Well done — you kept your arm straight throughout the session."
        )

    # ------------------------------------------------------------------
    # Trunk compensation across session
    # ------------------------------------------------------------------
    trunk_issues_per_round = []
    for r in rounds:
        leans = [abs(f.get("trunk_lean_2d", 0.0) or 0.0) for f in r.get("frames", [])]
        trunk_issues_per_round.append(max(leans) > 10.0 if leans else False)

    if all(trunk_issues_per_round) and len(trunk_issues_per_round) > 0:
        lines.append(
            "Try to keep your upper body upright throughout — trunk leaning was present in every round."
        )
    elif any(trunk_issues_per_round):
        problem_rounds = [i + 1 for i, v in enumerate(trunk_issues_per_round) if v]
        if len(problem_rounds) == 1:
            lines.append(
                f"In round {problem_rounds[0]}, some trunk leaning was detected — "
                "focus on keeping your core stable."
            )

    # ------------------------------------------------------------------
    # Check round_feedback for persistent issues
    # ------------------------------------------------------------------
    if round_feedback:
        first_rf = round_feedback[0] if round_feedback else []
        last_rf = round_feedback[-1] if round_feedback else []
        first_text = " ".join(first_rf).lower()
        last_text = " ".join(last_rf).lower()

        if "elbow" in first_text and "elbow" not in last_text and len(round_feedback) >= 2:
            lines.append(
                "Good improvement — the elbow bending issue noted early in the session was no longer present by the final round."
            )
        if "trunk" in first_text and "trunk" not in last_text and len(round_feedback) >= 2:
            lines.append(
                "Nice correction — you reduced trunk leaning as the session progressed."
            )

    lines.append(
        "Keep practising this exercise to gradually build up your shoulder range of motion."
    )

    return lines


def get_relevant_joints() -> list:
    return [
        ("Arm elevation", "left_arm_elevation"),
        ("Elbow bend", "left_elbow_bend_2d"),
        ("Trunk lean", "trunk_lean_2d"),
    ]