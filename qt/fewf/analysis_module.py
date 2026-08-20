import math
import statistics

# ---------------------------------------------------------------------------
# Module-level state for detect_rep
# ---------------------------------------------------------------------------
_phase = "ready"          # "ready" | "moving"
_VIS_THRESHOLD = 0.35


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _right_elbow_bend(pose_data: dict) -> float:
    val = pose_data.get("right_elbow_bend_2d", 0.0) or 0.0
    return float(val)


def _right_arm_elevation(pose_data: dict) -> float:
    val = pose_data.get("right_arm_elevation", 0.0) or 0.0
    return float(val)


# ---------------------------------------------------------------------------
# Required API
# ---------------------------------------------------------------------------

def get_instructions() -> list:
    return [
        "Stand facing the camera with your right arm at your side.",
        "Curl your right arm up and raise it, then lower it back down."
    ]


def detect_rep(pose_data: dict) -> bool:
    """Two-phase state machine: ready → moving → ready."""
    global _phase

    elbow_bend = _right_elbow_bend(pose_data)
    arm_elev = _right_arm_elevation(pose_data)

    # Movement started: elbow bend > ~51° (12 + 30% of 130 ≈ 51) or arm elevation > ~95° (71 + 30% of 80)
    MOVE_BEND_START = 51.0
    MOVE_ELEV_START = 95.0

    # Return to near-rest: elbow bend < 35° and arm elevation < 88°
    REST_BEND = 35.0
    REST_ELEV = 88.0

    if _phase == "ready":
        if elbow_bend > MOVE_BEND_START or arm_elev > MOVE_ELEV_START:
            _phase = "moving"
    elif _phase == "moving":
        if elbow_bend < REST_BEND and arm_elev < REST_ELEV:
            _phase = "ready"
            return True

    return False


def reset_round():
    global _phase
    _phase = "ready"


def generate_rep_feedback(rep_data: dict) -> list:
    frames = rep_data.get("frames", [])
    rep_number = rep_data.get("rep_number", 1)

    feedback = []
    feedback.append(f"Rep {rep_number} done.")

    if not frames:
        return feedback

    # --- Elbow bend (curl depth) ---
    elbow_bends = [_right_elbow_bend(f) for f in frames if _right_elbow_bend(f) > 5.0]
    peak_bend = max(elbow_bends) if elbow_bends else 0.0

    # Full rep threshold: 12 + 60% of 130 ≈ 90°
    FULL_BEND_THRESHOLD = 90.0
    # Movement started threshold: ~51°
    MOVE_BEND_THRESHOLD = 51.0

    if peak_bend < MOVE_BEND_THRESHOLD:
        feedback.append("Try to bend your elbow more as you curl — aim to bring your hand up toward your shoulder.")
    elif peak_bend < FULL_BEND_THRESHOLD:
        feedback.append("Good effort — try to curl a little higher and bring your hand closer to your shoulder.")

    # --- Arm elevation ---
    elev_vals = [_right_arm_elevation(f) for f in frames if _right_arm_elevation(f) > 5.0]
    peak_elev = max(elev_vals) if elev_vals else 0.0

    # Full elevation threshold: 71 + 60% of 80 ≈ 119°
    FULL_ELEV_THRESHOLD = 119.0
    MOVE_ELEV_THRESHOLD = 95.0

    if peak_elev < MOVE_ELEV_THRESHOLD:
        feedback.append("Make sure you are also raising your arm as part of the movement.")
    elif peak_elev < FULL_ELEV_THRESHOLD:
        feedback.append("Try to raise your arm a little higher during the movement.")

    if len(feedback) == 1:
        if peak_bend >= FULL_BEND_THRESHOLD and peak_elev >= FULL_ELEV_THRESHOLD:
            feedback.append("Excellent range — great curl and arm raise!")
        else:
            feedback.append("Good work, keep it up!")

    return feedback


def get_session_summary(session_data: dict) -> list:
    total_reps = session_data.get("total_reps", 0)
    rounds = session_data.get("rounds", [])
    round_feedback = session_data.get("round_feedback", [])

    lines = []

    if total_reps == 0:
        lines.append("No reps were completed this session.")
        lines.append("Try to bend your elbow and raise your arm more clearly so the movement can be detected.")
        lines.append("Stand facing the camera and make sure your right arm and elbow are visible throughout.")
        return lines

    lines.append(f"Great effort today — you completed {total_reps} rep{'s' if total_reps != 1 else ''} in total.")

    # Per-round peak elbow bend
    round_peaks = []
    for r in rounds:
        r_frames = r.get("frames", [])
        vals = [_right_elbow_bend(f) for f in r_frames if _right_elbow_bend(f) > 5.0]
        round_peaks.append(max(vals) if vals else 0.0)

    FULL_BEND_THRESHOLD = 90.0

    if len(round_peaks) >= 2:
        if round_peaks[-1] > round_peaks[0] + 15:
            lines.append(f"Your elbow curl range improved across the session — well done on building into it.")
        elif round_peaks[0] > round_peaks[-1] + 15:
            lines.append(f"Your curl range was stronger in the early rounds — try to maintain that depth as you fatigue.")
        elif all(p >= FULL_BEND_THRESHOLD for p in round_peaks):
            lines.append("You maintained a good curl depth across all rounds.")

    if round_peaks:
        overall_peak = max(round_peaks)
        if overall_peak >= FULL_BEND_THRESHOLD:
            lines.append("You reached a strong elbow bend during your best reps — keep aiming for that height.")
        elif overall_peak >= 51.0:
            lines.append("You showed some elbow curl movement — focus on bending your elbow further to get more out of each rep.")
        else:
            lines.append("The elbow bend was limited this session — try to curl your hand up higher toward your shoulder.")

    # Check round feedback for recurring issues
    elbow_issue_rounds = []
    elev_issue_rounds = []
    for i, rf in enumerate(round_feedback):
        round_text = " ".join(rf).lower()
        if "elbow" in round_text or "curl" in round_text or "hand" in round_text:
            elbow_issue_rounds.append(i + 1)
        if "raise" in round_text or "arm" in round_text:
            elev_issue_rounds.append(i + 1)

    if elbow_issue_rounds and len(elbow_issue_rounds) == len(round_feedback) and len(round_feedback) > 1:
        lines.append("Elbow curl depth was flagged in every round — this is the main area to work on next session.")
    elif elbow_issue_rounds and elbow_issue_rounds[-1] == len(round_feedback):
        lines.append(f"Elbow curl depth became an issue toward the end of the session — try to keep your range consistent throughout.")
    elif elbow_issue_rounds and elbow_issue_rounds[-1] < len(round_feedback):
        lines.append(f"Elbow curl improved by round {round_feedback.__len__()} — good correction.")

    if elev_issue_rounds and len(elev_issue_rounds) == len(round_feedback) and len(round_feedback) > 1:
        lines.append("Remember to lift your arm throughout the movement — this was noted in every round.")

    lines.append("Well done for completing the session. Keep practising and your range will continue to improve.")

    return lines


def get_relevant_joints() -> list:
    return [
        ("R Elbow Bend", "right_elbow_bend_2d"),
        ("R Arm Elev", "right_arm_elevation"),
        ("R Shoulder", "right_shoulder_angle"),
        ("R Wrist", "right_wrist_angle"),
    ]