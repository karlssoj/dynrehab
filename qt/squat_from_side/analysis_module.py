import math
import statistics

# ---------------------------------------------------------------------------
# Module-level state for detect_rep
# ---------------------------------------------------------------------------
_phase = "ready"          # "ready" | "descending" | "ascending"
_rep_count = 0

_VIS_THRESHOLD = 0.35

# ---------------------------------------------------------------------------
# Boundary values
# ---------------------------------------------------------------------------
_KNEE_BEND_DOWN_THRESHOLD = 90   # degrees — must exceed this at bottom
_KNEE_BEND_UP_THRESHOLD = 10     # degrees — must return below this to complete rep
_TORSO_LEAN_MIN = 20
_TORSO_LEAN_MAX = 45

# detect_rep loose thresholds
_REP_BEND_START = 25    # knee bend to leave "ready"
_REP_BEND_PEAK = 50     # knee bend to confirm "ascending" phase
_REP_BEND_RETURN = 15   # knee bend to confirm rep complete

# ---------------------------------------------------------------------------
# Lower-body side-view helpers (verbatim)
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

def _heel_rise(pose_data, side="auto"):
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

# ---------------------------------------------------------------------------
# Shoulder drift helper (side view — track horizontal shoulder movement)
# ---------------------------------------------------------------------------
def _shoulder_x_range(frames: list) -> float:
    """Return horizontal range of shoulder x-position across frames (normalised 0-1)."""
    kpts_list = [f.get("keypoints", {}) for f in frames]
    xs = []
    for kpts in kpts_list:
        ls = kpts.get("left_shoulder")
        rs = kpts.get("right_shoulder")
        # Pick whichever shoulder is more visible
        if ls and rs:
            s = ls if ls[3] >= rs[3] else rs
        elif ls:
            s = ls
        elif rs:
            s = rs
        else:
            continue
        if s[3] >= _VIS_THRESHOLD:
            xs.append(s[0])
    if not xs:
        return 0.0
    return max(xs) - min(xs)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_instructions() -> list:
    return [
        "Stand sideways to the camera with your feet shoulder-width apart.",
        "Perform slow, controlled squats aiming for a 90-degree knee bend at the bottom of each rep.",
    ]


def reset_round():
    global _phase, _rep_count
    _phase = "ready"
    _rep_count = 0


def detect_rep(pose_data: dict) -> bool:
    """
    Two-phase state machine: ready → descending → ready (count rep).
    Counts ANY attempt where the knee bends past the start threshold and
    returns to near-straight — shallow attempts are counted and corrected
    in feedback, not filtered out here.
    """
    global _phase, _rep_count

    bend = _best_knee_bend(pose_data)

    if _phase == "ready":
        if bend > _REP_BEND_START:
            _phase = "descending"

    elif _phase == "descending":
        if bend < _REP_BEND_RETURN:
            _phase = "ready"
            _rep_count += 1
            return True

    return False


def generate_rep_feedback(rep_data: dict) -> list:
    rep_number = rep_data.get("rep_number", 1)
    frames = rep_data.get("frames", [])

    lines = []

    # --- Opening acknowledgment ---
    lines.append(f"Rep {rep_number} done.")

    if not frames:
        lines.append("Keep going — focus on control through the full movement.")
        return lines

    # --- 1. Depth check (highest priority after safety) ---
    peak_bend = max((_best_knee_bend(f) for f in frames), default=0.0)
    if peak_bend < _KNEE_BEND_DOWN_THRESHOLD:
        lines.append("Try to squat a little deeper — aim for a 90-degree bend at the knee.")
    else:
        lines.append("Good depth — you reached the target knee bend.")

    # --- 2. Heel rise (safety / injury risk) ---
    bent_frames = _depth_frames(frames)
    if bent_frames:
        rises = [_heel_rise(f) for f in bent_frames]
        if rises and max(rises) > 0.02:
            lines.append("Keep your heels flat on the floor as you squat down.")

    # --- 3. Knees over toes ---
    avg_shin = _avg_shin_angle(bent_frames if bent_frames else frames)
    if avg_shin > 30:
        lines.append("Push your hips back — your knees are travelling too far over your toes.")

    # --- 4. Torso lean ---
    lean_vals = [f.get("trunk_lean_2d", 0.0) or 0.0 for f in bent_frames if f.get("trunk_lean_2d") is not None]
    if lean_vals:
        avg_lean = statistics.mean(lean_vals)
        if avg_lean > _TORSO_LEAN_MAX:
            lines.append("Try to keep your chest more upright — you're leaning too far forward.")
        elif avg_lean < _TORSO_LEAN_MIN:
            lines.append("A slight forward lean is natural — let your torso incline gently as you descend.")

    # --- 5. Shoulder drift ---
    sh_range = _shoulder_x_range(frames)
    if sh_range > 0.08:
        lines.append("Try to keep your shoulders moving vertically — avoid drifting forward or backward.")

    # --- Positive fallback if nothing flagged ---
    if len(lines) == 2 and "Good depth" in lines[1]:
        lines.append("Excellent control — keep that up.")
    elif len(lines) == 1:
        lines.append("Good effort — focus on keeping smooth control throughout the movement.")

    return lines


def get_session_summary(session_data: dict) -> list:
    total_reps = session_data.get("total_reps", 0)
    rounds = session_data.get("rounds", [])
    round_feedback = session_data.get("round_feedback", [])

    lines = []

    # --- Zero reps guard ---
    if total_reps == 0:
        lines.append("No reps were completed this session.")
        lines.append("Make sure you are standing sideways to the camera and bending your knees fully as you squat down.")
        lines.append("Try to reach a 90-degree knee bend at the bottom — that is the target for a completed rep.")
        return lines

    lines.append(f"Session complete — you performed {total_reps} squat{'s' if total_reps != 1 else ''} in total.")

    # --- Depth progression across rounds ---
    round_peaks = []
    for r in rounds:
        r_frames = r.get("frames", [])
        vals = [_best_knee_bend(f) for f in r_frames if _best_knee_bend(f) > 5]
        round_peaks.append(max(vals) if vals else 0.0)

    if len(round_peaks) >= 2:
        if round_peaks[-1] > round_peaks[0] + 10:
            lines.append("Your squat depth improved as the session progressed — great work warming into the movement.")
        elif round_peaks[0] > round_peaks[-1] + 10:
            lines.append("Your squat depth was a little shallower towards the end — fatigue may have been a factor.")
        elif all(p >= _KNEE_BEND_DOWN_THRESHOLD for p in round_peaks):
            lines.append("You maintained good squat depth in every round.")
        elif all(p < _KNEE_BEND_DOWN_THRESHOLD for p in round_peaks):
            lines.append("Work on reaching a deeper squat — aim for a 90-degree knee bend at the bottom of each rep.")
    elif len(round_peaks) == 1:
        if round_peaks[0] >= _KNEE_BEND_DOWN_THRESHOLD:
            lines.append("You hit a good squat depth this session.")
        else:
            lines.append("Try to squat a little deeper next time — the target is a 90-degree knee bend.")

    # --- Persistent issues from round_feedback ---
    heel_issue_rounds = 0
    knee_forward_rounds = 0
    torso_issue_rounds = 0
    shoulder_drift_rounds = 0

    for rf in round_feedback:
        rf_text = " ".join(rf) if isinstance(rf, list) else str(rf)
        if "heel" in rf_text.lower():
            heel_issue_rounds += 1
        if "toes" in rf_text.lower() or "knees" in rf_text.lower() and "forward" in rf_text.lower():
            knee_forward_rounds += 1
        if "chest" in rf_text.lower() or "upright" in rf_text.lower() or "lean" in rf_text.lower():
            torso_issue_rounds += 1
        if "shoulder" in rf_text.lower() and "drift" in rf_text.lower():
            shoulder_drift_rounds += 1

    num_rounds = len(round_feedback)

    if num_rounds > 0:
        if heel_issue_rounds == num_rounds:
            lines.append("Your heels were lifting throughout the session — focus on ankle mobility and try to keep your feet flat on the ground.")
        elif heel_issue_rounds > 0:
            lines.append("There were moments where your heels lifted — work on keeping them grounded, especially as you squat deeper.")

        if knee_forward_rounds == num_rounds:
            lines.append("Your knees consistently travelled too far over your toes — try sitting your hips back more as you descend.")
        elif knee_forward_rounds > 0:
            lines.append("In some reps your knees drifted too far forward — keep pushing your hips back to maintain shin alignment.")

        if torso_issue_rounds == num_rounds:
            lines.append("Your torso lean was outside the ideal range in every round — focus on keeping your chest facing forward throughout the squat.")
        elif torso_issue_rounds > 0:
            lines.append("Your torso lean was off in some rounds — aim for a slight, controlled forward lean rather than hunching over.")

        if shoulder_drift_rounds > 0:
            lines.append("Try to keep your shoulders moving straight up and down without drifting forward or backward.")

    # --- Positive close ---
    if len(lines) < 3:
        lines.append("Good session — keep practising these squats to build strength and control.")

    return lines


def get_relevant_joints() -> list:
    return [
        ("Left Knee", "left_knee_bend_2d"),
        ("Right Knee", "right_knee_bend_2d"),
        ("Torso Lean", "trunk_lean_2d"),
    ]