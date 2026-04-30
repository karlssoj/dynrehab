import math
import statistics

# ---------------------------------------------------------------------------
# Module-level state for detect_rep
# ---------------------------------------------------------------------------
_phase = "ready"          # "ready" | "descending" | "bottom"
_rep_count = 0

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_VIS_THRESHOLD = 0.35

# Loose thresholds for detect_rep (count any recognisable attempt)
_REP_START_BEND = 20.0    # knee bend to register start of descent
_REP_BOTTOM_BEND = 45.0   # knee bend to register bottom reached
_REP_UP_BEND = 15.0       # knee bend to register return to standing

# Quality boundary values (from spec)
_DEPTH_THRESHOLD = 90.0          # knee bend for adequate squat depth
_TORSO_LEAN_MIN = 20.0
_TORSO_LEAN_MAX = 45.0
_SHIN_ANGLE_MAX = 30.0           # avg shin angle above this = knees too far forward
_HEEL_RISE_THRESHOLD = 0.02


# ---------------------------------------------------------------------------
# Helpers (verbatim from spec)
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


def _get_max_knee_bend(frames: list) -> float:
    """Return the maximum knee bend across all frames."""
    best = 0.0
    for f in frames:
        best = max(best, _best_knee_bend(f))
    return best


def _get_torso_lean_at_bottom(frames: list) -> float:
    """Return the trunk lean when the knee is most bent (bottom of squat)."""
    max_bend = 0.0
    torso_at_bottom = 0.0
    for f in frames:
        bend = _best_knee_bend(f)
        if bend > max_bend:
            max_bend = bend
            torso_at_bottom = f.get("trunk_lean_2d", 0.0) or 0.0
    return torso_at_bottom


def _get_shoulder_horizontal_drift(frames: list) -> float:
    """
    Measure how much the mid-shoulder x-coordinate drifts horizontally
    across the session (max - min), as a fraction of frame width.
    """
    vals = []
    for f in frames:
        kpts = f.get("keypoints", {})
        ls = kpts.get("left_shoulder")
        rs = kpts.get("right_shoulder")
        if ls and rs and min(ls[3], rs[3]) >= _VIS_THRESHOLD:
            mid_x = (ls[0] + rs[0]) / 2.0
            vals.append(mid_x)
        elif ls and ls[3] >= _VIS_THRESHOLD:
            vals.append(ls[0])
        elif rs and rs[3] >= _VIS_THRESHOLD:
            vals.append(rs[0])
    if not vals:
        return 0.0
    return max(vals) - min(vals)


# ---------------------------------------------------------------------------
# Required API
# ---------------------------------------------------------------------------

def get_instructions() -> list[str]:
    return [
        "Stand sideways to the camera with your feet shoulder-width apart.",
        "Perform slow, controlled squats, aiming for a ninety-degree knee bend at the bottom of each rep.",
    ]


def reset_round():
    global _phase, _rep_count
    _phase = "ready"
    _rep_count = 0


def detect_rep(pose_data: dict) -> bool:
    global _phase, _rep_count

    bend = _best_knee_bend(pose_data)

    if _phase == "ready":
        if bend >= _REP_START_BEND:
            _phase = "descending"
        return False

    if _phase == "descending":
        if bend >= _REP_BOTTOM_BEND:
            _phase = "bottom"
        return False

    if _phase == "bottom":
        if bend <= _REP_UP_BEND:
            _phase = "ready"
            _rep_count += 1
            return True

    return False


def generate_rep_feedback(rep_data: dict) -> list[str]:
    rep_number = rep_data.get("rep_number", 0)
    frames = rep_data.get("frames", [])

    lines = []
    lines.append(f"Rep {rep_number} done.")

    if not frames:
        return lines

    # ---- 1. Depth check ----
    max_bend = _get_max_knee_bend(frames)
    if max_bend < _DEPTH_THRESHOLD:
        lines.append("Try to squat a little deeper — aim to get your thighs parallel to the floor.")
    else:
        lines.append("Good depth on that rep — you reached a solid ninety-degree bend.")

    # ---- 2. Heel rise (highest safety priority — side view) ----
    bent_frames = _depth_frames(frames)
    if bent_frames:
        rises = [_heel_rise(f) for f in bent_frames]
        if rises and max(rises) > _HEEL_RISE_THRESHOLD:
            lines.append("Keep your heels flat on the floor as you squat down — your heels were lifting slightly.")

    # ---- 3. Knees over toes / shin angle ----
    avg_shin = _avg_shin_angle(bent_frames if bent_frames else frames)
    if avg_shin > _SHIN_ANGLE_MAX:
        lines.append("Push your hips back a little more — your knees are travelling too far over your toes.")

    # ---- 4. Torso lean at bottom ----
    torso_lean = _get_torso_lean_at_bottom(frames)
    if torso_lean < _TORSO_LEAN_MIN:
        lines.append("You can afford a slight forward lean — try to let your hips hinge back naturally.")
    elif torso_lean > _TORSO_LEAN_MAX:
        lines.append("Try to keep your chest more upright — you are leaning forward a little too much at the bottom.")

    # ---- 5. Shoulder drift ----
    drift = _get_shoulder_horizontal_drift(frames)
    if drift > 0.10:
        lines.append("Try to keep your shoulders tracking straight up and down rather than drifting forward or backward.")

    # ---- Encouragement if no issues found (no corrective cues added beyond depth) ----
    if len(lines) == 2 and max_bend >= _DEPTH_THRESHOLD:
        lines.append("Great form — keep it up!")

    return lines


def get_session_summary(session_data: dict) -> list[str]:
    total_reps = session_data.get("total_reps", 0)
    rounds = session_data.get("rounds", [])
    round_feedback = session_data.get("round_feedback", [])

    lines = []

    if total_reps == 0:
        lines.append("No reps were completed this session.")
        lines.append("Try standing sideways to the camera and bend your knees to at least a quarter squat depth to register a rep.")
        lines.append("Focus on controlling the movement slowly downward and then pushing back up to standing.")
        return lines

    lines.append(f"Great work — you completed {total_reps} squat{'s' if total_reps != 1 else ''} this session.")

    # ---- Depth progression across rounds ----
    round_peaks = []
    for r in rounds:
        r_frames = r.get("frames", [])
        vals = [_best_knee_bend(f) for f in r_frames if _best_knee_bend(f) > 5.0]
        round_peaks.append(max(vals) if vals else 0.0)

    if len(round_peaks) >= 2:
        if round_peaks[-1] > round_peaks[0] + 10:
            lines.append(f"Your squat depth improved across the session — by round {len(round_peaks)} you were squatting noticeably deeper than in round 1.")
        elif round_peaks[0] > round_peaks[-1] + 10:
            lines.append(f"Your depth was best in the early rounds — try to maintain that same depth even as you fatigue in later rounds.")
        elif all(p >= _DEPTH_THRESHOLD for p in round_peaks):
            lines.append("You consistently reached good squat depth across all rounds.")

    # ---- Persistent issues from round feedback ----
    heel_issue_rounds = 0
    shin_issue_rounds = 0
    torso_issue_rounds = 0
    shoulder_issue_rounds = 0

    for rf_list in round_feedback:
        rf_text = " ".join(rf_list).lower()
        if "heel" in rf_text:
            heel_issue_rounds += 1
        if "toes" in rf_text or "shin" in rf_text or "hips back" in rf_text:
            shin_issue_rounds += 1
        if "chest" in rf_text or "leaning forward" in rf_text or "upright" in rf_text:
            torso_issue_rounds += 1
        if "shoulder" in rf_text or "drifting" in rf_text:
            shoulder_issue_rounds += 1

    num_rounds = max(len(round_feedback), 1)

    if heel_issue_rounds == num_rounds and num_rounds > 1:
        lines.append("Your heels were lifting throughout the session — working on ankle mobility stretches between sessions will help.")
    elif heel_issue_rounds == 1 and num_rounds > 1:
        lines.append("Heel rise appeared in one round — keep focusing on keeping your heels grounded as you squat.")

    if shin_issue_rounds == num_rounds and num_rounds > 1:
        lines.append("Your knees were travelling forward past your toes in every round — focus on sitting your hips back at the start of each squat.")
    elif shin_issue_rounds >= 1 and shin_issue_rounds < num_rounds:
        lines.append("In some rounds your knees drifted forward over your toes — push your hips back a little more to correct this.")

    if torso_issue_rounds == num_rounds and num_rounds > 1:
        lines.append("You were leaning forward more than ideal in every round — try to keep your chest facing forward as you squat.")
    elif torso_issue_rounds >= 1 and torso_issue_rounds < num_rounds:
        lines.append("Your torso lean was a little excessive in some rounds — aim for a modest, controlled forward tilt.")

    if shoulder_issue_rounds >= 1:
        lines.append("Watch your shoulder position — try to keep them tracking vertically without drifting forward or back.")

    # ---- Positive summary if few issues ----
    total_issue_rounds = heel_issue_rounds + shin_issue_rounds + torso_issue_rounds + shoulder_issue_rounds
    if total_issue_rounds == 0:
        lines.append("Your form was solid — good shin angle, stable heels, and a controlled torso lean throughout.")

    return lines


def get_relevant_joints() -> list:
    return [
        ("Left Knee", "left_knee_bend_2d"),
        ("Right Knee", "right_knee_bend_2d"),
        ("Torso Lean", "trunk_lean_2d"),
    ]