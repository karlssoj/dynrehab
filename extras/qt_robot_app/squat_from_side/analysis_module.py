import math
import statistics

# ---------------------------------------------------------------------------
# Module-level state for detect_rep
# ---------------------------------------------------------------------------
_phase = "ready"          # "ready" | "moving"
_rep_count = 0

# Loose thresholds for rep detection (independent of quality boundary values)
_REP_START_BEND = 25.0    # knee bend degrees to register "moving"
_REP_RESET_BEND = 12.0    # knee bend degrees to return to "ready"

_VIS_THRESHOLD = 0.35


# ---------------------------------------------------------------------------
# Low-level helpers
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


def _avg_trunk_lean(frames: list) -> float:
    vals = [f.get("trunk_lean_2d", 0.0) or 0.0 for f in frames]
    vals = [v for v in vals if v >= 0.0]
    return statistics.mean(vals) if vals else 0.0


def _max_knee_bend(frames: list) -> float:
    vals = [_best_knee_bend(f) for f in frames]
    return max(vals) if vals else 0.0


def _shoulder_drift(frames: list) -> float:
    """
    Returns the range (max - min) of the shoulder x-position across frames,
    normalised to frame width [0,1].  Large values indicate excessive
    forward/backward shoulder drift.
    """
    xs = []
    for f in frames:
        kpts = f.get("keypoints", {})
        for side in ("left_shoulder", "right_shoulder"):
            lm = kpts.get(side)
            if lm and lm[3] >= _VIS_THRESHOLD:
                xs.append(lm[0])
    if len(xs) < 2:
        return 0.0
    return max(xs) - min(xs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_instructions() -> list:
    return [
        "Stand sideways to the camera with your feet shoulder-width apart.",
        "Perform slow, controlled squats, bending your knees to about ninety degrees, "
        "then return to standing."
    ]


def detect_rep(pose_data: dict) -> bool:
    global _phase, _rep_count

    bend = _best_knee_bend(pose_data)

    if _phase == "ready":
        if bend >= _REP_START_BEND:
            _phase = "moving"
            return False

    elif _phase == "moving":
        if bend <= _REP_RESET_BEND:
            _phase = "ready"
            _rep_count += 1
            return True

    return False


def reset_round():
    global _phase, _rep_count
    _phase = "ready"
    _rep_count = 0


def generate_rep_feedback(rep_data: dict) -> list:
    rep_number = rep_data.get("rep_number", 1)
    frames = rep_data.get("frames", [])

    lines = []

    # --- Opening acknowledgement ---
    lines.append(f"Rep {rep_number} done.")

    if not frames:
        lines.append("Keep going — focus on your form.")
        return lines

    bent_frames = _depth_frames(frames)
    max_bend = _max_knee_bend(frames)
    avg_lean = _avg_trunk_lean(bent_frames if bent_frames else frames)

    # --- 1. Heel rise (highest priority — injury risk) ---
    if bent_frames:
        rises = [_heel_rise(f) for f in bent_frames]
        if rises and max(rises) > 0.02:
            lines.append("Keep your heels flat on the floor as you squat down.")

    # --- 2. Knees over toes ---
    avg_shin = _avg_shin_angle(bent_frames if bent_frames else frames)
    if avg_shin > 30:
        lines.append(
            "Push your hips back a little more — your knees are travelling too far over your toes."
        )

    # --- 3. Squat depth ---
    # Boundary value: knee bend should exceed 90 degrees (i.e. bend > 90)
    if max_bend < 90:
        lines.append(
            "Try to squat a little deeper — aim for a full ninety-degree bend at the knee."
        )
    else:
        lines.append("Good depth on that rep.")

    # --- 4. Torso lean ---
    # Boundary: 20–45 degrees is acceptable
    if avg_lean < 20:
        lines.append(
            "You can allow a slight forward lean through your torso — that is completely normal."
        )
    elif avg_lean > 45:
        lines.append(
            "Try to keep your chest a little more upright — you are leaning forward too much."
        )

    # --- Positive close if no corrections were needed ---
    correction_keywords = [
        "heels", "knees", "deeper", "lean"
    ]
    corrections_given = any(
        any(kw in sentence for kw in correction_keywords)
        for sentence in lines[1:]
    )
    if not corrections_given:
        lines.append("Great form — keep that controlled movement going.")

    return lines


def get_session_summary(session_data: dict) -> list:
    total_reps = session_data.get("total_reps", 0)
    rounds = session_data.get("rounds", [])
    round_feedback = session_data.get("round_feedback", [])

    lines = []

    # --- Zero reps guard ---
    if total_reps == 0:
        lines.append("No reps were completed this session.")
        lines.append(
            "Next time, stand directly sideways to the camera and try to bend your knees "
            "as far as you comfortably can before returning to standing."
        )
        lines.append(
            "If you are unsure of the movement, ask your physiotherapist to demonstrate it first."
        )
        return lines

    lines.append(
        f"Well done — you completed {total_reps} squat{'s' if total_reps != 1 else ''} this session."
    )

    # --- Per-round depth analysis ---
    round_peaks = []
    for r in rounds:
        r_frames = r.get("frames", [])
        vals = [_best_knee_bend(f) for f in r_frames if _best_knee_bend(f) > 5]
        round_peaks.append(max(vals) if vals else 0.0)

    depth_threshold = 90.0

    if len(round_peaks) >= 2:
        first = round_peaks[0]
        last = round_peaks[-1]
        if last > first + 10:
            lines.append("Your squat depth improved as the session went on — nice progression.")
        elif first > last + 10:
            lines.append(
                f"Your depth was a little shallower in the later rounds — "
                "try to maintain the same range throughout."
            )
        elif all(p >= depth_threshold for p in round_peaks):
            lines.append("You reached a good knee bend depth in every round.")
        else:
            shallow_rounds = [i + 1 for i, p in enumerate(round_peaks) if p < depth_threshold]
            if shallow_rounds:
                round_str = ", ".join(f"round {r}" for r in shallow_rounds)
                lines.append(
                    f"The squats were a little shallow in {round_str} — "
                    "aim for a full ninety-degree bend each time."
                )
    elif len(round_peaks) == 1:
        if round_peaks[0] >= depth_threshold:
            lines.append("You reached a good squat depth.")
        else:
            lines.append(
                "Your squat depth was a little shallow — try to bend your knees to ninety degrees."
            )

    # --- Persistent issues from round feedback ---
    def _issue_in_feedback(fb_list, keyword):
        return any(
            any(keyword in sentence for sentence in round_fb)
            for round_fb in fb_list
            if round_fb
        )

    def _issue_in_all_feedback(fb_list, keyword):
        if not fb_list:
            return False
        return all(
            any(keyword in sentence for sentence in round_fb)
            for round_fb in fb_list
            if round_fb
        )

    # Heel rise
    if _issue_in_all_feedback(round_feedback, "heels"):
        lines.append(
            "Your heels were lifting throughout the session — working on ankle mobility "
            "exercises between sessions would help."
        )
    elif _issue_in_feedback(round_feedback, "heels"):
        lines.append(
            "There were some moments where your heels lifted — focus on keeping them "
            "grounded as you squat deeper."
        )

    # Knees over toes
    if _issue_in_all_feedback(round_feedback, "knees are travelling"):
        lines.append(
            "Your knees were consistently travelling too far forward — "
            "practise sitting your hips back as you descend."
        )
    elif _issue_in_feedback(round_feedback, "knees are travelling"):
        lines.append(
            "At times your knees moved too far over your toes — "
            "think about pushing your hips back as you squat down."
        )

    # Torso lean
    if _issue_in_all_feedback(round_feedback, "leaning forward too much"):
        lines.append(
            "Your torso was leaning forward throughout — "
            "focus on keeping your chest up and your back tall."
        )
    elif _issue_in_feedback(round_feedback, "leaning forward too much"):
        lines.append(
            "There were moments of excessive forward lean — "
            "keep your chest facing forward as much as you can."
        )

    # --- Closing encouragement ---
    lines.append(
        "Keep practising your squats — consistency will build both strength and control."
    )

    return lines


def get_relevant_joints() -> list:
    return [
        ("Left Knee", "left_knee_bend_2d"),
        ("Right Knee", "right_knee_bend_2d"),
        ("Torso Lean", "trunk_lean_2d"),
    ]
