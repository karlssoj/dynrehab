import math
import statistics

_VIS_THRESHOLD = 0.5

# ── detect_rep state ──────────────────────────────────────────────────────────
_phase = "ready"   # "ready" | "moving"
_in_rep = False

# ── threshold constants (bend convention: 0=straight, higher=more bent) ───────
# knee_bend thresholds derived from reference data:
#   min_bend=3, range=106 → started=3+0.30*106≈35, full=3+0.60*106≈67
_KNEE_BEND_START   = 35.0   # knee bend to enter "moving" phase (loose)
_KNEE_BEND_FULL    = 67.0   # knee bend for a "completed" rep attempt (loose)
_KNEE_BEND_RETURN  = 20.0   # knee bend must fall below this to reset

# boundary-value targets (used only in feedback, NOT in detect_rep)
_KNEE_BEND_TARGET  = 90.0   # ideally >90° bend at bottom (≈90° knee angle)
_KNEE_RETURN_TARGET = 10.0  # ideally <10° bend at top

_TORSO_LEAN_MIN    = 20.0
_TORSO_LEAN_MAX    = 45.0

_SHIN_ANGLE_LIMIT  = 30.0   # degrees from vertical — above this = knees too far forward
_HEEL_RISE_LIMIT   = 0.04   # unitless y-fraction


# ─────────────────────────────────────────────────────────────────────────────
# Helper: best (largest) knee bend in a single frame
# ─────────────────────────────────────────────────────────────────────────────
def _best_knee_bend(pose_data: dict) -> float:
    l = pose_data.get("left_knee_bend_2d", 0.0) or 0.0
    r = pose_data.get("right_knee_bend_2d", 0.0) or 0.0
    return max(l, r)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: frames where knee is meaningfully bent
# ─────────────────────────────────────────────────────────────────────────────
def _depth_frames(frames: list) -> list:
    return [f for f in frames if _best_knee_bend(f) > 20.0]


# ─────────────────────────────────────────────────────────────────────────────
# Helper: average maximum shin angle across frames
# ─────────────────────────────────────────────────────────────────────────────
def _avg_shin_angle(frames: list) -> float:
    vals = [max(f.get("left_shin_angle", 0.0) or 0.0,
                f.get("right_shin_angle", 0.0) or 0.0)
            for f in frames]
    vals = [v for v in vals if v > 1.0]
    return statistics.mean(vals) if vals else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Helper: heel rise (side-view only)
# ─────────────────────────────────────────────────────────────────────────────
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
    # Foot landmarks score 0.30–0.45 from side-view; don't gate on _VIS_THRESHOLD
    if min(heel[3], foot_index[3]) < 0.30:
        return 0.0
    return foot_index[1] - heel[1]


# ─────────────────────────────────────────────────────────────────────────────
# Helper: peak knee bend across a list of frames
# ─────────────────────────────────────────────────────────────────────────────
def _peak_knee_bend(frames: list) -> float:
    vals = [_best_knee_bend(f) for f in frames]
    return max(vals) if vals else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Helper: trunk lean at bottom of squat (max lean in bent frames)
# ─────────────────────────────────────────────────────────────────────────────
def _max_torso_lean(frames: list) -> float:
    vals = [f.get("trunk_lean_2d", 0.0) or 0.0 for f in frames]
    vals = [v for v in vals if v > 0.5]
    return max(vals) if vals else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_instructions() -> list:
    return [
        "Stand sideways to the camera with your feet shoulder-width apart.",
        "Perform slow, controlled squats aiming to reach a 90-degree knee bend at the bottom of each rep.",
    ]


def reset_round():
    global _phase, _in_rep
    _phase = "ready"
    _in_rep = False


def detect_rep(pose_data: dict) -> bool:
    global _phase, _in_rep

    bend = _best_knee_bend(pose_data)

    if _phase == "ready":
        if bend >= _KNEE_BEND_START:
            _phase = "moving"
        return False

    if _phase == "moving":
        if bend >= _KNEE_BEND_FULL:
            _in_rep = True
        if _in_rep and bend < _KNEE_BEND_RETURN:
            _phase = "ready"
            _in_rep = False
            return True

    return False


def generate_rep_feedback(rep_data: dict) -> list:
    rep_number = rep_data.get("rep_number", 1)
    frames = rep_data.get("frames", [])

    lines = [f"Rep {rep_number} done."]

    if not frames:
        lines.append("Keep going — focus on depth and control.")
        return lines

    bent_frames = _depth_frames(frames)
    peak_bend = _peak_knee_bend(frames)

    # ── 1. Heel rise (highest injury-risk priority) ───────────────────────────
    if bent_frames:
        rises = [_heel_rise(f) for f in bent_frames]
        if rises and max(rises) > _HEEL_RISE_LIMIT:
            lines.append("Your heels were lifting off the ground — try to keep them flat throughout the squat.")
            return lines

    # ── 2. Knees travelling too far forward ──────────────────────────────────
    avg_shin = _avg_shin_angle(bent_frames if bent_frames else frames)
    if avg_shin > _SHIN_ANGLE_LIMIT:
        lines.append("Your knees are travelling too far over your toes — push your hips back as you squat down.")

    # ── 3. Squat depth ───────────────────────────────────────────────────────
    if peak_bend < _KNEE_BEND_TARGET:
        lines.append("Try to squat a little deeper — aim for a 90-degree knee bend at the bottom.")
    else:
        lines.append("Good depth — you reached a full 90-degree squat.")

    # ── 4. Torso lean ────────────────────────────────────────────────────────
    max_lean = _max_torso_lean(bent_frames if bent_frames else frames)
    if max_lean > _TORSO_LEAN_MAX:
        lines.append("You leaned forward quite a bit — try to keep your chest more upright as you squat.")
    elif max_lean < _TORSO_LEAN_MIN and max_lean > 1.0:
        lines.append("Good upright posture — just remember a small forward lean is completely natural.")

    if len(lines) == 1:
        lines.append("Nice squat — keep your movement controlled and steady.")

    return lines


def get_session_summary(session_data: dict) -> list:
    total_reps = session_data.get("total_reps", 0)
    rounds = session_data.get("rounds", [])
    round_feedback = session_data.get("round_feedback", [])

    lines = []

    # ── Zero reps edge-case ───────────────────────────────────────────────────
    if total_reps == 0:
        lines.append("No reps were completed this session.")
        lines.append(
            "Next time, make sure you are standing sideways to the camera and bend your knees enough "
            "for the system to detect the movement."
        )
        lines.append("Focus on lowering slowly until you feel a 90-degree bend in your knees.")
        return lines

    lines.append(f"Great effort today — you completed {total_reps} squat{'s' if total_reps != 1 else ''} in total.")

    # ── Per-round depth progression ──────────────────────────────────────────
    round_peaks = []
    for r in rounds:
        r_frames = r.get("frames", [])
        vals = [_best_knee_bend(f) for f in r_frames if _best_knee_bend(f) > 5]
        round_peaks.append(max(vals) if vals else 0.0)

    if len(round_peaks) >= 2:
        first = round_peaks[0]
        last = round_peaks[-1]
        if last > first + 10:
            lines.append("Your squat depth improved as the session went on — great progression.")
        elif first > last + 10:
            lines.append(
                "Your depth was deeper in the earlier rounds — fatigue may have set in towards the end. "
                "Focus on maintaining depth even when tired."
            )
        elif all(p >= _KNEE_BEND_TARGET for p in round_peaks):
            lines.append("You maintained good squat depth in every round.")
        elif all(p < _KNEE_BEND_TARGET for p in round_peaks):
            lines.append(
                "Your squat depth was a little shallow across the session — aim to reach a full 90-degree "
                "knee bend on every rep."
            )

    # ── Heel rise (persistent vs improved) ───────────────────────────────────
    heel_issue_rounds = []
    for i, r in enumerate(rounds):
        r_frames = r.get("frames", [])
        bent = _depth_frames(r_frames)
        if bent:
            rises = [_heel_rise(f) for f in bent]
            if rises and max(rises) > _HEEL_RISE_LIMIT:
                heel_issue_rounds.append(i + 1)

    if heel_issue_rounds:
        if len(heel_issue_rounds) == len(rounds) and len(rounds) > 1:
            lines.append(
                "Heel lifting was present in every round — continued work on ankle mobility will help you "
                "keep your heels flat throughout the squat."
            )
        elif len(heel_issue_rounds) == 1:
            lines.append(
                f"There was some heel rise in round {heel_issue_rounds[0]}, but it improved — "
                "keep focusing on keeping those heels down."
            )
        else:
            round_str = " and ".join(f"round {n}" for n in heel_issue_rounds)
            lines.append(
                f"Heel lifting appeared in {round_str} — ankle mobility work between sessions will help."
            )

    # ── Shin angle / knees over toes ─────────────────────────────────────────
    shin_issue_rounds = []
    for i, r in enumerate(rounds):
        r_frames = r.get("frames", [])
        bent = _depth_frames(r_frames)
        avg_shin = _avg_shin_angle(bent if bent else r_frames)
        if avg_shin > _SHIN_ANGLE_LIMIT:
            shin_issue_rounds.append(i + 1)

    if shin_issue_rounds:
        if len(shin_issue_rounds) == len(rounds) and len(rounds) > 1:
            lines.append(
                "Your knees were consistently travelling past your toes — practise pushing your hips "
                "further back as you lower into the squat."
            )
        else:
            round_str = " and ".join(f"round {n}" for n in shin_issue_rounds)
            lines.append(
                f"Knees travelling too far forward was noted in {round_str} — "
                "keep practising the hip-back cue."
            )

    # ── Torso lean ───────────────────────────────────────────────────────────
    torso_issue_rounds = []
    for i, r in enumerate(rounds):
        r_frames = r.get("frames", [])
        bent = _depth_frames(r_frames)
        max_lean = _max_torso_lean(bent if bent else r_frames)
        if max_lean > _TORSO_LEAN_MAX:
            torso_issue_rounds.append(i + 1)

    if torso_issue_rounds:
        if len(torso_issue_rounds) == len(rounds) and len(rounds) > 1:
            lines.append(
                "You showed quite a forward lean throughout the session — focus on keeping your chest "
                "facing forward and your torso upright as you squat."
            )
        else:
            round_str = " and ".join(f"round {n}" for n in torso_issue_rounds)
            lines.append(
                f"Excessive forward lean appeared in {round_str} — "
                "try to keep your chest lifted in future reps."
            )

    # ── Use round_feedback as secondary evidence for improvement signals ──────
    if round_feedback and len(round_feedback) >= 2:
        first_fb = " ".join(round_feedback[0]) if isinstance(round_feedback[0], list) else str(round_feedback[0])
        last_fb = " ".join(round_feedback[-1]) if isinstance(round_feedback[-1], list) else str(round_feedback[-1])
        heel_in_first = "heel" in first_fb.lower()
        heel_in_last = "heel" in last_fb.lower()
        if heel_in_first and not heel_in_last and not heel_issue_rounds:
            lines.append("Your heel control improved noticeably by the final round — well done.")

    # ── Closing encouragement ────────────────────────────────────────────────
    if not any("great" in l.lower() or "good" in l.lower() or "well done" in l.lower() for l in lines[1:]):
        lines.append(
            "Overall a solid session — keep practising controlled squats and the movement will continue to improve."
        )

    return lines


def get_relevant_joints() -> list:
    return [
        ("Left Knee", "left_knee_bend_2d"),
        ("Right Knee", "right_knee_bend_2d"),
        ("Torso Lean", "trunk_lean_2d"),
    ]