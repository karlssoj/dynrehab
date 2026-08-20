from __future__ import annotations

import math
import statistics

# ---------------------------------------------------------------------------
# Module-level state for detect_rep
# ---------------------------------------------------------------------------
_phase = "ready"
_start_hip_y = 0.0
_peak_hip_y = 0.0

_VIS_THRESHOLD = 0.35

# ---------------------------------------------------------------------------
# Thresholds (bend convention: 0=straight, higher=more bent)
# ---------------------------------------------------------------------------
# Knee bend: range ~109-112°, 60% ≈ 67° → raise to clinical minimum 75°
_KNEE_BEND_START = 25.0   # 30% of ~109 ≈ 33°, use 25 for loose detection
_KNEE_BEND_FULL  = 75.0   # clinical minimum
_KNEE_BEND_RETURN = 15.0  # back to near-straight

# Hip bend: range ~129-130°, 60% ≈ 78° → above clinical minimum 60°
_HIP_BEND_FULL = 60.0

# Trunk lean: range ~37°, max=37°
_TRUNK_LEAN_FULL = 22.0   # 60% of 37

# Shin angle over-toes threshold
_SHIN_ANGLE_LIMIT = 30.0


# ---------------------------------------------------------------------------
# Helpers
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


def _get_hip_y(pose_data: dict) -> float:
    kpts = pose_data.get("keypoints", {})
    pts = []
    for side in ("left_hip", "right_hip"):
        h = kpts.get(side)
        if h and h[3] > 0.3:
            pts.append(h[1])
    return sum(pts) / len(pts) if pts else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_instructions() -> list:
    return [
        "Stand sideways to the camera with your feet shoulder-width apart.",
        "Slowly bend both knees to squat down, then return to standing.",
    ]


def detect_rep(pose_data: dict) -> bool:
    global _phase, _start_hip_y, _peak_hip_y

    # Side-view orientation guard
    if pose_data.get("shoulder_lateral_span", 1.0) > 0.15:
        _phase = "ready"
        return False

    knee_bend = _best_knee_bend(pose_data)

    if _phase == "ready":
        if knee_bend > _KNEE_BEND_START:
            _phase = "moving"
            _start_hip_y = _get_hip_y(pose_data)
            _peak_hip_y = _start_hip_y

    elif _phase == "moving":
        hip_y = _get_hip_y(pose_data)
        if hip_y > 0.0 and hip_y > _peak_hip_y:
            _peak_hip_y = hip_y

        if knee_bend < _KNEE_BEND_RETURN:
            # Check hip descent coordination
            if _start_hip_y > 0.0:
                descended = _peak_hip_y > _start_hip_y + 0.03
            else:
                descended = True  # no reliable reading → allow

            _phase = "ready"
            if descended:
                return True

    return False


def reset_round():
    global _phase, _start_hip_y, _peak_hip_y
    _phase = "ready"
    _start_hip_y = 0.0
    _peak_hip_y = 0.0


def classify_movement(frames: list) -> str | None:
    if not frames:
        return None

    bends = [_best_knee_bend(f) for f in frames if _best_knee_bend(f) > 1.0]
    if not bends:
        return "Start the exercise by bending both knees to squat down."

    bend_range = max(bends) - min(bends)
    if bend_range < 10.0:
        return "Start the exercise by bending both knees to squat down."

    bent_frames  = [f for f in frames if _best_knee_bend(f) > 20.0]
    stand_frames = [f for f in frames if _best_knee_bend(f) <  5.0]

    if bent_frames and stand_frames:
        def _hy(f):
            kpts = f.get("keypoints", {})
            pts = [kpts[s][1] for s in ("left_hip", "right_hip")
                   if kpts.get(s) and kpts[s][3] > 0.3]
            return sum(pts) / len(pts) if pts else 0.0

        bent_hy  = [_hy(f) for f in bent_frames  if _hy(f) > 0]
        stand_hy = [_hy(f) for f in stand_frames if _hy(f) > 0]

        if bent_hy and stand_hy:
            avg_bent  = sum(bent_hy)  / len(bent_hy)
            avg_stand = sum(stand_hy) / len(stand_hy)
            if avg_bent < avg_stand + 0.03:
                return "Lower your hips as you squat down — bend both knees together."

    return None


def generate_rep_feedback(rep_data: dict) -> list:
    frames = rep_data.get("frames", [])
    rep_number = rep_data.get("rep_number", 1)
    feedback = []

    feedback.append(f"Rep {rep_number} done.")

    if not frames:
        return feedback

    # --- Squat depth ---
    knee_bends = [_best_knee_bend(f) for f in frames if _best_knee_bend(f) > 5.0]
    max_knee_bend = max(knee_bends) if knee_bends else 0.0

    if max_knee_bend < 45.0:
        feedback.append("Try to squat deeper — aim to lower your hips until your thighs are closer to horizontal.")
    elif max_knee_bend < _KNEE_BEND_FULL:
        feedback.append("Good effort — try to squat a little deeper to reach a parallel position.")

    # --- Stance check (ankle lateral span) ---
    ankle_spans = [f.get("ankle_lateral_span", 0.0) for f in frames]
    valid_spans = [v for v in ankle_spans if v is not None]
    avg_ankle_span = statistics.mean(valid_spans) if valid_spans else 0.0
    if avg_ankle_span > 0.20:
        feedback.append("Make sure both feet are side by side — one foot appears to be in front of the other.")

    bent_frames = _depth_frames(frames)

    # --- Heel rise (highest priority in bent phase) ---
    if bent_frames:
        rises = [_heel_rise(f) for f in bent_frames]
        if rises and max(rises) > 0.02:
            feedback.append("Keep your heels flat on the floor as you squat down — try to improve your ankle mobility.")

    # --- Knees too far forward ---
    avg_shin = _avg_shin_angle(bent_frames if bent_frames else frames)
    if avg_shin > _SHIN_ANGLE_LIMIT:
        feedback.append("Push your hips back a little more — your knees are travelling too far over your toes.")

    # --- Trunk lean ---
    trunk_leans = [f.get("trunk_lean_2d", 0.0) or 0.0 for f in frames]
    trunk_leans = [v for v in trunk_leans if v > 1.0]
    max_trunk_lean = max(trunk_leans) if trunk_leans else 0.0
    if max_trunk_lean > 35.0:
        feedback.append("Try to keep your chest up and your trunk more upright as you squat.")

    # --- Hip bend (checks hips are actually flexing) ---
    hip_bends = []
    for f in frames:
        l = f.get("left_hip_bend_2d", 0.0) or 0.0
        r = f.get("right_hip_bend_2d", 0.0) or 0.0
        hip_bends.append(max(l, r))
    max_hip_bend = max(hip_bends) if hip_bends else 0.0
    if max_knee_bend >= _KNEE_BEND_FULL and max_hip_bend < _HIP_BEND_FULL:
        feedback.append("Make sure you are pushing your hips back as you lower down.")

    # --- Positive feedback if no issues ---
    if len(feedback) == 1:
        if max_knee_bend >= _KNEE_BEND_FULL:
            feedback.append("Great squat — excellent depth and good overall form!")
        else:
            feedback.append("Good effort — keep working on your squat depth.")

    return feedback


def get_session_summary(session_data: dict) -> list:
    total_reps = session_data.get("total_reps", 0)
    rounds = session_data.get("rounds", [])
    round_feedback = session_data.get("round_feedback", [])
    lines = []

    if total_reps == 0:
        lines.append("No reps were completed this session.")
        lines.append("To perform a squat, stand sideways to the camera, then bend both knees to lower your hips, and push back up to standing.")
        lines.append("Try to keep your heels on the floor and your feet side by side throughout the movement.")
        return lines

    lines.append(f"You completed {total_reps} squat{'s' if total_reps != 1 else ''} this session.")

    # --- Per-round depth progression ---
    round_peaks = []
    for r in rounds:
        r_frames = r.get("frames", [])
        vals = [_best_knee_bend(f) for f in r_frames if _best_knee_bend(f) > 5.0]
        round_peaks.append(max(vals) if vals else 0.0)

    if len(round_peaks) >= 2:
        first = round_peaks[0]
        last  = round_peaks[-1]
        if last > first + 10.0:
            lines.append(f"Your squat depth improved across the session — well done on deepening your squats by round {len(round_peaks)}.")
        elif first > last + 10.0:
            lines.append(f"Your squat depth was better in round 1 — try to maintain that depth in later rounds.")
        elif all(p >= _KNEE_BEND_FULL for p in round_peaks):
            lines.append("You maintained good squat depth in every round.")
        elif all(p < _KNEE_BEND_FULL for p in round_peaks):
            lines.append("Work on squatting a little deeper each rep — aim to lower your hips until your thighs are roughly horizontal.")

    elif len(round_peaks) == 1:
        if round_peaks[0] >= _KNEE_BEND_FULL:
            lines.append("You reached a good squat depth in your session.")
        else:
            lines.append("Try to squat a little deeper next time — aim for your thighs to be roughly horizontal at the bottom.")

    # --- Persistent issues from round feedback ---
    heel_issue_rounds = 0
    knee_forward_rounds = 0
    trunk_lean_rounds = 0
    stance_issue_rounds = 0

    for rf in round_feedback:
        if not rf:
            continue
        combined = " ".join(rf).lower()
        if "heel" in combined:
            heel_issue_rounds += 1
        if "knees are travelling" in combined or "over your toes" in combined:
            knee_forward_rounds += 1
        if "chest up" in combined or "trunk" in combined:
            trunk_lean_rounds += 1
        if "side by side" in combined or "foot appears" in combined:
            stance_issue_rounds += 1

    total_rounds = len(round_feedback) if round_feedback else 1

    if heel_issue_rounds > 0:
        if heel_issue_rounds == total_rounds:
            lines.append("Your heels were lifting throughout the session — ankle mobility exercises would help you keep them flat.")
        else:
            lines.append(f"Heel rise was noticed in {heel_issue_rounds} of your rounds — focus on keeping your heels grounded as you squat.")

    if knee_forward_rounds > 0:
        if knee_forward_rounds == total_rounds:
            lines.append("Your knees consistently travelled too far forward — concentrate on pushing your hips back as you lower down.")
        else:
            lines.append(f"Knees travelling too far forward appeared in {knee_forward_rounds} round{'s' if knee_forward_rounds != 1 else ''} — keep working on sitting your hips back.")

    if trunk_lean_rounds > 0:
        if trunk_lean_rounds == total_rounds:
            lines.append("Try to keep your chest up throughout the squat to reduce excessive forward lean.")
        else:
            lines.append(f"Trunk lean was flagged in {trunk_lean_rounds} round{'s' if trunk_lean_rounds != 1 else ''} — focus on keeping your torso upright.")

    if stance_issue_rounds > 0:
        lines.append("Remember to stand with both feet side by side, not one in front of the other, for proper squat technique.")

    # --- Closing positive note ---
    if not any("heel" in l or "knees" in l or "trunk" in l or "feet" in l for l in lines[1:]):
        lines.append("Overall great work today — your squat technique looks solid.")
    else:
        lines.append("Keep practising and focus on the points raised — your squats will continue to improve.")

    return lines


def get_relevant_joints() -> list:
    return [
        ("Knee Bend",  "right_knee_bend_2d"),
        ("Hip Bend",   "right_hip_bend_2d"),
        ("Trunk Lean", "trunk_lean_2d"),
        ("Shin Angle", "right_shin_angle"),
    ]