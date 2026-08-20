from __future__ import annotations

import math
import statistics

# ---------------------------------------------------------------------------
# Module-level state for detect_rep
# ---------------------------------------------------------------------------
_phase = "ready"   # "ready" | "moving"
_BEND_START = 15.0   # degrees – knee bend to enter "moving"
_BEND_RETURN = 10.0  # degrees – knee bend to complete rep back to "ready"

_VIS_THRESHOLD = 0.35

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _best_knee_bend(pose_data: dict) -> float:
    l = pose_data.get("left_knee_bend_2d", 0.0) or 0.0
    r = pose_data.get("right_knee_bend_2d", 0.0) or 0.0
    return max(l, r)


def _depth_frames(frames: list) -> list:
    return [f for f in frames if _best_knee_bend(f) > 20.0]


def _knee_forward_ratio(pose_data: dict) -> float:
    """abs(knee_x - ankle_x) / shin_length. ~0.7 = knee over toes."""
    kpts = pose_data.get("keypoints", {})
    lk = kpts.get("left_knee")
    la = kpts.get("left_ankle")
    rk = kpts.get("right_knee")
    ra = kpts.get("right_ankle")
    l_vis = min(lk[3] if lk else 0, la[3] if la else 0)
    r_vis = min(rk[3] if rk else 0, ra[3] if ra else 0)
    if l_vis >= r_vis and l_vis > 0.3:
        knee, ankle = lk, la
    elif r_vis > 0.3:
        knee, ankle = rk, ra
    else:
        return 0.0
    shin_len = math.hypot(knee[0] - ankle[0], knee[1] - ankle[1])
    if shin_len < 0.02:
        return 0.0
    return abs(knee[0] - ankle[0]) / shin_len


def _get_trunk_lean(pose_data: dict) -> float:
    """Return trunk lean from vertical (degrees). Uses trunk_lean_2d if available."""
    val = pose_data.get("trunk_lean_2d", None)
    if val is not None:
        return float(val)
    val = pose_data.get("trunk_lean_angle", None)
    if val is not None:
        return float(val)
    return 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_instructions() -> list[str]:
    return [
        "Ställ dig med ena sidan mot kameran.",
        "Utför lugna och kontrollerade hukningar och försök nå minst 90 graders böjning i knäet.",
    ]


def detect_rep(pose_data: dict) -> bool:
    global _phase
    bend = _best_knee_bend(pose_data)
    if _phase == "ready":
        if bend >= _BEND_START:
            _phase = "moving"
        return False
    elif _phase == "moving":
        if bend < _BEND_RETURN:
            _phase = "ready"
            return True
    return False


def reset_round():
    global _phase
    _phase = "ready"


def generate_rep_feedback(rep_data: dict) -> list[str]:
    rep_number = rep_data.get("rep_number", 1)
    frames = rep_data.get("frames", [])
    feedback = []

    feedback.append(f"Rep {rep_number} klar.")

    if not frames:
        return feedback

    # --- Knee depth check ---
    bent_frames = _depth_frames(frames)
    check_frames = bent_frames if bent_frames else frames

    knee_bends = [_best_knee_bend(f) for f in check_frames if _best_knee_bend(f) > 1.0]
    max_bend = max(knee_bends) if knee_bends else 0.0

    if max_bend < 90.0:
        feedback.append(
            "Försök att böja knäet djupare – sikta på minst 90 graders böjning i knäleden."
        )

    # --- Trunk lean check ---
    lean_vals = [_get_trunk_lean(f) for f in check_frames if _get_trunk_lean(f) > 1.0]
    if lean_vals:
        avg_lean = statistics.mean(lean_vals)
        if avg_lean < 25.0:
            feedback.append(
                "Du lutar dig lite för lite framåt – en lutning på 25 till 45 grader är lämplig för balans och knäskydd."
            )
        elif avg_lean > 45.0:
            feedback.append(
                "Du lutar dig för mycket framåt med överkroppen – försök hålla ryggen mer upprätt, mellan 25 och 45 graders framåtlutning."
            )

    # --- Knees over toes check ---
    ratios = [_knee_forward_ratio(f) for f in check_frames]
    peak_ratio = max((r for r in ratios if r > 0.05), default=0.0)
    if peak_ratio > 0.5:
        feedback.append(
            "Knäna sträcker sig för långt framför vristen – försök hålla knäna mer i linje med vristen."
        )

    # --- No issues ---
    if len(feedback) == 1:
        feedback.append("Bra jobbat! Bra djup och kontrollerad rörelse – håll den stilen!")

    return feedback


def get_session_summary(session_data: dict) -> list[str]:
    total_reps = session_data.get("total_reps", 0)
    rounds = session_data.get("rounds", [])
    round_feedback = session_data.get("round_feedback", [])
    lines = []

    if total_reps == 0:
        lines.append("Inga repetitioner slutfördes under den här sessionen.")
        lines.append(
            "Nästa gång, försök böja knäet minst 90 grader och håll överkroppen i en kontrollerad framåtlutning på 25 till 45 grader."
        )
        return lines

    lines.append(f"Du slutförde totalt {total_reps} repetitioner – bra insats!")

    # --- Per-round knee depth analysis ---
    round_peaks = []
    for r in rounds:
        r_frames = r.get("frames", [])
        bent = _depth_frames(r_frames)
        check = bent if bent else r_frames
        vals = [_best_knee_bend(f) for f in check if _best_knee_bend(f) > 5.0]
        round_peaks.append(max(vals) if vals else 0.0)

    if len(round_peaks) >= 2:
        if round_peaks[-1] > round_peaks[0] + 10:
            lines.append(
                f"Ditt djup förbättrades under sessionen – du nådde ett bättre knädjup i runda {len(round_peaks)} jämfört med runda 1."
            )
        elif round_peaks[0] > round_peaks[-1] + 10:
            lines.append(
                f"Ditt knädjup var bättre i de tidiga rundorna – försök hålla energin och djupet även i de senare rundorna."
            )
        elif all(p >= 90.0 for p in round_peaks):
            lines.append("Du nådde ett bra knädjup på minst 90 grader i varje runda.")
        elif all(p < 90.0 for p in round_peaks):
            lines.append(
                "Knädjupet nådde inte riktigt 90 grader i någon runda – fokusera på att sjunka lite djupare nästa session."
            )

    # --- Per-round trunk lean analysis ---
    round_lean_avgs = []
    for r in rounds:
        r_frames = r.get("frames", [])
        bent = _depth_frames(r_frames)
        check = bent if bent else r_frames
        lean_vals = [_get_trunk_lean(f) for f in check if _get_trunk_lean(f) > 1.0]
        round_lean_avgs.append(statistics.mean(lean_vals) if lean_vals else 0.0)

    lean_issues_rounds = []
    for i, avg_lean in enumerate(round_lean_avgs):
        if avg_lean > 0.0 and (avg_lean < 25.0 or avg_lean > 45.0):
            lean_issues_rounds.append(i + 1)

    if lean_issues_rounds:
        if len(lean_issues_rounds) == len(rounds) and len(rounds) > 1:
            lines.append(
                "Överkroppens lutning var utanför det optimala intervallet (25–45 grader) i samtliga rundor – arbeta aktivt med att hålla en kontrollerad framåtlutning."
            )
        else:
            round_labels = ", ".join(f"runda {n}" for n in lean_issues_rounds)
            lines.append(
                f"Överkroppens lutning behöver justeras – den var utanför 25–45 grader i {round_labels}."
            )

    # --- Knees over toes across rounds ---
    knee_fwd_issues = []
    for i, r in enumerate(rounds):
        r_frames = r.get("frames", [])
        bent = _depth_frames(r_frames)
        check = bent if bent else r_frames
        ratios = [_knee_forward_ratio(f) for f in check]
        peak_ratio = max((rv for rv in ratios if rv > 0.05), default=0.0)
        if peak_ratio > 0.5:
            knee_fwd_issues.append(i + 1)

    if knee_fwd_issues:
        if len(knee_fwd_issues) == len(rounds) and len(rounds) > 1:
            lines.append(
                "Knäna sträckte sig för långt framför vristen i samtliga rundor – försök skjuta höfterna bakåt och hålla vikten mer i hälarna."
            )
        else:
            round_labels = ", ".join(f"runda {n}" for n in knee_fwd_issues)
            lines.append(
                f"Knäna sträckte sig för långt framför vristen i {round_labels} – fokusera på att hålla knäna i linje med vristen."
            )

    # --- Use round_feedback as secondary evidence for improvement ---
    if round_feedback and len(round_feedback) >= 2:
        first_issues = set(round_feedback[0]) if round_feedback[0] else set()
        last_issues = set(round_feedback[-1]) if round_feedback[-1] else set()
        if first_issues and not (first_issues & last_issues):
            lines.append(
                "Bra jobbat – du åtgärdade de inledande formkommentarerna och avslutade med bättre teknik."
            )

    # --- Closing encouragement ---
    lines.append(
        "Fortsätt träna med kontrollerad rörelse och fokus på djup och rak överkropp – du gör bra framsteg!"
    )

    return lines


def get_relevant_joints() -> list:
    return [
        ("Knäböjning", "left_knee_bend_2d"),
        ("Överkroppslutn.", "trunk_lean_2d"),
    ]