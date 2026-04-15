import sqlite3
import uuid
import anthropic
from core.module_validator import validate_module
from core.video_analyzer import analyze_reference_video
from physio_app.services.exercise_service import ExerciseService

_POSE_DATA_DESCRIPTION = """\
The `pose_data` dict has these fields (all angles in degrees, calculated in 3D using x,y,z):
  timestamp: float — seconds since session start
  left_knee_angle / right_knee_angle — raw 3-point angle; ~170° standing, ~90° deep squat
  left_hip_angle / right_hip_angle — raw 3-point angle; ~180° standing, ~90° hip flexion
  left_ankle_angle / right_ankle_angle — knee-ankle-foot_index angle
  left_shoulder_angle / right_shoulder_angle — elbow-shoulder-hip angle
  left_elbow_angle / right_elbow_angle — raw 3-point angle; ~180° straight, ~45° fully bent
    (3D; noisy on side view — use left_elbow_bend_2d / right_elbow_bend_2d instead)
  left_wrist_angle / right_wrist_angle — elbow-wrist-index_finger angle
  trunk_lean_angle — trunk from vertical; 0°=upright, increases when leaning forward
    (3D; noisy on side view — use trunk_lean_2d instead)
  neck_angle — angle at shoulder-midpoint between trunk direction and nose
  pelvic_tilt — left-hip→right-hip line from horizontal.
    WARNING: this field is unreliable for front-facing camera — reads ~180° when level.
    For frontal exercises compute pelvic tilt from keypoints directly (see helper below).
  left_hka_alignment / right_hka_alignment — frontal-plane hip-knee-ankle angle; ~180°=straight.
    WARNING: this is a SCALAR angle — it decreases for BOTH valgus (inward) AND varus (outward).
    Do NOT use hka_alignment < threshold to detect valgus; it is directionless.
    For frontal-view valgus detection use the _knee_lateral_deviation helper below instead.
  left_arm_elevation / right_arm_elevation — angle at the shoulder between the shoulder-hip line and the shoulder-wrist line; 0°=arm hanging at side, 90°=arm horizontal, 180°=arm straight overhead. Use this for shoulder flexion/extension/abduction exercises.
  --- RELIABLE PRE-COMPUTED VALUES (prefer these over keypoint helpers) ---
  left_elbow_bend_2d / right_elbow_bend_2d — elbow bend, x,y only. 0°=arm straight,
    ~135°=fully curled. USE THIS for arm curl / bicep curl exercises.
  left_knee_bend_2d / right_knee_bend_2d — knee bend, x,y only. 0°=leg straight,
    ~90°=deep squat. USE THIS for side-view squat and lunge exercises.
  left_hip_bend_2d / right_hip_bend_2d — hip flexion, x,y only. 0°=trunk upright,
    increases as hip flexes. USE THIS for side-view hip hinge exercises.
  trunk_lean_2d — trunk lean from vertical, x,y only. 0°=upright, increases leaning.
    USE THIS instead of trunk_lean_angle for all side-view exercises.
  left_knee_valgus / right_knee_valgus — SIGNED lateral knee deviation (normalised units).
    Positive = valgus (knee inward/medial), negative = varus (knee outward/lateral).
    Threshold: ±0.03 normalised units = clinically meaningful. ±0.05 = clearly visible.
    USE THIS for front-view squat/lunge valgus/varus detection instead of hka_alignment.
  keypoints: dict[str, tuple[float,float,float,float]] — name→(x,y,z,visibility)
    x,y in [0,1] (y increases downward), z=depth, visibility in [0,1]
    landmark names: nose, left_shoulder, right_shoulder, left_elbow, right_elbow,
    left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee,
    left_ankle, right_ankle, left_foot_index, right_foot_index, left_heel, right_heel

ANGLE CONVENTION (use this consistently across ALL exercises):
  0° = fully straight / fully extended joint
  Higher values = more bent / more flexed
  This means: for elbow, knee, hip — compute BEND AMOUNT = 180° − raw_angle.
  Example: raw knee angle 90° → knee_bend = 180 − 90 = 90° (deeply bent).
           raw knee angle 170° → knee_bend = 180 − 170 = 10° (nearly straight).
  The _elbow_bend_2d helper in the example already uses this convention.
  Always write helpers for other joints the same way: 0=straight, higher=more bent.

HELPER PATTERN for knee/hip/ankle bend (use x,y only — immune to z-depth noise):
  NOTE: knee, hip, and elbow bend are now pre-computed in pose_data (left_knee_bend_2d,
  left_hip_bend_2d, left_elbow_bend_2d, etc.) — use those directly. The helper below is
  kept for reference and for computing ankle bend or other custom angles.
  def _knee_bend_2d(pose_data, side):
      kpts = pose_data.get("keypoints", {})
      h = kpts.get(f"{side}_hip")
      k = kpts.get(f"{side}_knee")
      a = kpts.get(f"{side}_ankle")
      if not (h and k and a) or min(h[3], k[3], a[3]) < _VIS_THRESHOLD:
          return 0.0
      return 180.0 - _angle_2d(h[0], h[1], k[0], k[1], a[0], a[1])
  # Returns 0 when leg is straight, increases as knee bends.
  # Same pattern applies for _hip_bend_2d (shoulder, hip, knee) and
  # _ankle_bend_2d (knee, ankle, foot_index).

HELPER PATTERN for frontal-view pelvic tilt (do not use pose_data["pelvic_tilt"]):
  def _pelvic_tilt_2d(pose_data):
      # Returns 0 when hips are level, increases when one hip is higher.
      # Works regardless of camera mirror mode.
      kpts = pose_data.get("keypoints", {})
      lh = kpts.get("left_hip")
      rh = kpts.get("right_hip")
      if not (lh and rh) or min(lh[3], rh[3]) < _VIS_THRESHOLD:
          return 0.0
      dx = abs(lh[0] - rh[0])
      dy = abs(lh[1] - rh[1])
      if dx < 1e-10:
          return 90.0
      return math.degrees(math.atan2(dy, dx))

HELPER PATTERN for frontal-view valgus (knee caving inward) detection:
  NOTE: Knee valgus is now pre-computed as left_knee_valgus / right_knee_valgus in pose_data
  — use those directly. The helper below is kept for reference.
  Because hka_alignment is directionless, use lateral knee deviation from keypoints:

  def _knee_lateral_deviation(pose_data, side):
      # Returns positive if knee is MEDIAL (valgus/inward), negative if LATERAL (varus/outward).
      # Requires front-facing camera. Accounts for the mirror effect: patient's LEFT
      # appears on the RIGHT side of the image (high x), RIGHT on the left (low x).
      kpts = pose_data.get("keypoints", {})
      h = kpts.get(f"{side}_hip")
      k = kpts.get(f"{side}_knee")
      a = kpts.get(f"{side}_ankle")
      if not (h and k and a) or min(h[3], k[3], a[3]) < _VIS_THRESHOLD:
          return 0.0
      midpoint_x = (h[0] + a[0]) / 2.0
      raw = k[0] - midpoint_x
      # Left leg: inward = lower x in image → negate so positive = inward
      # Right leg: inward = higher x in image → keep sign
      return -raw if side == "left" else raw

  # Valgus check: _knee_lateral_deviation(pose_data, "left") > VALGUS_THRESHOLD
  # A value of 0.02–0.03 (in normalised [0,1] x coords) is a meaningful threshold.

LYING / SEATED / KNEELING exercises:
  _trunk_lean_2d() ONLY works when the patient is standing upright.
  For lying, seated, or kneeling exercises it returns ~0 regardless of movement.
  DO NOT use trunk lean as the primary detection signal for these exercises.
  Instead, use the bend helper for the joint being exercised (e.g. _knee_bend_2d
  for lying knee flexion, _elbow_bend_2d for seated arm curl, etc.).
  For lying exercises, set the 'return to straight' threshold generously (e.g. < 20°
  bend) because a relaxed lying leg may have 5-15° of apparent bend from soft tissue.
"""


_FUNCTION_SPEC_TEMPLATE = """\
Implement exactly these five functions (three required, two optional):

def get_instructions() -> list[str]:
    # OPTIONAL. Return 1-2 short sentences that tell the patient:
    # (1) how to stand relative to the camera, and (2) what movement to do.
    # Keep it brief — it is spoken aloud immediately before the countdown starts.

def detect_rep(pose_data: dict) -> bool:
    # REQUIRED. Called every frame during the exercise window.
    # Return True exactly once when a rep attempt is detected. Silent — no feedback.
    # CRITICAL: Do NOT use the physiotherapist's boundary values as detection thresholds.
    # Boundary values are quality targets for feedback — they are NOT gates for counting reps.
    # A patient who squats to 60° when the boundary is 90° MUST still have a rep counted
    # so generate_round_feedback() can coach them on the gap.
    # Use your own LOOSE anatomical thresholds (e.g. any knee bend > 20° counts as a squat
    # attempt, any elbow bend > 20° counts as a curl attempt).
    # Use a simple phase state machine: ready → moving → ready.
    # For exercises where a joint moves away from rest and returns:
    #   Phase "moving" starts when angle crosses a LOOSE threshold in movement direction
    #   Rep counted when angle returns past a separate return threshold
    # Use module-level variables for phase state. reset_round() resets them.

def reset_round():
    # OPTIONAL. Reset detect_rep state variables before each new exercise window.
    # Reset phase back to "ready" or "start", clear any history deques, etc.

def generate_round_feedback(round_data: dict) -> list[str]:
    # REQUIRED. Called once after each exercise window.
    # round_data: {
    #   "round_number": int,
    #   "rep_count": int,
    #   "frames": list[dict],       # pose_data frames collected during exercise window
    #   "duration_seconds": float
    # }
    # Return 2-4 spoken sentences as a list of strings.
    # BOUNDARY VALUES USAGE: The physiotherapist's boundary values are quality targets.
    # Compare the patient's actual measured values (from frames) against those targets
    # and report HOW CLOSE they came. Describe the gap and give specific coaching.
    # NEVER return "no movement detected" or similar if frames contain any recognizable
    # movement attempt — even a partial attempt deserves specific feedback.
    # "No movement" should only appear if frames show literally no joint movement
    # (e.g. the patient walked away from camera or was completely still).
    # - ALWAYS run all quality checks (pelvic tilt, knee valgus, trunk lean, etc.)
    #   regardless of rep_count. Do NOT return early after the rep-count message.
    # - If rep_count is 0 but movement was detected in frames: describe what the patient
    #   did ("you squatted about halfway down") and what they need to do differently
    #   ("bend your knees further — aim for a deeper squat").
    # - If rep_count > 0, open with acknowledgement + depth feedback, then quality checks.
    # - Give verbal coaching cues a physiotherapist would say out loud.
    #   Describe movement quality in plain language ("bend your arms more",
    #   "lean further forward", "snap back upright between each rep").
    #   By default do NOT say raw angle values — they mean nothing to most patients.
    #   EXCEPTION: if the physiotherapist's instructions explicitly ask for angle
    #   values in feedback (e.g. "report the knee angle"), then include them.
    # - Use separate if statements (NOT elif) for independent quality checks.
    # - Keep each sentence concise — they will be spoken aloud.

def get_relevant_joints() -> list:
    # OPTIONAL. Returns which joint values to display prominently on screen during the session.
    # Each entry is a (display_label, pose_data_key) pair.
    # pose_data_key must be a key that exists in the pose_data dict.
    # Return 1-4 joints — the ones most informative for THIS exercise.
    # Example: [("Trunk lean", "trunk_lean_angle"), ("L knee", "left_knee_angle")]

def get_session_summary(session_data: dict) -> str:
    # REQUIRED. Called once when the patient ends the session.
    # session_data: {
    #   "total_reps": int,
    #   "rounds": [{"round_number": int, "rep_count": int, "frames": list[dict],
    #               "duration_seconds": float}, ...],
    #   "duration_seconds": float,
    #   "angle_stats": {angle_name: {"min": float, "max": float}, ...}
    # }
    # "angle_stats" has the min and max of every angle seen across the whole session.
    # REQUIREMENTS:
    # - Only report on joints clinically relevant to THIS exercise.
    # - Use angle_stats to determine what quality level the patient reached.
    #   Phrase feedback as verbal coaching by default — no raw degree values —
    #   unless the physiotherapist's instructions explicitly request angle values.
    # - If total_reps is 0, explain what the patient should do differently.
    # - Ignore any angle_stats entry whose "min" < 10.0 — it is a detection artifact.
    # - Return a single string, 2-4 sentences maximum.
"""

_FEW_SHOT = """\
Example — skiing double-pole from the side (tested pattern; follow its structure closely):

```python
import math
import statistics

# ── thresholds ────────────────────────────────────────────────────────────────
_HINGE_START   = 20    # trunk_lean_angle > this → hinge begins
_HINGE_PEAK    = 40    # trunk_lean_angle >= this → counts as a real stroke
_HINGE_IDEAL   = 50    # trunk_lean_angle >= this → excellent depth
_RETURN        = 15    # trunk_lean_angle <= this → upright again
_HYSTERESIS    = 10    # trunk must drop this much from peak before "extending"
_ELBOW_IDEAL   = 40    # elbow_bend_approx >= this → good arm position
_ARM_RAISE     = 50    # arm_elevation >= this → arms raised high enough
_VIS_THRESHOLD = 0.35  # minimum MediaPipe visibility to trust a joint

# ── module-level rep detection state ─────────────────────────────────────────
_phase         = "ready"   # "ready" | "hinging" | "extending"
_max_trunk     = 0.0
_max_elbow     = 0.0
_peak_reached  = False
_rep_frames    = []        # frames for the current in-progress rep


def _pick_side(keypoints):
    lv = keypoints.get("left_shoulder",  (0, 0, 0, 0))[3]
    rv = keypoints.get("right_shoulder", (0, 0, 0, 0))[3]
    return "left" if lv >= rv else "right"


def _joints_visible(keypoints, side):
    for part in ("shoulder", "hip", "knee", "elbow"):
        kp = keypoints.get(f"{side}_{part}")
        if kp is None or kp[3] < _VIS_THRESHOLD:
            return False
    return True


def _angle_2d(ax, ay, bx, by, cx, cy):
    # Angle at B in the x-y plane. Returns degrees; 180 = straight.
    vax, vay = ax - bx, ay - by
    vcx, vcy = cx - bx, cy - by
    mag = math.hypot(vax, vay) * math.hypot(vcx, vcy)
    if mag < 1e-10:
        return 180.0
    return math.degrees(math.acos(max(-1.0, min(1.0, (vax*vcx + vay*vcy) / mag))))


def _elbow_bend_2d(pose_data, side):
    # Use x,y only — immune to MediaPipe z-depth noise on side-view exercises.
    # Returns bend amount: 0 = straight arm, higher = more bent.
    kpts = pose_data.get("keypoints", {})
    s = kpts.get(f"{side}_shoulder")
    e = kpts.get(f"{side}_elbow")
    w = kpts.get(f"{side}_wrist")
    if not (s and e and w) or min(s[3], e[3], w[3]) < _VIS_THRESHOLD:
        return 0.0
    return 180.0 - _angle_2d(s[0], s[1], e[0], e[1], w[0], w[1])


def _trunk_lean_2d(pose_data, side):
    # Trunk forward lean using x,y only — immune to MediaPipe z-depth noise.
    # Returns degrees from vertical: 0=upright, increases when leaning forward.
    # Always prefer this over pose_data["trunk_lean_angle"] for side-view exercises.
    kpts = pose_data.get("keypoints", {})
    sh = kpts.get(f"{side}_shoulder")
    h  = kpts.get(f"{side}_hip")
    if not (sh and h) or min(sh[3], h[3]) < _VIS_THRESHOLD:
        return pose_data.get("trunk_lean_angle", 0.0)
    dx = sh[0] - h[0]
    dy = sh[1] - h[1]  # negative when upright (shoulder above hip in image)
    dist = math.hypot(dx, dy)
    if dist < 1e-10 or dy >= 0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, -dy / dist))))


def get_instructions():
    return [
        "Stand side-on to the camera so your full body is visible.",
        "When the countdown ends, perform double pole skiing movements — raise your arms high and crunch forward.",
    ]


def detect_rep(pose_data):
    global _phase, _max_trunk, _max_elbow, _peak_reached, _rep_frames

    kpts = pose_data.get("keypoints", {})
    side = _pick_side(kpts)

    if not _joints_visible(kpts, side):
        return False

    trunk  = _trunk_lean_2d(pose_data, side)
    elbow  = _elbow_bend_2d(pose_data, side)

    if _phase == "ready":
        if trunk > _HINGE_START:
            _phase       = "hinging"
            _max_trunk   = trunk
            _max_elbow   = elbow
            _peak_reached = False
            _rep_frames  = [pose_data]

    elif _phase == "hinging":
        _rep_frames.append(pose_data)
        if trunk > _max_trunk:
            _max_trunk = trunk
        if elbow > _max_elbow:
            _max_elbow = elbow
        if not _peak_reached and trunk >= _HINGE_PEAK:
            _peak_reached = True
        if trunk < _max_trunk - _HYSTERESIS:
            _phase = "extending"

    elif _phase == "extending":
        _rep_frames.append(pose_data)
        if trunk <= _RETURN:
            _phase = "ready"
            if _peak_reached:
                _rep_frames = []
                return True
            _rep_frames = []

    return False


def reset_round():
    global _phase, _max_trunk, _max_elbow, _peak_reached, _rep_frames
    _phase        = "ready"
    _max_trunk    = 0.0
    _max_elbow    = 0.0
    _peak_reached = False
    _rep_frames   = []


def generate_round_feedback(round_data):
    rep_count = round_data.get("rep_count", 0)
    frames    = round_data.get("frames", [])

    if not frames:
        return ["I couldn't see you. Make sure your full body is visible side-on to the camera."]

    kpts = frames[0].get("keypoints", {})
    side = _pick_side(kpts)

    trunk_vals = [_trunk_lean_2d(f, side) for f in frames]
    elbow_vals = [_elbow_bend_2d(f, side) for f in frames]
    arm_vals   = [f.get(f"{side}_arm_elevation", 0.0) for f in frames]

    trunk_range = max(trunk_vals) - min(trunk_vals) if trunk_vals else 0.0
    peak_trunk  = max(trunk_vals) if trunk_vals else 0.0
    med_elbow   = statistics.median(elbow_vals) if elbow_vals else 0.0
    max_arm     = max(arm_vals) if arm_vals else 0.0

    lines = []

    if rep_count == 0:
        lines.append("No complete reps were detected that round.")
        if trunk_range < 10.0:
            lines.append(
                "It looks like you didn't move much. "
                "Crunch your upper body forward and snap back upright for each stroke."
            )
        elif peak_trunk < _HINGE_PEAK:
            lines.append(
                "You leaned forward but not enough to count as a rep. "
                "Drive your chest further toward your knees."
            )
        else:
            lines.append(
                "You leaned forward but didn't return fully upright between strokes. "
                "Snap back tall after each pole push so the rep counts."
            )
        return lines

    rep_word = "rep" if rep_count == 1 else "reps"
    lines.append(f"Good effort — you completed {rep_count} {rep_word}.")

    if peak_trunk >= _HINGE_IDEAL:
        lines.append("Your forward crunch was deep and powerful — great technique.")
    else:
        lines.append(
            "You could lean forward more during the stroke — "
            "drive your chest closer to your thighs for more power."
        )

    if med_elbow >= _ELBOW_IDEAL:
        lines.append("Good arm position — your elbows were nicely bent as strong levers.")
    else:
        lines.append(
            "Keep your elbows more bent during the stroke — "
            "they should act as stiff levers, not loose and extended."
        )

    if max_arm >= _ARM_RAISE:
        lines.append("You raised your arms high, giving a full powerful arc — well done.")
    else:
        lines.append(
            "Raise your arms higher before each stroke — hands up to forehead level "
            "for a full arc and more power."
        )

    return lines


def get_relevant_joints():
    return [
        ("Trunk lean", "trunk_lean_angle"),
        ("L elbow",    "left_elbow_angle"),
        ("R elbow",    "right_elbow_angle"),
        ("L arm elev", "left_arm_elevation"),
    ]


def get_session_summary(session_data):
    total_reps = session_data.get("total_reps", 0)
    stats      = session_data.get("angle_stats", {})

    peak_trunk = stats.get("trunk_lean_angle", {}).get("max", 0.0)
    l_arm      = stats.get("left_arm_elevation",  {}).get("max", 0.0)
    r_arm      = stats.get("right_arm_elevation", {}).get("max", 0.0)
    best_arm   = max(l_arm, r_arm)

    parts = []

    if total_reps == 0:
        parts.append("No complete double-pole reps were recorded this session.")
        if peak_trunk < _HINGE_PEAK:
            parts.append(
                "Focus on leaning your upper body further forward — "
                "drive your chest down toward your knees on each stroke."
            )
        return " ".join(parts)

    rep_word = "rep" if total_reps == 1 else "reps"
    parts.append(f"Session complete — you performed {total_reps} {rep_word} across all rounds.")

    if peak_trunk >= _HINGE_IDEAL:
        parts.append("Your forward lean was excellent — really powerful technique throughout.")
    elif peak_trunk > 10.0:
        parts.append(
            "Work on leaning further forward during each stroke — "
            "drive your chest closer to your knees for more power."
        )

    if best_arm >= _ARM_RAISE:
        parts.append("Good arm height — you raised your arms high before each stroke.")
    elif best_arm > 10.0:
        parts.append(
            "Try to raise your arms higher before each stroke — "
            "hands up to forehead level so you get a full powerful arc."
        )

    return " ".join(parts)
```
"""


def _build_function_spec(session_duration_secs: int) -> str:
    return _FUNCTION_SPEC_TEMPLATE.replace("10-second", f"{session_duration_secs}-second")


_DECREASING_ANGLE_JOINTS = {
    "left_elbow_angle", "right_elbow_angle",
    "left_knee_angle", "right_knee_angle",
    "left_hip_angle", "right_hip_angle",
}
_BENDING_JOINT_MIN_RANGE = 60  # degrees — below this the demo likely didn't show full range


def _format_reference_data(reference_data: dict) -> str:
    # Sort by range descending so the most active joints appear first
    sorted_joints = sorted(reference_data.items(), key=lambda kv: kv[1]["range"], reverse=True)
    lines = [
        "REFERENCE VIDEO ANALYSIS",
        "========================",
        "The therapist recorded a demo of the exact movement the patient should perform.",
        "Pose estimation was run on that video to measure which joints moved and how much.",
        "The joints below are listed from most active to least active.",
        "Joints NOT listed barely moved — they are not part of this exercise.",
        "",
        "Use this data to understand the exercise biomechanics and set your thresholds:",
        "",
    ]
    warnings = []
    for joint, stats in sorted_joints:
        rng = stats["range"]
        mn  = stats["min"]
        mx  = stats["max"]
        note = ""
        if joint in _DECREASING_ANGLE_JOINTS:
            # Convert to bend convention (0=straight) for bending joints
            bend_min = round(180 - mx)  # raw max → least bend
            bend_max = round(180 - mn)  # raw min → most bend
            suffix = f"  (bend: {bend_min}-{bend_max} deg, 0=straight convention)"
            if rng < _BENDING_JOINT_MIN_RANGE:
                note = (
                    f"  *** WARNING: bend range only {rng:.0f}° — demo likely did NOT show full range. "
                    f"Use clinical defaults instead (see below). ***"
                )
                warnings.append((joint, bend_min, bend_max, rng))
        else:
            suffix = ""
        lines.append(
            f"  {joint}: min={mn:.0f}°  max={mx:.0f}°  range={rng:.0f}°{suffix}"
        )
        if note:
            lines.append(note)

    lines += [
        "",
        "THRESHOLD CALIBRATION (use bend convention: 0=straight, higher=more bent):",
        "",
        "For joints whose value INCREASES with movement (trunk lean, arm elevation, bend amount):",
        "  'movement started' threshold = min_value + 30% of range",
        "  'full rep' threshold         = min_value + 60% of range",
        "",
        "For bending joints — use the bend values shown above (180 − raw_angle):",
        "  'movement started' = bend_min + 30% of bend_range",
        "  'full rep'         = bend_min + 60% of bend_range",
        "",
    ]

    if warnings:
        lines += [
            "*** INCOMPLETE DEMO OVERRIDE ***",
            "The following bending joints had a suspiciously small range — the demo video",
            "did not show the full movement. Use these CLINICAL defaults (bend convention) instead:",
        ]
        for joint, bend_min, bend_max, rng in warnings:
            if "elbow" in joint:
                lines.append(
                    f"  {joint}: ignore reference (only {rng:.0f}° range). "
                    f"Use: 'movement started' at ~30° bend, 'full bend' at ~80° bend."
                )
            elif "knee" in joint:
                lines.append(
                    f"  {joint}: ignore reference (only {rng:.0f}° range). "
                    f"Use: 'movement started' at ~25° bend, 'full bend' at ~80° bend."
                )
            elif "hip" in joint:
                lines.append(
                    f"  {joint}: ignore reference (only {rng:.0f}° range). "
                    f"Use: 'movement started' at ~20° bend, 'full bend' at ~80° bend."
                )
        lines.append("")

    lines += [
        "FEEDBACK LABEL ACCURACY:",
        "  The verbal feedback you generate must match the actual thresholds.",
        "  - 'hand close to shoulder' requires elbow bend threshold ~80°+ (very bent).",
        "  - 'deep squat' / 'deep bend' requires knee bend threshold ~80°+.",
        "  - Do NOT praise an achievement the threshold does not actually require.",
        "",
        "Do NOT change which angles you use as the primary detection signal — follow the",
        "example's structural pattern (trunk_lean_2d for forward-lean exercises, etc.).",
    ]
    return "\n".join(lines)


def build_prompt(exercise_name: str, camera_view: str,
                 client_instructions: str,
                 llm_instructions: str,
                 boundary_values: str,
                 display_values: str,
                 session_duration_secs: int,
                 reference_data: dict | None = None) -> str:
    ref_section = ""
    if reference_data:
        ref_section = "\n" + _format_reference_data(reference_data) + "\n"
    function_spec = _build_function_spec(session_duration_secs)
    return f"""\
You are generating a Python movement analysis module for a physiotherapy application.

Exercise: {exercise_name}
Camera view: {camera_view}

Client instructions (spoken to patient before and during exercise):
{client_instructions}

Analysis instructions (what to look for and what feedback to give):
{llm_instructions}

Boundary values (quality targets for generate_round_feedback — NOT thresholds for detect_rep. detect_rep must count any recognizable movement attempt using its own loose anatomical thresholds, independently of these values):
{boundary_values}

Display values (which values to show on screen during motion analysis):
{display_values}

Session duration: {session_duration_secs} seconds per exercise window

Available pose data fields:
{_POSE_DATA_DESCRIPTION}
{_FEW_SHOT}

Now generate the analysis module for '{exercise_name}' following the same structure as the example above.
{ref_section}
{function_spec}

Rules:
- Only import math, statistics, collections, itertools, or functools if needed. Do NOT import os, sys, subprocess, socket, or requests.
- Use module-level variables for state (rep phase tracking, etc.).
- Implement get_instructions, detect_rep, reset_round, generate_round_feedback, and get_session_summary. The required functions are detect_rep, generate_round_feedback, and get_session_summary.
- Feedback language: use plain verbal coaching by default. Only include numeric angle values if the physiotherapist's instructions explicitly request them.
- For side-view exercises, compute elbow/hip/knee bend AND trunk lean from keypoints x,y only (like _elbow_bend_2d and _trunk_lean_2d in the example). Do NOT use left/right_elbow_angle, left/right_hip_angle, or trunk_lean_angle directly — they are 3D and z-depth noise inflates them in side view.
- Use a CONSISTENT angle convention across all helpers: 0° = fully straight, higher = more bent/flexed. Always compute bend amount as (180° − raw_angle) so thresholds are comparable regardless of exercise. E.g. knee_bend_2d = 180 − knee_2d_angle, hip_bend_2d = 180 − hip_2d_angle.
- For frontal-view exercises: do NOT use pose_data["pelvic_tilt"] — compute _pelvic_tilt_2d from keypoints instead. Do NOT use hka_alignment < threshold to detect knee valgus — it is a scalar angle that fires for both valgus AND varus. Use _knee_lateral_deviation from keypoints (positive = inward) and only flag valgus when the value is actually positive above a threshold.
- For lying, seated, or kneeling exercises: do NOT use trunk lean as the primary detection signal. Use the bend helper for the joint being exercised (_knee_bend_2d, _elbow_bend_2d, etc.). Set the 'return to straight' threshold at ~15-20° bend to account for natural resting position noise.
- Implement get_relevant_joints() returning 1-4 (label, pose_data_key) pairs for the joints most relevant to this exercise. Use keys that exist in pose_data (e.g. "left_knee_angle", "trunk_lean_angle", "left_arm_elevation").
- Return ONLY valid Python code. No markdown fences. No explanations.
"""


class LLMService:
    def __init__(self, conn: sqlite3.Connection, api_key: str):
        self.conn = conn
        self.ex_svc = ExerciseService(conn)
        self._client = anthropic.Anthropic(api_key=api_key)

    def generate_module(self, exercise_id: str, name: str, camera_view: str,
                        client_instructions: str,
                        llm_instructions: str,
                        boundary_values: str,
                        display_values: str,
                        session_duration_secs: int) -> dict:
        """Call Claude, validate the result, store it. Returns the saved module dict."""
        reference_data = None
        ex = self.ex_svc.get(exercise_id)
        if ex and ex.reference_video_path:
            try:
                reference_data = analyze_reference_video(ex.reference_video_path) or None
            except Exception:
                reference_data = None

        prompt = build_prompt(
            exercise_name=name,
            camera_view=camera_view,
            client_instructions=client_instructions,
            llm_instructions=llm_instructions,
            boundary_values=boundary_values,
            display_values=display_values,
            session_duration_secs=session_duration_secs,
            reference_data=reference_data,
        )
        response_text = ""
        status = "failed"
        for attempt in range(1, 3):
            try:
                message = self._client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=8192,
                    messages=[{"role": "user", "content": prompt}],
                )
                response_text = message.content[0].text.strip()
                validation = validate_module(response_text)
                if validation["valid"]:
                    status = "validated"
                    break
                print(f"[llm_service] attempt {attempt} validation failed: {validation['error']}")
            except Exception as e:
                print(f"[llm_service] attempt {attempt} API error: {e}")
                response_text = str(e)

        self._log(exercise_id, prompt, response_text)
        return self.ex_svc.save_module(exercise_id, response_text, status)

    def _log(self, exercise_id: str, prompt: str, response: str):
        self.conn.execute(
            "INSERT INTO llm_logs (id, exercise_id, prompt, response) VALUES (?, ?, ?, ?)",
            (str(uuid.uuid4()), exercise_id, prompt, response),
        )
        self.conn.commit()
