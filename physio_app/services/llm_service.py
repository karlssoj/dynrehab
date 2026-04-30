import sqlite3
import uuid
import anthropic
from core.module_validator import validate_module
from core.video_analyzer import analyze_reference_video
from physio_app.services.exercise_service import ExerciseService

_POSE_DATA_DESCRIPTION = """\
All angles are in degrees, computed from x,y coordinates only (z ignored).
Fields marked [SIDE] are most informative from a side-view camera.
Fields marked [FRONT] are most informative from a front-view camera.
Fields marked [BOTH] are useful from either view.

--- JOINT ANGLES (raw, ~180°=straight, decreases as joint bends) ---
  left_knee_angle / right_knee_angle [BOTH] — ~170° standing, ~90° deep squat
  left_hip_angle / right_hip_angle [BOTH] — ~180° standing, ~90° hip flexion
  left_ankle_angle / right_ankle_angle [BOTH] — knee-ankle-foot_index angle
  left_shoulder_angle / right_shoulder_angle [BOTH] — elbow-shoulder-hip angle
  left_elbow_angle / right_elbow_angle [BOTH] — ~180° straight, ~45° fully bent
  left_wrist_angle / right_wrist_angle [BOTH] — elbow-wrist-index_finger angle
  neck_angle [BOTH] — angle at shoulder-midpoint between trunk and nose direction
  left_hka_alignment / right_hka_alignment [FRONT] — hip-knee-ankle angle ~180°=straight.
    WARNING: SCALAR — decreases for BOTH valgus and varus. Do NOT use to detect direction.
    Use left_knee_valgus / right_knee_valgus instead.

--- BEND VALUES (0°=straight, increases as joint bends — prefer these over raw angles) ---
  left_elbow_bend_2d / right_elbow_bend_2d [BOTH] — 0°=straight, ~135°=fully curled.
  left_knee_bend_2d / right_knee_bend_2d [BOTH] — 0°=straight, ~90°=deep squat.
    Also reliable for FRONT-view squats: hip descent in y captures depth.
  left_hip_bend_2d / right_hip_bend_2d [BOTH] — 0°=upright, increases as hip flexes.

--- SEGMENT-FROM-VERTICAL ANGLES (0°=segment vertical, increases as it tilts) ---
  trunk_lean_angle / trunk_lean_2d [BOTH] — trunk lean from vertical.
    SIDE view: measures forward lean. FRONT view: measures lateral lean (side-bend).
  left_shin_angle / right_shin_angle [SIDE] — shin (ankle→knee) from vertical.
    0°=shin vertical, ~20-30°=typical squat lean. Key for squat/lunge form.
  left_thigh_angle / right_thigh_angle [SIDE] — thigh (knee→hip) from vertical.
    0°=standing, 45°=quarter squat, 90°=thigh horizontal (parallel squat).
    This is the most direct squat depth measure from a side-view camera.
  left_arm_elevation / right_arm_elevation [BOTH] — arm from trunk line at shoulder.
    0°=arm at side, 90°=arm horizontal, 180°=arm overhead.

--- BILATERAL TILT ANGLES (0°=level, increases as one side is higher than other) ---
  pelvic_tilt [FRONT] — left-hip→right-hip line from horizontal.
    0°=hips level. For frontal exercises use this to detect hip drop (Trendelenburg).
    WARNING: reads ~180° on front camera when hips are level — use helper below instead.
  shoulder_tilt [FRONT] — left-shoulder→right-shoulder line from horizontal.
    0°=shoulders level. Detects dropped shoulder, useful for overhead press, scoliosis.

--- SIGNED VALGUS ANGLE ---
  left_knee_valgus / right_knee_valgus [FRONT] — signed angular deviation of knee from hip-ankle line (degrees).
    0° = hip, knee, ankle perfectly collinear (straight alignment).
    Positive = valgus (knee inward/medial = knees caving in).
    Negative = varus (knee outward/lateral = knees bowing out).
    Threshold: ±3° = clinically meaningful. ±5° = clearly visible. ±10° = severe.
    IMPORTANT: NEVER use abs() — the sign is the direction. Use it like this:
      if pose_data["left_knee_valgus"] > 5:
          # knee caving inward (valgus)
      elif pose_data["left_knee_valgus"] < -5:
          # knee bowing outward (varus)
      # values between -5 and +5 are acceptable alignment

--- KEYPOINTS ---
  keypoints: dict[str, tuple[float,float,float,float]] — name→(x,y,z,visibility)
    x,y in [0,1] (y increases downward), visibility in [0,1]
    landmarks: nose, left/right_shoulder, left/right_elbow, left/right_wrist,
    left/right_hip, left/right_knee, left/right_ankle,
    left/right_foot_index, left/right_heel

ANGLE CONVENTION (use consistently across ALL exercises):
  0° = fully straight / fully extended joint
  Higher values = more bent / more flexed
  Use bend amount = 180° − raw_angle (pre-computed as _bend_2d fields).
  Example: raw knee angle 90° → knee_bend = 90° (deeply bent).

HELPER for frontal pelvic tilt (use instead of pose_data["pelvic_tilt"] from front camera):
  def _pelvic_tilt_2d(pose_data):
      kpts = pose_data.get("keypoints", {})
      lh = kpts.get("left_hip"); rh = kpts.get("right_hip")
      if not (lh and rh) or min(lh[3], rh[3]) < _VIS_THRESHOLD: return 0.0
      dx = abs(lh[0] - rh[0]); dy = abs(lh[1] - rh[1])
      return 0.0 if dx < 1e-10 else math.degrees(math.atan2(dy, dx))

HELPER for custom angles (x,y only — use for ankle bend or any unlisted angle):
  def _angle_2d(ax, ay, bx, by, cx, cy):
      # Angle at B. Returns ~180° when straight, decreases as bent.
      vax,vay = ax-bx, ay-by; vcx,vcy = cx-bx, cy-by
      mag = math.hypot(vax,vay)*math.hypot(vcx,vcy)
      if mag < 1e-10: return 180.0
      return math.degrees(math.acos(max(-1.0, min(1.0, (vax*vcx+vay*vcy)/mag))))

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

FRONTAL-VIEW VALGUS — use pre-computed fields, do NOT recompute from keypoints:
  left_knee_valgus / right_knee_valgus are already the signed angular deviation.
  Positive = valgus (knee caving inward), Negative = varus (knee bowing outward).
  Example usage:
    if pose_data["left_knee_valgus"] > 5:
        # knee caving inward (valgus)
    elif pose_data["left_knee_valgus"] < -5:
        # knee bowing outward (varus)
  NEVER use hka_alignment to detect direction — it is unsigned and fires for both.
  NEVER recompute from keypoints — left_knee_valgus / right_knee_valgus are correct and ready.

SIDE-VIEW SPECIFIC — knees over toes / shin forward lean:
  left_shin_angle / right_shin_angle = angle of shin (ankle→knee) from vertical.
  0° = shin perfectly vertical (knee directly above ankle).
  Increases as the knee travels forward past the toes.
  Typical acceptable range during squats: up to ~30°. Above ~35-40° = knees too far forward.
  This is the ONLY reliable field for detecting knees-over-toes. It requires a SIDE-VIEW camera.
  From a FRONT-VIEW camera, shin_angle measures lateral shin tilt (not forward lean) and
  CANNOT be used to detect knees over toes — do NOT attempt this from a front view.
  Example side-view usage:
    avg_shin = statistics.mean(f["left_shin_angle"] for f in frames if f["left_shin_angle"] > 1)
    if avg_shin > 35:
        feedback.append("Your knees are travelling too far over your toes — shift your weight back.")

SIDE-VIEW SPECIFIC — heel rise detection:
  Heel rise cannot be read from a pre-computed field. Compute it per-frame from keypoints:
    def _heel_rise(pose_data, side="auto"):
        kpts = pose_data.get("keypoints", {})
        # Pick the side with better visibility, or use the specified side.
        if side == "auto":
            lh = kpts.get("left_heel"); rh = kpts.get("right_heel")
            lf = kpts.get("left_foot_index"); rf = kpts.get("right_foot_index")
            l_vis = min(lh[3] if lh else 0, lf[3] if lf else 0)
            r_vis = min(rh[3] if rh else 0, rf[3] if rf else 0)
            side = "left" if l_vis >= r_vis else "right"
        heel = kpts.get(f"{side}_heel")
        foot_index = kpts.get(f"{side}_foot_index")
        if not heel or not foot_index:
            return 0.0
        # *** IMPORTANT: use 0.20, NOT _VIS_THRESHOLD, for foot landmarks.
        # From a side-view camera MediaPipe assigns heel/foot_index visibility of only
        # 0.20–0.45 even when the foot is clearly visible. Using _VIS_THRESHOLD (typically
        # 0.35–0.50) causes this function to silently return 0.0 on almost every frame.
        if min(heel[3], foot_index[3]) < 0.20:
            return 0.0
        # y increases downward. When the heel lifts, heel_y decreases relative to foot_index_y.
        # rise > 0 means the heel is above the ball of the foot (heel has risen off the ground).
        return foot_index[1] - heel[1]
  Interpretation (SIDE VIEW only — unreliable from front/back view):
    The values are small because both landmarks are near the bottom of the frame:
    ~0.00–0.01 = flat foot
    > 0.02     = noticeable heel rise (~5 cm off the ground) — USE THIS as the detection threshold
    > 0.03     = clear heel rise (~7–8 cm)
    *** NEVER use 0.04 or higher — that requires ~10 cm of heel lift which is extreme. ***
  Detect during the downward phase only (e.g., while knee is bending):
    rises = [_heel_rise(f) for f in frames if f.get("left_knee_bend_2d", 0) > 20]
    if rises and max(rises) > 0.02:   # 0.02 ≈ 5 cm heel rise — DO NOT raise this
        feedback.append("Your heels were lifting off the ground — work on ankle mobility.")

LOWER-BODY SIDE-VIEW HELPERS — copy these verbatim for squat, lunge, deadlift, step-up, calf raise:
  These three helpers work together. _depth_frames() filters to frames where the knee is
  meaningfully bent so heel rise and shin checks are only evaluated during the actual movement,
  not while the patient is standing still.

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

  Usage pattern in generate_rep_cue (copy this for the knees-over-toes + heel-rise checks):
    bent_frames = _depth_frames(frames)
    # 1. Highest priority: heel rise
    if bent_frames:
        rises = [_heel_rise(f) for f in bent_frames]
        if rises and max(rises) > 0.02:   # 0.02 ≈ 5 cm — do not raise threshold
            return "Keep your heels flat on the floor as you squat down."
    # 2. Next: knees too far forward
    avg_shin = _avg_shin_angle(bent_frames if bent_frames else frames)
    if avg_shin > 30:
        return "Push your hips back — your knees are travelling too far over your toes."

LYING / SEATED / KNEELING exercises:
  _trunk_lean_2d() ONLY works when the patient is standing upright.
  For lying, seated, or kneeling exercises it returns ~0 regardless of movement.
  DO NOT use trunk lean as the primary detection signal for these exercises.
  Instead, use the bend helper for the joint being exercised (e.g. _knee_bend_2d
  for lying knee flexion, _elbow_bend_2d for seated arm curl, etc.).
  For lying exercises, set the 'return to straight' threshold generously (e.g. < 20°
  bend) because a relaxed lying leg may have 5-15° of apparent bend from soft tissue.
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


def _build_function_spec(session_duration_secs: int, feedback_mode: list) -> str:
    modes = set(feedback_mode) if feedback_mode else {"after_window"}

    required = []
    if "during_exercise" in modes:
        required.append("generate_rep_cue")
    if "after_window" in modes:
        required.append("generate_round_feedback")
    if "after_rep" in modes:
        required.append("generate_rep_feedback")
    if "after_exercise" in modes:
        required.append("get_session_summary")

    parts = [f"Implement these functions (required: {', '.join(required) or 'detect_rep'}):\n"]

    parts.append("""\
def get_instructions() -> list[str]:
    # OPTIONAL. Return 1-2 short sentences spoken aloud before countdown.
    # (1) where to stand, (2) what movement to do.

def detect_rep(pose_data: dict) -> bool:
    # REQUIRED. Called every frame during exercise.
    # Return True exactly once per completed rep attempt.
    # Use LOOSE anatomical thresholds — count any recognisable attempt.
    # Use a phase state machine: ready → moving → ready.
    # Use module-level variables. reset_round() resets them.

def reset_round():
    # OPTIONAL. Reset detect_rep state before each new window / rep cycle.
""")

    if "during_exercise" in modes:
        parts.append("""\
def generate_rep_cue(cue_data: dict) -> str:
    # REQUIRED (during_exercise mode). Called after each completed rep, and after
    # 5 seconds with no rep detected.
    # cue_data: {
    #   "trigger": "rep" | "timeout",
    #   "rep_number": int,
    #   "round_number": int | None,
    #   "frames": list[dict]   # recent pose frames
    # }
    # Return EXACTLY ONE short spoken sentence. Patient is still moving.
    #
    # "rep" trigger — PRIORITY RULE: check all issues, then speak only the MOST
    # clinically important one (highest injury risk). Never list multiple issues
    # in a single cue — the patient is mid-exercise and cannot act on more than
    # one correction at a time.
    #
    # Severity ranking principle (apply to the specific exercise at hand):
    #   1. Joint-loading / injury-risk errors (e.g., knees tracking far over toes,
    #      heel rise, valgus collapse, hyperextension) — speak these first.
    #   2. Depth / range-of-motion errors (e.g., not squatting deep enough,
    #      insufficient elbow bend) — speak if no higher-priority issue present.
    #   3. Posture / alignment secondary notes (e.g., mild trunk lean, slight
    #      shoulder shrug) — speak only if everything above is acceptable.
    #   4. No issues detected → give brief encouragement or a positive cue.
    #
    # Implementation pattern:
    #   frames = cue_data.get("frames", [])
    #   if frames:
    #       if <highest-severity check fails>:
    #           return "<injury-risk correction>"
    #       if <depth check fails>:
    #           return "<depth correction>"
    #       if <posture check fails>:
    #           return "<posture cue>"
    #   return "Good rep!"   # all checks passed
    #
    # "timeout" trigger: brief encouragement or movement reminder (no severity ranking needed).
    # Examples: "Good squat!", "Go deeper!", "Keep going!", "Lean forward more!", "Heels down!"
    # Do NOT return multiple sentences. Do NOT mention raw angle values.
    # Do NOT save all-issue coverage for this function — that belongs in generate_round_feedback.
""")

    if "after_window" in modes:
        parts.append(f"""\
def generate_round_feedback(round_data: dict) -> list[str]:
    # REQUIRED (after_window mode). Called once after each {session_duration_secs}-second
    # exercise window. Exercise pauses while feedback is spoken.
    # round_data: {{
    #   "round_number": int,
    #   "rep_count": int,
    #   "frames": list[dict],
    #   "duration_seconds": float
    # }}
    # Return a list of spoken sentences. Use as many sentences as needed.
    # - Always open with rep count + one specific positive observation.
    # - Cover ALL quality dimensions (depth, alignment, posture, timing, etc.) —
    #   not just the most severe. The patient has stopped moving and can absorb
    #   a full critique. Use a separate if statement per independent check.
    # - Add corrective cues only for issues that actually occurred.
    # - If rep_count is 0: NEVER give positive form feedback — you have no completed rep
    #   to evaluate. Open with "No reps were completed this round." If movement was
    #   detected (frames non-empty), describe what was missing (e.g. didn't reach depth).
    # - Use separate if statements (NOT elif) for independent quality checks.
    # - Give verbal coaching a physiotherapist would say aloud.
    # - By default do NOT report raw angle values unless instructions explicitly ask.
""")

    if "after_rep" in modes:
        parts.append("""\
def generate_rep_feedback(rep_data: dict) -> list[str]:
    # REQUIRED (after_rep mode). Called once after each completed rep.
    # Exercise is paused while this feedback is spoken — give a full per-rep critique.
    # rep_data: {
    #   "rep_number": int,      # total reps completed so far (1-indexed)
    #   "round_number": int,    # current round number
    #   "frames": list[dict]    # recent pose frames (last ~2 seconds)
    # }
    # Return a list of spoken sentences. 2-4 sentences — brief but complete.
    # - Open with a short rep acknowledgment (e.g., "Rep 3 done.").
    # - Cover the most important quality dimension first (depth, alignment, posture).
    # - Give a corrective cue only if an issue was clearly present in these frames.
    # - If no issues: give brief encouragement and one positive observation.
    # - NEVER report raw angle values unless instructions explicitly request them.
    # - Severity order same as generate_rep_cue: injury risk > depth > posture.
    # - Use separate if statements (NOT elif) for independent quality checks.
""")

    if "after_exercise" in modes:
        parts.append("""\
def get_session_summary(session_data: dict) -> list[str]:
    # REQUIRED (after_exercise mode). Called once when patient ends the session.
    # session_data: {
    #   "total_reps": int,
    #   "total_duration_seconds": float,
    #   "rounds": list[dict],               # [{round_number, rep_count, frames, duration_seconds}, ...]
    #   "rep_cues": list[str],              # cues spoken in during_exercise mode
    #   "round_feedback": list[list[str]],  # per-round feedback spoken after each window
    #   "angle_stats": {name: {"min": float, "max": float}, ...}  # GLOBAL — do NOT use for temporal claims
    # }
    # Return a list of spoken sentences synthesising the full session arc.
    #
    # *** DO NOT USE angle_stats FOR TEMPORAL/PROGRESSION CLAIMS ***
    # angle_stats holds global min/max across the ENTIRE session. It CANNOT tell you whether
    # performance improved, declined, or stayed consistent across rounds. Never say things like
    # "your depth was good throughout" based on angle_stats — you cannot know that from it.
    #
    # *** HOW TO DETECT PROGRESSION HONESTLY ***
    # Compute the primary angle per round from rounds[i]["frames"], then compare:
    #   round_peaks = []
    #   for r in session_data["rounds"]:
    #       vals = [f.get("KEY_ANGLE", 0) for f in r["frames"] if f.get("KEY_ANGLE", 0) > 5]
    #       round_peaks.append(max(vals) if vals else 0)
    #   if len(round_peaks) >= 2:
    #       if round_peaks[-1] > round_peaks[0] + 10:
    #           lines.append("Your depth improved across the session.")
    #       elif round_peaks[0] > round_peaks[-1] + 10:
    #           lines.append("Your depth was better in the early rounds.")
    #       elif all(p >= THRESHOLD for p in round_peaks):
    #           lines.append("You maintained good depth throughout.")  # only if ALL rounds confirm it
    #       # else: omit the temporal claim entirely
    # Replace KEY_ANGLE with the same primary angle used in detect_rep.
    #
    # Use round_feedback as secondary evidence:
    # - If an issue is raised in round_feedback[0] but absent in round_feedback[-1] → patient improved
    # - If the same issue appears in every round's feedback → persistent problem that needs work
    #
    # Requirements:
    # - Name specific rounds when describing change ("in round 1", "by round 2")
    # - NEVER say "throughout" or "consistently" unless per-round data confirms every round
    # - NEVER claim improvement unless per-round data shows a measurable increase
    # - Always mention at least one specific positive finding from the data
    # - CRITICAL — zero reps: if total_reps == 0, you have NO movement data to comment
    #   positively on. NEVER say form was good, depth was fine, or anything praising
    #   technique. Always open with "No reps were completed this session." then give
    #   specific, actionable guidance for what to do differently next time.
    # - Use verbal coaching language; no raw angle values unless instructions request them
""")

    parts.append("""\
def get_relevant_joints() -> list:
    # OPTIONAL. Returns [(display_label, pose_data_key), ...] for sidebar display.
    # Return only joints listed in Display Values (1-2 if none specified).
    # Labels max 12 chars. Keys must exist in pose_data.
""")

    return "\n".join(parts)


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
                 reference_data: dict | None = None,
                 feedback_mode: list | None = None) -> str:
    modes = set(feedback_mode) if feedback_mode else {"after_window"}
    ref_section = ""
    if reference_data:
        ref_section = "\n" + _format_reference_data(reference_data) + "\n"
    function_spec = _build_function_spec(session_duration_secs, list(modes))
    # Include the few-shot example only for after_window mode (it demonstrates that function)
    few_shot_section = _FEW_SHOT if "after_window" in modes else ""
    # Boundary values label: only mention the feedback function when relevant
    boundary_label = (
        "Boundary values (quality targets for the feedback functions — NOT thresholds for "
        "detect_rep. detect_rep must count any recognizable movement attempt using its own "
        "loose anatomical thresholds, independently of these values):"
    )
    return f"""\
You are generating a Python movement analysis module for a physiotherapy application.

Exercise: {exercise_name}
Camera view: {camera_view}

Client instructions (spoken to patient before and during exercise):
{client_instructions}

Analysis instructions (what to look for and what feedback to give):
{llm_instructions}

{boundary_label}
{boundary_values}

Display values (which values to show on screen during motion analysis):
{display_values}

Available pose data fields:
{_POSE_DATA_DESCRIPTION}
{few_shot_section}

Now generate the analysis module for '{exercise_name}' following the same structure as the example above.
{ref_section}
{function_spec}

Rules:
- Only import math, statistics, collections, itertools, or functools if needed. Do NOT import os, sys, subprocess, socket, or requests.
- Use module-level variables for state (rep phase tracking, etc.).
- Implement exactly the functions listed in the function spec above. detect_rep is always required.
- Feedback language: use plain verbal coaching by default. Only include numeric angle values if the physiotherapist's instructions explicitly request them.
- All angles in pose_data are computed from x,y only (z is ignored). You may use left/right_elbow_angle, trunk_lean_angle etc. directly — they are already 2D. Prefer the pre-computed _bend_2d fields (0=straight convention) wherever available.
- Use a CONSISTENT angle convention across all helpers: 0° = fully straight, higher = more bent/flexed. Always compute bend amount as (180° − raw_angle).
- For frontal-view exercises: do NOT use pose_data["pelvic_tilt"] — compute _pelvic_tilt_2d from keypoints instead. Do NOT use hka_alignment to detect knee valgus direction — it is unsigned and fires for both valgus and varus. Use pose_data["left_knee_valgus"] / pose_data["right_knee_valgus"] directly (positive = inward/valgus, negative = outward/varus). Do NOT attempt to detect knees-over-toes from a front view — it requires a side-view camera and shin_angle.
- For lying, seated, or kneeling exercises: do NOT use trunk lean as the primary detection signal. Use the bend helper for the joint being exercised (_knee_bend_2d, _elbow_bend_2d, etc.). Set the 'return to straight' threshold at ~15-20° bend to account for natural resting position noise.
- Implement get_relevant_joints() returning 1-4 (label, pose_data_key) pairs for the joints most relevant to this exercise. Use keys that exist in pose_data (e.g. "left_knee_angle", "trunk_lean_angle", "left_arm_elevation").
- Implement ONLY the checks the physiotherapist's instructions ask for. Do NOT add extra checks
  not mentioned in the instructions. Let the instructions drive what you detect and report.
- When the instructions mention heel rise, heels lifting, or ankle mobility: include the
  _heel_rise() helper copied verbatim from the LOWER-BODY SIDE-VIEW HELPERS section above, and
  use _depth_frames() to check it only during the bent phase. Use threshold > 0.04 unless the
  instructions specify otherwise. NEVER attempt to detect heel rise from keypoint y-positions
  directly — use only the _heel_rise() helper.
- When the instructions mention knees over toes, shin angle, or forward knee travel: include
  _best_knee_bend(), _depth_frames(), and _avg_shin_angle() copied verbatim from the
  LOWER-BODY SIDE-VIEW HELPERS section above. Use threshold > 30° unless instructions specify
  otherwise. NEVER compare knee x-position to foot_index x-position — this is geometrically
  unreliable. ALWAYS use left_shin_angle / right_shin_angle from pose_data.
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
                        session_duration_secs: int,
                        feedback_mode: list | None = None) -> dict:
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
            feedback_mode=feedback_mode or ["after_window"],
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
                validation = validate_module(response_text, feedback_mode=feedback_mode or ["after_window"])
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
