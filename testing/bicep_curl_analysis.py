"""
Skiing double-pole movement analysis — live webcam coaching version.

CAMERA: Stand SIDE-ON (shoulder pointing at the lens).

HOW IT WORKS:
  1. Instructions are shown first — waits until fully spoken.
  2. A 5-to-0 countdown starts.
  3. You perform for 10 seconds (silent analysis).
  4. Short, honest coaching feedback — waits until ALL audio finishes.
  5. Another countdown begins — repeats until you quit.

Run:  python skiing_coach.py
Quit: press 'q' in the video window.
"""

import sys
import os
import time
import math
import statistics
import collections
import cv2
import mediapipe as mp

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from core.angle_calculator import calculate_angles
from core.pose_engine import LANDMARK_NAMES
from client_app.services.tts_service import TTSService


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _angle_2d(a: tuple, b: tuple, c: tuple) -> float:
    ax, ay = a[0] - b[0], a[1] - b[1]
    cx, cy = c[0] - b[0], c[1] - b[1]
    dot = ax * cx + ay * cy
    mag = math.hypot(ax, ay) * math.hypot(cx, cy)
    if mag < 1e-10:
        return 180.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))


def _bend_angle(raw_angle: float) -> float:
    """Convert anatomical angle (180=straight) to bend angle (0=straight)."""
    return 180.0 - raw_angle


def _pick_side(keypoints: dict) -> str:
    lv = keypoints.get("left_shoulder", (0, 0, 0, 0))[3]
    rv = keypoints.get("right_shoulder", (0, 0, 0, 0))[3]
    return "left" if lv >= rv else "right"


def _elbow_bend_2d(keypoints: dict, side: str) -> float | None:
    """Elbow bend using only x,y — immune to z noise. 0 = straight."""
    s = keypoints.get(f"{side}_shoulder")
    e = keypoints.get(f"{side}_elbow")
    w = keypoints.get(f"{side}_wrist")
    if not (s and e and w):
        return None
    if min(s[3], e[3], w[3]) < _VIS_THRESHOLD:
        return None
    raw = _angle_2d(s, e, w)
    return 180.0 - raw


def _hip_bend_2d(keypoints: dict, side: str) -> float | None:
    """Hip bend using only x,y. 0 = straight."""
    s = keypoints.get(f"{side}_shoulder")
    h = keypoints.get(f"{side}_hip")
    k = keypoints.get(f"{side}_knee")
    if not (s and h and k):
        return None
    if min(s[3], h[3], k[3]) < _VIS_THRESHOLD:
        return None
    raw = _angle_2d(s, h, k)
    return 180.0 - raw


# ── Thresholds (0 = straight, bigger = more bent) ────────────────────────────

_HINGE_START_TRUNK      = 20.0
_HINGE_PEAK_TRUNK       = 40.0
_HINGE_IDEAL_TRUNK      = 50.0
_RETURN_TRUNK           = 15.0

_ELBOW_IDEAL_BENT       = 40.0
_HIP_HINGE_IDEAL        = 40.0
_HIP_HINGE_LOOSE        = 25.0

_ARM_RAISE_THRESHOLD    = 50.0

_HYSTERESIS             = 10.0
_VIS_THRESHOLD          = 0.35

# ── Motion detection thresholds ───────────────────────────────────────────────

_MOTION_TRUNK_RANGE_MIN = 12.0
_MOTION_SHOULDER_PX_MIN = 0.03

# ── Exercise cycle timing ─────────────────────────────────────────────────────

_COUNTDOWN_DURATION     = 5.0
_EXERCISE_DURATION      = 10.0
_POST_SPEECH_LINGER     = 2.0


# ── TTS queue helper ──────────────────────────────────────────────────────────

class _TTSQueue:
    def __init__(self, tts: TTSService):
        self._tts = tts
        self._queue: list[str] = []
        self._current: str = ""
        self._current_start: float = 0.0
        self._all_done: bool = True
        self._done_at: float = 0.0

    def load(self, lines: list[str], now: float):
        self._queue = list(lines)
        self._current = ""
        self._all_done = False
        self._done_at = 0.0
        self._advance(now)

    def tick(self, now: float):
        if self._all_done:
            return
        if self._current == "":
            self._advance(now)
            return
        if self._line_finished(now):
            self._advance(now)

    @property
    def all_done(self) -> bool:
        return self._all_done

    @property
    def done_at(self) -> float:
        return self._done_at

    @property
    def lines_remaining(self) -> int:
        return len(self._queue) + (1 if self._current and not self._all_done else 0)

    def _advance(self, now: float):
        if self._queue:
            self._current = self._queue.pop(0)
            self._current_start = now
            self._tts.speak(self._current)
        else:
            self._current = ""
            self._all_done = True
            self._done_at = now

    def _line_finished(self, now: float) -> bool:
        if hasattr(self._tts, 'is_speaking'):
            if not self._tts.is_speaking():
                return True
        est = max(2.0, len(self._current) / 10.0)
        return (now - self._current_start) >= est


# ── Module-level rep-tracking state ───────────────────────────────────────────

_phase              = "start"
_max_trunk_lean     = 0.0
_max_elbow_bend     = 0.0
_max_hip_bend       = 0.0
_peak_reached       = False

_trunk_history: collections.deque = collections.deque(maxlen=60)
_elbow_history: collections.deque = collections.deque(maxlen=60)

_round_rep_peaks: list[dict] = []
_current_rep_frames: list[dict] = []


def _joints_visible(keypoints: dict, side: str) -> bool:
    for part in ("shoulder", "hip", "knee"):
        kp = keypoints.get(f"{side}_{part}")
        if kp is None or kp[3] < _VIS_THRESHOLD:
            return False
    kp = keypoints.get(f"{side}_elbow")
    if kp is None or kp[3] < _VIS_THRESHOLD:
        return False
    return True


def _reset_rep_state():
    global _phase, _max_trunk_lean, _max_elbow_bend, _max_hip_bend, _peak_reached
    global _round_rep_peaks, _current_rep_frames
    _phase = "start"
    _max_trunk_lean = 0.0
    _max_elbow_bend = 0.0
    _max_hip_bend = 0.0
    _peak_reached = False
    _trunk_history.clear()
    _elbow_history.clear()
    _round_rep_peaks = []
    _current_rep_frames = []


# ── Silent rep detection ──────────────────────────────────────────────────────

def _detect_rep_silent(pose_data: dict) -> bool:
    global _phase, _max_trunk_lean, _max_elbow_bend, _max_hip_bend, _peak_reached
    global _current_rep_frames

    kpts = pose_data.get("keypoints", {})
    side = _pick_side(kpts)

    if not _joints_visible(kpts, side):
        if _phase != "start":
            _phase = "start"
            _trunk_history.clear()
            _elbow_history.clear()
            _current_rep_frames = []
        return False

    trunk_angle = pose_data.get("trunk_lean_angle", 0.0)
    elbow_bend  = _elbow_bend_2d(kpts, side) or 0.0
    hip_bend    = _hip_bend_2d(kpts, side) or 0.0

    if _phase == "start":
        if trunk_angle <= _RETURN_TRUNK:
            _phase = "ready"
            _trunk_history.clear()
            _elbow_history.clear()

    elif _phase == "ready":
        if trunk_angle > _HINGE_START_TRUNK:
            _phase = "hinging"
            _max_trunk_lean = trunk_angle
            _max_elbow_bend = elbow_bend
            _max_hip_bend = hip_bend
            _peak_reached = False
            _current_rep_frames = [pose_data]

    elif _phase == "hinging":
        _current_rep_frames.append(pose_data)
        if trunk_angle > _max_trunk_lean:
            _max_trunk_lean = trunk_angle
        if elbow_bend > _max_elbow_bend:
            _max_elbow_bend = elbow_bend
        if hip_bend > _max_hip_bend:
            _max_hip_bend = hip_bend
        if not _peak_reached and trunk_angle >= _HINGE_PEAK_TRUNK:
            _peak_reached = True
        if trunk_angle < _max_trunk_lean - _HYSTERESIS:
            _phase = "extending"

    elif _phase == "extending":
        _current_rep_frames.append(pose_data)
        if trunk_angle <= _RETURN_TRUNK:
            _phase = "ready"
            _trunk_history.clear()
            _elbow_history.clear()
            if _max_trunk_lean >= _HINGE_START_TRUNK:
                side_r = _pick_side(kpts)
                rep_trunk_angles = [f.get("trunk_lean_angle", 0.0) for f in _current_rep_frames]
                rep_elbow_bends  = [
                    b for f in _current_rep_frames
                    if (b := _elbow_bend_2d(f.get("keypoints", {}), side_r)) is not None
                ]
                rep_hip_bends = [
                    b for f in _current_rep_frames
                    if (b := _hip_bend_2d(f.get("keypoints", {}), side_r)) is not None
                ]
                rep_arm_elevs = [f.get(f"{side_r}_arm_elevation", 0.0) for f in _current_rep_frames]

                _round_rep_peaks.append({
                    "peak_trunk":     max(rep_trunk_angles),
                    "min_trunk":      min(rep_trunk_angles),
                    "max_elbow_bend": max(rep_elbow_bends) if rep_elbow_bends else 0.0,
                    "max_hip_bend":   max(rep_hip_bends) if rep_hip_bends else 0.0,
                    "max_arm_elev":   max(rep_arm_elevs),
                    "frame_count":    len(_current_rep_frames),
                    "peak_reached":   _peak_reached,
                })
                _current_rep_frames = []
                return True

    return False


# ── Detect whether the person actually moved ──────────────────────────────────

def _detect_activity(all_frames: list[dict]) -> bool:
    if len(all_frames) < 10:
        return False

    side = _pick_side(all_frames[0].get("keypoints", {}))

    trunk_angles = [f.get("trunk_lean_angle", 0.0) for f in all_frames]
    trunk_range = max(trunk_angles) - min(trunk_angles)

    shoulder_ys = []
    for f in all_frames:
        kp = f.get("keypoints", {}).get(f"{side}_shoulder")
        if kp and kp[3] > _VIS_THRESHOLD:
            shoulder_ys.append(kp[1])

    shoulder_travel = 0.0
    if len(shoulder_ys) >= 10:
        shoulder_travel = max(shoulder_ys) - min(shoulder_ys)

    return trunk_range >= _MOTION_TRUNK_RANGE_MIN or shoulder_travel >= _MOTION_SHOULDER_PX_MIN


# ── Concise coaching feedback ─────────────────────────────────────────────────

def _generate_exercise_feedback(rep_count: int, all_frames: list[dict]) -> list[str]:
    if not all_frames:
        return ["I couldn't see you. Make sure your full body is visible side-on to the camera."]

    active = _detect_activity(all_frames)

    if not active:
        return [
            "It doesn't look like you moved during that round.",
            "You need to stand up, raise your arms, and perform the double pole movement. "
            "Crunch your upper body forward, then snap back upright. Let's try again.",
        ]

    kpts = all_frames[0].get("keypoints", {})
    side = _pick_side(kpts)

    # ── Collect per-frame stats using 2D angles ──────────────────────────
    trunk_angles = []
    elbow_angles = []
    hip_angles   = []
    arm_elevs    = []

    for f in all_frames:
        fk = f.get("keypoints", {})

        trunk = f.get("trunk_lean_angle", 0.0)
        trunk_angles.append(trunk)

        eb = _elbow_bend_2d(fk, side)
        if eb is not None:
            elbow_angles.append(eb)

        hb = _hip_bend_2d(fk, side)
        if hb is not None:
            hip_angles.append(hb)

        arm_elevs.append(f.get(f"{side}_arm_elevation", 0.0))

    peak_trunk     = max(trunk_angles) if trunk_angles else 0.0
    median_elbow   = statistics.median(elbow_angles) if elbow_angles else 0.0
    median_hip     = statistics.median(hip_angles)   if hip_angles   else 0.0
    max_arm        = max(arm_elevs)    if arm_elevs    else 0.0

    # ── Debug print — remove after confirming fix ────────────────────────
    if elbow_angles:
        print(f"  [DEBUG] elbow bend 2D — min={min(elbow_angles):.1f}  "
              f"median={median_elbow:.1f}  max={max(elbow_angles):.1f}  "
              f"count={len(elbow_angles)}")
    if hip_angles:
        print(f"  [DEBUG] hip bend 2D — min={min(hip_angles):.1f}  "
              f"median={median_hip:.1f}  max={max(hip_angles):.1f}  "
              f"count={len(hip_angles)}")

    # ── No completed reps but there was movement ─────────────────────────
    if rep_count == 0:
        feedback = ["I saw some movement but no complete reps."]
        if peak_trunk < _HINGE_PEAK_TRUNK:
            feedback.append(
                "You didn't lean forward enough. Really crunch your upper body down "
                "and make sure you come back fully upright between each stroke."
            )
        else:
            feedback.append(
                "You leaned forward but didn't return fully upright between strokes. "
                "Snap back tall after each pole so the rep counts."
            )
        return feedback

    # ── Had reps — collect positives and issues ──────────────────────────
    positives = []
    issues = []

    # Hinge depth
    if peak_trunk >= _HINGE_IDEAL_TRUNK:
        positives.append(
            "Your forward crunch was deep and powerful — great power transfer through the poles."
        )
    elif peak_trunk >= _HINGE_PEAK_TRUNK:
        issues.append(
            "You could lean forward more during the stroke. "
            "Drive your chest closer to your thighs for more power."
        )
    else:
        issues.append(
            "You were standing too upright during the stroke. "
            "Bend forward much more — really crunch your upper body down toward your knees."
        )

    # Arm bend (2D: 0 = straight, 40+ median = good)
    if median_elbow >= _ELBOW_IDEAL_BENT:
        positives.append(
            "Good arm position — your elbows were nicely bent, acting as strong levers."
        )
    elif median_elbow >= _ELBOW_IDEAL_BENT - 15.0:
        issues.append(
            "Your arms were a bit too stretched out when poling. "
            "Try to bend your elbows more and keep them locked in that position."
        )
    else:
        issues.append(
            "Your arms were too straight. Bend your elbows and keep them bent — "
            "they should act as stiff levers, not loose and extended."
        )

    # Hip hinge (2D: 0 = straight legs, 40+ median = good)
    if median_hip >= _HIP_HINGE_IDEAL:
        positives.append(
            "Nice hip hinge — you folded well at the hips, loading your core for power."
        )
    elif median_hip >= _HIP_HINGE_LOOSE:
        issues.append(
            "Bend at the hips and knees a bit more to generate more power."
        )
    else:
        issues.append(
            "Your hips and legs were too stiff. Soften your knees and fold more at the hips — "
            "that's where the explosive power comes from."
        )

    # Arm elevation
    if max_arm >= _ARM_RAISE_THRESHOLD:
        positives.append(
            "You raised your arms high before each stroke, giving you a full powerful arc."
        )
    else:
        issues.append(
            "Raise your arms higher before each stroke — hands up to forehead level "
            "so you get a full powerful arc."
        )

    # ── Build final feedback ─────────────────────────────────────────────
    feedback = []

    if not issues:
        feedback.append(f"Great round — {rep_count} reps with solid technique!")
        feedback.extend(positives[:2])
        feedback.append("Keep that up and try to push the tempo a bit faster next time.")
    elif not positives:
        feedback.append(f"You did {rep_count} reps.")
        feedback.extend(issues[:3])
    else:
        feedback.append(f"You did {rep_count} reps.")
        feedback.append(positives[0])
        feedback.extend(issues[:2])

    return feedback


# ── Live webcam loop ──────────────────────────────────────────────────────────

def run_live(camera_index: int = 0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    tts = TTSService(cooldown_seconds=0.5)
    tts_q = _TTSQueue(tts)
    start_time = time.time()

    print("\n=== SKIING DOUBLE-POLE COACH ===")
    print("Stand SIDE-ON to the camera (shoulder pointing at the lens).")
    print("Press 'q' to quit.\n")

    cycle_state       = "instructions"
    cycle_state_start = 0.0
    round_number      = 0
    round_rep_count   = 0
    round_frames: list[dict] = []
    feedback_lines: list[str] = []
    last_countdown_spoken = -1
    instructions_loaded = False

    INSTRUCTION_LINES_DISPLAY = [
        "SKIING DOUBLE-POLE COACH",
        "",
        "Stand SIDE-ON to the camera.",
        "Stand tall, push hips forward,",
        "raise hands to forehead level.",
        "",
        "Perform double-pole movements",
        "during the 10-second exercise window.",
        "",
        "Get ready...",
    ]

    INSTRUCTION_SPEECH = [
        "Welcome to the skiing double pole coach.",
        "Stand side-on to the camera so your full body is visible.",
        "Stand tall, raise your hands high, and when the countdown ends, perform double pole movements.",
        "Get ready!",
    ]

    with mp.solutions.pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True
            display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            now = time.time() - start_time
            elapsed_in_state = now - cycle_state_start

            pose_data: dict = {"timestamp": now, "keypoints": {}}

            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark
                keypoints = {
                    LANDMARK_NAMES[i]: (lm.x, lm.y, lm.z, lm.visibility)
                    for i, lm in enumerate(lms)
                    if i in LANDMARK_NAMES
                }
                angles = calculate_angles(keypoints)
                pose_data.update(angles)
                pose_data["keypoints"] = keypoints

                h, w = display.shape[:2]
                for name, (x, y, z, vis) in keypoints.items():
                    if vis > 0.5:
                        cv2.circle(display, (int(x * w), int(y * h)), 4, (180, 180, 180), -1)
                for joint, color in [
                    ("left_hip", (0, 255, 0)), ("right_hip", (0, 180, 255)),
                    ("left_shoulder", (255, 200, 0)), ("right_shoulder", (255, 100, 0)),
                ]:
                    kp = keypoints.get(joint)
                    if kp and kp[3] > 0.5:
                        cv2.circle(display, (int(kp[0] * w), int(kp[1] * h)), 12, color, 2)

            h_disp, w_disp = display.shape[:2]

            tts_q.tick(now)

            # ── State machine ─────────────────────────────────────────────

            if cycle_state == "instructions":
                if not instructions_loaded:
                    tts_q.load(INSTRUCTION_SPEECH, now)
                    instructions_loaded = True
                    print("[instructions] Speaking instructions...")

                overlay = display.copy()
                cv2.rectangle(overlay, (0, 0), (w_disp, h_disp), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
                for i, line in enumerate(INSTRUCTION_LINES_DISPLAY):
                    color = (0, 220, 255) if i == 0 else (255, 255, 255)
                    scale = 0.85 if i == 0 else 0.65
                    thick = 2 if i == 0 else 1
                    cv2.putText(display, line,
                                (30, 50 + i * 35),
                                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

                if not tts_q.all_done:
                    cv2.putText(display, f"Speaking... ({tts_q.lines_remaining} left)",
                                (30, h_disp - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

                if tts_q.all_done and (now - tts_q.done_at) >= _POST_SPEECH_LINGER:
                    cycle_state = "countdown"
                    cycle_state_start = now
                    last_countdown_spoken = -1
                    print("[countdown] Starting countdown...")

            elif cycle_state == "countdown":
                remaining = _COUNTDOWN_DURATION - elapsed_in_state
                count_num = max(0, math.ceil(remaining))

                if count_num != last_countdown_spoken and count_num > 0:
                    last_countdown_spoken = count_num
                    tts.speak(str(count_num))
                    print(f"  Countdown: {count_num}")
                elif count_num == 0 and last_countdown_spoken != 0:
                    last_countdown_spoken = 0
                    tts.speak("Go!")
                    print("  GO!")

                overlay = display.copy()
                cv2.rectangle(overlay, (0, 0), (w_disp, h_disp), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)

                if round_number > 0:
                    cv2.putText(display, f"Round {round_number + 1} starting...",
                                (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)

                if count_num > 0:
                    text = str(count_num)
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 5.0, 8)[0]
                    tx = (w_disp - text_size[0]) // 2
                    ty = (h_disp + text_size[1]) // 2
                    cv2.putText(display, text, (tx, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0, 255, 255), 8)
                else:
                    text = "GO!"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 4.0, 8)[0]
                    tx = (w_disp - text_size[0]) // 2
                    ty = (h_disp + text_size[1]) // 2
                    cv2.putText(display, text, (tx, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 0), 8)

                if elapsed_in_state >= _COUNTDOWN_DURATION + 0.5:
                    cycle_state = "exercise"
                    cycle_state_start = now
                    round_number += 1
                    round_rep_count = 0
                    round_frames = []
                    _reset_rep_state()
                    print(f"[exercise] Round {round_number} — exercise for {_EXERCISE_DURATION:.0f}s...")

            elif cycle_state == "exercise":
                time_left = _EXERCISE_DURATION - elapsed_in_state

                round_frames.append(pose_data)
                if _detect_rep_silent(pose_data):
                    round_rep_count += 1
                    print(f"  Rep detected: {round_rep_count}")

                cv2.putText(display, f"Round {round_number}  |  EXERCISE",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display, f"Time left: {max(0, time_left):.1f}s",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display, f"Reps: {round_rep_count}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                bar_width = int((elapsed_in_state / _EXERCISE_DURATION) * w_disp)
                cv2.rectangle(display, (0, h_disp - 8), (bar_width, h_disp), (0, 200, 255), -1)

                if elapsed_in_state >= _EXERCISE_DURATION:
                    feedback_lines = _generate_exercise_feedback(round_rep_count, round_frames)
                    tts_q.load(feedback_lines, now)
                    cycle_state = "feedback"
                    cycle_state_start = now
                    print(f"[feedback] Round {round_number} — {round_rep_count} reps")
                    for line in feedback_lines:
                        print(f"  {line}")

            elif cycle_state == "feedback":
                overlay = display.copy()
                cv2.rectangle(overlay, (0, 0), (w_disp, h_disp), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.65, display, 0.35, 0, display)

                cv2.putText(display, f"Round {round_number} Feedback",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)

                line_height = 30
                y_cursor = 80
                for i, line in enumerate(feedback_lines):
                    words = line.split()
                    rows = []
                    current = ""
                    for word in words:
                        test = current + " " + word if current else word
                        if len(test) > 70:
                            if current:
                                rows.append(current)
                            current = word
                        else:
                            current = test
                    if current:
                        rows.append(current)
                    for row in rows:
                        if y_cursor < h_disp - 40:
                            cv2.putText(display, row,
                                        (20, y_cursor),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                            y_cursor += line_height

                if not tts_q.all_done:
                    cv2.putText(display, "Speaking...",
                                (20, h_disp - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
                else:
                    linger_left = max(0, _POST_SPEECH_LINGER - (now - tts_q.done_at))
                    cv2.putText(display, f"Next round in {linger_left:.0f}s...",
                                (20, h_disp - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (130, 130, 130), 1)

                if tts_q.all_done and (now - tts_q.done_at) >= _POST_SPEECH_LINGER:
                    cycle_state = "countdown"
                    cycle_state_start = now
                    last_countdown_spoken = -1
                    print("[countdown] Starting next countdown...")

            cv2.imshow("Skiing Double-Pole Coach", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession ended after {round_number} rounds.")


if __name__ == "__main__":
    import __main__
    print(f"[version check] running: {__main__.__file__}")
    run_live(camera_index=0)