import builtins
import math
import time
from typing import Optional

ALLOWED_IMPORT_NAMES = {"math", "statistics", "collections", "itertools", "functools"}

_CALIB_VIS_LOW = 0.2
_CALIB_VIS_HIGH = 0.4
_CALIB_MIN_HEIGHT = 0.60       # body (nose→ankle) must span ≥60% of frame height
_CALIB_TOO_CLOSE_UPPER = 0.50  # nose→hip > this while ankles absent/low-vis → too close
_CALIB_SIDE_THRESHOLD = 0.13   # shoulder x-sep above this → facing camera (frontal)
_CALIB_HOLD_SECS = 1.5
_CALIB_SPEAK_COOLDOWN = 4.0

_CALIB_CRITICAL_KPS = [
    "nose", "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

COUNTDOWN_SECS = 5


def _make_safe_import():
    real_import = builtins.__import__

    def safe_import(name, *args, **kwargs):
        top = name.split(".")[0]
        if top not in ALLOWED_IMPORT_NAMES:
            raise ImportError(f"Import of '{name}' is not allowed in analysis modules")
        return real_import(name, *args, **kwargs)

    return safe_import


SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in [
        "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
        "len", "list", "max", "min", "print", "range", "round", "set",
        "str", "sum", "tuple", "zip", "isinstance", "issubclass", "type",
        "hasattr", "getattr", "None", "True", "False",
    ]
    if hasattr(builtins, name)
}
SAFE_BUILTINS["__build_class__"] = builtins.__build_class__
SAFE_BUILTINS["__import__"] = _make_safe_import()


class SessionService:
    def __init__(self, exercise_id: str, module_code: str, exercise_secs: int = 10,
                 feedback_mode: list[str] | None = None, camera_view: str = "side"):
        self.exercise_id = exercise_id
        self._exercise_secs = exercise_secs
        self._feedback_mode = set(feedback_mode or ["after_window"])
        self._camera_view = camera_view
        self.rep_count = 0
        self._round_number = 0
        self._round_rep_count = 0
        self._round_frames: list[dict] = []
        self._all_rounds: list[dict] = []
        self._angle_stats: dict = {}
        self._started_at = time.time()
        self._state = "instructions"  # instructions|calibration|countdown|exercise|feedback
        self._state_wall_start = time.time()
        self._countdown_last: Optional[int] = None
        self._feedback_emitted = False
        self._last_cue_time: float = 0.0
        self._rep_cues: list[str] = []
        self._round_feedback_history: list[list[str]] = []
        self._calibration_last_speak: float = 0.0
        self._calibration_ready_since: float = 0.0
        self._load_module(module_code)

    def _load_module(self, code: str):
        namespace = {"__builtins__": SAFE_BUILTINS}
        exec(code, namespace)
        self._detect_rep = namespace["detect_rep"]
        self._generate_round_feedback = namespace.get("generate_round_feedback")
        self._generate_rep_cue = namespace.get("generate_rep_cue")
        self._get_session_summary = namespace.get("get_session_summary")
        self._get_instructions = namespace.get("get_instructions")
        self._reset_round = namespace.get("reset_round")
        self._get_relevant_joints = namespace.get("get_relevant_joints")

    def get_instructions(self) -> list[str]:
        if self._get_instructions:
            try:
                return list(self._get_instructions())
            except Exception:
                return []
        return []

    def get_relevant_joints(self) -> list:
        """Returns list of (display_label, pose_data_key) pairs from the module."""
        if self._get_relevant_joints:
            try:
                return list(self._get_relevant_joints())
            except Exception:
                return []
        return []

    def _call_rep_cue(self, trigger: str) -> Optional[str]:
        if not self._generate_rep_cue:
            return None
        cue_data = {
            "trigger": trigger,
            "rep_number": self.rep_count,
            "round_number": self._round_number if "after_window" in self._feedback_mode else None,
            "frames": list(self._round_frames[-30:]),
        }
        try:
            result = self._generate_rep_cue(cue_data)
            return str(result).strip() or None
        except Exception as e:
            print(f"[session] generate_rep_cue error: {e}")
            return None

    def start_calibration(self):
        """Called by view when instructions speech is done."""
        self._state = "calibration"
        self._state_wall_start = time.time()
        self._calibration_last_speak = 0.0
        self._calibration_ready_since = 0.0

    def start_countdown(self):
        """Transition to countdown — called internally and by tests."""
        self._state = "countdown"
        self._state_wall_start = time.time()
        self._countdown_last = None

    def end_feedback(self):
        """Called by view when feedback speech is done."""
        self.start_countdown()

    def process_frame(self, pose_data: dict) -> dict:
        # Track angle stats (skip zeros and tiny values)
        for key, val in pose_data.items():
            if key in ("timestamp", "keypoints"):
                continue
            if isinstance(val, (int, float)) and val > 5.0:
                if key not in self._angle_stats:
                    self._angle_stats[key] = {"min": float(val), "max": float(val)}
                else:
                    if val < self._angle_stats[key]["min"]:
                        self._angle_stats[key]["min"] = float(val)
                    if val > self._angle_stats[key]["max"]:
                        self._angle_stats[key]["max"] = float(val)

        elapsed = time.time() - self._state_wall_start
        result = {
            "state": self._state,
            "round_number": self._round_number,
            "round_rep_count": self._round_rep_count,
            "total_reps": self.rep_count,
            "time_remaining": 0.0,
            "countdown_speak": None,   # int (1-5) or 0 for "Go!" — view speaks it
            "feedback_lines": None,    # list[str], non-None only on first frame of feedback
            "highlight_joints": set(),
        }

        if self._state == "instructions":
            pass  # view handles instruction speech and calls start_calibration()

        elif self._state == "calibration":
            status, message = self._check_position(pose_data)
            result["calibration_status"] = status
            result["calibration_message"] = message or ""
            now = time.time()
            if status == "ready":
                if self._calibration_ready_since == 0.0:
                    self._calibration_ready_since = now
                if now - self._calibration_ready_since >= _CALIB_HOLD_SECS:
                    result["calibration_speak"] = "Good! Starting now."
                    self.start_countdown()
            else:
                self._calibration_ready_since = 0.0
                if message and now - self._calibration_last_speak >= _CALIB_SPEAK_COOLDOWN:
                    result["calibration_speak"] = message
                    self._calibration_last_speak = now

        elif self._state == "countdown":
            remaining = COUNTDOWN_SECS - elapsed
            count_num = max(0, math.ceil(remaining))
            if count_num != self._countdown_last:
                self._countdown_last = count_num
                result["countdown_speak"] = count_num  # 0 means "Go!"
            result["time_remaining"] = max(0.0, remaining)
            if elapsed >= COUNTDOWN_SECS + 0.5:
                self._enter_exercise()
                result["state"] = "exercise"

        elif self._state == "exercise":
            time_left = self._exercise_secs - elapsed
            result["time_remaining"] = max(0.0, time_left)
            self._round_frames.append(pose_data)
            _rep_detected = False
            try:
                if bool(self._detect_rep(pose_data)):
                    _rep_detected = True
                    self._round_rep_count += 1
                    self.rep_count += 1
                    if "during_exercise" in self._feedback_mode:
                        self._last_cue_time = time.time()
                        cue = self._call_rep_cue("rep")
                        if cue:
                            result["rep_cue"] = cue
                            self._rep_cues.append(cue)
            except Exception as e:
                print(f"[session] detect_rep error: {e}")
            result["round_rep_count"] = self._round_rep_count
            result["total_reps"] = self.rep_count

            if (not _rep_detected
                    and "during_exercise" in self._feedback_mode
                    and self._last_cue_time > 0
                    and time.time() - self._last_cue_time >= 5.0):
                self._last_cue_time = time.time()
                cue = self._call_rep_cue("timeout")
                if cue:
                    result["rep_cue"] = cue
                    self._rep_cues.append(cue)

            if elapsed >= self._exercise_secs and "after_window" in self._feedback_mode:
                feedback = self._enter_feedback()
                result["feedback_lines"] = feedback
                result["time_remaining"] = 0.0
                result["state"] = "feedback"
                self._feedback_emitted = True

        elif self._state == "feedback":
            if not self._feedback_emitted:
                # Shouldn't happen but guard anyway
                pass

        result["state"] = self._state
        return result

    def _enter_exercise(self):
        self._last_cue_time = time.time()
        self._state = "exercise"
        self._state_wall_start = time.time()
        self._round_number += 1
        self._round_rep_count = 0
        self._round_frames = []
        self._feedback_emitted = False
        if self._reset_round:
            try:
                self._reset_round()
            except Exception:
                pass

    def _enter_feedback(self) -> list[str]:
        self._state = "feedback"
        self._state_wall_start = time.time()
        round_data = {
            "round_number": self._round_number,
            "rep_count": self._round_rep_count,
            "frames": list(self._round_frames),
            "duration_seconds": self._exercise_secs,
        }
        self._all_rounds.append(round_data)
        try:
            lines = self._generate_round_feedback(round_data) if self._generate_round_feedback else None
            lines = list(lines) if lines else ["Round complete."]
        except Exception as e:
            print(f"[session] generate_round_feedback error: {e}")
            lines = ["Round complete."]
        self._round_feedback_history.append(lines)
        return lines

    def _check_position(self, pose_data: dict) -> tuple[str, str | None]:
        kpts = pose_data.get("keypoints", {})
        visible_count = sum(
            1 for k in _CALIB_CRITICAL_KPS
            if kpts.get(k) and kpts[k][3] > _CALIB_VIS_LOW
        )
        if visible_count < 4:
            return "no_person", "Step in front of the camera so your whole body is visible."
        top_y = min(
            (kpts[k][1] for k in ("nose", "left_shoulder", "right_shoulder")
             if kpts.get(k) and kpts[k][3] > _CALIB_VIS_LOW),
            default=None,
        )
        ankle_y = max(
            (kpts[k][1] for k in ("left_ankle", "right_ankle", "left_heel", "right_heel")
             if kpts.get(k) and kpts[k][3] > _CALIB_VIS_LOW),
            default=None,
        )
        hip_y = max(
            (kpts[k][1] for k in ("left_hip", "right_hip")
             if kpts.get(k) and kpts[k][3] > _CALIB_VIS_LOW),
            default=None,
        )
        # Upper body large → person is too close (feet at or below frame edge)
        upper_body_large = (
            top_y is not None and hip_y is not None
            and hip_y - top_y > _CALIB_TOO_CLOSE_UPPER
        )
        if ankle_y is not None and top_y is not None:
            if ankle_y - top_y < _CALIB_MIN_HEIGHT:
                if upper_body_large:
                    return "too_close", "Step back from the camera."
                return "too_far", "Move closer to the camera."
        elif top_y is not None and hip_y is not None:
            if upper_body_large:
                return "too_close", "Step back from the camera."
            return "too_far", "Move closer to the camera."
        else:
            return "too_far", "Move closer to the camera."
        # Check orientation before demanding full visibility — shoulders are nearly always
        # visible, and a wrong-orientation person will have naturally occluded far-side joints.
        ls = kpts.get("left_shoulder")
        rs = kpts.get("right_shoulder")
        if ls and rs and min(ls[3], rs[3]) > _CALIB_VIS_LOW:
            shoulder_sep = abs(ls[0] - rs[0])
            if self._camera_view == "side" and shoulder_sep > _CALIB_SIDE_THRESHOLD:
                return "wrong_orientation", "Turn sideways to the camera."
            if self._camera_view == "front" and shoulder_sep < _CALIB_SIDE_THRESHOLD:
                return "wrong_orientation", "Turn to face the camera."
            if self._camera_view == "back":
                nose = kpts.get("nose")
                if nose and nose[3] > _CALIB_VIS_HIGH:
                    return "wrong_orientation", "Turn your back to the camera."
        if any(not kpts.get(k) or kpts[k][3] < _CALIB_VIS_HIGH for k in _CALIB_CRITICAL_KPS):
            if upper_body_large:
                return "too_close", "Step back from the camera."
            return "too_far", "Move closer to the camera."
        return "ready", None

    def get_session_summary_speech(self) -> list[str]:
        if not self._get_session_summary:
            return []
        session_data = {
            "total_reps": self.rep_count,
            "total_duration_seconds": time.time() - self._started_at,
            "rounds": list(self._all_rounds),
            "rep_cues": list(self._rep_cues),
            "round_feedback": list(self._round_feedback_history),
            "angle_stats": dict(self._angle_stats),
        }
        try:
            result = self._get_session_summary(session_data)
            if isinstance(result, list):
                return [str(s) for s in result if s]
            return [str(result)] if result else []
        except Exception as e:
            print(f"[session] get_session_summary error: {e}")
            return []

    def get_summary(self) -> dict:
        return {
            "rep_count": self.rep_count,
            "quality_pct": 0,
            "feedback_log": [],
            "duration_seconds": time.time() - self._started_at,
        }
