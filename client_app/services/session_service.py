import builtins
import time
from typing import Optional

ALLOWED_IMPORT_NAMES = {"math", "statistics", "collections", "itertools", "functools"}


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
    def __init__(self, exercise_id: str, module_code: str):
        self.exercise_id = exercise_id
        self.rep_count = 0
        self._rep_frames: list[dict] = []
        self._rep_start_time: Optional[float] = None
        self._feedback_log: list[dict] = []
        self._good_reps = 0
        self._started_at = time.time()
        self._load_module(module_code)

    def _load_module(self, code: str):
        namespace = {"__builtins__": SAFE_BUILTINS}
        exec(code, namespace)
        self._analyze_frame = namespace["analyze_frame"]
        self._detect_rep = namespace["detect_rep"]
        self._on_rep_complete = namespace["on_rep_complete"]

    def process_frame(self, pose_data: dict) -> dict:
        feedback = self._analyze_frame(pose_data)
        highlight_joints = {f["joint"] for f in feedback if f.get("joint")}

        for msg in feedback:
            self._feedback_log.append({
                "timestamp": pose_data.get("timestamp", 0.0),
                "message": msg["message"],
            })

        if self._rep_start_time is None:
            self._rep_start_time = pose_data.get("timestamp", 0.0)
        self._rep_frames.append(pose_data)

        rep_completed = bool(self._detect_rep(pose_data))
        post_rep_feedback = []

        if rep_completed:
            self.rep_count += 1
            rep_data = {
                "rep_number": self.rep_count,
                "frames": list(self._rep_frames),
                "duration_seconds": (
                    pose_data.get("timestamp", 0.0) - self._rep_start_time
                ),
            }
            post_rep_feedback = self._on_rep_complete(rep_data)
            is_good = all(
                f.get("message", "").lower().startswith(("great", "good", "perfect", "well"))
                for f in post_rep_feedback
            ) or len(post_rep_feedback) == 0
            if is_good:
                self._good_reps += 1
            for msg in post_rep_feedback:
                self._feedback_log.append({
                    "timestamp": pose_data.get("timestamp", 0.0),
                    "message": f"[Rep {self.rep_count}] {msg['message']}",
                })
            self._rep_frames = []
            self._rep_start_time = None

        return {
            "feedback": feedback,
            "rep_completed": rep_completed,
            "post_rep_feedback": post_rep_feedback,
            "rep_count": self.rep_count,
            "highlight_joints": highlight_joints,
        }

    def get_summary(self) -> dict:
        quality_pct = (self._good_reps / self.rep_count * 100) if self.rep_count > 0 else 0
        return {
            "rep_count": self.rep_count,
            "quality_pct": round(quality_pct),
            "feedback_log": list(self._feedback_log),
            "duration_seconds": time.time() - self._started_at,
        }
