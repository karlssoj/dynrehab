"""Standalone viewer: shows every detected pose keypoint (skeleton) plus the
back-contour spine line, live from a webcam. No exercise/session logic --
just the raw PoseEngine pipeline drawn on screen.

Usage:
    pip install -r requirements.txt
    python main.py [--camera N] [--backend yolo|mediapipe]

Press 'f' for fullscreen, 'q' or close the window to quit. Stand sideways
to the camera (profile view) for the spine line to appear -- it's only
sampled when the person is detected in profile.

Backends:
    yolo (default) -- segments the cropped torso region directly each
        tick. Needs yolo11n-pose.pt + yolo11n-seg.pt (bundled in this
        folder).
    mediapipe -- segments the whole frame once per tick (mediapipe can't
        reliably re-segment an already-cropped torso region on its own),
        then warps that cached mask into the torso crop's coordinate space
        using the same transform used to build the crop. Needs the
        `mediapipe` package (see requirements.txt).
"""
import argparse
import os
import time

import cv2
import numpy as np

# Resample the spine every 0.05s (20Hz) instead of PoseEngine's own 0.25s
# (4Hz) default, so the drawn line refreshes about as often as the skeleton
# (~20fps) instead of visibly lagging/flickering behind it. Matches the value
# already validated in the main app. Set before importing PoseEngine so its
# __init__ picks it up via os.getenv; doesn't override a real env var the
# caller may have already set.
os.environ.setdefault("SPINE_SAMPLE_INTERVAL_SECS", "0.05")
# Lower than YOLOBackend's own default (0.3): the hip/shoulder keypoints
# needed for spine sampling were intermittently dropping below that
# threshold (held briefly, then removed from the keypoints dict), making the
# spine-sampling gate in pose_engine.py fail even while genuinely in
# profile -- that's the "line only shows now and then" symptom. Must be set
# before pose_backends.py is imported, since it reads this env var once at
# module load. Informed starting point -- lower further (e.g. 0.1) if it's
# still dropping out, or raise it if the line starts appearing on
# clearly-wrong keypoint guesses.
os.environ.setdefault("YOLO_KEYPOINT_CONF_THRESHOLD", "0.1")
# Lower than pose_engine.py's own default (0.3): with --backend mediapipe, the
# hip landmark's own reported visibility was observed to sit well below 0.3 for
# extended stretches even while genuinely standing in profile (a mediapipe
# quirk -- it applies no confidence pre-filter of its own, unlike YOLOBackend),
# which silently suppressed the drawn line for long stretches. Lower further if
# it's still suppressed, or raise it if the line starts appearing on
# clearly-wrong keypoint guesses.
os.environ.setdefault("SPINE_MIN_KEYPOINT_CONFIDENCE", "0.1")
# TEMPORARY diagnostic for the "top point slips up toward the head"
# report -- prints raw_hip_px/hip_px/shoulder_px/scan_hip_px etc. once per
# successful spine sample. Set via os.environ here (not just documented as
# an env var to set yourself) because a prior attempt to set SPINE_DEBUG=1
# from the shell before running didn't actually reach the process. Safe to
# remove/comment out once the top-point issue is diagnosed.
os.environ.setdefault("SPINE_DEBUG", "1")

from core.pose_backends import create_backend
from core.pose_engine import PoseEngine

_WINDOW_NAME = "Keypoints + spine line ('f' fullscreen, 'q' quit)"


def _letterbox_to_size(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Scale `frame` to fit within target_w x target_h, preserving its
    aspect ratio (no stretching/squashing) -- letterboxed with black bars
    on whichever axis has leftover space, like a video player would."""
    h, w = frame.shape[:2]
    if w <= 0 or h <= 0 or target_w <= 0 or target_h <= 0:
        return frame
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, frame.shape[2]), dtype=frame.dtype)
    x_off, y_off = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def _knee_angle(pose_frame):
    """Whichever side's hip keypoint is present this tick (same
    left-preferred convention used throughout pose_engine.py). Uses
    left/right_knee_bend_2d (0 = straight leg, increasing as the knee
    bends) rather than the raw left/right_knee_angle (180 = straight,
    decreasing as it bends) -- the "bend" convention reads as 0 for a
    straight leg, matching what people expect. 0.0 is also
    calculate_angles' guard value when a required landmark (hip, knee or
    ankle) is missing, so -- unlike the raw angle, where 0.0 was an
    implausible real reading and thus a safe "no data" sentinel -- missing
    data is detected here by checking keypoint presence directly."""
    keypoints = pose_frame.keypoints or {}
    side = "left" if keypoints.get("left_hip") is not None else "right"
    if any(keypoints.get(f"{side}_{joint}") is None for joint in ("hip", "knee", "ankle")):
        return None
    return getattr(pose_frame, f"{side}_knee_bend_2d")


# On-screen readout box: (label, getter). Plain-language labels instead of
# the clinical field names.
_READOUTS = [
    ("Knee angle", _knee_angle),
    ("Trunk lean", lambda pf: pf.trunk_lean_2d),
    ("Thoracic curve", lambda pf: pf.spine_thoracic_bend_2d),
    ("Lumbar curve", lambda pf: pf.spine_lumbar_bend_2d),
]

_READOUT_FONT_SCALE = 0.35
_READOUT_LINE_HEIGHT = 16

# A rep counts once the knee has bent to at least this far (see
# left/right_knee_bend_2d, 0 = straight leg, used by _knee_angle above)...
_SQUAT_DOWN_BEND_DEG = 60.0
# ...and has then come back up to at most this far.
_SQUAT_UP_BEND_DEG = 20.0

# All 17 YOLO pose keypoint names (see YOLOBackend._YOLO_TO_NAME in
# pose_backends.py). Counting doesn't start until every one of these has
# been detected simultaneously at least once -- otherwise a partial/
# low-confidence reading during e.g. standing up from a chair (knee still
# reads as "bent", then straightens as you stand) could look exactly like
# a completed rep before you've even started exercising. NOTE: in a strict
# profile stance the far-side wrist/ear can stay occluded indefinitely, so
# if the counter never arms, turn briefly to face the camera once at the
# start so every joint is seen, then turn back to profile.
_ALL_BODY_JOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


class SquatCounter:
    """Counts full squat reps from the knee-bend readout: bending to at
    least _SQUAT_DOWN_BEND_DEG "arms" a rep, then straightening back to at
    most _SQUAT_UP_BEND_DEG counts it. Ticks with no knee reading (None,
    e.g. person briefly out of frame) are ignored rather than resetting
    progress, so a momentary dropout mid-rep doesn't lose the count."""

    def __init__(self):
        self.count = 0
        self._armed = False

    def update(self, knee_bend_deg):
        if knee_bend_deg is None:
            return
        if not self._armed:
            if knee_bend_deg >= _SQUAT_DOWN_BEND_DEG:
                self._armed = True
        elif knee_bend_deg <= _SQUAT_UP_BEND_DEG:
            self.count += 1
            self._armed = False


def _draw_readouts(frame, values: dict, squat_count: int):
    """Draw a small semi-transparent box in the top-left corner with the
    squat rep count and the latest angle readouts. `values` maps each
    _READOUTS label to its current (possibly held-over/stale) value, or
    None if never seen yet."""
    box_w = 220
    box_h = _READOUT_LINE_HEIGHT * (len(_READOUTS) + 1) + 10
    roi = frame[0:box_h, 0:box_w]
    overlay = roi.copy()
    overlay[:] = (0, 0, 0)
    cv2.addWeighted(overlay, 0.55, roi, 0.45, 0, roi)
    cv2.putText(frame, f"SQUATS: {squat_count}", (6, 14),
                cv2.FONT_HERSHEY_SIMPLEX, _READOUT_FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA)
    for i, (label, _) in enumerate(_READOUTS, start=1):
        value = values.get(label)
        text = f"{label}: {value:.0f} deg" if value is not None else f"{label}: --"
        cv2.putText(frame, text, (6, 14 + i * _READOUT_LINE_HEIGHT),
                    cv2.FONT_HERSHEY_SIMPLEX, _READOUT_FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--backend", choices=("yolo", "mediapipe"), default="yolo",
                         help="pose backend to use for keypoints + the spine line's "
                              "segmentation mask (default yolo)")
    return parser.parse_args()


def main():
    args = _parse_args()

    backend = create_backend({
        "pose_backend": args.backend,
        "yolo_model": "yolo11n-pose.pt",
    })
    engine = PoseEngine(backend=backend)

    latest_frame = {"annotated": None}
    # --- diagnostics: measure actual pipeline throughput AND latency, since
    # "laggy / spine line flickers" reports need real numbers before guessing
    # at a fix. Prints once per second: frames/sec PoseEngine delivers, what
    # fraction had a spine sample, and how far behind wall-clock time the
    # displayed frame is (pose_frame.timestamp is relative to engine start,
    # so latency = wall_clock_elapsed_since_engine_start - pose_frame.timestamp;
    # a value that stays near ~0.03-0.1s is healthy, one that keeps climbing
    # means frames are queueing up faster than we can process them).
    engine_start_wall = time.time()
    stats = {"frames": 0, "spine_hits": 0, "last_print": time.time(), "last_latency": 0.0}
    # Holds the most recent non-None reading for each readout, so the box
    # doesn't flash blank on ticks that don't have a fresh spine sample
    # (spine_thoracic_bend_2d/spine_lumbar_bend_2d are only set on ticks
    # where sampling succeeded, see pose_engine.py) -- same "hold last known
    # value" idea as the drawn line itself, just applied to the text too.
    readout_values = {label: None for label, _ in _READOUTS}
    squat_counter = SquatCounter()
    # Becomes True the first time all 17 body joints have been detected in
    # the same frame -- see _ALL_BODY_JOINTS. Counting is gated on this so
    # standing up from a chair before the whole skeleton is even in frame
    # can't look like a completed rep.
    ready = {"all_joints_seen": False}

    def on_frame(pose_frame, annotated):
        if not ready["all_joints_seen"]:
            keypoints = pose_frame.keypoints or {}
            if all(joint in keypoints for joint in _ALL_BODY_JOINTS):
                ready["all_joints_seen"] = True

        fresh_knee_bend = None
        for label, getter in _READOUTS:
            value = getter(pose_frame)
            if label == "Knee angle":
                fresh_knee_bend = value
            if value is not None:
                readout_values[label] = value
        if ready["all_joints_seen"]:
            squat_counter.update(fresh_knee_bend)
        _draw_readouts(annotated, readout_values, squat_counter.count)
        latest_frame["annotated"] = annotated
        stats["frames"] += 1
        if pose_frame.spine_curvature_ratio is not None:
            stats["spine_hits"] += 1
        now = time.time()
        stats["last_latency"] = (now - engine_start_wall) - pose_frame.timestamp
        elapsed = now - stats["last_print"]
        if elapsed >= 1.0:
            fps = stats["frames"] / elapsed
            spine_rate = stats["spine_hits"] / stats["frames"] if stats["frames"] else 0.0
            print(f"[diag] {fps:.1f} fps, spine sample present in "
                  f"{spine_rate * 100:.0f}% of frames, latency={stats['last_latency']:.2f}s")
            stats["frames"] = 0
            stats["spine_hits"] = 0
            stats["last_print"] = now

    engine.subscribe(on_frame)
    engine.start(source=args.camera)

    # WINDOW_NORMAL (not the default WINDOW_AUTOSIZE) makes the window
    # user-resizable/maximizable; _letterbox_to_size below then fits the
    # camera frame into whatever size the window actually is, preserving
    # aspect ratio, instead of cv2.imshow silently stretching it to fill
    # the window and distorting the proportions.
    cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
    fullscreen = {"on": False}
    try:
        while True:
            frame = latest_frame["annotated"]
            if frame is not None:
                try:
                    _, _, win_w, win_h = cv2.getWindowImageRect(_WINDOW_NAME)
                except cv2.error:
                    win_w, win_h = 0, 0
                if win_w > 0 and win_h > 0:
                    frame = _letterbox_to_size(frame, win_w, win_h)
                cv2.imshow(_WINDOW_NAME, frame)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q") or cv2.getWindowProperty(_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("f"):
                fullscreen["on"] = not fullscreen["on"]
                cv2.setWindowProperty(
                    _WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen["on"] else cv2.WINDOW_NORMAL)
            if frame is None:
                time.sleep(0.05)
    finally:
        engine.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
