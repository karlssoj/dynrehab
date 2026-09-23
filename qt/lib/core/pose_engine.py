import math
import os
import threading
import time
from typing import Callable
import cv2
import numpy as np
from core.data_contract import PoseFrame
from core.angle_calculator import calculate_angles
from core.pose_backends import POSE_CONNECTIONS, PoseBackend, MediaPipeBackend, draw_spine_zones
from core.spine_contour import (crop_and_rotate_roi, extract_back_contour,
                                 signed_curvature_ratio, signed_curvature_profile_indices,
                                 spine_zone_bend_angles, spine_bend_anchor_indices,
                                 spine_zone_for_index, sparse_points_in_frame, facing_left,
                                 back_contour_points_in_frame)
from core.one_euro_filter import OneEuroFilter

# Kept for backwards compatibility — callers that imported LANDMARK_NAMES from here still work.
LANDMARK_NAMES = {
    0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
    4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
    7: "left_ear", 8: "right_ear", 9: "mouth_left", 10: "mouth_right",
    11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist", 17: "left_pinky", 18: "right_pinky",
    19: "left_index", 20: "right_index", 21: "left_thumb", 22: "right_thumb",
    23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle", 29: "left_heel", 30: "right_heel",
    31: "left_foot_index", 32: "right_foot_index",
}

_SPINE_SAMPLE_INTERVAL_SECS = 0.25  # 4 Hz — see spec's measured performance baseline
# Override via the SPINE_SAMPLE_INTERVAL_SECS env var (e.g. in .env) — a
# smaller value samples more often (updates the drawn contour faster) at
# the cost of more CPU load per second; a larger value reduces load.

_SPINE_CONTOUR_POINT_COUNT = 10  # bucket count for the ANGLE MEASUREMENT's vertex
# search (signed_curvature_profile_indices) -- NOT the on-screen line, which is now
# drawn at full contour resolution regardless of this value (see spine_points in
# the sampling block below). Override via the SPINE_CONTOUR_POINT_COUNT env var.

_SPINE_MAX_SHOULDER_LATERAL_RATIO = 0.4  # informed starting point, not validated —
# shoulder_lateral_span alone is NOT scale-invariant: it shrinks with distance
# from the camera regardless of facing direction, so a person far away (or
# partly out of frame) could pass an absolute threshold while still facing
# the camera. Normalizing by hip-shoulder distance (torso length in the same
# frame) makes the check invariant to how large the person is in the image —
# true profile gives a ratio near 0 regardless of distance; facing the
# camera gives a ratio close to (shoulder width / torso length), roughly
# 0.8-1.0 for typical body proportions. Override via
# SPINE_MAX_SHOULDER_LATERAL_RATIO.

_SPINE_ZONE_FRACTION = 1.0 / 3.0  # see spine_contour.py's _SPINE_ZONE_FRACTION —
# governs the ANGLE MEASUREMENT's vertex search. Do not raise this to make the
# on-screen zone coloring look wider — see _SPINE_ZONE_DISPLAY_FRACTION below for
# that. Override via the SPINE_ZONE_FRACTION env var.

_SPINE_ZONE_DISPLAY_FRACTION = 0.4  # fraction of the drawn contour dots colored as
# thoracic/lumbar (see spine_zone_for_index), purely cosmetic — used ONLY for the
# on-screen overlay, never passed into the angle-measurement functions. Deliberately
# a SEPARATE constant from _SPINE_ZONE_FRACTION: widening the zone used by the
# actual angle math (rather than just the coloring) can silently invert genuine
# curvature readings once it reaches the profile's contaminated middle region — see
# _SPINE_ZONE_FRACTION's docstring for the measured cliff. Override via the
# SPINE_ZONE_DISPLAY_FRACTION env var.

_SPINE_BEND_BOUNDARY_FRACTION = 0.5  # see spine_contour.py's _SPINE_BEND_BOUNDARY_FRACTION —
# override via the SPINE_BEND_BOUNDARY_FRACTION env var.

_SPINE_SCAN_SHIFT_FRAC = 0.12  # informed starting point, not validated — shifts the
# HIP end of the sampled shoulder-to-hip window upward (toward the shoulder), by this
# fraction of the shoulder-hip chord length. Without this shift the window's bottom
# runs exactly at the hip joint, which already sits within the visible buttock
# curvature in profile (over-including it at the bottom). Override via the
# SPINE_SCAN_SHIFT_FRAC env var.
#
# NOTE: this constant now drives ONLY the hip end (see _spine_scan_shift_frac's
# standing/bent interpolation below). The shoulder end has its own, separate,
# fixed fraction -- _SPINE_SHOULDER_SCAN_SHIFT_FRAC -- rather than reusing this
# value: the two ends used to share a single fraction/vector (a pure translation
# of the whole window, keeping its length unchanged), but once the hip end's
# fraction grew to accommodate standing posture (see
# _SPINE_SCAN_SHIFT_FRAC_STANDING, up to 0.22), applying that SAME larger
# fraction to the shoulder end pushed it visibly past the real shoulder into the
# neck -- the shoulder end never had the standing/bent buttock problem the hip
# end has, so it doesn't need to track hip_bend_deg at all.
#
# An ear-keypoint-based version of the shoulder shift (reaching further up,
# toward the real ear landmark, when confidently visible) was tried and
# reverted: even after projecting the ear onto the hip-shoulder axis (to avoid
# rotating crop_and_rotate_roi's whole coordinate system toward the head), the
# resulting anchor point could still land far up near/behind the head for a
# long neck or noisy ear reading -- visibly wrong on the overlay and unreliable
# for the angle math. The fixed, modest percentage shift below stays close to
# the real shoulder and has not shown that failure mode.
_SPINE_SHOULDER_SCAN_SHIFT_FRAC = 0.12  # override via SPINE_SHOULDER_SCAN_SHIFT_FRAC
#
# This value alone (_SPINE_SCAN_SHIFT_FRAC below is really "the BENT-forward
# reference value") was tuned/observed to look right once the person is
# meaningfully hip-hinged (e.g. partway into a squat) -- but reported wrong when
# standing upright: the bottom of the drawn line still sat on the buttock instead
# of the low back. This makes anatomical sense -- when standing tall (hip
# extended, more lordosis) the glutes protrude further up along the back's
# silhouette than when hip-flexed (glutes flatten/rotate away as the pelvis tilts
# forward), so a FIXED shift can't be correct at both ends of that range. See
# _spine_scan_shift_frac, which interpolates between this and
# _SPINE_SCAN_SHIFT_FRAC_STANDING using the live hip-flexion ("hip bend") angle.
_SPINE_SCAN_SHIFT_FRAC_STANDING = 0.22  # informed starting point, not validated —
# window-shift fraction used when the hip is essentially straight (standing tall,
# hip_bend_deg near 0). Override via SPINE_SCAN_SHIFT_FRAC_STANDING.
_SPINE_SCAN_SHIFT_BEND_LOW_DEG = 10.0   # hip_bend_deg at/below which the STANDING
# fraction is used at full strength. Override via SPINE_SCAN_SHIFT_BEND_LOW_DEG.
_SPINE_SCAN_SHIFT_BEND_HIGH_DEG = 45.0  # hip_bend_deg at/above which the BENT
# (_SPINE_SCAN_SHIFT_FRAC) fraction is used at full strength -- linearly
# interpolated between the two thresholds. Override via
# SPINE_SCAN_SHIFT_BEND_HIGH_DEG.


def _spine_scan_shift_frac(hip_bend_deg: float, standing_frac: float, bent_frac: float,
                            bend_low_deg: float, bend_high_deg: float) -> float:
    """Interpolate the scan-window shift fraction (see _SPINE_SCAN_SHIFT_FRAC)
    between standing_frac (used at/below bend_low_deg, i.e. hip essentially
    straight) and bent_frac (used at/above bend_high_deg, i.e. meaningfully
    hip-hinged), linearly in between. hip_bend_deg is 0=hip straight,
    increasing as the hip flexes (see left_hip_bend_2d/right_hip_bend_2d in
    angle_calculator.py) -- NOT the spine's own thoracic/lumbar bend, which
    this shift is computed upstream of. Falls back to bent_frac if the
    threshold range is degenerate (bend_high_deg <= bend_low_deg)."""
    if bend_high_deg <= bend_low_deg:
        return bent_frac
    t = (hip_bend_deg - bend_low_deg) / (bend_high_deg - bend_low_deg)
    t = max(0.0, min(1.0, t))
    return standing_frac + t * (bent_frac - standing_frac)

# One-euro-filter tuning for the per-zone spine bend angle. Deliberately
# separate from YOLOBackend's keypoint-smoothing constants (min_cutoff=1.0,
# beta=0.05, d_cutoff=1.0 there) — those were tuned for a ~30fps per-frame
# pixel-position stream, whereas spine sampling runs at ~4Hz on a derived
# scalar (bend angle in degrees), a much lower and more irregular cadence.
# Informed starting point, not validated — see Risks in the spine-posture
# baseline design plan.
_SPINE_FILTER_MIN_CUTOFF = 1.0  # override via SPINE_FILTER_MIN_CUTOFF
_SPINE_FILTER_BETA = 0.1        # override via SPINE_FILTER_BETA
_SPINE_FILTER_D_CUTOFF = 1.0    # override via SPINE_FILTER_D_CUTOFF

# One-euro-filter tuning for the HIP and SHOULDER KEYPOINTS' PIXEL POSITION
# (x, y) feeding the spine scan window, applied before
# crop_and_rotate_roi/chord_len/n_raw are derived from them. Deliberately
# heavier (lower min_cutoff) than YOLOBackend's own per-frame keypoint
# smoothing (min_cutoff=1.0) -- that smoothing alone was not enough to stop
# the chord's ends from visibly drifting a few pixels between ticks even
# during ordinary standing (sensor/segmentation noise, minor sway), which
# translates into the whole rotated/cropped ROI -- and so the drawn
# contour's on-screen position -- jittering up and down. Unlike the
# earlier, reverted attempt to smooth the derived chord_len/
# boundary_reference_len as a separate scalar (which caused chord_len,
# n_raw and boundary_idx to disagree with each other because only the
# boundary calculation used the smoothed value while everything else kept
# using the tick's raw, unsmoothed length), smoothing the keypoint
# POSITIONS themselves keeps every downstream quantity (chord_len, n_raw,
# the contour, the drawn anchor) consistently derived from the same
# smoothed points -- no mismatch, just an intentional lag/stability
# trade-off. Deliberately NOT a hard freeze/calibration lock, and NOT a
# plain fixed-cutoff low-pass filter either: with beta > 0, OneEuroFilter's
# cutoff frequency rises with the point's estimated speed
# (cutoff = min_cutoff + beta * |dx_hat|, see one_euro_filter.py), so it
# damps jitter during near-stationary standing (cutoff stays at min_cutoff)
# while still tracking fast, genuine movement (e.g. standing up quickly out
# of a squat) with much less lag once the point is actually moving fast --
# the drawn curve must always keep following real back movement during the
# exercise, only the standing-still jitter should be smoothed away.
# IMPORTANT: beta=0.0 disables that velocity adaptation entirely (cutoff
# becomes a constant min_cutoff regardless of speed) -- this was the actual
# bug behind a reported ~0.5s-lag artifact (drawn line's bottom point
# sagging onto the buttock for a moment after standing up quickly from a
# squat, before catching up): beta was left at 0.0 here, so the "adaptive"
# part of the filter was never actually active. Was previously hip-only
# (see git history); extended to the shoulder side too since it has since
# shown the same drift. Informed starting point, not validated. Override
# via SPINE_HIP_POSITION_MIN_CUTOFF/BETA / SPINE_SHOULDER_POSITION_MIN_CUTOFF/BETA.
_SPINE_HIP_POSITION_MIN_CUTOFF = 0.3
_SPINE_HIP_POSITION_BETA = 0.015      # override via SPINE_HIP_POSITION_BETA
_SPINE_HIP_POSITION_D_CUTOFF = 1.0    # override via SPINE_HIP_POSITION_D_CUTOFF
_SPINE_SHOULDER_POSITION_MIN_CUTOFF = 0.3  # override via SPINE_SHOULDER_POSITION_MIN_CUTOFF
_SPINE_SHOULDER_POSITION_BETA = 0.015      # override via SPINE_SHOULDER_POSITION_BETA
_SPINE_SHOULDER_POSITION_D_CUTOFF = 1.0    # override via SPINE_SHOULDER_POSITION_D_CUTOFF

# Same One-euro-filter tuning again, for the KNEE keypoint's pixel position --
# used only to compute _hip_bend_deg (the hip-flexion angle that drives
# _spine_scan_shift_frac), not for the crop/ROI geometry itself. Deliberately
# defaulted to the SAME values as the hip/shoulder position filters above:
# the whole point of filtering the knee this way is so hip_bend_deg is
# derived from points with the SAME lag as hip_px/shoulder_px, so the shift
# fraction it selects never disagrees with where the anchor geometry
# actually is (see the mismatch this fixed, documented at the call site).
# Override via SPINE_KNEE_POSITION_MIN_CUTOFF, but keep it equal to the hip/
# shoulder values unless you're deliberately reintroducing that mismatch.
_SPINE_KNEE_POSITION_MIN_CUTOFF = 0.3
_SPINE_KNEE_POSITION_BETA = 0.015
_SPINE_KNEE_POSITION_D_CUTOFF = 1.0


def _should_sample_spine(last_sample_time: float, now: float,
                          interval: float = _SPINE_SAMPLE_INTERVAL_SECS) -> bool:
    return now - last_sample_time >= interval


def _hip_bend_deg(shoulder_px: tuple[float, float], hip_px: tuple[float, float],
                   knee_px: tuple[float, float]) -> float:
    """Hip-flexion angle in degrees at hip_px, between rays to shoulder_px
    and knee_px: 0 = hip straight (standing), increasing as the hip flexes
    forward. Same convention as angle_calculator.calculate_angles'
    left_hip_bend_2d/right_hip_bend_2d (bend = 180 - raw angle), but computed
    directly from the caller's own already-smoothed pixel points instead of
    pose_frame's version (see the call site in _run for why that distinction
    matters -- mixing lag characteristics between the two produced a visible
    artifact). Returns 0.0 (i.e. "no bend detected") if any two of the three
    points coincide, mirroring spine_contour._angle_2d's degenerate-case
    convention."""
    vax, vay = shoulder_px[0] - hip_px[0], shoulder_px[1] - hip_px[1]
    vcx, vcy = knee_px[0] - hip_px[0], knee_px[1] - hip_px[1]
    mag = math.hypot(vax, vay) * math.hypot(vcx, vcy)
    if mag < 1e-10:
        return 0.0
    cosine = (vax * vcx + vay * vcy) / mag
    angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    return 180.0 - angle


class PoseEngine:
    def __init__(self, backend: PoseBackend = None):
        self._backend = backend if backend is not None else MediaPipeBackend()
        self._subscribers: list[Callable] = []
        self._running = False
        self._thread: threading.Thread = None
        self._highlight_joints: set[str] = set()
        self._lock = threading.Lock()
        self._seek_start = False
        self._last_spine_sample = 0.0
        self._last_spine_points: list[tuple[float, float]] | None = None
        self._last_spine_zone_labels: list[str] | None = None
        self._last_spine_anchor_points: list[tuple[float, float]] | None = None
        self._last_spine_thoracic_vertex: tuple[float, float] | None = None
        self._last_spine_lumbar_vertex: tuple[float, float] | None = None
        self._last_spine_thoracic_deg: float | None = None
        self._last_spine_lumbar_deg: float | None = None
        self._spine_sample_interval = float(
            os.getenv("SPINE_SAMPLE_INTERVAL_SECS", _SPINE_SAMPLE_INTERVAL_SECS))
        self._spine_contour_point_count = int(
            os.getenv("SPINE_CONTOUR_POINT_COUNT", _SPINE_CONTOUR_POINT_COUNT))
        self._spine_max_shoulder_lateral_ratio = float(
            os.getenv("SPINE_MAX_SHOULDER_LATERAL_RATIO", _SPINE_MAX_SHOULDER_LATERAL_RATIO))
        self._spine_zone_fraction = float(
            os.getenv("SPINE_ZONE_FRACTION", _SPINE_ZONE_FRACTION))
        self._spine_zone_display_fraction = float(
            os.getenv("SPINE_ZONE_DISPLAY_FRACTION", _SPINE_ZONE_DISPLAY_FRACTION))
        self._spine_bend_boundary_fraction = float(
            os.getenv("SPINE_BEND_BOUNDARY_FRACTION", _SPINE_BEND_BOUNDARY_FRACTION))
        self._spine_scan_shift_frac = float(
            os.getenv("SPINE_SCAN_SHIFT_FRAC", _SPINE_SCAN_SHIFT_FRAC))
        self._spine_scan_shift_frac_standing = float(
            os.getenv("SPINE_SCAN_SHIFT_FRAC_STANDING", _SPINE_SCAN_SHIFT_FRAC_STANDING))
        self._spine_scan_shift_bend_low_deg = float(
            os.getenv("SPINE_SCAN_SHIFT_BEND_LOW_DEG", _SPINE_SCAN_SHIFT_BEND_LOW_DEG))
        self._spine_scan_shift_bend_high_deg = float(
            os.getenv("SPINE_SCAN_SHIFT_BEND_HIGH_DEG", _SPINE_SCAN_SHIFT_BEND_HIGH_DEG))
        self._spine_shoulder_scan_shift_frac = float(
            os.getenv("SPINE_SHOULDER_SCAN_SHIFT_FRAC", _SPINE_SHOULDER_SCAN_SHIFT_FRAC))
        spine_filter_min_cutoff = float(os.getenv("SPINE_FILTER_MIN_CUTOFF", _SPINE_FILTER_MIN_CUTOFF))
        spine_filter_beta = float(os.getenv("SPINE_FILTER_BETA", _SPINE_FILTER_BETA))
        spine_filter_d_cutoff = float(os.getenv("SPINE_FILTER_D_CUTOFF", _SPINE_FILTER_D_CUTOFF))
        self._spine_thoracic_angle_filter = OneEuroFilter(
            spine_filter_min_cutoff, spine_filter_beta, spine_filter_d_cutoff)
        self._spine_lumbar_angle_filter = OneEuroFilter(
            spine_filter_min_cutoff, spine_filter_beta, spine_filter_d_cutoff)
        spine_hip_pos_min_cutoff = float(
            os.getenv("SPINE_HIP_POSITION_MIN_CUTOFF", _SPINE_HIP_POSITION_MIN_CUTOFF))
        spine_hip_pos_beta = float(os.getenv("SPINE_HIP_POSITION_BETA", _SPINE_HIP_POSITION_BETA))
        spine_hip_pos_d_cutoff = float(
            os.getenv("SPINE_HIP_POSITION_D_CUTOFF", _SPINE_HIP_POSITION_D_CUTOFF))
        self._spine_hip_position_filter_x = OneEuroFilter(
            spine_hip_pos_min_cutoff, spine_hip_pos_beta, spine_hip_pos_d_cutoff)
        self._spine_hip_position_filter_y = OneEuroFilter(
            spine_hip_pos_min_cutoff, spine_hip_pos_beta, spine_hip_pos_d_cutoff)
        spine_shoulder_pos_min_cutoff = float(
            os.getenv("SPINE_SHOULDER_POSITION_MIN_CUTOFF", _SPINE_SHOULDER_POSITION_MIN_CUTOFF))
        spine_shoulder_pos_beta = float(
            os.getenv("SPINE_SHOULDER_POSITION_BETA", _SPINE_SHOULDER_POSITION_BETA))
        spine_shoulder_pos_d_cutoff = float(
            os.getenv("SPINE_SHOULDER_POSITION_D_CUTOFF", _SPINE_SHOULDER_POSITION_D_CUTOFF))
        self._spine_shoulder_position_filter_x = OneEuroFilter(
            spine_shoulder_pos_min_cutoff, spine_shoulder_pos_beta, spine_shoulder_pos_d_cutoff)
        self._spine_shoulder_position_filter_y = OneEuroFilter(
            spine_shoulder_pos_min_cutoff, spine_shoulder_pos_beta, spine_shoulder_pos_d_cutoff)
        spine_knee_pos_min_cutoff = float(
            os.getenv("SPINE_KNEE_POSITION_MIN_CUTOFF", _SPINE_KNEE_POSITION_MIN_CUTOFF))
        spine_knee_pos_beta = float(os.getenv("SPINE_KNEE_POSITION_BETA", _SPINE_KNEE_POSITION_BETA))
        spine_knee_pos_d_cutoff = float(
            os.getenv("SPINE_KNEE_POSITION_D_CUTOFF", _SPINE_KNEE_POSITION_D_CUTOFF))
        self._spine_knee_position_filter_x = OneEuroFilter(
            spine_knee_pos_min_cutoff, spine_knee_pos_beta, spine_knee_pos_d_cutoff)
        self._spine_knee_position_filter_y = OneEuroFilter(
            spine_knee_pos_min_cutoff, spine_knee_pos_beta, spine_knee_pos_d_cutoff)
        self._spine_error_logged = False
        self._spine_debug = os.getenv("SPINE_DEBUG", "").lower() in ("1", "true", "yes")
        print(f"[pose_engine] spine sample interval: {self._spine_sample_interval}s "
              f"({'from SPINE_SAMPLE_INTERVAL_SECS env var' if 'SPINE_SAMPLE_INTERVAL_SECS' in os.environ else 'default'})")

    def seek_to_start(self):
        with self._lock:
            self._seek_start = True

    def subscribe(self, callback: Callable[[PoseFrame, np.ndarray], None]):
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable):
        with self._lock:
            self._subscribers = [s for s in self._subscribers if s != callback]

    def set_highlight_joints(self, joints: set[str]):
        with self._lock:
            self._highlight_joints = set(joints)

    def start(self, source=0):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, args=(source,), daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _run(self, source):
        cap = cv2.VideoCapture(source)
        is_file = isinstance(source, str)
        raw_fps = cap.get(cv2.CAP_PROP_FPS) if is_file else 0.0
        frame_delay = 1.0 / raw_fps if raw_fps > 0 else 0.0
        try:
            start_time = time.time()
            while self._running and cap.isOpened():
                with self._lock:
                    if self._seek_start:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self._seek_start = False
                frame_start = time.time()
                ret, frame = cap.read()
                if not ret:
                    if is_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    self._running = False
                    break

                with self._lock:
                    highlight = frozenset(self._highlight_joints)

                keypoints, annotated = self._backend.process(frame, highlight)

                pose_frame = PoseFrame(timestamp=time.time() - start_time)
                if keypoints:
                    angles = calculate_angles(keypoints)
                    for k, v in angles.items():
                        setattr(pose_frame, k, v)
                    pose_frame.keypoints = keypoints

                    if _should_sample_spine(self._last_spine_sample, frame_start,
                                             self._spine_sample_interval):
                        self._last_spine_sample = frame_start
                        spine_points = None  # cleared unless this attempt succeeds below
                        zone_labels = None
                        anchor_points = None
                        thoracic_vertex_pt = None
                        lumbar_vertex_pt = None
                        smoothed_angles = None  # cleared unless this attempt succeeds below
                        try:
                            hip_is_left = keypoints.get("left_hip") is not None
                            hip = keypoints.get("left_hip") or keypoints.get("right_hip")
                            shoulder = keypoints.get("left_shoulder") or keypoints.get("right_shoulder")
                            knee = keypoints.get("left_knee") if hip_is_left else keypoints.get("right_knee")
                            in_profile = False
                            if hip and shoulder:
                                torso_span = math.hypot(shoulder[0] - hip[0], shoulder[1] - hip[1])
                                if torso_span > 1e-6:
                                    lateral_ratio = pose_frame.shoulder_lateral_span / torso_span
                                    in_profile = lateral_ratio <= self._spine_max_shoulder_lateral_ratio
                            if hip and shoulder and min(hip[3], shoulder[3]) >= 0.3 and in_profile:
                                h, w = frame.shape[:2]
                                raw_hip_px = (hip[0] * w, hip[1] * h)
                                shoulder_px = (shoulder[0] * w, shoulder[1] * h)
                                # Smooth the hip keypoint's pixel position (heavier than
                                # YOLOBackend's own per-frame smoothing, see
                                # _SPINE_HIP_POSITION_MIN_CUTOFF) before deriving anything
                                # else from it -- chord_len, n_raw, the contour and the
                                # drawn hip-end anchor are all consistently based on this
                                # same smoothed point, so there's no risk of the mismatch
                                # the earlier (reverted) chord-length-only smoothing had.
                                hip_px = (
                                    self._spine_hip_position_filter_x.filter(raw_hip_px[0], frame_start),
                                    self._spine_hip_position_filter_y.filter(raw_hip_px[1], frame_start))
                                # Smooth the shoulder keypoint's pixel position the same way
                                # (see _SPINE_SHOULDER_POSITION_MIN_CUTOFF) -- both chord ends
                                # need equal damping, otherwise an unsmoothed shoulder end
                                # still jitters the whole rotated ROI (and thus the drawn
                                # line's on-screen position) even while the hip end is stable.
                                shoulder_px = (
                                    self._spine_shoulder_position_filter_x.filter(shoulder_px[0], frame_start),
                                    self._spine_shoulder_position_filter_y.filter(shoulder_px[1], frame_start))
                                # Shift the hip side uniformly toward the shoulder (see
                                # _SPINE_SCAN_SHIFT_FRAC) -- a pure translation along the
                                # shoulder-hip direction -- to trim the buttock curvature at
                                # the bottom. There's no equivalent landmark to reach for on
                                # the hip side, so this stays percentage-based. The fraction
                                # itself is NOT constant: how much of the buttock silhouette
                                # needs trimming depends on hip flexion (see
                                # _spine_scan_shift_frac's docstring), so it's interpolated
                                # live between the standing and bent-forward reference values
                                # using a hip-flexion angle (0=standing straight, increases as
                                # the person hinges forward at the hip).
                                #
                                # Deliberately NOT pose_frame.left_hip_bend_2d/
                                # right_hip_bend_2d here -- those are computed from the pose
                                # backend's OWN (more lightly smoothed) keypoints, a different
                                # lag than hip_px/shoulder_px above (heavier SPINE_*_POSITION
                                # smoothing). Standing up quickly from a squat exposed that
                                # mismatch: the less-lagged bend value would already read
                                # "standing" while hip_px/shoulder_px were still catching up,
                                # transiently selecting the wrong (too small) shift fraction
                                # and dropping the drawn line's bottom point onto the buttock
                                # for a moment. Computing the bend angle from the SAME smoothed
                                # hip_px/shoulder_px (plus a matching smoothed knee_px) keeps
                                # both quantities on the same lag, removing that mismatch.
                                if knee is not None and knee[3] >= 0.3:
                                    knee_raw_px = (knee[0] * w, knee[1] * h)
                                    knee_px = (
                                        self._spine_knee_position_filter_x.filter(knee_raw_px[0], frame_start),
                                        self._spine_knee_position_filter_y.filter(knee_raw_px[1], frame_start))
                                    hip_bend_deg = _hip_bend_deg(shoulder_px, hip_px, knee_px)
                                else:
                                    # No visible knee this tick -- fall back to the
                                    # backend-keypoint-derived value rather than guessing.
                                    hip_bend_deg = (pose_frame.left_hip_bend_2d if hip_is_left
                                                     else pose_frame.right_hip_bend_2d)
                                scan_shift_frac = _spine_scan_shift_frac(
                                    hip_bend_deg, self._spine_scan_shift_frac_standing,
                                    self._spine_scan_shift_frac,
                                    self._spine_scan_shift_bend_low_deg,
                                    self._spine_scan_shift_bend_high_deg)
                                hip_shift_dx = (shoulder_px[0] - hip_px[0]) * scan_shift_frac
                                hip_shift_dy = (shoulder_px[1] - hip_px[1]) * scan_shift_frac
                                scan_hip_px = (hip_px[0] + hip_shift_dx, hip_px[1] + hip_shift_dy)

                                # Shoulder side: its OWN fixed percentage-of-chord shift
                                # (_SPINE_SHOULDER_SCAN_SHIFT_FRAC), independent of the hip
                                # side's standing/bent-adaptive fraction above -- see that
                                # constant's docstring for why the two ends can no longer
                                # share one fraction/vector. An earlier version of this
                                # reached toward the real ear keypoint instead, which was
                                # reverted: even projected onto the hip-shoulder axis (to
                                # avoid rotating the ROI's coordinate system toward the
                                # head), the resulting anchor point could still land far up
                                # near/behind the head when the neck is long or the ear
                                # reading noisy, visibly wrong in the on-screen overlay and
                                # unreliable for the angle math. The fixed, modest shift
                                # below stays anchored close to the real shoulder and has
                                # not shown this failure mode.
                                shoulder_shift_dx = (shoulder_px[0] - hip_px[0]) * self._spine_shoulder_scan_shift_frac
                                shoulder_shift_dy = (shoulder_px[1] - hip_px[1]) * self._spine_shoulder_scan_shift_frac
                                scan_shoulder_px = (shoulder_px[0] + shoulder_shift_dx, shoulder_px[1] + shoulder_shift_dy)
                                roi_result = crop_and_rotate_roi(frame, scan_hip_px, scan_shoulder_px)
                                if roi_result:
                                    roi_image, hip_point, shoulder_point, chord_len, M = roi_result
                                    mask = self._backend.get_segmentation_mask(roi_image)
                                    if mask is not None:
                                        profiles = extract_back_contour(mask, hip_point, shoulder_point)
                                        if profiles:
                                            face_left = facing_left(keypoints)
                                            if face_left is not None:
                                                left_profile, right_profile = profiles
                                                back_profile = right_profile if face_left else left_profile
                                                full_points = back_contour_points_in_frame(
                                                    M, hip_point, shoulder_point, back_profile,
                                                    on_right_side=face_left)
                                                # Draw the FULL-resolution contour (one point per
                                                # raw row), not a downsampled handful of dots --
                                                # the 10-bucket downsampling is only meaningful for
                                                # the angle math's vertex search (see
                                                # signed_curvature_profile_indices below); for the
                                                # on-screen line there's no reason to throw away
                                                # resolution the segmentation mask already gave us.
                                                spine_points = full_points
                                                zone_labels = [
                                                    spine_zone_for_index(i, len(back_profile), self._spine_zone_display_fraction)
                                                    for i in range(len(back_profile))
                                                ]
                                                pose_frame.spine_curvature_ratio = signed_curvature_ratio(
                                                    left_profile, right_profile, chord_len, face_left)
                                                profile_indices = signed_curvature_profile_indices(
                                                    back_profile, chord_len, self._spine_contour_point_count)
                                                if profile_indices is not None:
                                                    if self._spine_debug:
                                                        n_raw = len(back_profile)
                                                        boundary_idx_preview = min(
                                                            n_raw - 1, max(0, round(
                                                                self._spine_bend_boundary_fraction * (n_raw - 1))))
                                                        print(f"[spine_debug] chord_len={chord_len:.1f} "
                                                              f"n_raw={n_raw} boundary_idx={boundary_idx_preview} "
                                                              f"trunk_lean_2d={pose_frame.trunk_lean_2d:.1f} "
                                                              f"hip_bend_deg={hip_bend_deg:.1f} "
                                                              f"scan_shift_frac={scan_shift_frac:.3f} "
                                                              f"raw_hip_px=({raw_hip_px[0]:.1f},{raw_hip_px[1]:.1f}) "
                                                              f"hip_px=({hip_px[0]:.1f},{hip_px[1]:.1f}) "
                                                              f"shoulder_px=({shoulder_px[0]:.1f},{shoulder_px[1]:.1f}) "
                                                              f"scan_hip_px=({scan_hip_px[0]:.1f},{scan_hip_px[1]:.1f})")
                                                    raw_thoracic_deg, raw_lumbar_deg = spine_zone_bend_angles(
                                                        back_profile, chord_len, profile_indices,
                                                        self._spine_zone_fraction,
                                                        self._spine_bend_boundary_fraction)
                                                    if raw_thoracic_deg is not None and raw_lumbar_deg is not None:
                                                        smoothed_thoracic = self._spine_thoracic_angle_filter.filter(
                                                            raw_thoracic_deg, frame_start)
                                                        smoothed_lumbar = self._spine_lumbar_angle_filter.filter(
                                                            raw_lumbar_deg, frame_start)
                                                        pose_frame.spine_thoracic_bend_2d = smoothed_thoracic
                                                        pose_frame.spine_lumbar_bend_2d = smoothed_lumbar
                                                        smoothed_angles = (smoothed_thoracic, smoothed_lumbar)

                                                        anchors = spine_bend_anchor_indices(
                                                            back_profile, profile_indices,
                                                            self._spine_zone_fraction,
                                                            self._spine_bend_boundary_fraction)
                                                        if anchors is not None:
                                                            shoulder_idx, thoracic_idx, boundary_idx, lumbar_idx, hip_idx = anchors
                                                            anchor_points = sparse_points_in_frame(
                                                                M, hip_point, shoulder_point, back_profile,
                                                                [shoulder_idx, boundary_idx, hip_idx],
                                                                on_right_side=face_left)
                                                            vertex_points = sparse_points_in_frame(
                                                                M, hip_point, shoulder_point, back_profile,
                                                                [thoracic_idx, lumbar_idx],
                                                                on_right_side=face_left)
                                                            thoracic_vertex_pt, lumbar_vertex_pt = vertex_points
                                                            if self._spine_debug:
                                                                shoulder_anchor_pt, boundary_anchor_pt, hip_anchor_pt = anchor_points
                                                                print(f"[spine_debug] shoulder_anchor={shoulder_anchor_pt} "
                                                                      f"boundary_anchor={boundary_anchor_pt} "
                                                                      f"hip_anchor={hip_anchor_pt}")
                        except Exception as e:
                            # A failure anywhere in the spine-sampling pipeline (e.g. a
                            # missing/corrupt segmentation model file, an internal
                            # ultralytics/torch error) must degrade to "no sample this
                            # tick" rather than crashing the capture loop — the rest of
                            # _run() (frame capture, angle calculation, subscriber
                            # callbacks) must keep working regardless. Logged once (not
                            # every ~250ms tick) so a persistent failure is still visible.
                            if not self._spine_error_logged:
                                self._spine_error_logged = True
                                print(f"[pose_engine] spine sampling failed (will keep "
                                      f"retrying silently, this is logged once only): {e}")
                        # Replace the cached points on every sampling attempt, success or
                        # not -- a failed attempt (lost person, bad mask, exception) must
                        # clear a stale line rather than let it linger from the last
                        # successful sample.
                        self._last_spine_points = spine_points
                        self._last_spine_zone_labels = zone_labels
                        if smoothed_angles is None:
                            # No fresh angles this tick (lost person, bad mask, wrong
                            # orientation, exception) -- reset both filters so a stale
                            # _t_prev doesn't blend a much-later fresh reading with an
                            # ancient one, producing an unpredictable dx_hat. Mirrors
                            # _KeypointState's reset-on-loss handling in pose_backends.py.
                            self._spine_thoracic_angle_filter.reset()
                            self._spine_lumbar_angle_filter.reset()
                            self._spine_hip_position_filter_x.reset()
                            self._spine_hip_position_filter_y.reset()
                            self._spine_shoulder_position_filter_x.reset()
                            self._spine_shoulder_position_filter_y.reset()
                            self._spine_knee_position_filter_x.reset()
                            self._spine_knee_position_filter_y.reset()
                            # Clear the measurement-triangle overlay too -- a failed tick
                            # must not leave a stale triangle/label hanging on screen.
                            self._last_spine_anchor_points = None
                            self._last_spine_thoracic_vertex = None
                            self._last_spine_lumbar_vertex = None
                            self._last_spine_thoracic_deg = None
                            self._last_spine_lumbar_deg = None
                        else:
                            self._last_spine_anchor_points = anchor_points
                            self._last_spine_thoracic_vertex = thoracic_vertex_pt
                            self._last_spine_lumbar_vertex = lumbar_vertex_pt
                            self._last_spine_thoracic_deg = smoothed_angles[0]
                            self._last_spine_lumbar_deg = smoothed_angles[1]

                    if self._last_spine_points:
                        zone_points = list(zip(self._last_spine_points,
                                                self._last_spine_zone_labels or []))
                        draw_spine_zones(
                            annotated, zone_points,
                            self._last_spine_anchor_points or [],
                            self._last_spine_thoracic_vertex, self._last_spine_lumbar_vertex,
                            self._last_spine_thoracic_deg, self._last_spine_lumbar_deg)

                with self._lock:
                    subs = list(self._subscribers)

                for cb in subs:
                    try:
                        cb(pose_frame, annotated)
                    except Exception:
                        pass

                if frame_delay > 0:
                    elapsed = time.time() - frame_start
                    remaining = frame_delay - elapsed
                    if remaining > 0:
                        time.sleep(remaining)
        finally:
            cap.release()
            self._backend.close()
