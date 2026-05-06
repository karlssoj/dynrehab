from __future__ import annotations

import cv2
import numpy as np

_FONT = cv2.FONT_HERSHEY_SIMPLEX


class Display:
    def __init__(self, window_name: str = "QT Exercise", exercise_secs: int = 60):
        self._window = window_name
        self._exercise_secs = exercise_secs
        self._last_feedback: list[str] = []
        cv2.namedWindow(self._window, cv2.WINDOW_NORMAL)

    def update(self, frame: np.ndarray, state: dict) -> bool:
        """Render state overlays and show in window. Returns False when ESC is pressed."""
        display = frame.copy()
        h, w = display.shape[:2]
        s = state.get("state", "")

        new_feedback = state.get("feedback_lines")
        if new_feedback:
            self._last_feedback = list(new_feedback)

        if s == "calibration":
            self._render_calibration(display, h, w, state)
        elif s == "countdown":
            self._render_countdown(display, h, w, state)
        elif s == "exercise":
            self._render_exercise(display, h, w, state)
        elif s == "feedback":
            self._render_feedback(display, h, w)

        cv2.imshow(self._window, display)
        return cv2.waitKey(1) != 27

    def _render_calibration(self, display: np.ndarray, h: int, w: int, state: dict):
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)
        cv2.putText(display, "Positioning", (30, 46), _FONT, 1.0, (0, 220, 255), 2)
        msg = state.get("calibration_message", "")
        if msg:
            color = (0, 255, 0) if state.get("calibration_status") == "ready" else (0, 180, 255)
            cv2.putText(display, msg.rstrip("."), (30, h - 40), _FONT, 1.0, color, 2)

    def _render_countdown(self, display: np.ndarray, h: int, w: int, state: dict):
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)
        cnt = state.get("countdown_speak")
        remaining = state.get("time_remaining", 0.0)
        num = cnt if cnt is not None else max(0, int(remaining))
        if num > 0:
            text = str(num)
            ts = cv2.getTextSize(text, _FONT, 5.0, 8)[0]
            cv2.putText(display, text,
                        ((w - ts[0]) // 2, (h + ts[1]) // 2),
                        _FONT, 5.0, (0, 255, 255), 8)
        else:
            ts = cv2.getTextSize("GO!", _FONT, 4.0, 8)[0]
            cv2.putText(display, "GO!",
                        ((w - ts[0]) // 2, (h + ts[1]) // 2),
                        _FONT, 4.0, (0, 255, 0), 8)

    def _render_exercise(self, display: np.ndarray, h: int, w: int, state: dict):
        reps = state.get("total_reps", 0)
        time_left = state.get("time_remaining", 0.0)
        cv2.putText(display, f"Reps: {reps}", (10, 35), _FONT, 1.0, (0, 255, 0), 2)
        cv2.putText(display, f"Time: {max(0.0, time_left):.0f}s", (10, 65), _FONT, 0.7, (0, 255, 255), 2)
        elapsed = self._exercise_secs - time_left
        if self._exercise_secs > 0:
            bar_w = int(min(1.0, elapsed / self._exercise_secs) * w)
            cv2.rectangle(display, (0, h - 8), (bar_w, h), (0, 200, 255), -1)
        joint_values = state.get("joint_values", {})
        y = 35
        for label_text, val in joint_values.items():
            text = f"{label_text}: {val:.1f}" if isinstance(val, float) else f"{label_text}: {val}"
            tw = cv2.getTextSize(text, _FONT, 0.55, 1)[0][0]
            cv2.putText(display, text, (w - tw - 10, y), _FONT, 0.55, (200, 200, 200), 1)
            y += 24

    def _render_feedback(self, display: np.ndarray, h: int, w: int):
        feedback = self._last_feedback
        if not feedback:
            return
        line_h = 30
        block_h = len(feedback) * line_h + 20
        top_y = max(0, h - block_h)
        overlay = display.copy()
        cv2.rectangle(overlay, (0, top_y), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
        for i, line in enumerate(feedback):
            cv2.putText(display, line,
                        (10, top_y + 20 + i * line_h),
                        _FONT, 0.65, (255, 255, 100), 2)

    def close(self):
        cv2.destroyAllWindows()
