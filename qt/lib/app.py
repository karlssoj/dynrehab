import math
import sys
from pathlib import Path
from typing import Callable

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image

from core.pose_engine import PoseEngine
from core.data_contract import PoseFrame
from session_runner import SessionRunner
from tts_service import TTSService


class StandaloneApp(ctk.CTk):
    def __init__(self, config: dict, analysis_module):
        super().__init__()
        self._config = config
        self._module = analysis_module
        self.title(config.get("name", "Exercise"))
        self.geometry("1100x700")
        ctk.set_appearance_mode("dark")
        self._current_frame = None
        self._show_instructions()

    def _clear(self):
        if self._current_frame is not None:
            self._current_frame.destroy()
            self._current_frame = None

    def _show_instructions(self):
        self._clear()
        self._current_frame = InstructionsFrame(
            self, self._config, on_start=self._show_session
        )
        self._current_frame.pack(fill="both", expand=True)

    def _show_session(self):
        self._clear()
        self._current_frame = SessionFrame(
            self, self._config, self._module, on_done=self._show_summary
        )
        self._current_frame.pack(fill="both", expand=True)

    def _show_summary(self, summary: dict):
        self._clear()
        self._current_frame = SummaryFrame(
            self, summary, self._config.get("name", "Exercise"), on_done=self.quit
        )
        self._current_frame.pack(fill="both", expand=True)


class InstructionsFrame(ctk.CTkFrame):
    def __init__(self, parent, config: dict, on_start: Callable, **kwargs):
        super().__init__(parent, **kwargs)
        name = config.get("name", "Exercise")
        instructions = config.get("client_instructions", "")

        ctk.CTkLabel(
            self, text=name, font=ctk.CTkFont(size=32, weight="bold")
        ).pack(pady=(60, 16))

        ctk.CTkLabel(
            self, text="Instructions", font=ctk.CTkFont(size=18), text_color="gray"
        ).pack(pady=(0, 12))

        box = ctk.CTkTextbox(
            self, width=700, height=300, font=ctk.CTkFont(size=16),
            wrap="word", state="normal"
        )
        box.pack(pady=(0, 30))
        box.insert("1.0", instructions)
        box.configure(state="disabled")

        self.after(3000, on_start)


class SessionFrame(ctk.CTkFrame):
    _BEND_FIELDS = {
        "left_elbow_angle", "right_elbow_angle",
        "left_knee_angle", "right_knee_angle",
        "left_hip_angle", "right_hip_angle",
        "left_ankle_angle", "right_ankle_angle",
        "left_hka_alignment", "right_hka_alignment",
    }
    _VALGUS_FIELDS = {"left_knee_valgus", "right_knee_valgus"}

    def __init__(self, parent, config: dict, analysis_module,
                 on_done: Callable[[dict], None], **kwargs):
        super().__init__(parent, **kwargs)
        self._config = config
        self._on_done = on_done
        self._engine = PoseEngine()
        self._runner = SessionRunner(config, analysis_module)
        self._tts = TTSService(cooldown_seconds=0.0)
        self._feedback_lines: list[str] = []
        self._instructions_spoken = False
        self._countdown_triggered = False
        self._exercise_secs: int = config.get("session_duration_secs", 60)
        self._feedback_panel_visible = False
        self._feedback_end_scheduled = False
        self._feedback_tts_started = False
        self._joint_value_labels: dict[str, ctk.CTkLabel] = {}
        self._active = True
        self._build()
        self._start()

    def _build(self):
        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x", padx=20, pady=(12, 0))
        ctk.CTkLabel(
            top, text=self._config.get("name", "Exercise"),
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(side="left")
        ctk.CTkButton(
            top, text="End Session", fg_color="#c0392b", hover_color="#922b21",
            command=self._end_session
        ).pack(side="right")

        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=20, pady=10)
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=1)

        self.camera_label = ctk.CTkLabel(
            main, text="Starting camera…", width=768, height=576, fg_color="#1a1a2e"
        )
        self.camera_label.grid(row=0, column=0, sticky="nsew", padx=(0, 15))

        self._feedback_panel = ctk.CTkFrame(main, fg_color="#000000", corner_radius=0)
        self._feedback_round_label = ctk.CTkLabel(
            self._feedback_panel, text="",
            font=ctk.CTkFont(size=22, weight="bold"), text_color="#00dcff"
        )
        self._feedback_round_label.pack(anchor="w", padx=30, pady=(24, 12))
        self._feedback_body = ctk.CTkTextbox(
            self._feedback_panel, fg_color="#111111", text_color="white",
            font=ctk.CTkFont(size=18), wrap="word", corner_radius=8, state="disabled"
        )
        self._feedback_body.pack(fill="both", expand=True, padx=30, pady=(0, 24))

        right = ctk.CTkFrame(main)
        right.grid(row=0, column=1, sticky="nsew")

        ctk.CTkLabel(right, text="Reps", font=ctk.CTkFont(size=16)).pack(pady=(20, 4))
        self.rep_label = ctk.CTkLabel(
            right, text="0", font=ctk.CTkFont(size=64, weight="bold")
        )
        self.rep_label.pack()

        ctk.CTkLabel(right, text="Live Values", font=ctk.CTkFont(size=14)).pack(pady=(20, 4))
        self._joints_frame = ctk.CTkFrame(right, fg_color="#111122", corner_radius=8)
        self._joints_frame.pack(fill="x", padx=10, pady=(0, 4))

        ctk.CTkLabel(right, text="Feedback", font=ctk.CTkFont(size=14)).pack(pady=(16, 6))
        self.feedback_label = ctk.CTkLabel(
            right, text="", wraplength=200, justify="center",
            font=ctk.CTkFont(size=13)
        )
        self.feedback_label.pack(padx=10)

    def _start(self):
        instructions = self._runner.get_instructions()
        if instructions:
            self._tts.speak_immediate(". ".join(instructions))
            self._instructions_spoken = True
        else:
            self._runner.start_countdown()
            self._countdown_triggered = True
        self._relevant_joints = self._runner.get_relevant_joints()
        self._build_joint_labels()
        self._engine.subscribe(self._on_frame)
        self._engine.start()

    def _on_frame(self, pose_frame: PoseFrame, annotated: np.ndarray):
        result = self._runner.process_frame(pose_frame.to_dict())
        self.after(0, lambda r=result, f=annotated, p=pose_frame: self._update_ui(r, f, p))

    def _update_ui(self, result: dict, frame: np.ndarray, pose_frame: PoseFrame):
        if not self._active:
            return
        state = result["state"]
        display = frame.copy()
        h, w = display.shape[:2]

        if state == "instructions":
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
            text = "Listening to instructions..."
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            tx = (w - text_size[0]) // 2
            cv2.putText(display, text, (tx, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
            if (self._instructions_spoken and not self._countdown_triggered
                    and not self._tts.is_speaking()):
                self._countdown_triggered = True
                self._runner.start_countdown()

        elif state == "countdown":
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)
            round_num = result["round_number"] + 1
            cv2.putText(display, f"Round {round_num} starting...",
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
            countdown_speak = result.get("countdown_speak")
            if countdown_speak is not None:
                if countdown_speak > 0:
                    count_str = str(countdown_speak)
                    ts = cv2.getTextSize(count_str, cv2.FONT_HERSHEY_SIMPLEX, 5.0, 8)[0]
                    cv2.putText(display, count_str,
                                ((w - ts[0]) // 2, (h + ts[1]) // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0, 255, 255), 8)
                else:
                    ts = cv2.getTextSize("GO!", cv2.FONT_HERSHEY_SIMPLEX, 4.0, 8)[0]
                    cv2.putText(display, "GO!",
                                ((w - ts[0]) // 2, (h + ts[1]) // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 0), 8)
            else:
                remaining = result.get("time_remaining", 0.0)
                count_num = max(0, math.ceil(remaining))
                if count_num > 0:
                    count_str = str(count_num)
                    ts = cv2.getTextSize(count_str, cv2.FONT_HERSHEY_SIMPLEX, 5.0, 8)[0]
                    cv2.putText(display, count_str,
                                ((w - ts[0]) // 2, (h + ts[1]) // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0, 255, 255), 8)

        elif state == "exercise":
            time_left = result.get("time_remaining", 0.0)
            cv2.putText(display,
                        f"Round {result['round_number']}  |  Reps: {result['round_rep_count']}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, f"Time left: {max(0.0, time_left):.1f}s",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            elapsed = self._exercise_secs - time_left
            bar_width = int(min(1.0, elapsed / self._exercise_secs) * w)
            cv2.rectangle(display, (0, h - 8), (bar_width, h), (0, 200, 255), -1)

        countdown_speak = result.get("countdown_speak")
        if countdown_speak is not None and state == "countdown":
            self._tts.speak(str(countdown_speak) if countdown_speak > 0 else "Go!")

        feedback_lines = result.get("feedback_lines")
        if feedback_lines is not None:
            self._feedback_lines = list(feedback_lines)
            self._tts.speak_immediate(". ".join(feedback_lines))
            self._feedback_end_scheduled = False
            self._feedback_tts_started = True

        if state == "feedback" and not self._feedback_panel_visible:
            self._show_feedback_panel(result["round_number"])
        elif state != "feedback" and self._feedback_panel_visible:
            self._hide_feedback_panel()

        if state == "feedback" and not self._feedback_end_scheduled:
            if self._tts.is_speaking():
                self._feedback_tts_started = True
            if self._feedback_tts_started and not self._tts.is_speaking():
                self._feedback_end_scheduled = True
                self._runner.end_feedback()

        if not self._feedback_panel_visible:
            self._update_camera(display)

        if state == "exercise" and self._joint_value_labels:
            self._update_joint_labels(pose_frame)

        label_map = {
            "instructions": ("—", "Instructions..."),
            "countdown": (str(result["round_number"] + 1), "Get ready!"),
            "exercise": (str(result["round_rep_count"]), "EXERCISE"),
            "feedback": (
                str(result["round_rep_count"]),
                self._feedback_lines[0] if self._feedback_lines else "",
            ),
        }
        rep_text, fb_text = label_map.get(state, ("—", ""))
        self.rep_label.configure(text=rep_text)
        self.feedback_label.configure(text=fb_text)

    def _update_camera(self, bgr_frame: np.ndarray):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2.resize(rgb, (768, 576)))
        ctk_img = ctk.CTkImage(light_image=img, size=(768, 576))
        self.camera_label.configure(image=ctk_img, text="")
        self.camera_label.image = ctk_img

    def _build_joint_labels(self):
        for widget in self._joints_frame.winfo_children():
            widget.destroy()
        self._joint_value_labels.clear()
        if not self._relevant_joints:
            ctk.CTkLabel(self._joints_frame, text="None configured",
                         text_color="gray", font=ctk.CTkFont(size=11)).pack(pady=6)
            return
        for display_label, key in self._relevant_joints:
            row = ctk.CTkFrame(self._joints_frame, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=3)
            ctk.CTkLabel(row, text=display_label, font=ctk.CTkFont(size=24),
                         text_color="#aaaacc", anchor="w").pack(side="left")
            val_lbl = ctk.CTkLabel(row, text="—", font=ctk.CTkFont(size=28, weight="bold"),
                                   text_color="#00dcff", anchor="e")
            val_lbl.pack(side="right")
            self._joint_value_labels[key] = val_lbl

    def _update_joint_labels(self, pose_frame: PoseFrame):
        for key, lbl in self._joint_value_labels.items():
            val = getattr(pose_frame, key, None)
            if val is None:
                lbl.configure(text="—")
                continue
            val = float(val)
            if key in self._BEND_FIELDS:
                val = 180.0 - val
            if key in self._VALGUS_FIELDS:
                lbl.configure(text=f"{val:+.1f}°")
            else:
                lbl.configure(text=f"{val:.1f}°")

    def _show_feedback_panel(self, round_number: int):
        self._feedback_round_label.configure(text=f"Round {round_number}  —  Feedback")
        lines = [ln.strip() for ln in self._feedback_lines if ln.strip()]
        self._feedback_body.configure(state="normal")
        self._feedback_body.delete("1.0", "end")
        self._feedback_body.insert("1.0", "\n\n".join(lines))
        self._feedback_body.configure(state="disabled")
        self.camera_label.grid_remove()
        self._feedback_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 15))
        self._feedback_panel_visible = True

    def _hide_feedback_panel(self):
        self._feedback_panel.grid_remove()
        self.camera_label.grid(row=0, column=0, sticky="nsew", padx=(0, 15))
        self._feedback_panel_visible = False

    def _end_session(self):
        self._active = False
        self._tts.stop()
        self._engine.stop()
        self._engine.unsubscribe(self._on_frame)
        self._on_done(self._runner.get_summary())


class SummaryFrame(ctk.CTkFrame):
    def __init__(self, parent, summary: dict, exercise_name: str,
                 on_done: Callable, **kwargs):
        super().__init__(parent, **kwargs)
        ctk.CTkLabel(self, text="Session Complete",
                     font=ctk.CTkFont(size=28, weight="bold")).pack(pady=(30, 6))
        ctk.CTkLabel(self, text=exercise_name,
                     text_color="gray", font=ctk.CTkFont(size=16)).pack()

        stats = ctk.CTkFrame(self, fg_color="transparent")
        stats.pack(pady=20)
        self._stat_box(stats, "Reps", str(summary["rep_count"])).pack(side="left", padx=20)
        self._stat_box(stats, "Quality", f"{summary['quality_pct']}%").pack(side="left", padx=20)
        mins = int(summary.get("duration_seconds", 0) // 60)
        secs = int(summary.get("duration_seconds", 0) % 60)
        self._stat_box(stats, "Duration", f"{mins}:{secs:02d}").pack(side="left", padx=20)

        ctk.CTkLabel(self, text="Feedback Log",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=40, pady=(10, 4))
        log_frame = ctk.CTkScrollableFrame(self, height=250)
        log_frame.pack(fill="x", padx=40, pady=(0, 20))
        feedback_log = summary.get("feedback_log", [])
        if not feedback_log:
            ctk.CTkLabel(log_frame, text="No feedback recorded.",
                         text_color="gray").pack(pady=20)
        else:
            for entry in feedback_log:
                ts = entry.get("timestamp", 0.0)
                msg = entry.get("message", "")
                row = ctk.CTkFrame(log_frame, fg_color="transparent")
                row.pack(fill="x", pady=2)
                ctk.CTkLabel(row, text=f"{ts:.1f}s",
                             text_color="gray", width=60).pack(side="left")
                ctk.CTkLabel(row, text=msg, anchor="w").pack(
                    side="left", fill="x", expand=True)

        ctk.CTkButton(self, text="Done", height=44, width=160, command=on_done).pack(pady=10)

    @staticmethod
    def _stat_box(parent, label: str, value: str) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(parent, width=140, height=100)
        ctk.CTkLabel(frame, text=value,
                     font=ctk.CTkFont(size=36, weight="bold")).pack(pady=(14, 2))
        ctk.CTkLabel(frame, text=label,
                     text_color="gray", font=ctk.CTkFont(size=13)).pack()
        return frame
