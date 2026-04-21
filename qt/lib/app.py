import json
import math
import sys
import time
from pathlib import Path
from typing import Callable

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image

from core.pose_engine import PoseEngine
from core.data_contract import PoseFrame
from session_runner import SessionRunner

_SECS_PER_WORD = 0.3   # ~200 WPM, used for instructions display only
_MIN_DISPLAY_SECS = 3
_VOICE_DONE_TIMEOUT_SECS = 30  # safety fallback if voice module is absent or crashes


def _display_ms(text: str) -> int:
    words = len(text.split())
    return int(max(_MIN_DISPLAY_SECS, words * _SECS_PER_WORD) * 1000)


class StandaloneApp(ctk.CTk):
    def __init__(self, config: dict, analysis_module,
                 feedback_dir: Path = None):
        super().__init__()
        self._config = config
        self._module = analysis_module
        self._feedback_dir = feedback_dir or Path(sys.argv[0]).resolve().parent
        self.title(config.get("name", "Exercise"))
        self.geometry("1100x700")
        ctk.set_appearance_mode("dark")
        self._current_frame = None
        self.protocol("WM_DELETE_WINDOW", self.quit)
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
            self, self._config, self._module,
            feedback_dir=self._feedback_dir,
            on_done=lambda _: self.quit(),
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

        self.after(_display_ms(instructions), on_start)


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
                 feedback_dir: Path,
                 on_done: Callable[[dict], None], **kwargs):
        super().__init__(parent, **kwargs)
        self._config = config
        self._on_done = on_done
        self._feedback_dir = feedback_dir
        self._engine = PoseEngine()
        self._runner = SessionRunner(config, analysis_module)
        self._feedback_lines: list[str] = []
        self._exercise_secs: int = config.get("session_duration_secs", 60)
        self._feedback_panel_visible = False
        self._feedback_end_scheduled = False
        self._last_message_ts: float = 0.0
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
        self._relevant_joints = self._runner.get_relevant_joints()
        self._build_joint_labels()
        instructions = self._runner.get_instructions()
        if instructions:
            self._write_message(". ".join(instructions), "instructions")
        self._runner.start_countdown()
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

        if state == "countdown":
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
                    self._write_message(count_str, "countdown")
                else:
                    ts = cv2.getTextSize("GO!", cv2.FONT_HERSHEY_SIMPLEX, 4.0, 8)[0]
                    cv2.putText(display, "GO!",
                                ((w - ts[0]) // 2, (h + ts[1]) // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 0), 8)
                    self._write_message("Go!", "countdown")
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

        feedback_lines = result.get("feedback_lines")
        if feedback_lines is not None:
            self._feedback_lines = list(feedback_lines)
            self._write_message(". ".join(feedback_lines), "feedback")
            self._feedback_end_scheduled = False

        if state == "feedback" and not self._feedback_panel_visible:
            self._show_feedback_panel(result["round_number"])
        elif state != "feedback" and self._feedback_panel_visible:
            self._hide_feedback_panel()

        if state == "feedback" and not self._feedback_end_scheduled:
            self._feedback_end_scheduled = True
            deadline = time.time() + _VOICE_DONE_TIMEOUT_SECS
            self.after(200, lambda: self._check_voice_done(self._last_message_ts, deadline))

        if not self._feedback_panel_visible:
            self._update_camera(display)

        if state == "exercise" and self._joint_value_labels:
            self._update_joint_labels(pose_frame)

        label_map = {
            "countdown": (str(result["round_number"] + 1), "Get ready!"),
            "exercise": (str(result["round_rep_count"]), ""),
            "feedback": (
                str(result["round_rep_count"]),
                self._feedback_lines[0] if self._feedback_lines else "",
            ),
        }
        rep_text, fb_text = label_map.get(state, ("—", ""))
        self.rep_label.configure(text=rep_text)
        self.feedback_label.configure(text=fb_text)

    def _check_voice_done(self, ts: float, deadline: float):
        if not self._active:
            return
        done = False
        try:
            done_file = self._feedback_dir / "voice_done.json"
            if done_file.exists():
                data = json.loads(done_file.read_text(encoding="utf-8"))
                if data.get("timestamp", 0) >= ts:
                    done = True
        except Exception:
            pass
        if done or time.time() >= deadline:
            self._runner.end_feedback()
        else:
            self.after(200, lambda: self._check_voice_done(ts, deadline))

    def _write_message(self, text: str, msg_type: str):
        ts = time.time()
        payload = {"timestamp": ts, "type": msg_type, "text": text}
        try:
            path = self._feedback_dir / "message.json"
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                            encoding="utf-8")
            self._last_message_ts = ts
        except Exception as e:
            print(f"[app] failed to write message.json: {e}")

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
        self._engine.stop()
        self._engine.unsubscribe(self._on_frame)
        self._on_done(self._runner.get_summary())


class SummaryFrame(ctk.CTkFrame):
    """Unused in standalone mode — kept for potential future use."""

    def __init__(self, parent, summary: dict, exercise_name: str,
                 on_done: Callable, **kwargs):
        super().__init__(parent, **kwargs)
        ctk.CTkLabel(self, text="Session Complete",
                     font=ctk.CTkFont(size=28, weight="bold")).pack(pady=(30, 6))
        ctk.CTkLabel(self, text=exercise_name,
                     text_color="gray", font=ctk.CTkFont(size=16)).pack()
        ctk.CTkButton(self, text="Done", height=44, width=160, command=on_done).pack(pady=30)
