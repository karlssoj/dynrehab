import customtkinter as ctk
import sqlite3
import time
import json
import numpy as np
import cv2
from PIL import Image
from core.pose_engine import PoseEngine
from core.data_contract import PoseFrame
from client_app.services.session_service import SessionService
from client_app.services.tts_service import TTSService
from physio_app.services.exercise_service import ExerciseService


class SessionViewFrame(ctk.CTkFrame):
    def __init__(self, parent, db_conn: sqlite3.Connection,
                 exercise_id: str, **kwargs):
        super().__init__(parent, **kwargs)
        self.app = parent
        self.exercise_id = exercise_id
        self.ex_svc = ExerciseService(db_conn)
        self._engine = PoseEngine()
        self._session: SessionService = None
        self._tts = TTSService(cooldown_seconds=0.0)
        self._feedback_lines: list[str] = []
        self._instructions_spoken = False   # True once instructions TTS has been queued
        self._countdown_triggered = False   # True once start_countdown() has been called
        self._exercise_secs: int = 0   # set properly in _start_session
        self._feedback_mode: list[str] = ["after_window"]
        self._last_rep_cue_time: float = 0.0
        self._build()
        self._start_session()

    def _build(self):
        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x", padx=20, pady=(12, 0))
        ctk.CTkButton(top, text="← Home", width=80,
                      command=self._go_home).pack(side="left", padx=(0, 10))
        self.title_label = ctk.CTkLabel(top, text="",
                                        font=ctk.CTkFont(size=18, weight="bold"))
        self.title_label.pack(side="left")
        ctk.CTkButton(top, text="End Session", fg_color="#c0392b",
                      hover_color="#922b21", command=self._end_session).pack(side="right")

        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=20, pady=10)
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=1)

        self.camera_label = ctk.CTkLabel(main, text="Starting camera…",
                                         width=768, height=576, fg_color="#1a1a2e")
        self.camera_label.grid(row=0, column=0, sticky="nsew", padx=(0, 15))

        # Feedback panel — same grid cell, shown instead of camera during feedback state
        self._feedback_panel = ctk.CTkFrame(main, fg_color="#000000", corner_radius=0)
        self._feedback_panel_visible = False
        self._feedback_round_label = ctk.CTkLabel(
            self._feedback_panel, text="",
            font=ctk.CTkFont(size=22, weight="bold"), text_color="#00dcff")
        self._feedback_round_label.pack(anchor="w", padx=30, pady=(24, 12))
        self._feedback_body = ctk.CTkTextbox(
            self._feedback_panel,
            fg_color="#111111", text_color="white",
            font=ctk.CTkFont(size=18), wrap="word",
            corner_radius=8, state="disabled")
        self._feedback_body.pack(fill="both", expand=True, padx=30, pady=(0, 24))

        right = ctk.CTkFrame(main)
        right.grid(row=0, column=1, sticky="nsew")

        ctk.CTkLabel(right, text="Reps",
                     font=ctk.CTkFont(size=16)).pack(pady=(20, 4))
        self.rep_label = ctk.CTkLabel(right, text="0",
                                      font=ctk.CTkFont(size=64, weight="bold"))
        self.rep_label.pack()

        ctk.CTkLabel(right, text="Live Values",
                     font=ctk.CTkFont(size=14)).pack(pady=(20, 4))
        self._joints_frame = ctk.CTkFrame(right, fg_color="#111122", corner_radius=8)
        self._joints_frame.pack(fill="x", padx=10, pady=(0, 4))
        self._joint_value_labels: dict[str, ctk.CTkLabel] = {}

        ctk.CTkLabel(right, text="Feedback",
                     font=ctk.CTkFont(size=14)).pack(pady=(16, 6))
        self.feedback_label = ctk.CTkLabel(right, text="",
                                           wraplength=200, justify="center",
                                           font=ctk.CTkFont(size=13))
        self.feedback_label.pack(padx=10)

    def _start_session(self):
        ex = self.ex_svc.get(self.exercise_id)
        if ex is None:
            return
        self.title_label.configure(text=ex.name)
        module = self.ex_svc.get_active_module(self.exercise_id)
        if module is None:
            self.feedback_label.configure(text="No analysis module found for this exercise.")
            return
        self._exercise_secs = ex.session_duration_secs
        try:
            self._feedback_mode = json.loads(ex.feedback_mode) if ex.feedback_mode else ["after_window"]
        except (json.JSONDecodeError, TypeError):
            self._feedback_mode = ["after_window"]
        self._session = SessionService(exercise_id=self.exercise_id,
                                       module_code=module["code"],
                                       exercise_secs=self._exercise_secs,
                                       feedback_mode=self._feedback_mode,
                                       camera_view=ex.camera_view or "side")
        instructions = self._session.get_instructions()
        if instructions:
            self._tts.speak_immediate(". ".join(instructions))
            self._instructions_spoken = True
        else:
            self._session.start_calibration()
            self._countdown_triggered = True
        self._relevant_joints: list = self._session.get_relevant_joints()
        self._build_joint_labels()
        self._engine.subscribe(self._on_frame)
        self._engine.start()

    def _on_frame(self, pose_frame: PoseFrame, annotated: np.ndarray):
        if self._session is None:
            self.after(0, lambda f=annotated: self._update_camera(f, None))
            return

        result = self._session.process_frame(pose_frame.to_dict())

        self.after(0, lambda r=result, f=annotated, p=pose_frame: self._update_ui(r, f, p))

    def _update_ui(self, result: dict, frame: np.ndarray, pose_frame: PoseFrame):
        state = result["state"]
        display = frame.copy()

        # Draw state overlays onto the OpenCV frame before passing to camera
        h, w = display.shape[:2]

        if state == "instructions":
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
            text = "Listening to instructions..."
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 0.8, 2)[0]
            tx = (w - text_size[0]) // 2
            ty = h // 2
            cv2.putText(display, text, (tx, ty), font, 0.8, (0, 220, 255), 2)
            if (self._instructions_spoken and not self._countdown_triggered
                    and not self._tts.is_speaking()):
                self._countdown_triggered = True
                self._session.start_calibration()

        elif state == "calibration":
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)
            calib_status = result.get("calibration_status", "")
            _STATUS_TEXT = {
                "no_person":         "Step in front of the camera",
                "too_far":           "Move closer to the camera",
                "too_close":         "Step back from the camera",
                "wrong_orientation": "Wrong orientation",
                "ready":             "Good position!",
            }
            status_text = _STATUS_TEXT.get(calib_status, "Positioning...")
            color = (0, 255, 0) if calib_status == "ready" else (0, 180, 255)
            cv2.putText(display, "Positioning",
                        (30, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2)
            cv2.putText(display, status_text,
                        (30, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

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
                    text_size = cv2.getTextSize(count_str, cv2.FONT_HERSHEY_SIMPLEX, 5.0, 8)[0]
                    tx = (w - text_size[0]) // 2
                    ty = (h + text_size[1]) // 2
                    cv2.putText(display, count_str, (tx, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0, 255, 255), 8)
                else:
                    text_size = cv2.getTextSize("GO!", cv2.FONT_HERSHEY_SIMPLEX, 4.0, 8)[0]
                    tx = (w - text_size[0]) // 2
                    ty = (h + text_size[1]) // 2
                    cv2.putText(display, "GO!", (tx, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 0), 8)
            else:
                # Draw the current countdown number based on time_remaining
                remaining = result.get("time_remaining", 0.0)
                import math
                count_num = max(0, math.ceil(remaining))
                if count_num > 0:
                    count_str = str(count_num)
                    text_size = cv2.getTextSize(count_str, cv2.FONT_HERSHEY_SIMPLEX, 5.0, 8)[0]
                    tx = (w - text_size[0]) // 2
                    ty = (h + text_size[1]) // 2
                    cv2.putText(display, count_str, (tx, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0, 255, 255), 8)

        elif state == "exercise":
            time_left = result.get("time_remaining", 0.0)
            round_num = result["round_number"]
            round_reps = result["round_rep_count"]

            cv2.putText(display, f"Round {round_num}  |  Reps: {round_reps}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, f"Time left: {max(0.0, time_left):.1f}s",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            elapsed = self._exercise_secs - time_left
            bar_width = int(min(1.0, elapsed / self._exercise_secs) * w)
            cv2.rectangle(display, (0, h - 8), (bar_width, h), (0, 200, 255), -1)

        calibration_speak = result.get("calibration_speak")
        if calibration_speak:
            self._tts.speak(calibration_speak)

        # Handle countdown speech
        countdown_speak = result.get("countdown_speak")
        if countdown_speak is not None and state == "countdown":
            if countdown_speak > 0:
                self._tts.speak(str(countdown_speak))
            else:
                self._tts.speak("Go!")

        # Handle feedback_lines — only non-None on the first frame of feedback state
        feedback_lines = result.get("feedback_lines")
        if feedback_lines is not None:
            self._feedback_lines = list(feedback_lines)
            self._tts.speak_immediate(". ".join(feedback_lines))
            self._feedback_end_scheduled = False
            self._feedback_tts_started = False   # must observe TTS speaking before ending

        rep_cue = result.get("rep_cue")
        if rep_cue:
            self._tts.speak(rep_cue)
            self.feedback_label.configure(text=rep_cue)
            self._last_rep_cue_time = time.time()

        # Show/hide feedback panel based on state
        if state == "feedback" and not self._feedback_panel_visible:
            self._show_feedback_panel(result["round_number"])
        elif state != "feedback" and self._feedback_panel_visible:
            self._hide_feedback_panel()

        # Poll: wait until TTS has been observed speaking, then end when it stops.
        # Two-stage guard prevents firing on the same frame as speak_immediate() when
        # the TTS thread hasn't started yet and is_speaking() still returns False.
        if state == "feedback" and not getattr(self, "_feedback_end_scheduled", False):
            if self._tts.is_speaking():
                self._feedback_tts_started = True
            if getattr(self, "_feedback_tts_started", False) and not self._tts.is_speaking():
                self._feedback_end_scheduled = True
                self._session.end_feedback()

        # Update camera only when it is visible
        if not self._feedback_panel_visible:
            self._update_camera(display, pose_frame)

        # Update sidebar joint value labels during exercise
        if state == "exercise" and pose_frame is not None and self._joint_value_labels:
            self._update_joint_labels(pose_frame)

        # Update right panel labels based on state
        if state == "instructions":
            self.rep_label.configure(text="—")
            self.feedback_label.configure(text="Instructions...")
        elif state == "calibration":
            self.rep_label.configure(text="—")
            self.feedback_label.configure(text="Positioning...")
        elif state == "countdown":
            self.rep_label.configure(text=str(result["round_number"] + 1))
            self.feedback_label.configure(text="Get ready!")
        elif state == "exercise":
            self.rep_label.configure(text=str(result["round_rep_count"]))
            if time.time() - self._last_rep_cue_time >= 2.0:
                self.feedback_label.configure(text="EXERCISE")
        elif state == "feedback":
            self.rep_label.configure(text=str(result["round_rep_count"]))
            first_line = self._feedback_lines[0] if self._feedback_lines else ""
            self.feedback_label.configure(text=first_line)

    def _update_camera(self, bgr_frame: np.ndarray, pose_frame: PoseFrame | None):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2.resize(rgb, (768, 576)))
        ctk_img = ctk.CTkImage(light_image=img, size=(768, 576))
        self.camera_label.configure(image=ctk_img, text="")
        self.camera_label.image = ctk_img

    # These fields use 180°=straight — convert to 0°=straight for display.
    # Includes both the 3D raw angles AND the frontal-plane HKA alignment angles.
    _BEND_FIELDS = {
        "left_elbow_angle",    "right_elbow_angle",
        "left_knee_angle",     "right_knee_angle",
        "left_hip_angle",      "right_hip_angle",
        "left_ankle_angle",    "right_ankle_angle",
        "left_hka_alignment",  "right_hka_alignment",
    }

    def _build_joint_labels(self):
        """Create one label row per relevant joint in the joints panel."""
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
            ctk.CTkLabel(row, text=display_label,
                         font=ctk.CTkFont(size=24), text_color="#aaaacc",
                         anchor="w").pack(side="left")
            val_lbl = ctk.CTkLabel(row, text="—",
                                   font=ctk.CTkFont(size=28, weight="bold"),
                                   text_color="#00dcff", anchor="e")
            val_lbl.pack(side="right")
            self._joint_value_labels[key] = val_lbl

    # Valgus fields are dimensionless scaled units, not degrees
    _VALGUS_FIELDS = {"left_knee_valgus", "right_knee_valgus"}

    def _update_joint_labels(self, pose_frame: PoseFrame):
        """Refresh the sidebar value labels from the current pose frame."""
        for key, lbl in self._joint_value_labels.items():
            val = getattr(pose_frame, key, None)
            if val is None:
                lbl.configure(text="—")
                continue
            val = float(val)
            if key in self._BEND_FIELDS:
                val = 180.0 - val
            if key in self._VALGUS_FIELDS:
                lbl.configure(text=f"{val:+.1f}°")  # show sign: +3.2° / -1.4°
            else:
                lbl.configure(text=f"{val:.1f}°")

    def _show_feedback_panel(self, round_number: int):
        """Replace camera with the feedback text panel."""
        self._feedback_round_label.configure(text=f"Round {round_number}  —  Feedback")
        lines = [ln.strip() for ln in self._feedback_lines if ln.strip()]
        body_text = "\n\n".join(lines)
        self._feedback_body.configure(state="normal")
        self._feedback_body.delete("1.0", "end")
        self._feedback_body.insert("1.0", body_text)
        self._feedback_body.configure(state="disabled")
        self.camera_label.grid_remove()
        self._feedback_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 15))
        self._feedback_panel_visible = True

    def _hide_feedback_panel(self):
        """Restore camera view."""
        self._feedback_panel.grid_remove()
        self.camera_label.grid(row=0, column=0, sticky="nsew", padx=(0, 15))
        self._feedback_panel_visible = False

    def _show_summary_panel(self, lines: list[str]):
        self._feedback_round_label.configure(text="Session Summary")
        body_text = "\n\n".join(ln.strip() for ln in lines if ln.strip())
        self._feedback_body.configure(state="normal")
        self._feedback_body.delete("1.0", "end")
        self._feedback_body.insert("1.0", body_text)
        self._feedback_body.configure(state="disabled")
        self.camera_label.grid_remove()
        self._feedback_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 15))
        self._feedback_panel_visible = True

    def _wait_for_summary_then_home(self, _seen_speaking: bool = False, _deadline: float = 0.0):
        if _deadline == 0.0:
            _deadline = time.time() + 10.0
        if time.time() >= _deadline:
            self.app.show_launcher()
            return
        speaking = self._tts.is_speaking()
        if speaking:
            self.after(200, lambda: self._wait_for_summary_then_home(True, _deadline))
        elif _seen_speaking:
            self.app.show_launcher()
        else:
            self.after(200, lambda: self._wait_for_summary_then_home(False, _deadline))

    def _end_session(self):
        self._tts.stop()
        self._engine.stop()
        self._engine.unsubscribe(self._on_frame)
        if self._session and "after_exercise" in self._feedback_mode:
            summary_lines = self._session.get_session_summary_speech()
            if summary_lines:
                self._show_summary_panel(summary_lines)
                self._tts.speak_immediate(". ".join(summary_lines))
                self._wait_for_summary_then_home()
                return
        self.app.show_launcher()

    def _go_home(self):
        self._tts.stop()
        self._engine.stop()
        self._engine.unsubscribe(self._on_frame)
        self.app.show_launcher()
