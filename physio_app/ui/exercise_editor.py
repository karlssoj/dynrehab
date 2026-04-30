import customtkinter as ctk
import sqlite3
import shutil
import threading
import os
import json
from pathlib import Path
from tkinter import filedialog
from PIL import Image
import cv2
from physio_app.services.exercise_service import ExerciseService
from physio_app.services.llm_service import LLMService
from physio_app.services.video_service import VideoRecorder


class ExerciseEditorFrame(ctk.CTkFrame):
    def __init__(self, parent, db_conn: sqlite3.Connection,
                 exercise_id: str = None, **kwargs):
        super().__init__(parent, **kwargs)
        self.app = parent
        self.db_conn = db_conn
        self.ex_svc = ExerciseService(db_conn)
        self.exercise_id = exercise_id
        self._original_exercise_id = exercise_id  # track if exercise pre-existed
        self._recorder: VideoRecorder = None
        self._recording = False
        self._recording_time_left = 0
        self._video_path = None
        self._build()
        if exercise_id:
            self._load(exercise_id)

    def _build(self):
        nav = ctk.CTkFrame(self, fg_color="transparent")
        nav.pack(fill="x", padx=20, pady=(15, 0))
        ctk.CTkButton(nav, text="← Back", width=80,
                      command=self._back).pack(side="left")
        ctk.CTkButton(nav, text="← Home", width=80,
                      command=self.app.show_launcher).pack(side="left", padx=(8, 0))
        ctk.CTkLabel(self, text="Exercise Editor",
                     font=ctk.CTkFont(size=22, weight="bold")).pack(pady=(5, 10))

        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)   # add this line

        # Left — scrollable fields
        left = ctk.CTkScrollableFrame(content, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        ctk.CTkLabel(left, text="Exercise Name").pack(anchor="w")
        self.name_entry = ctk.CTkEntry(left, placeholder_text="e.g. Squat from side")
        self.name_entry.pack(fill="x", pady=(0, 12))

        ctk.CTkLabel(left, text="Camera View").pack(anchor="w")
        self.view_var = ctk.StringVar(value="side")
        ctk.CTkOptionMenu(left, variable=self.view_var,
                          values=["side", "front", "back"]).pack(fill="x", pady=(0, 12))

        ctk.CTkLabel(left, text="Client Instructions").pack(anchor="w")
        ctk.CTkLabel(left, text="What the patient sees/hears before starting",
                     text_color="gray", font=ctk.CTkFont(size=11)).pack(anchor="w")
        self.client_instructions_text = ctk.CTkTextbox(left, height=100)
        self.client_instructions_text.pack(fill="x", pady=(0, 12))

        ctk.CTkLabel(left, text="Analysis Instructions").pack(anchor="w")
        ctk.CTkLabel(left, text="What the LLM should look for and what feedback to give",
                     text_color="gray", font=ctk.CTkFont(size=11)).pack(anchor="w")
        self.llm_instructions_text = ctk.CTkTextbox(left, height=120)
        self.llm_instructions_text.pack(fill="x", pady=(0, 12))

        ctk.CTkLabel(left, text="Boundary Values").pack(anchor="w")
        ctk.CTkLabel(left, text="Specific thresholds, e.g. knee must reach at least 90°",
                     text_color="gray", font=ctk.CTkFont(size=11)).pack(anchor="w")
        self.boundary_values_text = ctk.CTkTextbox(left, height=80)
        self.boundary_values_text.pack(fill="x", pady=(0, 12))

        ctk.CTkLabel(left, text="Display Values").pack(anchor="w")
        ctk.CTkLabel(left, text="Which angle values to show on screen during analysis",
                     text_color="gray", font=ctk.CTkFont(size=11)).pack(anchor="w")
        self.display_values_text = ctk.CTkTextbox(left, height=80)
        self.display_values_text.pack(fill="x", pady=(0, 12))

        self._duration_frame = ctk.CTkFrame(left, fg_color="transparent")
        self._duration_frame.pack(fill="x")
        ctk.CTkLabel(self._duration_frame, text="Session Duration (seconds)").pack(anchor="w")
        self.session_duration_entry = ctk.CTkEntry(self._duration_frame, placeholder_text="10")
        self.session_duration_entry.pack(fill="x", pady=(0, 12))

        self._feedback_mode_label = ctk.CTkLabel(left, text="Feedback Mode")
        self._feedback_mode_label.pack(anchor="w")
        ctk.CTkLabel(left, text="When should feedback be given to the patient?",
                     text_color="gray", font=ctk.CTkFont(size=11)).pack(anchor="w")
        self._mode_window_var = ctk.BooleanVar(value=True)
        self._mode_rep_var = ctk.BooleanVar(value=False)
        self._mode_during_var = ctk.BooleanVar(value=False)
        self._mode_after_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(left, text="After time window  (pause for feedback each round)",
                        variable=self._mode_window_var,
                        command=self._on_mode_changed).pack(anchor="w")
        ctk.CTkCheckBox(left, text="After each rep  (pause for detailed feedback per rep)",
                        variable=self._mode_rep_var,
                        command=self._on_mode_changed).pack(anchor="w")
        ctk.CTkCheckBox(left, text="During exercise  (short cue after each rep, no pause)",
                        variable=self._mode_during_var).pack(anchor="w")
        ctk.CTkCheckBox(left, text="After exercise  (summary when session ends)",
                        variable=self._mode_after_var).pack(anchor="w", pady=(0, 12))

        # Right — video
        right = ctk.CTkFrame(content, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew")

        ctk.CTkLabel(right, text="Reference Video").pack(anchor="w")
        self.video_label = ctk.CTkLabel(right, text="No camera preview",
                                        width=320, height=240, fg_color="#1a1a2e")
        self.video_label.pack(pady=(0, 8))

        btn_row = ctk.CTkFrame(right, fg_color="transparent")
        btn_row.pack(fill="x")
        self.record_btn = ctk.CTkButton(btn_row, text="Record",
                                        command=self._start_recording)
        self.record_btn.pack(side="left", padx=(0, 8))
        self.stop_btn = ctk.CTkButton(btn_row, text="Stop", state="disabled",
                                      command=self._stop_recording)
        self.stop_btn.pack(side="left", padx=(0, 8))
        ctk.CTkButton(btn_row, text="Browse file…",
                      command=self._browse_video).pack(side="left")

        self.video_status = ctk.CTkLabel(right, text="", text_color="gray")
        self.video_status.pack(anchor="w", pady=(4, 0))

        # Bottom — save
        bottom = ctk.CTkFrame(self, fg_color="transparent")
        bottom.pack(fill="x", padx=20, pady=15)
        self._save_btn = ctk.CTkButton(bottom, text="Save & Generate Analysis",
                                       command=self._save_and_generate)
        self._save_btn.pack(side="left")
        self.status_label = ctk.CTkLabel(bottom, text="")
        self.status_label.pack(side="left", padx=12)

    def _on_mode_changed(self):
        if self._mode_window_var.get():
            self._duration_frame.pack(fill="x", before=self._feedback_mode_label)
        else:
            self._duration_frame.pack_forget()

    def _load(self, exercise_id: str):
        ex = self.ex_svc.get(exercise_id)
        if ex is None:
            return
        self.name_entry.insert(0, ex.name)
        self.view_var.set(ex.camera_view)
        self.client_instructions_text.insert("1.0", ex.client_instructions)
        self.llm_instructions_text.insert("1.0", ex.llm_instructions)
        self.boundary_values_text.insert("1.0", ex.boundary_values)
        self.display_values_text.insert("1.0", ex.display_values)
        self.session_duration_entry.insert(0, str(ex.session_duration_secs))
        try:
            modes = json.loads(ex.feedback_mode) if ex.feedback_mode else ["after_window"]
        except (json.JSONDecodeError, TypeError):
            modes = ["after_window"]
        self._mode_during_var.set("during_exercise" in modes)
        self._mode_window_var.set("after_window" in modes)
        self._mode_rep_var.set("after_rep" in modes)
        self._mode_after_var.set("after_exercise" in modes)
        self._on_mode_changed()
        if ex.reference_video_path:
            self._video_path = ex.reference_video_path
            self.video_status.configure(text=f"Video: {Path(ex.reference_video_path).name}")

    def _get_video_path(self, exercise_id: str) -> str:
        project_root = Path(__file__).parent.parent.parent
        return str(project_root / "data" / "exercises" / exercise_id / "reference.mp4")

    _COUNTDOWN_SECS = 5
    _RECORD_SECS    = 10

    def _start_recording(self):
        if self.exercise_id is None:
            self.exercise_id = self._save_exercise()
        self.record_btn.configure(state="disabled")
        self.stop_btn.configure(state="disabled")
        self.video_status.configure(text="Get ready…", text_color="#f39c12")
        self._countdown_tick(self._COUNTDOWN_SECS)

    def _countdown_tick(self, count: int):
        if count > 0:
            self.video_label.configure(
                text=str(count),
                font=ctk.CTkFont(size=80, weight="bold"),
                text_color="#00dcff",
            )
            self.after(1000, lambda: self._countdown_tick(count - 1))
        else:
            self.video_label.configure(
                text="GO!",
                font=ctk.CTkFont(size=72, weight="bold"),
                text_color="#00ff44",
            )
            self.after(500, self._begin_recording)

    def _begin_recording(self):
        path = self._get_video_path(self.exercise_id)
        self._recorder = VideoRecorder(path)
        self._recorder.set_preview_callback(self._on_preview_frame)
        self._recorder.start()
        self._recording = True
        self.stop_btn.configure(state="normal")
        self.video_status.configure(text="Recording…", text_color="#e74c3c")
        self._recording_time_left = self._RECORD_SECS
        self._record_tick()
        self.after(self._RECORD_SECS * 1000, self._stop_recording)

    def _record_tick(self):
        if not self._recording:
            return
        self.video_status.configure(
            text=f"Recording… {self._recording_time_left}s remaining",
            text_color="#e74c3c",
        )
        self._recording_time_left -= 1
        if self._recording_time_left >= 0:
            self.after(1000, self._record_tick)

    def _stop_recording(self):
        if not self._recording:
            return
        if self._recorder:
            self._recorder.stop()
            self._recorder = None
        self._recording = False
        self.record_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        path = self._get_video_path(self.exercise_id)
        self._video_path = path
        self.ex_svc.update(self.exercise_id, reference_video_path=path)
        self.video_status.configure(text="Saved: reference.mp4", text_color="#2ecc71")

    def _browse_video(self):
        src = filedialog.askopenfilename(
            title="Choose reference video",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")],
        )
        if not src:
            return
        if self.exercise_id is None:
            self.exercise_id = self._save_exercise()
        dst = self._get_video_path(self.exercise_id)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        self._video_path = dst
        self.ex_svc.update(self.exercise_id, reference_video_path=dst)
        self.video_status.configure(text=f"Video: {Path(src).name}", text_color="#2ecc71")
        self._show_video_thumbnail(dst)

    def _show_video_thumbnail(self, path: str):
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return
        self._on_preview_frame(frame)

    def _on_preview_frame(self, bgr_frame):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2.resize(rgb, (320, 240)))
        ctk_img = ctk.CTkImage(light_image=img, size=(320, 240))
        self.after(0, lambda i=ctk_img: (
            self.video_label.configure(image=i, text=""),
            setattr(self.video_label, "image", i),
        ))

    def _get_fields(self) -> dict:
        """Read all form fields. Returns dict with all values."""
        try:
            duration = int(self.session_duration_entry.get().strip())
            if duration <= 0:
                duration = 10
        except (ValueError, TypeError):
            duration = 10
        modes = []
        if self._mode_window_var.get():
            modes.append("after_window")
        if self._mode_rep_var.get():
            modes.append("after_rep")
        if self._mode_during_var.get():
            modes.append("during_exercise")
        if self._mode_after_var.get():
            modes.append("after_exercise")
        if not modes:
            modes = ["after_window"]
        return {
            "name": self.name_entry.get().strip() or "Unnamed Exercise",
            "camera_view": self.view_var.get(),
            "client_instructions": self.client_instructions_text.get("1.0", "end").strip(),
            "llm_instructions": self.llm_instructions_text.get("1.0", "end").strip(),
            "boundary_values": self.boundary_values_text.get("1.0", "end").strip(),
            "display_values": self.display_values_text.get("1.0", "end").strip(),
            "session_duration_secs": duration,
            "feedback_mode": json.dumps(modes),
        }

    def _save_exercise(self) -> str:
        f = self._get_fields()
        ex = self.ex_svc.create(
            f["name"], f["camera_view"],
            client_instructions=f["client_instructions"],
            llm_instructions=f["llm_instructions"],
            boundary_values=f["boundary_values"],
            display_values=f["display_values"],
            session_duration_secs=f["session_duration_secs"],
            feedback_mode=f["feedback_mode"],
        )
        return ex.id

    def _save_and_generate(self):
        if not self.name_entry.get().strip():
            self.status_label.configure(text="Exercise name is required.",
                                        text_color="#e74c3c")
            return
        f = self._get_fields()

        if self.exercise_id:
            self.ex_svc.update(
                self.exercise_id,
                name=f["name"], camera_view=f["camera_view"],
                client_instructions=f["client_instructions"],
                llm_instructions=f["llm_instructions"],
                boundary_values=f["boundary_values"],
                display_values=f["display_values"],
                session_duration_secs=f["session_duration_secs"],
                feedback_mode=f["feedback_mode"],
            )
        else:
            self.exercise_id = self._save_exercise()

        self.status_label.configure(text="Generating analysis code…", text_color="white")
        self._save_btn.configure(state="disabled")

        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        llm_svc = LLMService(self.db_conn, api_key=api_key)

        def run():
            module = llm_svc.generate_module(
                self.exercise_id, f["name"], f["camera_view"],
                client_instructions=f["client_instructions"],
                llm_instructions=f["llm_instructions"],
                boundary_values=f["boundary_values"],
                display_values=f["display_values"],
                session_duration_secs=f["session_duration_secs"],
                feedback_mode=json.loads(f["feedback_mode"]),
            )
            self.after(0, lambda: self._on_generation_done(module["status"]))

        threading.Thread(target=run, daemon=True).start()

    def _on_generation_done(self, status: str):
        self._save_btn.configure(state="normal")
        if status == "validated":
            self.status_label.configure(text="✓ Code generated successfully",
                                        text_color="#2ecc71")
        else:
            self.status_label.configure(
                text="✗ Generation failed — check API key or instructions",
                text_color="#e74c3c")

    def _back(self):
        if self._recorder:
            self._recorder.stop()
        self._recording = False
        self._recorder = None
        # Clean up a new exercise created only for recording if it was never completed
        if self._original_exercise_id is None and self.exercise_id is not None:
            if self.ex_svc.get_active_module(self.exercise_id) is None:
                self.ex_svc.delete(self.exercise_id)
        self.app.show_exercise_list()
