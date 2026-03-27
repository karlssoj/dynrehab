import customtkinter as ctk
import sqlite3
import threading
import os
from pathlib import Path
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
        self._recorder: VideoRecorder = None
        self._recording = False
        self._video_path = None
        self._build()
        if exercise_id:
            self._load(exercise_id)

    def _build(self):
        ctk.CTkButton(self, text="← Back", width=80,
                      command=self._back).pack(anchor="nw", padx=20, pady=(15, 0))
        ctk.CTkLabel(self, text="Exercise Editor",
                     font=ctk.CTkFont(size=22, weight="bold")).pack(pady=(5, 10))

        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)

        # Left — fields
        left = ctk.CTkFrame(content, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        ctk.CTkLabel(left, text="Exercise Name").pack(anchor="w")
        self.name_entry = ctk.CTkEntry(left, placeholder_text="e.g. Squat from side")
        self.name_entry.pack(fill="x", pady=(0, 12))

        ctk.CTkLabel(left, text="Camera View").pack(anchor="w")
        self.view_var = ctk.StringVar(value="side")
        ctk.CTkOptionMenu(left, variable=self.view_var,
                          values=["side", "front", "back"]).pack(fill="x", pady=(0, 12))

        ctk.CTkLabel(left, text="Instructions").pack(anchor="w")
        self.instructions_text = ctk.CTkTextbox(left, height=200)
        self.instructions_text.pack(fill="both", expand=True, pady=(0, 12))

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
        self.stop_btn.pack(side="left")

        self.video_status = ctk.CTkLabel(right, text="", text_color="gray")
        self.video_status.pack(anchor="w", pady=(4, 0))

        # Bottom — save
        bottom = ctk.CTkFrame(self, fg_color="transparent")
        bottom.pack(fill="x", padx=20, pady=15)
        ctk.CTkButton(bottom, text="Save & Generate Analysis",
                      command=self._save_and_generate).pack(side="left")
        self.status_label = ctk.CTkLabel(bottom, text="")
        self.status_label.pack(side="left", padx=12)

    def _load(self, exercise_id: str):
        ex = self.ex_svc.get(exercise_id)
        if ex is None:
            return
        self.name_entry.insert(0, ex.name)
        self.view_var.set(ex.camera_view)
        self.instructions_text.insert("1.0", ex.instructions_text)
        if ex.reference_video_path:
            self._video_path = ex.reference_video_path
            self.video_status.configure(text=f"Video: {Path(ex.reference_video_path).name}")

    def _get_video_path(self, exercise_id: str) -> str:
        return str(Path("data/exercises") / exercise_id / "reference.mp4")

    def _start_recording(self):
        if self.exercise_id is None:
            self.exercise_id = self._save_exercise()
        path = self._get_video_path(self.exercise_id)
        self._recorder = VideoRecorder(path)
        self._recorder.set_preview_callback(self._on_preview_frame)
        self._recorder.start()
        self._recording = True
        self.record_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.video_status.configure(text="Recording…", text_color="#e74c3c")

    def _stop_recording(self):
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

    def _on_preview_frame(self, bgr_frame):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2.resize(rgb, (320, 240)))
        ctk_img = ctk.CTkImage(light_image=img, size=(320, 240))
        self.video_label.configure(image=ctk_img, text="")
        self.video_label.image = ctk_img

    def _save_exercise(self) -> str:
        name = self.name_entry.get().strip() or "Unnamed Exercise"
        instructions = self.instructions_text.get("1.0", "end").strip()
        ex = self.ex_svc.create(name, self.view_var.get(), instructions)
        return ex.id

    def _save_and_generate(self):
        name = self.name_entry.get().strip()
        if not name:
            self.status_label.configure(text="Exercise name is required.",
                                        text_color="#e74c3c")
            return
        instructions = self.instructions_text.get("1.0", "end").strip()
        camera_view = self.view_var.get()

        if self.exercise_id:
            self.ex_svc.update(self.exercise_id, name=name,
                               camera_view=camera_view, instructions_text=instructions)
        else:
            ex = self.ex_svc.create(name, camera_view, instructions)
            self.exercise_id = ex.id

        self.status_label.configure(text="Generating analysis code…", text_color="white")
        self.update()

        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        llm_svc = LLMService(self.db_conn, api_key=api_key)

        def run():
            module = llm_svc.generate_module(self.exercise_id, name, camera_view, instructions)
            self.after(0, lambda: self._on_generation_done(module["status"]))

        threading.Thread(target=run, daemon=True).start()

    def _on_generation_done(self, status: str):
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
        self.app.show_exercise_list()
