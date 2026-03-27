import customtkinter as ctk
import sqlite3
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
        self._tts = TTSService(cooldown_seconds=2.0)
        self._feedback_clear_job = None
        self._build()
        self._start_session()

    def _build(self):
        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x", padx=20, pady=(12, 0))
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
                                         width=640, height=480, fg_color="#1a1a2e")
        self.camera_label.grid(row=0, column=0, sticky="nsew", padx=(0, 15))

        right = ctk.CTkFrame(main)
        right.grid(row=0, column=1, sticky="nsew")

        ctk.CTkLabel(right, text="Reps",
                     font=ctk.CTkFont(size=16)).pack(pady=(20, 4))
        self.rep_label = ctk.CTkLabel(right, text="0",
                                      font=ctk.CTkFont(size=64, weight="bold"))
        self.rep_label.pack()

        ctk.CTkLabel(right, text="Feedback",
                     font=ctk.CTkFont(size=14)).pack(pady=(30, 6))
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
        self._session = SessionService(exercise_id=self.exercise_id,
                                       module_code=module["code"])
        self._engine.subscribe(self._on_frame)
        self._engine.start()

    def _on_frame(self, pose_frame: PoseFrame, annotated: np.ndarray):
        if self._session is None:
            self.after(0, lambda f=annotated: self._update_camera(f))
            return

        result = self._session.process_frame(pose_frame.to_dict())
        self._engine.set_highlight_joints(result["highlight_joints"])

        self.after(0, lambda r=result, f=annotated: self._update_ui(r, f))

    def _update_ui(self, result: dict, frame: np.ndarray):
        self._update_camera(frame)
        self.rep_label.configure(text=str(result["rep_count"]))

        all_feedback = result["feedback"] + result["post_rep_feedback"]
        if all_feedback:
            msg = all_feedback[0]["message"]
            self.feedback_label.configure(text=msg)
            self._tts.speak(msg)
            if self._feedback_clear_job is not None:
                self.after_cancel(self._feedback_clear_job)
            self._feedback_clear_job = self.after(
                3000, lambda: self.feedback_label.configure(text="")
            )

    def _update_camera(self, bgr_frame: np.ndarray):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2.resize(rgb, (640, 480)))
        ctk_img = ctk.CTkImage(light_image=img, size=(640, 480))
        self.camera_label.configure(image=ctk_img, text="")
        self.camera_label.image = ctk_img

    def _end_session(self):
        if self._feedback_clear_job is not None:
            self.after_cancel(self._feedback_clear_job)
            self._feedback_clear_job = None
        self._engine.stop()
        self._engine.unsubscribe(self._on_frame)
        summary = self._session.get_summary() if self._session else {
            "rep_count": 0, "quality_pct": 0, "feedback_log": [], "duration_seconds": 0
        }
        ex = self.ex_svc.get(self.exercise_id)
        self.app.show_session_summary(summary, ex.name if ex else "Exercise")
