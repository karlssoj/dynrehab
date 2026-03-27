import customtkinter as ctk
import sqlite3
import cv2
import threading
import time
from PIL import Image
from physio_app.services.exercise_service import ExerciseService


class ExercisePreviewFrame(ctk.CTkFrame):
    def __init__(self, parent, db_conn: sqlite3.Connection,
                 exercise_id: str, **kwargs):
        super().__init__(parent, **kwargs)
        self.app = parent
        self.exercise_id = exercise_id
        self.svc = ExerciseService(db_conn)
        self._playing = False
        self._video_path = ""
        self._build()
        self._load()

    def _build(self):
        ctk.CTkButton(self, text="← Back", width=80,
                      command=self._back).pack(anchor="nw", padx=20, pady=(15, 0))

        self.title_label = ctk.CTkLabel(self, text="",
                                        font=ctk.CTkFont(size=22, weight="bold"))
        self.title_label.pack(pady=(8, 4))
        self.view_label = ctk.CTkLabel(self, text="", text_color="gray")
        self.view_label.pack()

        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=30, pady=10)
        content.columnconfigure(0, weight=2)
        content.columnconfigure(1, weight=3)

        left = ctk.CTkFrame(content, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 15))
        ctk.CTkLabel(left, text="Instructions",
                     font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w")
        self.instructions_box = ctk.CTkTextbox(left, state="disabled")
        self.instructions_box.pack(fill="both", expand=True, pady=(6, 0))

        right = ctk.CTkFrame(content, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew")
        ctk.CTkLabel(right, text="Reference Video",
                     font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w")
        self.video_label = ctk.CTkLabel(right, text="No reference video",
                                        width=400, height=300, fg_color="#1a1a2e")
        self.video_label.pack(pady=(6, 8))
        self.play_btn = ctk.CTkButton(right, text="▶ Play", command=self._play)
        self.play_btn.pack()

        ctk.CTkButton(self, text="Start Session",
                      font=ctk.CTkFont(size=17, weight="bold"),
                      height=50,
                      command=self._start_session).pack(
            pady=20, ipadx=30)

    def _load(self):
        ex = self.svc.get(self.exercise_id)
        if ex is None:
            return
        self.title_label.configure(text=ex.name)
        self.view_label.configure(text=f"Camera view: {ex.camera_view}")
        self.instructions_box.configure(state="normal")
        self.instructions_box.insert("1.0", ex.instructions_text or "(No instructions provided)")
        self.instructions_box.configure(state="disabled")
        self._video_path = ex.reference_video_path or ""

    def _play(self):
        if not self._video_path or self._playing:
            return
        self._playing = True
        self.play_btn.configure(state="disabled")
        threading.Thread(target=self._stream_video, daemon=True).start()

    def _stream_video(self):
        cap = cv2.VideoCapture(self._video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        delay = max(0.001, 1.0 / fps)
        try:
            while self._playing and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2.resize(rgb, (400, 300)))
                ctk_img = ctk.CTkImage(light_image=img, size=(400, 300))
                self.after(0, lambda i=ctk_img: (
                    self.video_label.configure(image=i, text=""),
                    setattr(self.video_label, "image", i),
                ))
                time.sleep(delay)
        finally:
            cap.release()
        self._playing = False
        self.after(0, lambda: self.play_btn.configure(state="normal"))

    def _start_session(self):
        self._playing = False
        self.app.show_session(self.exercise_id)

    def _back(self):
        self._playing = False
        self.app.show_exercise_list()
