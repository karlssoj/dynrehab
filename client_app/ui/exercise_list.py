import customtkinter as ctk
import sqlite3
from physio_app.services.exercise_service import ExerciseService


class ClientExerciseListFrame(ctk.CTkFrame):
    def __init__(self, parent, db_conn: sqlite3.Connection, **kwargs):
        super().__init__(parent, **kwargs)
        self.app = parent
        self.svc = ExerciseService(db_conn)
        self._build()

    def _build(self):
        ctk.CTkLabel(self, text="Select an Exercise",
                     font=ctk.CTkFont(size=26, weight="bold")).pack(pady=(30, 20))

        scroll = ctk.CTkScrollableFrame(self)
        scroll.pack(fill="both", expand=True, padx=40, pady=(0, 20))

        exercises = self.svc.list_all()
        validated = [
            e for e in exercises
            if (m := self.svc.get_active_module(e.id)) and m["status"] == "validated"
        ]

        if not validated:
            ctk.CTkLabel(scroll,
                         text="No exercises available yet.\nAsk your physiotherapist to create some.",
                         text_color="gray", font=ctk.CTkFont(size=14)).pack(pady=60)
            return

        for ex in validated:
            row = ctk.CTkFrame(scroll)
            row.pack(fill="x", pady=6)
            ctk.CTkLabel(row, text=ex.name,
                         font=ctk.CTkFont(size=16)).pack(side="left", padx=16, pady=14)
            ctk.CTkLabel(row, text=f"Camera: {ex.camera_view}",
                         text_color="gray").pack(side="left")
            ctk.CTkButton(row, text="Select →", width=100,
                          command=lambda e=ex: self.app.show_exercise_preview(e.id)).pack(
                side="right", padx=12, pady=8)
