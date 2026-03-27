import customtkinter as ctk
import sqlite3
from physio_app.services.exercise_service import ExerciseService

STATUS_COLORS = {
    "validated": "#2ecc71",
    "failed": "#e74c3c",
    None: "#7f8c8d",
}


class ExerciseListFrame(ctk.CTkFrame):
    def __init__(self, parent, db_conn: sqlite3.Connection, **kwargs):
        super().__init__(parent, **kwargs)
        self.app = parent
        self.svc = ExerciseService(db_conn)
        self._build()

    def _build(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(20, 10))
        ctk.CTkLabel(header, text="Exercises",
                     font=ctk.CTkFont(size=24, weight="bold")).pack(side="left")
        ctk.CTkButton(header, text="+ New Exercise",
                      command=self._new).pack(side="right")

        self.scroll = ctk.CTkScrollableFrame(self)
        self.scroll.pack(fill="both", expand=True, padx=20, pady=10)
        self._refresh()

    def _refresh(self):
        for w in self.scroll.winfo_children():
            w.destroy()
        exercises = self.svc.list_all()
        if not exercises:
            ctk.CTkLabel(self.scroll,
                         text="No exercises yet. Click '+ New Exercise' to get started.",
                         text_color="gray").pack(pady=40)
            return
        for ex in exercises:
            module = self.svc.get_active_module(ex.id)
            status = module["status"] if module else None
            self._exercise_row(ex, status)

    def _exercise_row(self, ex, status):
        row = ctk.CTkFrame(self.scroll)
        row.pack(fill="x", pady=4)
        color = STATUS_COLORS.get(status, STATUS_COLORS[None])
        ctk.CTkLabel(row, text="●", text_color=color,
                     font=ctk.CTkFont(size=18)).pack(side="left", padx=(10, 6), pady=10)
        ctk.CTkLabel(row, text=ex.name,
                     font=ctk.CTkFont(size=15)).pack(side="left", pady=10)
        ctk.CTkLabel(row, text=f"[{ex.camera_view}]",
                     text_color="gray").pack(side="left", padx=6, pady=10)
        ctk.CTkButton(row, text="Edit", width=70,
                      command=lambda e=ex: self.app.show_exercise_editor(e.id)).pack(
            side="right", padx=4, pady=6)
        ctk.CTkButton(row, text="View Code", width=90,
                      command=lambda e=ex: self.app.show_code_viewer(e.id)).pack(
            side="right", padx=4, pady=6)
        ctk.CTkButton(row, text="Delete", width=70,
                      fg_color="#c0392b", hover_color="#922b21",
                      command=lambda e=ex: self._delete(e.id)).pack(
            side="right", padx=4, pady=6)

    def _new(self):
        self.app.show_exercise_editor(exercise_id=None)

    def _delete(self, exercise_id: str):
        self.svc.delete(exercise_id)
        self._refresh()
