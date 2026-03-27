import customtkinter as ctk
from dotenv import load_dotenv

load_dotenv()
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class PhysioApp(ctk.CTk):
    def __init__(self, db_conn):
        super().__init__()
        self.db_conn = db_conn
        self.title("PhysioMotion AI — Physiotherapist Portal")
        self.geometry("1100x700")
        self.resizable(True, True)
        self._current_frame = None
        self.show_exercise_list()

    def _switch_frame(self, frame_class, **kwargs):
        if self._current_frame is not None:
            self._current_frame.destroy()
        self._current_frame = frame_class(self, self.db_conn, **kwargs)
        self._current_frame.pack(fill="both", expand=True)

    def show_exercise_list(self):
        from physio_app.ui.exercise_list import ExerciseListFrame
        self._switch_frame(ExerciseListFrame)

    def show_exercise_editor(self, exercise_id: str = None):
        from physio_app.ui.exercise_editor import ExerciseEditorFrame
        self._switch_frame(ExerciseEditorFrame, exercise_id=exercise_id)

    def show_code_viewer(self, exercise_id: str):
        from physio_app.ui.code_viewer import CodeViewerFrame
        self._switch_frame(CodeViewerFrame, exercise_id=exercise_id)
