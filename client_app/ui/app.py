import customtkinter as ctk
import sqlite3

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class ClientApp(ctk.CTk):
    def __init__(self, db_conn: sqlite3.Connection):
        super().__init__()
        self.db_conn = db_conn
        self.title("PhysioMotion AI — Client")
        self.geometry("1000x680")
        self.resizable(True, True)
        self._current_frame = None
        self.show_exercise_list()

    def _switch_frame(self, frame_class, **kwargs):
        if self._current_frame is not None:
            self._current_frame.destroy()
        self._current_frame = frame_class(self, self.db_conn, **kwargs)
        self._current_frame.pack(fill="both", expand=True)

    def show_exercise_list(self):
        from client_app.ui.exercise_list import ClientExerciseListFrame
        self._switch_frame(ClientExerciseListFrame)

    def show_exercise_preview(self, exercise_id: str):
        from client_app.ui.exercise_preview import ExercisePreviewFrame
        self._switch_frame(ExercisePreviewFrame, exercise_id=exercise_id)

    def show_session(self, exercise_id: str):
        from client_app.ui.session_view import SessionViewFrame
        self._switch_frame(SessionViewFrame, exercise_id=exercise_id)

    def show_session_summary(self, summary: dict, exercise_name: str):
        from client_app.ui.session_summary import SessionSummaryFrame
        self._switch_frame(SessionSummaryFrame, summary=summary, exercise_name=exercise_name)
