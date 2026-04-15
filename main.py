import customtkinter as ctk
from dotenv import load_dotenv
from physio_app.db import init_db


class PhysioMotionApp(ctk.CTk):
    def __init__(self, db_conn):
        super().__init__()
        self.db_conn = db_conn
        self.geometry("1100x700")
        self.resizable(True, True)
        self._current_frame = None
        self.show_launcher()

    def _switch_frame(self, frame_class, **kwargs):
        if self._current_frame is not None:
            self._current_frame.destroy()
        self._current_frame = frame_class(self, self.db_conn, **kwargs)
        self._current_frame.pack(fill="both", expand=True)

    # ── Launcher ────────────────────────────────────────────────────────────

    def show_launcher(self):
        self.title("PhysioMotion AI")
        from launcher.launcher_frame import LauncherFrame
        self._switch_frame(LauncherFrame)

    # ── Physiotherapist Portal ───────────────────────────────────────────────

    def show_exercise_list(self):
        self.title("PhysioMotion AI — Physiotherapist Portal")
        from physio_app.ui.exercise_list import ExerciseListFrame
        self._switch_frame(ExerciseListFrame)

    def show_exercise_editor(self, exercise_id: str = None):
        self.title("PhysioMotion AI — Physiotherapist Portal")
        from physio_app.ui.exercise_editor import ExerciseEditorFrame
        self._switch_frame(ExerciseEditorFrame, exercise_id=exercise_id)

    def show_code_viewer(self, exercise_id: str):
        self.title("PhysioMotion AI — Physiotherapist Portal")
        from physio_app.ui.code_viewer import CodeViewerFrame
        self._switch_frame(CodeViewerFrame, exercise_id=exercise_id)

    # ── Rehabilitation Assistant ─────────────────────────────────────────────

    def show_client_exercise_list(self):
        self.title("PhysioMotion AI — Rehabilitation Assistant")
        from client_app.ui.exercise_list import ClientExerciseListFrame
        self._switch_frame(ClientExerciseListFrame)

    def show_exercise_preview(self, exercise_id: str):
        self.title("PhysioMotion AI — Rehabilitation Assistant")
        from client_app.ui.exercise_preview import ExercisePreviewFrame
        self._switch_frame(ExercisePreviewFrame, exercise_id=exercise_id)

    def show_session(self, exercise_id: str):
        self.title("PhysioMotion AI — Rehabilitation Assistant")
        from client_app.ui.session_view import SessionViewFrame
        self._switch_frame(SessionViewFrame, exercise_id=exercise_id)



def main():
    load_dotenv()
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    conn = init_db()
    app = PhysioMotionApp(conn)
    try:
        app.mainloop()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
