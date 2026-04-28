import sqlite3
import uuid
from dataclasses import dataclass
from typing import Optional


@dataclass
class Exercise:
    id: str
    name: str
    camera_view: str
    client_instructions: str
    llm_instructions: str
    boundary_values: str
    display_values: str
    session_duration_secs: int
    reference_video_path: str
    feedback_mode: str = '["after_window"]'
    created_at: str = ""


class ExerciseService:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def create(self, name: str, camera_view: str,
               client_instructions: str = "",
               llm_instructions: str = "",
               boundary_values: str = "",
               display_values: str = "",
               session_duration_secs: int = 10,
               feedback_mode: str = '["after_window"]') -> Exercise:
        ex_id = str(uuid.uuid4())
        self.conn.execute(
            "INSERT INTO exercises "
            "(id, name, camera_view, client_instructions, llm_instructions, "
            "boundary_values, display_values, session_duration_secs, feedback_mode) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ex_id, name, camera_view, client_instructions, llm_instructions,
             boundary_values, display_values, session_duration_secs, feedback_mode),
        )
        self.conn.commit()
        return self.get(ex_id)

    def get(self, exercise_id: str) -> Optional[Exercise]:
        row = self.conn.execute(
            "SELECT * FROM exercises WHERE id = ?", (exercise_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_exercise(row)

    def list_all(self) -> list[Exercise]:
        rows = self.conn.execute(
            "SELECT * FROM exercises ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_exercise(r) for r in rows]

    def update(self, exercise_id: str, name: Optional[str] = None, camera_view: Optional[str] = None,
               client_instructions: Optional[str] = None, llm_instructions: Optional[str] = None,
               boundary_values: Optional[str] = None, display_values: Optional[str] = None,
               session_duration_secs: Optional[int] = None,
               reference_video_path: Optional[str] = None,
               feedback_mode: Optional[str] = None) -> Exercise:
        updates, params = [], []
        for col, val in [
            ("name", name), ("camera_view", camera_view),
            ("client_instructions", client_instructions),
            ("llm_instructions", llm_instructions),
            ("boundary_values", boundary_values),
            ("display_values", display_values),
            ("session_duration_secs", session_duration_secs),
            ("reference_video_path", reference_video_path),
            ("feedback_mode", feedback_mode),
        ]:
            if val is not None:
                updates.append(f"{col} = ?")
                params.append(val)
        if updates:
            params.append(exercise_id)
            self.conn.execute(
                f"UPDATE exercises SET {', '.join(updates)} WHERE id = ?", params
            )
            self.conn.commit()
        return self.get(exercise_id)

    def delete(self, exercise_id: str):
        self.conn.execute("DELETE FROM exercises WHERE id = ?", (exercise_id,))
        self.conn.commit()

    def get_active_module(self, exercise_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM analysis_modules WHERE exercise_id = ? AND is_active = 1",
            (exercise_id,),
        ).fetchone()
        return dict(row) if row else None

    def save_module(self, exercise_id: str, code: str, status: str) -> dict:
        max_ver = self.conn.execute(
            "SELECT MAX(version) FROM analysis_modules WHERE exercise_id = ?",
            (exercise_id,),
        ).fetchone()[0]
        next_ver = (max_ver or 0) + 1
        self.conn.execute(
            "UPDATE analysis_modules SET is_active = 0 WHERE exercise_id = ?",
            (exercise_id,),
        )
        mod_id = str(uuid.uuid4())
        self.conn.execute(
            "INSERT INTO analysis_modules (id, exercise_id, version, code, status, is_active) "
            "VALUES (?, ?, ?, ?, ?, 1)",
            (mod_id, exercise_id, next_ver, code, status),
        )
        self.conn.commit()
        result = dict(self.conn.execute(
            "SELECT * FROM analysis_modules WHERE id = ?", (mod_id,)
        ).fetchone())

        # Commit above makes the new module active; generate() will see it correctly.
        if status == "validated":
            try:
                # Local import avoids circular dependency: standalone_generator imports ExerciseService.
                from physio_app.services.standalone_generator import StandaloneGenerator
                StandaloneGenerator.generate(exercise_id, self.conn)
            except Exception as e:
                import traceback
                print(f"[standalone] generation failed: {e}")
                traceback.print_exc()

        return result

    def list_modules(self, exercise_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM analysis_modules WHERE exercise_id = ? ORDER BY version DESC",
            (exercise_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def set_active_module(self, exercise_id: str, module_id: str):
        self.conn.execute(
            "UPDATE analysis_modules SET is_active = 0 WHERE exercise_id = ?",
            (exercise_id,),
        )
        self.conn.execute(
            "UPDATE analysis_modules SET is_active = 1 WHERE id = ?", (module_id,)
        )
        self.conn.commit()

    @staticmethod
    def _row_to_exercise(row) -> Exercise:
        d = dict(row)
        return Exercise(
            id=d["id"], name=d["name"], camera_view=d["camera_view"],
            client_instructions=d.get("client_instructions") or "",
            llm_instructions=d.get("llm_instructions") or "",
            boundary_values=d.get("boundary_values") or "",
            display_values=d.get("display_values") or "",
            session_duration_secs=int(d["session_duration_secs"]) if d.get("session_duration_secs") is not None else 10,
            reference_video_path=d.get("reference_video_path") or "",
            feedback_mode=d.get("feedback_mode") or '["after_window"]',
            created_at=d.get("created_at") or "",
        )
