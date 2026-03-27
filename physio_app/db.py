import sqlite3
from pathlib import Path


def get_db_path() -> str:
    base = Path(__file__).parent.parent / "data"
    base.mkdir(exist_ok=True)
    return str(base / "physiomotion.db")


def init_db(db_path: str = None) -> sqlite3.Connection:
    if db_path is None:
        db_path = get_db_path()
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    _create_tables(conn)
    return conn


def _create_tables(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS exercises (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            camera_view TEXT NOT NULL,
            instructions_text TEXT DEFAULT '',
            reference_video_path TEXT DEFAULT '',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS analysis_modules (
            id TEXT PRIMARY KEY,
            exercise_id TEXT NOT NULL,
            version INTEGER NOT NULL,
            code TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'generated',
            is_active INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (exercise_id) REFERENCES exercises(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            exercise_id TEXT NOT NULL,
            started_at DATETIME,
            ended_at DATETIME,
            rep_count INTEGER DEFAULT 0,
            feedback_log TEXT DEFAULT '[]',
            session_summary TEXT DEFAULT '',
            FOREIGN KEY (exercise_id) REFERENCES exercises(id)
        );

        CREATE TABLE IF NOT EXISTS llm_logs (
            id TEXT PRIMARY KEY,
            exercise_id TEXT NOT NULL,
            prompt TEXT NOT NULL,
            response TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
