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
    _migrate(conn)
    return conn


def _create_tables(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS exercises (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            camera_view TEXT NOT NULL,
            client_instructions TEXT DEFAULT '',
            llm_instructions TEXT DEFAULT '',
            boundary_values TEXT DEFAULT '',
            display_values TEXT DEFAULT '',
            session_duration_secs INTEGER DEFAULT 10,
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


def _migrate(conn: sqlite3.Connection):
    """Add new exercise fields to existing DBs, wiping all data first."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(exercises)")}
    if "client_instructions" not in cols:
        conn.executescript("""
            DELETE FROM analysis_modules;
            DELETE FROM sessions;
            DELETE FROM llm_logs;
            DELETE FROM exercises;
        """)
        conn.execute("ALTER TABLE exercises ADD COLUMN client_instructions TEXT DEFAULT ''")
        conn.execute("ALTER TABLE exercises ADD COLUMN llm_instructions TEXT DEFAULT ''")
        conn.execute("ALTER TABLE exercises ADD COLUMN boundary_values TEXT DEFAULT ''")
        conn.execute("ALTER TABLE exercises ADD COLUMN display_values TEXT DEFAULT ''")
        conn.execute(
            "ALTER TABLE exercises ADD COLUMN session_duration_secs INTEGER DEFAULT 10"
        )
        conn.commit()
