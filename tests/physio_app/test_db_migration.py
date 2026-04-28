import sqlite3


def _make_old_db(db_path: str):
    """Create a DB with the pre-migration schema (no new columns)."""
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE exercises (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            camera_view TEXT NOT NULL,
            instructions_text TEXT DEFAULT '',
            reference_video_path TEXT DEFAULT '',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE analysis_modules (
            id TEXT PRIMARY KEY,
            exercise_id TEXT NOT NULL,
            version INTEGER NOT NULL,
            code TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'generated',
            is_active INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            exercise_id TEXT NOT NULL
        );
        CREATE TABLE llm_logs (
            id TEXT PRIMARY KEY,
            exercise_id TEXT NOT NULL,
            prompt TEXT NOT NULL,
            response TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()


def test_migration_adds_new_columns(tmp_path):
    db_path = str(tmp_path / "test.db")
    _make_old_db(db_path)

    from physio_app.db import init_db
    conn = init_db(db_path)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(exercises)")}
    assert "client_instructions" in cols
    assert "llm_instructions" in cols
    assert "boundary_values" in cols
    assert "display_values" in cols
    assert "session_duration_secs" in cols
    conn.close()


def test_migration_wipes_existing_data(tmp_path):
    db_path = str(tmp_path / "test.db")
    _make_old_db(db_path)

    # Insert a row into the old DB
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO exercises (id, name, camera_view) VALUES ('x', 'Test', 'side')"
    )
    conn.commit()
    conn.close()

    from physio_app.db import init_db
    conn2 = init_db(db_path)
    count = conn2.execute("SELECT COUNT(*) FROM exercises").fetchone()[0]
    assert count == 0
    conn2.close()


def test_fresh_db_has_new_columns(tmp_path):
    from physio_app.db import init_db
    conn = init_db(str(tmp_path / "fresh.db"))
    cols = {row[1] for row in conn.execute("PRAGMA table_info(exercises)")}
    assert "client_instructions" in cols
    assert "session_duration_secs" in cols
    conn.close()


def test_feedback_mode_migration_non_destructive(tmp_path):
    """feedback_mode column added without wiping existing data."""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE exercises (
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
        CREATE TABLE analysis_modules (
            id TEXT PRIMARY KEY, exercise_id TEXT NOT NULL,
            version INTEGER NOT NULL, code TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'generated', is_active INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE sessions (id TEXT PRIMARY KEY, exercise_id TEXT NOT NULL);
        CREATE TABLE llm_logs (
            id TEXT PRIMARY KEY, exercise_id TEXT NOT NULL,
            prompt TEXT NOT NULL, response TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.execute("INSERT INTO exercises (id, name, camera_view) VALUES ('x', 'Test', 'side')")
    conn.commit()
    conn.close()

    from physio_app.db import init_db
    conn2 = init_db(db_path)
    cols = {row[1] for row in conn2.execute("PRAGMA table_info(exercises)")}
    assert "feedback_mode" in cols
    count = conn2.execute("SELECT COUNT(*) FROM exercises").fetchone()[0]
    assert count == 1  # data NOT wiped
    conn2.close()


def test_fresh_db_has_feedback_mode(tmp_path):
    from physio_app.db import init_db
    conn = init_db(str(tmp_path / "fresh.db"))
    cols = {row[1] for row in conn.execute("PRAGMA table_info(exercises)")}
    assert "feedback_mode" in cols
    conn.close()
