from physio_app.db import init_db


def test_init_db_creates_all_tables(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = init_db(db_path)
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert {"exercises", "analysis_modules", "sessions", "llm_logs"} <= tables
    conn.close()


def test_init_db_idempotent(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path).close()
    conn = init_db(db_path)  # second call must not raise
    conn.close()
