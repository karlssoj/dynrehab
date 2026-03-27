import pytest
import tempfile
import os


@pytest.fixture
def tmp_db():
    from physio_app.db import init_db
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    conn = init_db(db_path)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass
        try:
            os.unlink(db_path)
        except Exception:
            pass
