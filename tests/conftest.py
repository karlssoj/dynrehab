import pytest
import sqlite3
import tempfile
import os
from physio_app.db import init_db


@pytest.fixture
def tmp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    conn = init_db(db_path)
    yield conn
    conn.close()
    os.unlink(db_path)
