from physio_app.db import init_db
from physio_app.ui.app import PhysioApp


def main():
    conn = init_db()
    app = PhysioApp(conn)
    try:
        app.mainloop()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
