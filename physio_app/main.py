from physio_app.db import init_db
from physio_app.ui.app import PhysioApp


def main():
    conn = init_db()
    app = PhysioApp(conn)
    app.mainloop()
    conn.close()


if __name__ == "__main__":
    main()
