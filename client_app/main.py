from physio_app.db import init_db
from client_app.ui.app import ClientApp


def main():
    conn = init_db()
    app = ClientApp(conn)
    try:
        app.mainloop()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
