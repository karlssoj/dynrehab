import customtkinter as ctk
import sqlite3


class LauncherFrame(ctk.CTkFrame):
    def __init__(self, parent, db_conn: sqlite3.Connection, **kwargs):
        super().__init__(parent, **kwargs)
        self.app = parent
        self._build()

    def _build(self):
        ctk.CTkLabel(
            self, text="PhysioMotion AI",
            font=ctk.CTkFont(size=36, weight="bold"),
        ).pack(pady=(60, 6))
        ctk.CTkLabel(
            self, text="Movement analysis & rehabilitation platform",
            text_color="gray", font=ctk.CTkFont(size=15),
        ).pack()

        tiles = ctk.CTkFrame(self, fg_color="transparent")
        tiles.pack(expand=True, pady=50)

        # Physiotherapist Portal tile
        physio_tile = ctk.CTkFrame(
            tiles, width=320, height=300,
            corner_radius=16, border_width=2, border_color="#2980b9",
        )
        physio_tile.pack(side="left", padx=40)
        physio_tile.pack_propagate(False)
        ctk.CTkLabel(
            physio_tile, text="Physiotherapist Portal",
            font=ctk.CTkFont(size=19, weight="bold"),
        ).pack(pady=(40, 8))
        ctk.CTkLabel(
            physio_tile,
            text="Create and manage exercises\nfor your patients",
            text_color="gray", justify="center",
        ).pack(pady=(0, 20))
        ctk.CTkButton(
            physio_tile, text="Open", width=160, height=44,
            command=self.app.show_exercise_list,
        ).pack()

        # Rehabilitation Assistant tile
        client_tile = ctk.CTkFrame(
            tiles, width=320, height=300,
            corner_radius=16, border_width=2, border_color="#27ae60",
        )
        client_tile.pack(side="left", padx=40)
        client_tile.pack_propagate(False)
        ctk.CTkLabel(
            client_tile, text="Rehabilitation Assistant",
            font=ctk.CTkFont(size=19, weight="bold"),
        ).pack(pady=(40, 8))
        ctk.CTkLabel(
            client_tile,
            text="Perform movement analysis\nand guided exercises",
            text_color="gray", justify="center",
        ).pack(pady=(0, 20))
        ctk.CTkButton(
            client_tile, text="Open", width=160, height=44,
            fg_color="#27ae60", hover_color="#1e8449",
            command=self.app.show_client_exercise_list,
        ).pack()
