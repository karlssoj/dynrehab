import customtkinter as ctk
import sqlite3
import threading
import os
from physio_app.services.exercise_service import ExerciseService
from physio_app.services.llm_service import LLMService


class CodeViewerFrame(ctk.CTkFrame):
    def __init__(self, parent, db_conn: sqlite3.Connection,
                 exercise_id: str, **kwargs):
        super().__init__(parent, **kwargs)
        self.app = parent
        self.db_conn = db_conn
        self.ex_svc = ExerciseService(db_conn)
        self.exercise_id = exercise_id
        self._modules = []
        self._build()
        self._load()

    def _build(self):
        nav = ctk.CTkFrame(self, fg_color="transparent")
        nav.pack(fill="x", padx=20, pady=(15, 0))
        ctk.CTkButton(nav, text="← Back", width=80,
                      command=lambda: self.app.show_exercise_list()).pack(side="left")
        ctk.CTkButton(nav, text="← Home", width=80,
                      command=self.app.show_launcher).pack(side="left", padx=(8, 0))

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(5, 8))
        ctk.CTkLabel(header, text="Analysis Code",
                     font=ctk.CTkFont(size=22, weight="bold")).pack(side="left")

        controls = ctk.CTkFrame(self, fg_color="transparent")
        controls.pack(fill="x", padx=20, pady=(0, 8))
        ctk.CTkLabel(controls, text="Version:").pack(side="left", padx=(0, 6))
        self.version_var = ctk.StringVar()
        self.version_menu = ctk.CTkOptionMenu(controls, variable=self.version_var,
                                              values=["—"],
                                              command=self._on_version_selected)
        self.version_menu.pack(side="left")
        self.status_badge = ctk.CTkLabel(controls, text="",
                                         font=ctk.CTkFont(size=13))
        self.status_badge.pack(side="left", padx=12)
        self._regen_btn = ctk.CTkButton(controls, text="Regenerate",
                                        command=self._regenerate)
        self._regen_btn.pack(side="right")
        self.regen_status = ctk.CTkLabel(controls, text="")
        self.regen_status.pack(side="right", padx=8)

        self.code_box = ctk.CTkTextbox(
            self, font=ctk.CTkFont(family="Courier", size=12), state="disabled")
        self.code_box.pack(fill="both", expand=True, padx=20, pady=(0, 20))

    def _load(self):
        self._modules = self.ex_svc.list_modules(self.exercise_id)
        if not self._modules:
            self._set_code("No analysis module generated yet.")
            return
        labels = [f"v{m['version']} ({m['status']})" for m in self._modules]
        self.version_menu.configure(values=labels)
        self.version_var.set(labels[0])
        self._show_module(self._modules[0])

    def _on_version_selected(self, label: str):
        try:
            idx = list(self.version_menu.cget("values")).index(label)
        except ValueError:
            return
        self._show_module(self._modules[idx])

    def _show_module(self, module: dict):
        color = "#2ecc71" if module["status"] == "validated" else "#e74c3c"
        self.status_badge.configure(text=f"● {module['status']}", text_color=color)
        self._set_code(module["code"])

    def _set_code(self, text: str):
        self.code_box.configure(state="normal")
        self.code_box.delete("1.0", "end")
        self.code_box.insert("1.0", text)
        self.code_box.configure(state="disabled")

    def _regenerate(self):
        ex = self.ex_svc.get(self.exercise_id)
        if ex is None:
            return
        self.regen_status.configure(text="Regenerating…", text_color="white")
        self._regen_btn.configure(state="disabled")
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        llm_svc = LLMService(self.db_conn, api_key=api_key)

        def run():
            module = llm_svc.generate_module(
                ex.id, ex.name, ex.camera_view, ex.instructions_text)
            self.after(0, lambda: self._on_regen_done(module["status"]))

        threading.Thread(target=run, daemon=True).start()

    def _on_regen_done(self, status: str):
        self._regen_btn.configure(state="normal")
        if status == "validated":
            self.regen_status.configure(text="✓ Regenerated", text_color="#2ecc71")
        else:
            self.regen_status.configure(text="✗ Failed", text_color="#e74c3c")
        self._load()
