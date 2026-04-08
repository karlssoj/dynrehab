import customtkinter as ctk
import sqlite3


class SessionSummaryFrame(ctk.CTkFrame):
    def __init__(self, parent, db_conn: sqlite3.Connection,
                 summary: dict, exercise_name: str, **kwargs):
        super().__init__(parent, **kwargs)
        self.app = parent
        self._build(summary, exercise_name)

    def _build(self, summary: dict, exercise_name: str):
        ctk.CTkLabel(self, text="Session Complete",
                     font=ctk.CTkFont(size=28, weight="bold")).pack(pady=(30, 6))
        ctk.CTkLabel(self, text=exercise_name,
                     text_color="gray", font=ctk.CTkFont(size=16)).pack()

        stats = ctk.CTkFrame(self, fg_color="transparent")
        stats.pack(pady=20)

        self._stat_box(stats, "Reps", str(summary["rep_count"])).pack(side="left", padx=20)
        self._stat_box(stats, "Quality", f"{summary['quality_pct']}%").pack(side="left", padx=20)
        mins = int(summary.get("duration_seconds", 0) // 60)
        secs = int(summary.get("duration_seconds", 0) % 60)
        self._stat_box(stats, "Duration", f"{mins}:{secs:02d}").pack(side="left", padx=20)

        ctk.CTkLabel(self, text="Feedback Log",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=40, pady=(10, 4))

        log_frame = ctk.CTkScrollableFrame(self, height=250)
        log_frame.pack(fill="x", padx=40, pady=(0, 20))

        feedback_log = summary.get("feedback_log", [])
        if not feedback_log:
            ctk.CTkLabel(log_frame, text="No feedback recorded.", text_color="gray").pack(pady=20)
        else:
            for entry in feedback_log:
                ts = entry.get("timestamp", 0.0)
                msg = entry.get("message", "")
                row = ctk.CTkFrame(log_frame, fg_color="transparent")
                row.pack(fill="x", pady=2)
                ctk.CTkLabel(row, text=f"{ts:.1f}s", text_color="gray", width=60).pack(side="left")
                ctk.CTkLabel(row, text=msg, anchor="w").pack(side="left", fill="x", expand=True)

        ctk.CTkButton(self, text="Done", height=44, width=160,
                      command=lambda: self.app.show_client_exercise_list()).pack(pady=10)
        ctk.CTkButton(self, text="← Home", width=100,
                      command=self.app.show_launcher).pack(pady=(0, 20))

    @staticmethod
    def _stat_box(parent, label: str, value: str) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(parent, width=140, height=100)
        ctk.CTkLabel(frame, text=value,
                     font=ctk.CTkFont(size=36, weight="bold")).pack(pady=(14, 2))
        ctk.CTkLabel(frame, text=label,
                     text_color="gray", font=ctk.CTkFont(size=13)).pack()
        return frame
