import re
import shutil
from pathlib import Path

from physio_app.services.exercise_service import ExerciseService

_REPO_ROOT = Path(__file__).parent.parent.parent
_QT_DIR = _REPO_ROOT / "qt"
_CORE_SRC = _REPO_ROOT / "core"

_RUN_PY = '''\
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
sys.path.insert(0, str(Path(__file__).parent))

import exercise_config
import analysis_module
from app import StandaloneApp

if __name__ == "__main__":
    StandaloneApp(exercise_config.CONFIG, analysis_module).mainloop()
'''


def _slugify(name: str) -> str:
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return slug or "exercise"


def _render_config(ex) -> str:
    lines = [
        "CONFIG = {",
        f"    \"name\": {repr(ex.name)},",
        f"    \"camera_view\": {repr(ex.camera_view)},",
        f"    \"client_instructions\": {repr(ex.client_instructions)},",
        f"    \"display_values\": {repr(ex.display_values)},",
        f"    \"session_duration_secs\": {ex.session_duration_secs},",
        "}",
        "",
    ]
    return "\n".join(lines)


def _sync_lib_core():
    core_dst = _QT_DIR / "lib" / "core"
    core_dst.mkdir(parents=True, exist_ok=True)
    for fname in ("pose_engine.py", "angle_calculator.py", "data_contract.py"):
        src = _CORE_SRC / fname
        if src.exists():
            shutil.copy2(src, core_dst / fname)
        else:
            print(f"[standalone] warning: core source {fname} not found at {src}")
    init = core_dst / "__init__.py"
    if not init.exists():
        init.write_text("", encoding="utf-8")


class StandaloneGenerator:
    @staticmethod
    def generate(exercise_id: str, conn) -> None:
        svc = ExerciseService(conn)
        ex = svc.get(exercise_id)
        if ex is None:
            print(f"[standalone] exercise {exercise_id} not found, skipping")
            return
        module = svc.get_active_module(exercise_id)
        if module is None:
            print(f"[standalone] no active module for {exercise_id}, skipping")
            return

        slug = _slugify(ex.name)
        exercise_dir = _QT_DIR / slug
        if exercise_dir.exists():
            print(f"[standalone] warning: qt/{slug}/ already exists, overwriting")
        exercise_dir.mkdir(parents=True, exist_ok=True)

        (exercise_dir / "exercise_config.py").write_text(
            _render_config(ex), encoding="utf-8"
        )
        (exercise_dir / "analysis_module.py").write_text(
            module["code"], encoding="utf-8"
        )
        (exercise_dir / "run.py").write_text(_RUN_PY, encoding="utf-8")

        _sync_lib_core()
        print(f"[standalone] generated qt/{slug}/")
