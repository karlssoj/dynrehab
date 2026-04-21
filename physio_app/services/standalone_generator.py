import re
import shutil
from pathlib import Path

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
