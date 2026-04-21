import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
sys.path.insert(0, str(Path(__file__).parent))

import exercise_config
import analysis_module
from app import StandaloneApp

if __name__ == "__main__":
    StandaloneApp(exercise_config.CONFIG, analysis_module,
                  feedback_dir=Path(__file__).parent).mainloop()
