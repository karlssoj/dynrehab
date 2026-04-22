import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
sys.path.insert(0, str(Path(__file__).parent))

import exercise_config
import analysis_module
from app import StandaloneApp

if __name__ == "__main__":
    video_source = sys.argv[1] if len(sys.argv) > 1 else 0
    voice = subprocess.Popen([sys.executable, str(Path(__file__).parent / "voice.py")])
    try:
        StandaloneApp(exercise_config.CONFIG, analysis_module,
                      feedback_dir=Path(__file__).parent,
                      video_source=video_source).mainloop()
    finally:
        voice.terminate()
