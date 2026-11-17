import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
os.environ.setdefault("ANGLE_STUDIO_OPEN", "1")  # accounts off for the step tests; the account tests turn them on
