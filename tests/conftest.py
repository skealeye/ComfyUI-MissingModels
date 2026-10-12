import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # import core / service as plain modules (the package __init__ needs ComfyUI)
sys.path.insert(0, str(ROOT / "tests"))   # shared helpers between test files
FIXTURES = Path(__file__).resolve().parent / "fixtures"
