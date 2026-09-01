"""
Tracking Learning With Probes -- source package.

Importing this puts the repo root on sys.path, so `from src import stats` works from
inside src/ regardless of where the process was started (Colab, a notebook,
pytest). `src/runlog.py` is a package module and never shadows the standard
library as long as the repo root, not src/, is what sits on the path.
"""
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
