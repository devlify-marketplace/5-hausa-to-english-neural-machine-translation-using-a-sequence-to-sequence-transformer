#!/usr/bin/env python3
"""Run the training workflow from the project checkout."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hausa_nmt.train import main
if __name__ == "__main__":
    main()
