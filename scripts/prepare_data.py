#!/usr/bin/env python3
"""Prepare deterministic source-grouped train/validation/test partitions."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hausa_nmt.data import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = args.root.resolve()
    with (root / "configs" / "default.json").open(encoding="utf-8") as f:
        cfg = json.load(f)
    manifest = prepare_dataset(root / cfg["raw_data_path"], root / cfg["processed_data_dir"], seed=args.seed,
                              train_fraction=cfg["train_fraction"], validation_fraction=cfg["validation_fraction"])
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
