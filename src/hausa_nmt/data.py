"""Data loading and deterministic Hausa-English corpus preparation."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = ("hau", "eng")


def _clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    # Preserve Hausa orthography/diacritics and punctuation; normalize whitespace only.
    return " ".join(str(value).strip().split())


def load_parallel_pairs(path: str | Path) -> pd.DataFrame:
    """Load the public English/Hausa corpus and orient it Hausa -> English."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Run scripts/download_data.py first.")
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() in {".csv", ".tsv"}:
        frame = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
    else:
        raise ValueError("Supported data formats are .parquet, .csv and .tsv")

    # Dataset mirror has `eng` and `hau` columns; accept a few explicit variants.
    column_map = {str(c).strip().lower(): c for c in frame.columns}
    ha_col = next((column_map[k] for k in ("hau", "hausa", "ha") if k in column_map), None)
    en_col = next((column_map[k] for k in ("eng", "english", "en") if k in column_map), None)
    if ha_col is None or en_col is None:
        raise ValueError(f"Expected Hausa and English columns; found {list(frame.columns)}")

    pairs = pd.DataFrame({
        "source": frame[ha_col].map(_clean_text),
        "target": frame[en_col].map(_clean_text),
    })
    pairs = pairs[(pairs.source != "") & (pairs.target != "")]
    pairs = pairs.drop_duplicates(subset=["source", "target"], keep="first").reset_index(drop=True)
    if pairs.empty:
        raise ValueError("No usable parallel sentence pairs found")
    return pairs


def split_by_source_group(
    pairs: pd.DataFrame, seed: int = 42, train_fraction: float = 0.8, validation_fraction: float = 0.1
) -> Dict[str, pd.DataFrame]:
    """Split by unique Hausa source string so repeated source sentences cannot cross splits."""
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1 and train_fraction + validation_fraction < 1):
        raise ValueError("Split fractions must be positive and sum to less than one")
    rng = np.random.default_rng(seed)
    sources = pairs["source"].drop_duplicates().to_numpy(copy=True)
    rng.shuffle(sources)
    assignments: dict[str, str] = {}
    for source in sources:
        draw = rng.random()
        assignments[source] = "train" if draw < train_fraction else (
            "validation" if draw < train_fraction + validation_fraction else "test"
        )
    result = {
        name: pairs.loc[pairs.source.map(assignments) == name].reset_index(drop=True)
        for name in ("train", "validation", "test")
    }
    if any(result[name].empty for name in result):
        raise ValueError("A split is empty; increase the dataset or adjust the split fractions")
    # Defensive check: source strings are disjoint across all three splits.
    source_sets = [set(result[name].source) for name in ("train", "validation", "test")]
    if (source_sets[0] & source_sets[1]) or (source_sets[0] & source_sets[2]) or (source_sets[1] & source_sets[2]):
        raise AssertionError("Source sentence leakage detected between splits")
    return result


def prepare_dataset(
    input_path: str | Path, output_dir: str | Path, seed: int = 42,
    train_fraction: float = 0.8, validation_fraction: float = 0.1,
) -> dict:
    input_path, output_dir = Path(input_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = load_parallel_pairs(input_path)
    splits = split_by_source_group(pairs, seed, train_fraction, validation_fraction)
    for name, frame in splits.items():
        frame.to_csv(output_dir / f"ha_en_{name}.csv", index=False, encoding="utf-8")
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    project_root = output_dir.parent.parent.resolve()
    try:
        source_file = input_path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        source_file = input_path.name
    manifest = {
        "source_file": source_file,
        "source_sha256": digest,
        "direction": "Hausa (hau) -> English (eng)",
        "seed": seed,
        "split_method": "random assignment of unique Hausa source strings; repeated source sentences remain in one split",
        "fractions_requested": {"train": train_fraction, "validation": validation_fraction, "test": 1 - train_fraction - validation_fraction},
        "usable_unique_pair_count": len(pairs),
        "counts": {name: int(len(frame)) for name, frame in splits.items()},
        "unique_source_counts": {name: int(frame.source.nunique()) for name, frame in splits.items()},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_split(path: str | Path, max_samples: int = 0, seed: int = 42) -> pd.DataFrame:
    """Read a processed CSV, optionally taking a reproducible random subset."""
    frame = pd.read_csv(path, keep_default_na=False)
    if max_samples and max_samples > 0 and len(frame) > max_samples:
        frame = frame.sample(n=max_samples, random_state=seed).sort_index()
    return frame.reset_index(drop=True)
