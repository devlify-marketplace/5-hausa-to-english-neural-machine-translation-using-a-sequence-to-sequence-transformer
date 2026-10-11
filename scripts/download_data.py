#!/usr/bin/env python3
"""Download the CC BY 4.0 English-Hausa MT560 mirror; does not execute remote code."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "michsethowusu/english-hausa_sentence-pairs_mt560"
URL = "https://huggingface.co/datasets/michsethowusu/english-hausa_sentence-pairs_mt560/resolve/main/data/train-00000-of-00001.parquet?download=true"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(root: Path, overwrite: bool = False) -> Path:
    path = root / "data" / "raw" / "mt560_english_hausa.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        print(f"Using existing file: {path} ({path.stat().st_size:,} bytes; sha256 {sha256(path)})")
        return path
    tmp = path.with_suffix(path.suffix + ".part")
    request = Request(URL, headers={"User-Agent": "topic5-hausa-nmt-coursework/1.0"})
    started = time.time()
    with urlopen(request, timeout=120) as response, tmp.open("wb") as out:
        total = response.headers.get("Content-Length")
        copied = 0
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            out.write(block)
            copied += len(block)
            if copied % (5 * 1024 * 1024) < len(block):
                print(f"Downloaded {copied:,}" + (f"/{int(total):,} bytes" if total else " bytes"), flush=True)
    tmp.replace(path)
    info = {
        "dataset_id": DATASET_ID,
        "upstream": "https://opus.nlpl.eu/MT560",
        "dataset_card": f"https://huggingface.co/datasets/{DATASET_ID}",
        "file_url": URL,
        "license_reported_by_mirror": "CC BY 4.0",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "file": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "download_seconds": round(time.time() - started, 2),
        "remote_dataset_code_executed": False,
    }
    (root / "data" / "raw" / "download_manifest.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {path}; sha256={info['sha256']}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    download(args.root.resolve(), args.overwrite)


if __name__ == "__main__":
    main()
