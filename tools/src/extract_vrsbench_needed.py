#!/usr/bin/env python3
"""
Extract only the VRSBench images referenced by eval JSONs.

This avoids unpacking the entire multi-GB archives by pulling just the
filenames required by VQA/Caption/Referring eval files.

Usage:
  python tools/extract_vrsbench_needed.py --root .

It looks under {root}/VRSBench and tries Images_val.zip then Images_train.zip.
It extracts into {root}/VRSBench/Images_val and Images_train as needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile


def load_needed_ids(vrs_root: Path) -> set[str]:
    needed = set()
    for fname in [
        "VRSBench_EVAL_Cap.json",
        "VRSBench_EVAL_vqa.json",
        "VRSBench_EVAL_referring.json",
    ]:
        p = vrs_root / fname
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for it in data:
            img_id = it.get("image_id")
            if img_id:
                needed.add(img_id)
    return needed


def selective_extract(zip_path: Path, dest_dir: Path, needed: set[str]) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    matched = 0
    if not zip_path.exists():
        return 0
    with ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        # Images are usually at the top level of the zip
        to_get = [n for n in needed if n in names]
        for n in to_get:
            target = dest_dir / n
            if target.exists():
                matched += 1
                continue
            with zf.open(n) as src, open(target, "wb") as dst:
                dst.write(src.read())
            matched += 1
    return matched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".")
    args = ap.parse_args()
    root = Path(args.root)
    vrs = root / "VRSBench"
    needed = load_needed_ids(vrs)
    print(f"need {len(needed)} files from eval jsons")

    total = 0
    # Try val first
    total += selective_extract(vrs / "Images_val.zip", vrs / "Images_val", needed)
    # Then train
    remaining = needed - {p.name for p in (vrs / "Images_val").glob("*.png")}
    total += selective_extract(vrs / "Images_train.zip", vrs / "Images_train", remaining)
    print(f"extracted {total} files")


if __name__ == "__main__":
    main()

