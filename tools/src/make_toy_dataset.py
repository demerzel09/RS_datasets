#!/usr/bin/env python3
"""
Make a small, heterogeneous RS toy dataset from this workspace.

Creates task-split JSONL annotations and copies/normalizes images.

Tasks supported (separate outputs):
- caption: RSICD, RSITMD, VRSBench (caption)
- vqa: VRSBench (VQA)
- ref: VRSBench (referring), AIR-SLT (polygon)
- cls: AID, RESISC45, EuroSAT (directory labels)
- paired (optional): SSL4EO-S12 100 patches (RGB vs S2A multispectral vs S1 SAR, time-aligned)

Usage examples
  python tools/make_toy_dataset.py --root . \
    --out toy_datasets --seed 42 \
    --count-caption 300 --count-vqa 200 --count-ref 50 --count-cls 200 \
    --make-paired

Notes
- Images are converted to RGB PNG/JPEG with longest side=512 by default.
- TIF handling: if >=3 bands, use first 3; if single band, replicate to 3; values are min-max normalized per band.
- Ref polygons are preserved as-is; bboxes are also emitted for convenience.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# Lazy imports for heavy libs
def _lazy_imports():
    try:
        from PIL import Image
    except Exception as e:
        raise RuntimeError("Pillow is required: pip install pillow") from e
    try:
        import numpy as np
    except Exception as e:
        raise RuntimeError("NumPy is required: pip install numpy") from e
    try:
        import tifffile as tiff
    except Exception as e:
        raise RuntimeError("tifffile is required for .tif: pip install tifffile") from e
    return Image, np, tiff


@dataclass
class Cfg:
    root: Path
    out: Path
    seed: int
    # Task counts; None means skip that task
    caption_n: Optional[int] = 300
    vqa_n: Optional[int] = 200
    ref_n: Optional[int] = 50
    cls_n: Optional[int] = 200
    resize: int = 512
    jpeg: bool = False  # default PNG
    make_paired: bool = False
    max_per_image: int = 1


def set_seed(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def normalize_and_save_image(src: Path, dst: Path, resize: int, jpeg: bool):
    """Load image (jpg/png/tif), convert to 3ch uint8, resize longest side, save.
    - jpg/png: Pillow handles
    - tif: tifffile -> numpy -> Pillow
    """
    Image, np, tiff = _lazy_imports()

    # Load
    ext = src.suffix.lower()
    arr: np.ndarray
    if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        with Image.open(src) as im:
            im = im.convert("RGB")
            arr = np.array(im)
    elif ext in [".tif", ".tiff"]:
        arr = tiff.imread(str(src))
        # Shape handling: (H, W), (H, W, C), or (C, H, W)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3:
            # Heuristic: if first dim is small (<=12) treat as (C,H,W)
            if arr.shape[0] <= 16 and arr.shape[0] < arr.shape[-1]:
                arr = np.transpose(arr, (1, 2, 0))  # to HWC
            # If more than 3 bands, take first 3
            if arr.shape[-1] >= 3:
                arr = arr[..., :3]
            elif arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            else:
                # 2 bands -> pad with mean
                mean_band = np.mean(arr, axis=-1, keepdims=True)
                arr = np.concatenate([arr, mean_band], axis=-1)
        else:
            raise RuntimeError(f"Unsupported TIF shape: {arr.shape} for {src}")
        # Per-band min-max to 0-255
        arr = arr.astype("float32")
        # Avoid div by zero
        for c in range(3):
            band = arr[..., c]
            bmin = float(np.nanmin(band))
            bmax = float(np.nanmax(band))
            if not math.isfinite(bmin) or not math.isfinite(bmax) or bmax <= bmin:
                arr[..., c] = 0
            else:
                arr[..., c] = (band - bmin) / max(1e-6, (bmax - bmin))
        arr = (arr * 255.0).clip(0, 255).astype("uint8")
    else:
        raise RuntimeError(f"Unsupported image extension: {src}")

    # Resize longest side
    h, w = arr.shape[:2]
    scale = resize / float(max(h, w)) if max(h, w) > resize else 1.0
    if scale != 1.0:
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        im = Image.fromarray(arr).resize((nw, nh), resample=Image.BICUBIC)
    else:
        im = Image.fromarray(arr)

    # Save
    ensure_dir(dst.parent)
    if jpeg:
        im.save(dst.with_suffix(".jpg"), format="JPEG", quality=90, optimize=True)
        return dst.with_suffix(".jpg")
    else:
        im.save(dst.with_suffix(".png"), format="PNG", optimize=True)
        return dst.with_suffix(".png")


def save_array_image(arr, dst: Path, resize: int, jpeg: bool):
    """Save a HxWxC uint8 array as RGB image with resizing similar to normalize_and_save_image."""
    Image, np, _ = _lazy_imports()
    if arr.dtype != np.uint8:
        arr = arr.astype("uint8")
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    h, w = arr.shape[:2]
    scale = resize / float(max(h, w)) if max(h, w) > resize else 1.0
    im = Image.fromarray(arr)
    if scale != 1.0:
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        im = im.resize((nw, nh), resample=Image.BICUBIC)
    ensure_dir(dst.parent)
    if jpeg:
        im.save(dst.with_suffix(".jpg"), format="JPEG", quality=90, optimize=True)
        return dst.with_suffix(".jpg")
    else:
        im.save(dst.with_suffix(".png"), format="PNG", optimize=True)
        return dst.with_suffix(".png")


def write_jsonl(path: Path, rows: Iterable[dict]):
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------- Collectors per source ----------


def collect_rsicd_caption(cfg: Cfg, limit: int) -> List[dict]:
    root = cfg.root / "rs5m_test_data" / "rsicd"
    img_root = root / "RSICD_images"
    csv_path = root / "rsicd_test.csv"
    rows = []
    if not csv_path.exists():
        return rows
    seen = set()
    per_image: dict[str,int] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        items = list(reader)
    random.shuffle(items)
    for it in items:
        fn = it.get("filename")
        text = it.get("title")
        if not fn or not text:
            continue
        key = (fn, text)
        if key in seen:
            continue
        seen.add(key)
        src = img_root / fn
        if not src.exists():
            continue
        # enforce per-image cap
        cnt = per_image.get(fn, 0)
        if cnt >= cfg.max_per_image:
            continue
        out_rel = Path("images") / "caption" / "rsicd" / fn
        out_abs = cfg.out / out_rel
        try:
            saved = normalize_and_save_image(src, out_abs, cfg.resize, cfg.jpeg)
        except Exception:
            continue
        rows.append({
            "task": "caption",
            "image": str(saved.relative_to(cfg.out)).replace("\\", "/"),
            "text": text.strip(),
            "source": "RSICD",
        })
        per_image[fn] = cnt + 1
        if len(rows) >= limit:
            break
    return rows


def collect_rsitmd_caption(cfg: Cfg, limit: int) -> List[dict]:
    root = cfg.root / "rs5m_test_data" / "rsitmd"
    img_root = root / "images"
    csv_path = root / "rsitmd_test.csv"
    rows = []
    if not csv_path.exists():
        return rows
    seen = set()
    per_image: dict[str,int] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        items = list(reader)
    random.shuffle(items)
    for it in items:
        fn = it.get("filename")
        text = it.get("title")
        if not fn or not text:
            continue
        key = (fn, text)
        if key in seen:
            continue
        seen.add(key)
        src = img_root / fn
        if not src.exists():
            continue
        # per-image cap
        cnt = per_image.get(fn, 0)
        if cnt >= cfg.max_per_image:
            continue
        out_rel = Path("images") / "caption" / "rsitmd" / fn
        out_abs = cfg.out / out_rel
        try:
            saved = normalize_and_save_image(src, out_abs, cfg.resize, cfg.jpeg)
        except Exception:
            continue
        rows.append({
            "task": "caption",
            "image": str(saved.relative_to(cfg.out)).replace("\\", "/"),
            "text": text.strip(),
            "source": "RSITMD",
        })
        per_image[fn] = cnt + 1
        if len(rows) >= limit:
            break
    return rows


def _vrsbench_find_image(root: Path, image_id: str) -> Optional[Path]:
    for split in ["Images_val", "Images_train"]:
        p = root / split / image_id
        if p.exists():
            return p
    return None


def collect_vrsbench_caption(cfg: Cfg, limit: int) -> List[dict]:
    root = cfg.root / "VRSBench"
    ann = root / "VRSBench_EVAL_Cap.json"
    rows = []
    if not ann.exists():
        return rows
    items = json.loads(ann.read_text(encoding="utf-8"))
    random.shuffle(items)
    for it in items:
        img_id = it.get("image_id")
        text = it.get("ground_truth") or it.get("caption")
        if not img_id or not text:
            continue
        src = _vrsbench_find_image(root, img_id)
        if not src:
            # Log missing image id
            with (cfg.out / "logs" / "skipped.txt").open("a", encoding="utf-8") as lf:
                lf.write(f"[VRSBench-CAP] missing image {img_id}\n")
            continue
        out_rel = Path("images") / "caption" / "vrsbench" / img_id
        out_abs = cfg.out / out_rel
        try:
            saved = normalize_and_save_image(src, out_abs, cfg.resize, cfg.jpeg)
        except Exception:
            continue
        rows.append({
            "task": "caption",
            "image": str(saved.relative_to(cfg.out)).replace("\\", "/"),
            "text": text.strip(),
            "source": "VRSBench",
        })
        if len(rows) >= limit:
            break
    return rows


def collect_vrsbench_vqa(cfg: Cfg, limit: int) -> List[dict]:
    root = cfg.root / "VRSBench"
    ann = root / "VRSBench_EVAL_vqa.json"
    rows: List[dict] = []
    if not ann.exists():
        return rows
    items = json.loads(ann.read_text(encoding="utf-8"))
    # Group by image_id to include all Q&A per image
    by_img: Dict[str, List[dict]] = {}
    for it in items:
        img_id = it.get("image_id")
        if not img_id:
            continue
        by_img.setdefault(img_id, []).append(it)
    img_ids = list(by_img.keys())
    random.shuffle(img_ids)
    saved_cache: Dict[str, Path] = {}
    images_added = 0
    for img_id in img_ids:
        if images_added >= limit:
            break
        src = _vrsbench_find_image(root, img_id)
        if not src:
            with (cfg.out / "logs" / "skipped.txt").open("a", encoding="utf-8") as lf:
                lf.write(f"[VRSBench-VQA] missing image {img_id}\n")
            continue
        out_rel = Path("images") / "vqa" / "vrsbench" / img_id
        out_abs = cfg.out / out_rel
        try:
            saved = saved_cache.get(img_id)
            if saved is None:
                saved = normalize_and_save_image(src, out_abs, cfg.resize, cfg.jpeg)
                saved_cache[img_id] = saved
        except Exception:
            continue
        # Add all Q&A for this image
        qas = by_img[img_id]
        random.shuffle(qas)
        added_any = False
        for it in qas:
            q = it.get("question")
            a = it.get("ground_truth")
            if not q or a is None:
                continue
            rows.append({
                "task": "vqa",
                "image": str(saved.relative_to(cfg.out)).replace("\\", "/"),
                "question": str(q).strip(),
                "answer": str(a).strip(),
                "source": "VRSBench",
            })
            added_any = True
        if added_any:
            images_added += 1
    return rows


def collect_referring(cfg: Cfg, limit: int) -> List[dict]:
    rows: List[dict] = []
    limit = max(0, limit or 0)
    if limit == 0:
        return rows

    # Load metadata for both sources first so we can balance quotas.
    root_v = cfg.root / "VRSBench"
    ann_v = root_v / "VRSBench_EVAL_referring.json"
    vrs_by_img: Dict[str, List[dict]] = {}
    if ann_v.exists():
        items = json.loads(ann_v.read_text(encoding="utf-8"))
        for it in items:
            img_id = it.get("image_id")
            if not img_id:
                continue
            vrs_by_img.setdefault(img_id, []).append(it)
    vrs_img_ids = list(vrs_by_img.keys())
    random.shuffle(vrs_img_ids)

    root_a = cfg.root / "rs5m_test_data" / "AIR-SLT"
    ann_a = root_a / "annotations" / "anno.json"
    img_root_a = root_a / "imgs"
    air_by_img: Dict[str, List[dict]] = {}
    if ann_a.exists():
        items = json.loads(ann_a.read_text(encoding="utf-8"))
        for it in items:
            img_id = it.get("jpg_name")
            if not img_id:
                continue
            air_by_img.setdefault(img_id, []).append(it)
    air_img_ids = list(air_by_img.keys())
    random.shuffle(air_img_ids)

    availability = {
        "VRSBench": len(vrs_img_ids),
        "AIR-SLT": len(air_img_ids),
    }
    total_available = availability["VRSBench"] + availability["AIR-SLT"]
    if total_available == 0:
        return rows
    target_total = min(limit, total_available)

    # Desired ratio: prefer VRSBench but always reserve a chunk for AIR-SLT when possible.
    weights = {"VRSBench": 0.6, "AIR-SLT": 0.4}
    quotas: Dict[str, int] = {}
    fractions: List[Tuple[float, str]] = []
    for src, weight in weights.items():
        raw = target_total * weight
        base = int(math.floor(raw))
        quotas[src] = base
        fractions.append((raw - base, src))
    remaining = target_total - sum(quotas.values())
    fractions.sort(reverse=True)
    for frac, src in fractions:
        if remaining <= 0:
            break
        quotas[src] += 1
        remaining -= 1

    allocations = {}
    for src, quota in quotas.items():
        allocations[src] = min(quota, availability.get(src, 0))
    allocated = sum(allocations.values())
    leftover = target_total - allocated
    if leftover > 0:
        # Give leftover slots to sources that still have capacity, prioritizing larger availability gaps.
        order = sorted(
            allocations.keys(),
            key=lambda s: (availability.get(s, 0) - allocations[s]),
            reverse=True,
        )
        for src in order:
            if leftover <= 0:
                break
            capacity = availability.get(src, 0) - allocations[src]
            if capacity <= 0:
                continue
            take = min(capacity, leftover)
            allocations[src] += take
            leftover -= take

    total_added = 0
    used_images: set[str] = set()

    # Collect from VRSBench up to its allocation.
    vrs_target = allocations.get("VRSBench", 0)
    if vrs_target > 0:
        saved_cache: Dict[str, Path] = {}
        for img_id in vrs_img_ids:
            if total_added >= target_total or vrs_target <= 0:
                break
            group = vrs_by_img[img_id]
            src = _vrsbench_find_image(root_v, img_id)
            if not src:
                with (cfg.out / "logs" / "skipped.txt").open("a", encoding="utf-8") as lf:
                    lf.write(f"[VRSBench-REF] missing image {img_id}\n")
                continue
            out_rel = Path("images") / "ref" / "vrsbench" / img_id
            out_abs = cfg.out / out_rel
            try:
                saved = saved_cache.get(img_id)
                if saved is None:
                    saved = normalize_and_save_image(src, out_abs, cfg.resize, cfg.jpeg)
                    saved_cache[img_id] = saved
            except Exception:
                continue
            rel_path = str(saved.relative_to(cfg.out)).replace("\\", "/")
            if rel_path in used_images:
                continue
            random.shuffle(group)
            added_any = False
            for it in group:
                expr = it.get("question")
                if not expr:
                    continue
                bbox = None
                if isinstance(it.get("obj_corner"), list) and len(it["obj_corner"]) >= 4:
                    pts = it["obj_corner"]
                    xs = pts[0::2]
                    ys = pts[1::2]
                    bbox = [min(xs), min(ys), max(xs), max(ys)]
                rows.append({
                    "task": "ref",
                    "image": rel_path,
                    "expression": str(expr).strip(),
                    "bbox": bbox,
                    "polygon": it.get("obj_corner"),
                    "source": "VRSBench",
                })
                added_any = True
            if added_any:
                used_images.add(rel_path)
                total_added += 1
                vrs_target -= 1

    # Collect from AIR-SLT up to its allocation.
    air_target = allocations.get("AIR-SLT", 0)
    if air_target > 0:
        for img_id in air_img_ids:
            if total_added >= target_total or air_target <= 0:
                break
            group = air_by_img[img_id]
            src = img_root_a / img_id
            if not src.exists():
                with (cfg.out / "logs" / "skipped.txt").open("a", encoding="utf-8") as lf:
                    lf.write(f"[AIR-SLT] missing image {img_id}\n")
                continue
            out_rel = Path("images") / "ref" / "air-slt" / img_id
            out_abs = cfg.out / out_rel
            try:
                saved = normalize_and_save_image(src, out_abs, cfg.resize, cfg.jpeg)
            except Exception:
                continue
            rel_path = str(saved.relative_to(cfg.out)).replace("\\", "/")
            if rel_path in used_images:
                continue
            random.shuffle(group)
            added_any = False
            for it in group:
                expr = it.get("caption")
                pts = it.get("points")
                if not expr or not pts:
                    continue
                flat: List[float] = []
                for poly in pts:
                    for xy in poly:
                        if len(xy) >= 2:
                            flat.extend([float(xy[0]), float(xy[1])])
                bbox = None
                if flat:
                    xs = flat[0::2]
                    ys = flat[1::2]
                    bbox = [min(xs), min(ys), max(xs), max(ys)]
                rows.append({
                    "task": "ref",
                    "image": rel_path,
                    "expression": str(expr).strip(),
                    "bbox": bbox,
                    "polygon": flat,
                    "source": "AIR-SLT",
                })
                added_any = True
            if added_any:
                used_images.add(rel_path)
                total_added += 1
                air_target -= 1

    return rows


def collect_classification(cfg: Cfg, limit: int) -> List[dict]:
    rows: List[dict] = []
    # Strategy: sample per dataset quotas
    quotas = {
        "AID": int(limit * 0.4),  # 40%
        "RESISC45": int(limit * 0.4),  # 40%
        "EuroSAT": limit - int(limit * 0.4) - int(limit * 0.4),  # remainder
    }
    # AID (pass 1: roughly uniform per class)
    root_a = cfg.root / "rs5m_test_data" / "AID"
    if root_a.exists():
        classes = [d for d in root_a.iterdir() if d.is_dir()]
        per = max(1, quotas["AID"] // max(1, len(classes)))
        for clsdir in classes:
            if len(rows) >= quotas["AID"]:
                break
            imgs = list(clsdir.glob("*.jpg"))
            random.shuffle(imgs)
            for p in imgs[:per]:
                out_rel = Path("images") / "cls" / "AID" / clsdir.name / p.name
                out_abs = cfg.out / out_rel
                try:
                    saved = normalize_and_save_image(p, out_abs, cfg.resize, cfg.jpeg)
                except Exception:
                    with (cfg.out / "logs" / "skipped.txt").open("a", encoding="utf-8") as lf:
                        lf.write(f"[AID] failed to process {p.name}\n")
                    continue
                rows.append({
                    "task": "cls",
                    "image": str(saved.relative_to(cfg.out)).replace("\\", "/"),
                    "label": clsdir.name,
                    "source": "AID",
                })
                if len(rows) >= quotas["AID"]:
                    break
        # pass 2: fill remainder from any class
        remaining = max(0, quotas["AID"] - len([r for r in rows if r.get("source") == "AID"]))
        if remaining > 0:
            all_imgs = list(root_a.rglob("*.jpg"))
            random.shuffle(all_imgs)
            for p in all_imgs:
                if remaining <= 0:
                    break
                clsname = p.parent.name
                out_rel = Path("images") / "cls" / "AID" / clsname / p.name
                out_abs = cfg.out / out_rel
                try:
                    saved = normalize_and_save_image(p, out_abs, cfg.resize, cfg.jpeg)
                except Exception:
                    continue
                rows.append({
                    "task": "cls",
                    "image": str(saved.relative_to(cfg.out)).replace("\\", "/"),
                    "label": clsname,
                    "source": "AID",
                })
                remaining -= 1
    # RESISC45 (uniform, then fill remainder)
    root_r = cfg.root / "rs5m_test_data" / "RESISC45"
    if root_r.exists():
        classes = [d for d in root_r.iterdir() if d.is_dir()]
        per = max(1, quotas["RESISC45"] // max(1, len(classes)))
        for clsdir in classes:
            if len([r for r in rows if r.get("source") == "RESISC45"]) >= quotas["RESISC45"]:
                break
            imgs = list(clsdir.glob("*.jpg"))
            random.shuffle(imgs)
            for p in imgs[:per]:
                out_rel = Path("images") / "cls" / "RESISC45" / clsdir.name / p.name
                out_abs = cfg.out / out_rel
                try:
                    saved = normalize_and_save_image(p, out_abs, cfg.resize, cfg.jpeg)
                except Exception:
                    with (cfg.out / "logs" / "skipped.txt").open("a", encoding="utf-8") as lf:
                        lf.write(f"[RESISC45] failed to process {p.name}\n")
                    continue
                rows.append({
                    "task": "cls",
                    "image": str(saved.relative_to(cfg.out)).replace("\\", "/"),
                    "label": clsdir.name,
                    "source": "RESISC45",
                })
                if len([r for r in rows if r.get("source") == "RESISC45"]) >= quotas["RESISC45"]:
                    break
        remaining = max(0, quotas["RESISC45"] - len([r for r in rows if r.get("source") == "RESISC45"]))
        if remaining > 0:
            all_imgs = list(root_r.rglob("*.jpg"))
            random.shuffle(all_imgs)
            for p in all_imgs:
                if remaining <= 0:
                    break
                clsname = p.parent.name
                out_rel = Path("images") / "cls" / "RESISC45" / clsname / p.name
                out_abs = cfg.out / out_rel
                try:
                    saved = normalize_and_save_image(p, out_abs, cfg.resize, cfg.jpeg)
                except Exception:
                    continue
                rows.append({
                    "task": "cls",
                    "image": str(saved.relative_to(cfg.out)).replace("\\", "/"),
                    "label": clsname,
                    "source": "RESISC45",
                })
                remaining -= 1
    # EuroSAT RGB (uniform, then fill remainder)
    root_e = cfg.root / "rs5m_test_data" / "eurosat-rgb" / "2750"
    if root_e.exists():
        classes = [d for d in root_e.iterdir() if d.is_dir()]
        target = quotas["EuroSAT"]
        per = max(1, target // max(1, len(classes)))
        added = 0
        for clsdir in classes:
            if added >= target:
                break
            imgs = list(clsdir.glob("*.jpg"))
            random.shuffle(imgs)
            take = min(per, max(0, target - added))
            for p in imgs[:take]:
                out_rel = Path("images") / "cls" / "EuroSAT" / clsdir.name / p.name
                out_abs = cfg.out / out_rel
                try:
                    saved = normalize_and_save_image(p, out_abs, cfg.resize, cfg.jpeg)
                except Exception:
                    with (cfg.out / "logs" / "skipped.txt").open("a", encoding="utf-8") as lf:
                        lf.write(f"[EuroSAT] failed to process {p.name}\n")
                    continue
                rows.append({
                    "task": "cls",
                    "image": str(saved.relative_to(cfg.out)).replace("\\", "/"),
                    "label": clsdir.name,
                    "source": "EuroSAT",
                })
                added += 1
                if added >= target:
                    break
        # fill remainder from any class
        remaining = max(0, target - added)
        if remaining > 0:
            all_imgs = list(root_e.rglob("*.jpg"))
            random.shuffle(all_imgs)
            for p in all_imgs:
                if remaining <= 0:
                    break
                clsname = p.parent.name
                out_rel = Path("images") / "cls" / "EuroSAT" / clsname / p.name
                out_abs = cfg.out / out_rel
                try:
                    saved = normalize_and_save_image(p, out_abs, cfg.resize, cfg.jpeg)
                except Exception:
                    continue
                rows.append({
                    "task": "cls",
                    "image": str(saved.relative_to(cfg.out)).replace("\\", "/"),
                    "label": clsname,
                    "source": "EuroSAT",
                })
                remaining -= 1
    return rows[:limit]


_DATE_RE = re.compile(r"(\d{8})T(\d{6})")


def _parse_date_from_name(name: str) -> Optional[datetime]:
    m = _DATE_RE.search(name)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


def _s2a_make_rgb_from_bands(folder: Path, dst: Path, resize: int, jpeg: bool) -> Optional[Path]:
    """Compose RGB from Sentinel-2 bands in folder (B4,B3,B2)."""
    Image, np, tiff = _lazy_imports()
    bands = []
    for b in ["B4.tif", "B3.tif", "B2.tif"]:
        p = folder / b
        if not p.exists():
            return None
        arr = tiff.imread(str(p)).astype("float32")
        bands.append(arr)
    arr = __import__('numpy').stack(bands, axis=-1)
    # Per-band min-max
    for c in range(3):
        band = arr[..., c]
        bmin = float(__import__('numpy').nanmin(band))
        bmax = float(__import__('numpy').nanmax(band))
        if not math.isfinite(bmin) or not math.isfinite(bmax) or bmax <= bmin:
            arr[..., c] = 0
        else:
            arr[..., c] = (band - bmin) / max(1e-6, (bmax - bmin))
    arr = (arr * 255.0).clip(0, 255).astype("uint8")
    # Resize
    h, w = arr.shape[:2]
    scale = resize / float(max(h, w)) if max(h, w) > resize else 1.0
    im = Image.fromarray(arr)
    if scale != 1.0:
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        im = im.resize((nw, nh), resample=Image.BICUBIC)
    ensure_dir(dst.parent)
    if jpeg:
        im.save(dst.with_suffix(".jpg"), format="JPEG", quality=90, optimize=True)
        return dst.with_suffix(".jpg")
    else:
        im.save(dst.with_suffix(".png"), format="PNG", optimize=True)
        return dst.with_suffix(".png")


def _s1_make_rgb_from_vvvh(folder: Path, dst: Path, resize: int, jpeg: bool) -> Optional[Path]:
    """Compose pseudo-RGB from Sentinel-1 VV/VH."""
    Image, np, tiff = _lazy_imports()
    vv = folder / "VV.tif"
    vh = folder / "VH.tif"
    if not vv.exists() or not vh.exists():
        return None
    arr_vv = tiff.imread(str(vv)).astype("float32")
    arr_vh = tiff.imread(str(vh)).astype("float32")
    # Normalize each
    def norm(x):
        xmin = float(__import__('numpy').nanmin(x))
        xmax = float(__import__('numpy').nanmax(x))
        if not math.isfinite(xmin) or not math.isfinite(xmax) or xmax <= xmin:
            return __import__('numpy').zeros_like(x)
        y = (x - xmin) / max(1e-6, (xmax - xmin))
        return y
    vvn = norm(arr_vv)
    vhn = norm(arr_vh)
    comp = __import__('numpy').stack([vvn, vhn, 0.5 * (vvn + vhn)], axis=-1)
    comp = (comp * 255.0).clip(0, 255).astype("uint8")
    # Resize and save
    h, w = comp.shape[:2]
    scale = resize / float(max(h, w)) if max(h, w) > resize else 1.0
    im = Image.fromarray(comp)
    if scale != 1.0:
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        im = im.resize((nw, nh), resample=Image.BICUBIC)
    ensure_dir(dst.parent)
    if jpeg:
        im.save(dst.with_suffix(".jpg"), format="JPEG", quality=90, optimize=True)
        return dst.with_suffix(".jpg")
    else:
        im.save(dst.with_suffix(".png"), format="PNG", optimize=True)
        return dst.with_suffix(".png")


def _parse_band_name(name: str) -> Tuple[int, str]:
    # e.g., B8A -> (8, 'A'), B2 -> (2, '')
    m = re.match(r"B(\d+)([A-Z]?)", name.upper())
    if not m:
        return (999, name)
    return (int(m.group(1)), m.group(2))


def _label_resolution_tag(max_area: int, area: int) -> str:
    if area <= 0 or max_area <= 0:
        return "g"
    ratio = max_area / float(area)
    # Map ratio ~1->10m, ~4->20m, ~36->60m with tolerance
    def close(x, y, tol=0.35):
        return abs(x - y) / y <= tol
    if close(ratio, 1.0):
        return "10m"
    if close(ratio, 4.0):
        return "20m"
    if close(ratio, 36.0):
        return "60m"
    return f"g{int(round(ratio))}"


def _s2_stack_raw_groups(folder: Path, out_prefix: Path) -> Dict[str, Path]:
    """[Deprecated for 'raw' output] Kept for backward compatibility where stacking is desired.
    Prefer using _s2_copy_raw_groups to preserve original band files.
    """
    Image, np, tiff = _lazy_imports()
    band_files = sorted([p for p in folder.iterdir() if p.suffix.lower() in (".tif", ".tiff") and p.stem.upper().startswith("B")])
    if not band_files:
        return {}
    # Read shapes
    groups: Dict[Tuple[int, int], List[Tuple[str, Path]]] = {}
    for p in band_files:
        try:
            arr = tiff.imread(str(p), maxworkers=1)
        except Exception:
            continue
        if arr.ndim != 2:
            continue
        h, w = arr.shape
        key = (h, w)
        groups.setdefault(key, []).append((p.stem.upper(), p))
    if not groups:
        return {}
    # Sort groups by area desc
    items = sorted(groups.items(), key=lambda kv: kv[0][0] * kv[0][1], reverse=True)
    max_area = items[0][0][0] * items[0][0][1]
    saved: Dict[str, Path] = {}
    for (h, w), entries in items:
        tag = _label_resolution_tag(max_area, h * w)
        # sort bands using numeric then suffix
        entries.sort(key=lambda t: _parse_band_name(t[0]))
        # stack
        arrays = []
        dtypes = []
        for bname, path in entries:
            try:
                a = tiff.imread(str(path), maxworkers=1)
            except Exception:
                continue
            arrays.append(a)
            dtypes.append(a.dtype)
        if not arrays:
            continue
        common = arrays[0].dtype
        try:
            import numpy as np
            common = np.result_type(*dtypes)
            arr = np.stack(arrays, axis=0)  # (C,H,W)
            arr = arr.astype(common, copy=False)
        except Exception:
            continue
        out_path = out_prefix.with_name(out_prefix.stem + f"_{tag}.tif")
        ensure_dir(out_path.parent)
        try:
            tiff.imwrite(str(out_path), arr, photometric='minisblack')
            saved[tag] = out_path
        except Exception:
            continue
    return saved


def _s1_stack_raw(folder: Path, out_path: Path) -> Optional[Path]:
    Image, np, tiff = _lazy_imports()
    vv = folder / "VV.tif"
    vh = folder / "VH.tif"
    if not vv.exists() or not vh.exists():
        return None
    try:
        a_vv = tiff.imread(str(vv))
        a_vh = tiff.imread(str(vh))
    except Exception:
        return None


def _s2_copy_raw_groups(folder: Path, out_dir: Path) -> Dict[str, List[Path]]:
    """Copy Sentinel-2 single-band TIFs preserving originals.
    Groups by identical shape to infer resolution tags via _label_resolution_tag.
    Returns dict tag -> list of copied file paths.
    """
    Image, np, tiff = _lazy_imports()
    ensure_dir(out_dir)
    band_files = sorted([p for p in folder.iterdir() if p.suffix.lower() in (".tif", ".tiff") and p.stem.upper().startswith("B")])
    if not band_files:
        return {}
    shapes: Dict[Tuple[int, int], List[Path]] = {}
    for p in band_files:
        try:
            arr = tiff.imread(str(p), maxworkers=1)
        except Exception:
            continue
        if arr.ndim != 2:
            continue
        h, w = arr.shape
        shapes.setdefault((h, w), []).append(p)
    if not shapes:
        return {}
    items = sorted(shapes.items(), key=lambda kv: kv[0][0] * kv[0][1], reverse=True)
    max_area = items[0][0][0] * items[0][0][1]
    out: Dict[str, List[Path]] = {}
    for (h, w), files in items:
        tag = _label_resolution_tag(max_area, h * w)
        for src in files:
            dst = out_dir / src.name
            try:
                ensure_dir(dst.parent)
                shutil.copy2(src, dst)
                out.setdefault(tag, []).append(dst)
            except Exception:
                continue
    return out


def _s1_copy_raw(folder: Path, out_dir: Path) -> Dict[str, Optional[Path]]:
    """Copy Sentinel-1 VV/VH TIFs preserving originals. Returns dict with keys VV/VH."""
    ensure_dir(out_dir)
    vv = folder / "VV.tif"
    vh = folder / "VH.tif"
    out: Dict[str, Optional[Path]] = {"VV": None, "VH": None}
    if vv.exists():
        try:
            dst = out_dir / vv.name
            shutil.copy2(vv, dst)
            out["VV"] = dst
        except Exception:
            pass
    if vh.exists():
        try:
            dst = out_dir / vh.name
            shutil.copy2(vh, dst)
            out["VH"] = dst
        except Exception:
            pass
    return out
    if a_vv.shape != a_vh.shape or a_vv.ndim != 2:
        return None
    try:
        import numpy as np
        common = np.result_type(a_vv.dtype, a_vh.dtype)
        arr = np.stack([a_vv, a_vh], axis=0).astype(common, copy=False)
        ensure_dir(out_path.parent)
        import tifffile as tiff
        tiff.imwrite(str(out_path), arr, photometric='minisblack')
        return out_path
    except Exception:
        return None


def collect_paired_modalities(cfg: Cfg, max_pairs: int = 40) -> List[dict]:
    """Build small set of aligned RGB/S2A/S1 per time for SSL4EO-S12 100 patches.
    Returns list of rows for paired.jsonl
    """
    root = cfg.root / "SSL4EO-S12-v1.1" / "ssl4eo-s12_100patches"
    rows: List[dict] = []
    if not root.exists():
        return rows
    rgb_root = root / "rgb"
    s2a_root = root / "s2a"  # Sentinel-2 Level-2A
    s2c_root = root / "s2c"  # Sentinel-2 Level-1C (if available)
    s1_root = root / "s1"
    patch_ids = [d.name for d in sorted(rgb_root.iterdir()) if d.is_dir()]
    random.shuffle(patch_ids)
    for pid in patch_ids:
        if len(rows) >= max_pairs:
            break
        rgb_dir = rgb_root / pid
        s2a_dir = s2a_root / pid
        s2c_dir = s2c_root / pid
        s1_dir = s1_root / pid
        if not (s2a_dir.exists() and s1_dir.exists()):
            continue
        # Collect dates
        rgb_items = [(p, _parse_date_from_name(p.name)) for p in rgb_dir.glob("*.png")]
        rgb_items = [(p, dt) for p, dt in rgb_items if dt]
        if not rgb_items:
            continue
        s2a_items = []
        for d in s2a_dir.iterdir():
            if d.is_dir():
                dt = _parse_date_from_name(d.name)
                if dt:
                    s2a_items.append((d, dt))
        s2c_items = []
        if s2c_dir.exists():
            for d in s2c_dir.iterdir():
                if d.is_dir():
                    dt = _parse_date_from_name(d.name)
                    if dt:
                        s2c_items.append((d, dt))
        s1_items = []
        for d in s1_dir.iterdir():
            if d.is_dir():
                dt = _parse_date_from_name(d.name)
                if dt:
                    s1_items.append((d, dt))
        if not s2a_items or not s1_items:
            continue
        # For each rgb time, match nearest s2a and s1 within 7 days
        for rgb_p, rgb_dt in sorted(rgb_items, key=lambda x: x[1]):
            if len(rows) >= max_pairs:
                break
            nearest_s2 = min(s2a_items, key=lambda x: abs(x[1] - rgb_dt))
            nearest_s1 = min(s1_items, key=lambda x: abs(x[1] - rgb_dt))
            # relax thresholds to broaden matches if needed
            if abs(nearest_s2[1] - rgb_dt) > timedelta(days=30):
                continue
            if abs(nearest_s1[1] - rgb_dt) > timedelta(days=30):
                continue
            # Save composed images
            base = f"{pid}_{rgb_dt.strftime('%Y%m%d')}"
            out_rgb = cfg.out / "images" / "paired" / "ssl4eo-s12" / f"{base}_rgb.png"
            out_s2 = cfg.out / "images" / "paired" / "ssl4eo-s12" / f"{base}_s2rgb.png"
            out_s1 = cfg.out / "images" / "paired" / "ssl4eo-s12" / f"{base}_sar.png"
            raw_root_dir = cfg.out / "images" / "paired" / "ssl4eo-s12" / "raw"
            out_s2a_raw_dir = raw_root_dir / f"{base}_s2a"
            out_s2c_raw_dir = raw_root_dir / f"{base}_s2c"
            out_s1_raw_dir = raw_root_dir / f"{base}_s1"
            s2a_raw_groups = {}
            s2c_raw_groups = {}
            s1_raw_files = {}
            try:
                saved_rgb = normalize_and_save_image(rgb_p, out_rgb, cfg.resize, cfg.jpeg)
                saved_s2 = _s2a_make_rgb_from_bands(nearest_s2[0], out_s2, cfg.resize, cfg.jpeg)
                if not saved_s2:
                    continue
                saved_s1 = _s1_make_rgb_from_vvvh(nearest_s1[0], out_s1, cfg.resize, cfg.jpeg)
                if not saved_s1:
                    continue
                # Copy raw original files (no stacking)
                s2a_raw_groups = _s2_copy_raw_groups(nearest_s2[0], out_s2a_raw_dir)
                if s2c_items:
                    nearest_s2c = min(s2c_items, key=lambda x: abs(x[1] - rgb_dt))
                    if abs(nearest_s2c[1] - rgb_dt) <= timedelta(days=30):
                        s2c_raw_groups = _s2_copy_raw_groups(nearest_s2c[0], out_s2c_raw_dir)
                s1_raw_files = _s1_copy_raw(nearest_s1[0], out_s1_raw_dir)
            except Exception:
                continue
            rows.append({
                "task": "paired",
                "patch_id": pid,
                "date": rgb_dt.strftime("%Y-%m-%d"),
                "rgb": str(saved_rgb.relative_to(cfg.out)).replace("\\", "/"),
                "s2_rgb": str(saved_s2.relative_to(cfg.out)).replace("\\", "/"),
                "s1_rgb": str(saved_s1.relative_to(cfg.out)).replace("\\", "/"),
                "s2a_raw": {k: [str(p.relative_to(cfg.out)).replace("\\", "/") for p in v] for k, v in s2a_raw_groups.items()},
                "s2c_raw": {k: [str(p.relative_to(cfg.out)).replace("\\", "/") for p in v] for k, v in s2c_raw_groups.items()},
                "s1_raw": {k: (str(p.relative_to(cfg.out)).replace("\\", "/") if p else None) for k, p in s1_raw_files.items()},
                "source": "SSL4EO-S12",
            })
    return rows


def collect_vqa_temporal_ssl4eo(cfg: Cfg, series_limit: int, frames_per_series: int = 4) -> List[dict]:
    """Create temporal VQA rows aligned with paired.jsonl keys from SSL4EO-S12.
    Emits one row per frame with series_id and frame_index, incl. rgb/s2/s1 paths.
    """
    root = cfg.root / "SSL4EO-S12-v1.1" / "ssl4eo-s12_100patches"
    rows: List[dict] = []
    if not root.exists() or series_limit <= 0:
        return rows
    rgb_root = root / "rgb"
    s2a_root = root / "s2a"
    s2c_root = root / "s2c"
    s1_root = root / "s1"
    patch_ids = [d.name for d in sorted(rgb_root.iterdir()) if d.is_dir()]
    random.shuffle(patch_ids)
    series_added = 0
    for pid in patch_ids:
        if series_added >= series_limit:
            break
        rgb_dir = rgb_root / pid
        s2a_dir = s2a_root / pid
        s2c_dir = s2c_root / pid
        s1_dir = s1_root / pid
        # collect RGB frames with dates
        rgb_items = [(p, _parse_date_from_name(p.name)) for p in rgb_dir.glob("*.png")]
        rgb_items = [(p, dt) for p, dt in rgb_items if dt]
        if len(rgb_items) < 2:
            continue
        rgb_items.sort(key=lambda x: x[1])
        # pick up to frames_per_series roughly evenly spaced
        idxs = list(range(len(rgb_items)))
        if len(idxs) > frames_per_series:
            # evenly spaced indices
            idxs = [int(round(i * (len(rgb_items) - 1) / (frames_per_series - 1))) for i in range(frames_per_series)]
            idxs = sorted(set(idxs))
        picks = [rgb_items[i] for i in idxs]
        # verify other modalities availability optionally but do not enforce
        out_rows_for_series: List[dict] = []
        frame_index = 0
        for rgb_p, dt in picks:
            date_str = dt.strftime("%Y-%m-%d")
            base = f"{pid}_{dt.strftime('%Y%m%d')}"
            # Place under dataset subfolder for clarity: vqa_ts/ssl4eo-s12
            out_rgb = cfg.out / "images" / "vqa_ts" / "ssl4eo-s12" / f"{base}_rgb.png"
            out_s2 = cfg.out / "images" / "vqa_ts" / "ssl4eo-s12" / f"{base}_s2rgb.png"
            out_s1 = cfg.out / "images" / "vqa_ts" / "ssl4eo-s12" / f"{base}_sar.png"
            raw_prefix_dir = cfg.out / "images" / "vqa_ts" / "ssl4eo-s12" / "raw"
            out_s2a_raw_dir = raw_prefix_dir / f"{base}_s2a"
            out_s2c_raw_dir = raw_prefix_dir / f"{base}_s2c"
            out_s1_raw_dir = raw_prefix_dir / f"{base}_s1"
            try:
                saved_rgb = normalize_and_save_image(rgb_p, out_rgb, cfg.resize, cfg.jpeg)
            except Exception:
                continue
            saved_s2 = None
            saved_s1 = None
            s2a_raw = {}
            s2c_raw = {}
            s1_raw_path = None
            # find nearest same-date folders for s2a/s1
            s2a_raw_groups = {}
            s2c_raw_groups = {}
            s1_raw_files = {}
            try:
                if s2a_dir.exists():
                    # find folder by exact date match
                    cand = [d for d in s2a_dir.iterdir() if d.is_dir() and _parse_date_from_name(d.name)]
                    if cand:
                        nearest_s2 = min(cand, key=lambda x: abs(_parse_date_from_name(x.name) - dt))
                        if abs(_parse_date_from_name(nearest_s2.name) - dt) <= timedelta(days=30):
                            saved_s2 = _s2a_make_rgb_from_bands(nearest_s2, out_s2, cfg.resize, cfg.jpeg)
                            s2a_raw_groups = _s2_copy_raw_groups(nearest_s2, out_s2a_raw_dir)
                if s2c_dir.exists():
                    cand = [d for d in s2c_dir.iterdir() if d.is_dir() and _parse_date_from_name(d.name)]
                    if cand:
                        nearest_s2c = min(cand, key=lambda x: abs(_parse_date_from_name(x.name) - dt))
                        if abs(_parse_date_from_name(nearest_s2c.name) - dt) <= timedelta(days=30):
                            s2c_raw_groups = _s2_copy_raw_groups(nearest_s2c, out_s2c_raw_dir)
                if s1_dir.exists():
                    cand = [d for d in s1_dir.iterdir() if d.is_dir() and _parse_date_from_name(d.name)]
                    if cand:
                        nearest_s1 = min(cand, key=lambda x: abs(_parse_date_from_name(x.name) - dt))
                        if abs(_parse_date_from_name(nearest_s1.name) - dt) <= timedelta(days=30):
                            saved_s1 = _s1_make_rgb_from_vvvh(nearest_s1, out_s1, cfg.resize, cfg.jpeg)
                            s1_raw_files = _s1_copy_raw(nearest_s1, out_s1_raw_dir)
            except Exception:
                pass
            out_rows_for_series.append({
                "task": "vqa_ts",
                "series_id": pid,
                "frame_index": frame_index,
                "series_len": len(picks),
                "patch_id": pid,
                "date": date_str,
                "rgb": str(saved_rgb.relative_to(cfg.out)).replace("\\", "/"),
                "s2_rgb": str(saved_s2.relative_to(cfg.out)).replace("\\", "/") if saved_s2 else None,
                "s1_rgb": str(saved_s1.relative_to(cfg.out)).replace("\\", "/") if saved_s1 else None,
                "s2a_raw": {k: [str(p.relative_to(cfg.out)).replace("\\", "/") for p in v] for k, v in s2a_raw_groups.items()},
                "s2c_raw": {k: [str(p.relative_to(cfg.out)).replace("\\", "/") for p in v] for k, v in s2c_raw_groups.items()},
                "s1_raw": {k: (str(p.relative_to(cfg.out)).replace("\\", "/") if p else None) for k, p in s1_raw_files.items()},
                "question": None,
                "answer": None,
                "source": "SSL4EO-S12",
            })
            frame_index += 1
        if out_rows_for_series:
            rows.extend(out_rows_for_series)
            series_added += 1
    return rows


def _extract_video_frames(video_path: Path, max_frames: int = 4, log_path: Optional[Path] = None) -> List:
    """Extract up to max_frames frames from a video using imageio if available.
    Falls back to sequential read if length unknown. Logs errors if log_path provided.
    """
    try:
        import imageio
    except Exception as e:
        if log_path:
            ensure_dir(log_path.parent)
            with log_path.open("a", encoding="utf-8") as lf:
                lf.write(f"[GeoLLaVA] imageio not available: {e}\n")
        return []
    try:
        rdr = imageio.get_reader(str(video_path))
    except Exception as e:
        if log_path:
            ensure_dir(log_path.parent)
            with log_path.open("a", encoding="utf-8") as lf:
                lf.write(f"[GeoLLaVA] failed to open video {video_path.name}: {e}\n")
        return []
    frames = []
    # Try even spacing by length; if not available, take first max_frames frames
    try:
        n = rdr.get_length()
    except Exception:
        n = -1
    try:
        if isinstance(n, int) and n > 0:
            idxs = list(range(n))
            if len(idxs) > max_frames:
                idxs = [int(round(i * (n - 1) / (max_frames - 1))) for i in range(max_frames)]
                idxs = sorted(set(idxs))
            for i in idxs:
                try:
                    frames.append(rdr.get_data(i))
                except Exception:
                    continue
        else:
            # fallback: sequential read
            for i, frame in enumerate(rdr):
                frames.append(frame)
                if len(frames) >= max_frames:
                    break
    except Exception as e:
        if log_path:
            ensure_dir(log_path.parent)
            with log_path.open("a", encoding="utf-8") as lf:
                lf.write(f"[GeoLLaVA] failed to read frames {video_path.name}: {e}\n")
    try:
        rdr.close()
    except Exception:
        pass
    return frames


def collect_vqa_temporal_geollava(cfg: Cfg, series_limit: int, frames_per_series: int = 2) -> List[dict]:
    """Create temporal VQA rows from GeoLLaVA videos. Follows paired-like keys.
    Each output row is a frame with series_id and frame_index; question/answer left None.
    """
    root = cfg.root / "GeoLLaVA"
    ann = root / "updated_val_annotations.json"
    vid_root = root / "updated_val_videos"
    rows: List[dict] = []
    if not ann.exists() or not vid_root.exists() or series_limit <= 0:
        return rows
    try:
        items = json.loads(ann.read_text(encoding="utf-8"))
    except Exception:
        return rows
    random.shuffle(items)
    series_added = 0
    for it in items:
        if series_added >= series_limit:
            break
        vid_name = it.get("video")
        if not vid_name:
            continue
        vpath = vid_root / vid_name
        if not vpath.exists():
            # Log missing video
            with (cfg.out / "logs" / "skipped.txt").open("a", encoding="utf-8") as lf:
                lf.write(f"[GeoLLaVA] missing video {vid_name}\n")
            continue
        frames = _extract_video_frames(vpath, max_frames=frames_per_series, log_path=(cfg.out / "logs" / "skipped.txt"))
        if len(frames) < 2:
            # Log insufficient frames
            with (cfg.out / "logs" / "skipped.txt").open("a", encoding="utf-8") as lf:
                lf.write(f"[GeoLLaVA] insufficient frames extracted from {vid_name}\n")
            continue
        # Extract Q/A from conversations if available
        q_text: Optional[str] = None
        a_text: Optional[str] = None
        conv = it.get("conversations") or []
        for turn in conv:
            role = (turn.get("from") or "").lower()
            val = turn.get("value") or ""
            if role == "human" and q_text is None:
                q_text = val.replace("<video>", "").strip()
            elif role == "gpt" and a_text is None:
                a_text = val.strip()
            if q_text and a_text:
                break
        out_rows_for_series: List[dict] = []
        for fi, frame in enumerate(frames):
            base = f"{Path(vid_name).stem}_{fi:02d}"
            out_rgb = cfg.out / "images" / "vqa_ts" / "geollava" / f"{base}_rgb.png"
            try:
                saved_rgb = save_array_image(frame, out_rgb, cfg.resize, cfg.jpeg)
            except Exception:
                continue
            out_rows_for_series.append({
                "task": "vqa_ts",
                "series_id": Path(vid_name).stem,
                "frame_index": fi,
                "series_len": len(frames),
                "patch_id": Path(vid_name).stem,
                "date": None,
                "rgb": str(saved_rgb.relative_to(cfg.out)).replace("\\", "/"),
                "s2_rgb": None,
                "s1_rgb": None,
                "s2a_raw": {},
                "s2c_raw": {},
                "s1_raw": None,
                "question": q_text,
                "answer": a_text,
                "source": "GeoLLaVA",
            })
        if out_rows_for_series:
            rows.extend(out_rows_for_series)
            series_added += 1
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="Workspace RS root (this dir)")
    ap.add_argument("--out", type=str, default="toy_datasets", help="Output dir")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resize", type=int, default=512)
    ap.add_argument("--jpeg", action="store_true", help="Save JPEG instead of PNG")
    ap.add_argument("--count-caption", type=int, default=None, help="Number of caption samples (None to skip)")
    ap.add_argument("--count-vqa", type=int, default=None, help="Number of VQA images (None to skip; includes all Q&A per image)")
    ap.add_argument("--count-ref", type=int, default=None, help="Number of Referring images (None to skip; includes all expressions per image)")
    ap.add_argument("--count-cls", type=int, default=None, help="Number of classification images (None to skip)")
    ap.add_argument("--make-paired", action="store_true")
    ap.add_argument("--count-vqa-ts", type=int, default=0, help="Number of temporal VQA series (images) to include")
    ap.add_argument("--vqa-ts-frames", type=int, default=4, help="Frames per temporal VQA series")
    args = ap.parse_args()

    cfg = Cfg(
        root=Path(args.root),
        out=Path(args.out),
        seed=args.seed,
        # Keep None to indicate skip for each task unless explicitly set
        caption_n=args.count_caption,
        vqa_n=args.count_vqa,
        ref_n=args.count_ref,
        cls_n=args.count_cls,
        resize=args.resize,
        jpeg=bool(args.jpeg),
        make_paired=bool(args.make_paired),
    )
    set_seed(cfg.seed)

    # Prepare output
    ensure_dir(cfg.out / "images")
    ensure_dir(cfg.out / "logs")

    # Collect caption from multiple sources with randomized mix across RSICD/RSITMD/VRSBench
    cap_rows: List[dict] = []
    if cfg.caption_n is not None and cfg.caption_n > 0:
        cap_budget = cfg.caption_n
        # allocate counts randomly but roughly evenly across 3 sources
        probs = [1.0, 1.0, 1.0]
        total = sum(probs)
        ratios = [p / total for p in probs]
        alloc = [int(round(cap_budget * r)) for r in ratios]
        drift = cap_budget - sum(alloc)
        # fix rounding drift by distributing +/-1
        idx = 0
        while drift != 0:
            take = min(1, abs(drift))
            if drift > 0:
                alloc[idx % 3] += take
                drift -= take
            else:
                # avoid making negative
                if alloc[idx % 3] > 0:
                    alloc[idx % 3] -= take
                    drift += take
            idx += 1
        sources = [
            (collect_rsicd_caption, alloc[0]),
            (collect_rsitmd_caption, alloc[1]),
            (collect_vrsbench_caption, alloc[2]),
        ]
        random.shuffle(sources)
        for fn, cnt in sources:
            if cnt <= 0:
                continue
            cap_rows.extend(fn(cfg, cnt))
        # Fill any shortfall from remaining sources
        shortfall = cap_budget - len(cap_rows)
        if shortfall > 0:
            fillers = [collect_rsicd_caption, collect_rsitmd_caption, collect_vrsbench_caption]
            random.shuffle(fillers)
            for fn in fillers:
                if shortfall <= 0:
                    break
                got = fn(cfg, shortfall)
                cap_rows.extend(got)
                shortfall = cap_budget - len(cap_rows)
        random.shuffle(cap_rows)
        write_jsonl(cfg.out / "caption.jsonl", cap_rows)

    # VQA
    vqa_rows: List[dict] = []
    if cfg.vqa_n is not None and cfg.vqa_n > 0:
        vqa_rows = collect_vrsbench_vqa(cfg, cfg.vqa_n)
        write_jsonl(cfg.out / "vqa.jsonl", vqa_rows)

    # Referring
    ref_rows: List[dict] = []
    if cfg.ref_n is not None and cfg.ref_n > 0:
        ref_rows = collect_referring(cfg, cfg.ref_n)
        write_jsonl(cfg.out / "ref.jsonl", ref_rows)

    # Classification
    cls_rows: List[dict] = []
    if cfg.cls_n is not None and cfg.cls_n > 0:
        cls_rows = collect_classification(cfg, cfg.cls_n)
        write_jsonl(cfg.out / "cls.jsonl", cls_rows)

    # Paired modalities (optional)
    if cfg.make_paired:
        paired_rows = collect_paired_modalities(cfg, max_pairs=30)
        write_jsonl(cfg.out / "paired.jsonl", paired_rows)

    # Temporal VQA (optional)
    vqa_ts_rows: List[dict] = []
    if args.count_vqa_ts and args.count_vqa_ts > 0:
        # Split budget half-half between SSL4EO and GeoLLaVA by default
        ssl_quota = max(0, args.count_vqa_ts // 2)
        geo_quota = max(0, args.count_vqa_ts - ssl_quota)
        vqa_ts_rows.extend(collect_vqa_temporal_ssl4eo(cfg, ssl_quota, frames_per_series=args.vqa_ts_frames))
        vqa_ts_rows.extend(collect_vqa_temporal_geollava(cfg, geo_quota, frames_per_series=min(2, args.vqa_ts_frames)))
        write_jsonl(cfg.out / "vqa_ts.jsonl", vqa_ts_rows)

    # Summary
    summary = {}
    if cfg.caption_n is not None:
        summary["caption"] = len(cap_rows)
    if cfg.vqa_n is not None:
        summary["vqa"] = len(vqa_rows)
    if cfg.ref_n is not None:
        summary["ref"] = len(ref_rows)
    if cfg.cls_n is not None:
        summary["cls"] = len(cls_rows)
    if cfg.make_paired:
        summary["paired"] = len(paired_rows)
    if vqa_ts_rows:
        # count series by unique series_id
        series_ids = set(r.get("series_id") for r in vqa_ts_rows)
        summary["vqa_ts_rows"] = len(vqa_ts_rows)
        summary["vqa_ts_series"] = len([sid for sid in series_ids if sid])
    (cfg.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
