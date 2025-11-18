#!/usr/bin/env python3
"""
make_toy_dataset_to_json.py

Wrapper around the original `make_toy_dataset.py` collectors that emits
task-level JSON files (capture-first / series-aware) instead of JSONL.

Usage mirrors the original script. Example (PowerShell):
python .\tools\src\make_toy_dataset_to_json.py --root . --out toy_datasets --count-caption 50 --count-vqa 30 --make-paired

Notes:
- This script dynamically loads the original `make_toy_dataset.py` module
  and reuses its collection functions to avoid code duplication.
- Outputs: `caption.json`, `vqa.json`, `ref.json`, `cls.json`, `paired.json`, `vqa_ts.json`
  placed under the `--out` directory (same location as original script writes images).
"""
from __future__ import annotations

import argparse
import importlib.util
import runpy
import types
import json
import sys
from pathlib import Path
from typing import Dict, List


def extract_location_from_row(r: dict):
    """Try to extract a geo location from a collector row.
    Returns a normalized dict or None. Normalized form: {'lat': float, 'lon': float}
    Only fill when obvious latitude/longitude-like fields exist.
    """
    if not isinstance(r, dict):
        return None
    # direct location dict with lat/lon
    loc = r.get("location")
    if isinstance(loc, dict):
        lat = loc.get("lat") or loc.get("latitude")
        lon = loc.get("lon") or loc.get("longitude")
        if lat is not None and lon is not None:
            try:
                return {"lat": float(lat), "lon": float(lon)}
            except Exception:
                return None
        # if dict doesn't have lat/lon, return raw dict assuming it's already normalized
        return loc
    # common separate fields
    lat = r.get("lat") or r.get("latitude") or r.get("y")
    lon = r.get("lon") or r.get("longitude") or r.get("x")
    if lat is not None and lon is not None:
        try:
            return {"lat": float(lat), "lon": float(lon)}
        except Exception:
            return None
    # coords as list [lat, lon] or [lon, lat]
    coords = r.get("coords") or r.get("coordinate") or r.get("coordinates")
    if isinstance(coords, (list, tuple)) and len(coords) >= 2:
        try:
            # guess order: if lat looks in [-90,90] use [lat,lon]
            a = float(coords[0])
            b = float(coords[1])
            if -90 <= a <= 90 and -180 <= b <= 180:
                return {"lat": a, "lon": b}
            if -90 <= b <= 90 and -180 <= a <= 180:
                return {"lat": b, "lon": a}
        except Exception:
            return None
    # bbox/polygon (geo) - not handled unless explicit lat/lon present
    return None


def load_make_toy_module(path: Path):
    """
    Safely load the make_toy_dataset.py as a module-like object.

    Use runpy.run_path to execute the file in a fresh globals dict without
    triggering the `if __name__ == '__main__'` main() call. Return a
    SimpleNamespace wrapper so callers can access functions/objects via
    attribute access (e.g., mod.collect_vrsbench_vqa).
    """
    g = runpy.run_path(str(path), run_name="make_toy_dataset")
    return types.SimpleNamespace(**g)


def make_capture_id_from_path(p: str, used: Dict[str, int]) -> str:
    # sanitize filename to stable capture id
    name = Path(p).stem
    base = f"c_{name}"
    if base not in used:
        used[base] = 0
        return base
    used[base] += 1
    return f"{base}_{used[base]}"


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

    root = Path(args.root)
    out = Path(args.out)
    # load original module
    src_mod_path = Path(__file__).resolve().parents[0] / "make_toy_dataset.py"
    if not src_mod_path.exists():
        # try one level up (tools/src vs tools)
        src_mod_path = Path(__file__).resolve().parents[1] / "tools" / "make_toy_dataset.py"
    if not src_mod_path.exists():
        print("Could not find make_toy_dataset.py for reuse", file=sys.stderr)
        sys.exit(1)
    mtd = load_make_toy_module(src_mod_path)

    # Build Cfg object from original
    cfg = mtd.Cfg(
        root=root,
        out=out,
        seed=args.seed,
        caption_n=args.count_caption,
        vqa_n=args.count_vqa,
        ref_n=args.count_ref,
        cls_n=args.count_cls,
        resize=args.resize,
        jpeg=bool(args.jpeg),
        make_paired=bool(args.make_paired),
    )
    mtd.set_seed(cfg.seed)

    # Ensure out exists
    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)

    # We'll build per-task JSON in capture-first form.
    # Helper maps
    file_to_capture: Dict[str, str] = {}
    used_capture_bases: Dict[str, int] = {}
    captures: Dict[str, dict] = {}
    # track which captures are used by each task so we write only relevant captures
    caption_cids = set()
    vqa_cids = set()
    ref_cids = set()
    cls_cids = set()
    paired_cids = set()
    vqa_ts_cids = set()

    def ensure_capture_for_file(file_path: str, sensor_type: str = "rgb") -> str:
        # file_path is relative path string (as returned by collectors)
        if file_path in file_to_capture:
            return file_to_capture[file_path]
        cid = make_capture_id_from_path(file_path, used_capture_bases)
        file_to_capture[file_path] = cid
        # initialize capture entry minimal
        cap = {
            "capture_id": cid,
            "timestamp": None,
            "location": None,
            "sensors": [{"sensor_type": sensor_type, "file_name": file_path}],
        }
        captures[cid] = cap
        return cid

    def _add_raw_sensors_from_row(cap: dict, row: dict):
        """If the collector row contains raw modality fields (s2a_raw, s2c_raw, s1_raw),
        add them to the capture's sensors list. This preserves raw original files while
        annotations/reference images remain the RGB derivatives (as in original script).
        """
        if not isinstance(row, dict):
            return
        # s2a_raw and s2c_raw are dicts tag -> [paths]
        for key in ("s2a_raw", "s2c_raw"):
            grp = row.get(key) or {}
            if isinstance(grp, dict):
                for tag, paths in grp.items():
                    if not isinstance(paths, (list, tuple)):
                        continue
                    for p in paths:
                        if not p:
                            continue
                        if not any(s.get("file_name") == p for s in cap.get("sensors", [])):
                            cap.setdefault("sensors", []).append({"sensor_type": f"{key}", "file_name": p, "tag": tag})
        # s1_raw is usually a dict of keys (VV/VH) -> path or None
        s1 = row.get("s1_raw") or {}
        if isinstance(s1, dict):
            for subk, p in s1.items():
                if not p:
                    continue
                if not any(s.get("file_name") == p for s in cap.get("sensors", [])):
                    cap.setdefault("sensors", []).append({"sensor_type": "s1_raw", "file_name": p, "band": subk})

    # --- Caption ---
    caption_json = {"info": {"name": "caption", "version": "v1"}, "captures": []}
    if cfg.caption_n is not None and cfg.caption_n > 0:
        # reuse original allocation logic: call its caption collectors
        # We'll invoke the same collection strategy as original main by calling collect functions directly
        cap_rows = []
        # Use same allocation as original main
        cap_budget = cfg.caption_n
        probs = [1.0, 1.0, 1.0]
        total = sum(probs)
        ratios = [p / total for p in probs]
        alloc = [int(round(cap_budget * r)) for r in ratios]
        drift = cap_budget - sum(alloc)
        idx = 0
        while drift != 0:
            take = min(1, abs(drift))
            if drift > 0:
                alloc[idx % 3] += take
                drift -= take
            else:
                if alloc[idx % 3] > 0:
                    alloc[idx % 3] -= take
                    drift += take
            idx += 1
        sources = [
            (mtd.collect_rsicd_caption, alloc[0]),
            (mtd.collect_rsitmd_caption, alloc[1]),
            (mtd.collect_vrsbench_caption, alloc[2]),
        ]
        import random

        random.shuffle(sources)
        for fn, cnt in sources:
            if cnt <= 0:
                continue
            cap_rows.extend(fn(cfg, cnt))
        shortfall = cap_budget - len(cap_rows)
        if shortfall > 0:
            fillers = [mtd.collect_rsicd_caption, mtd.collect_rsitmd_caption, mtd.collect_vrsbench_caption]
            random.shuffle(fillers)
            for fn in fillers:
                if shortfall <= 0:
                    break
                got = fn(cfg, shortfall)
                cap_rows.extend(got)
                shortfall = cap_budget - len(cap_rows)

        # convert rows to capture-first
        for r in cap_rows:
            img = r.get("image")
            text = r.get("text")
            source = r.get("source")
            if not img or not text:
                continue
            cid = ensure_capture_for_file(img, sensor_type="rgb")
            # try to populate geo location if present in row
            loc = extract_location_from_row(r)
            if loc and not captures[cid].get("location"):
                captures[cid]["location"] = loc
            caption_cids.add(cid)
            cap = captures[cid]
            cap.setdefault("captions", [])
            cap["captions"].append({"text": text, "source": source})

    # --- VQA ---
    vqa_json = {"info": {"name": "vqa", "version": "v1"}, "questions": [], "captures": [], "annotations": []}
    if cfg.vqa_n is not None and cfg.vqa_n > 0:
        vqa_rows = mtd.collect_vrsbench_vqa(cfg, cfg.vqa_n)
        # build question bank (question text -> q_id)
        qmap: Dict[str, str] = {}
        # also map q_id -> question text to detect collisions
        qid_to_text: Dict[str, str] = {}
        q_used = 0
        for r in vqa_rows:
            qtext = r.get("question")
            if not qtext:
                continue
            # if source provided and a question_id exists in the source row, prefer it
            qid_field = r.get("question_id") or r.get("qid")
            if qid_field is not None:
                qid_candidate = str(qid_field)
                # if this qid was already seen with a different text, namespace with source
                existing = qid_to_text.get(qid_candidate)
                if existing is not None and existing != qtext:
                    src = r.get("source") or r.get("dataset") or "src"
                    qid_candidate = f"{src}_{qid_candidate}"
                # register mapping
                if qtext not in qmap:
                    qmap[qtext] = qid_candidate
                    qid_to_text[qid_candidate] = qtext
                continue
            # fallback: no question_id in row -> auto allocate by question text
            if qtext not in qmap:
                q_used += 1
                qid = f"q_{q_used:04d}"
                qmap[qtext] = qid
                qid_to_text[qid] = qtext
        # attach questions
        for qtext, qid in qmap.items():
            vqa_json["questions"].append({"q_id": qid, "question": qtext})
        # group rows by image
        by_img: Dict[str, List[dict]] = {}
        for r in vqa_rows:
            img = r.get("image")
            if not img:
                continue
            by_img.setdefault(img, []).append(r)
        for img, items in by_img.items():
            cid = ensure_capture_for_file(img, sensor_type="rgb")
            # try to extract location from any item for this image
            for it in items:
                loc = extract_location_from_row(it)
                if loc and not captures[cid].get("location"):
                    captures[cid]["location"] = loc
            vqa_cids.add(cid)
            ann = {"capture_id": cid, "vqa_instances": []}
            for it in items:
                qtext = it.get("question")
                ans = it.get("answer")
                if not qtext:
                    continue
                qid = qmap.get(qtext)
                if qid is None:
                    continue
                # answers may be single string
                answers = [ans] if isinstance(ans, str) else (ans or [])
                # include question text alongside q_id for convenience
                ann["vqa_instances"].append({"q_id": qid, "question": qtext, "answers": answers})
            vqa_json["annotations"].append(ann)

    # --- Referring ---
    ref_json = {"info": {"name": "ref", "version": "v1"}, "captures": [], "refs": []}
    if cfg.ref_n is not None and cfg.ref_n > 0:
        ref_rows = mtd.collect_referring(cfg, cfg.ref_n)
        # rows contain: image, expression, bbox, polygon, source
        for r in ref_rows:
            img = r.get("image")
            expr = r.get("expression")
            bbox = r.get("bbox")
            poly = r.get("polygon")
            src = r.get("source")
            if not img or not expr:
                continue
            cid = ensure_capture_for_file(img, sensor_type="rgb")
            # try to populate location if present
            loc = extract_location_from_row(r)
            if loc and not captures[cid].get("location"):
                captures[cid]["location"] = loc
            ref_cids.add(cid)
            cap = captures[cid]
            cap.setdefault("objects", [])
            # store ref as top-level refs as well
            ref_json.setdefault("refs", []).append({"capture_id": cid, "expression": expr, "bbox": bbox, "polygon": poly, "source": src})

    # --- Classification ---
    cls_json = {"info": {"name": "cls", "version": "v1"}, "categories": [], "captures": []}
    if cfg.cls_n is not None and cfg.cls_n > 0:
        cls_rows = mtd.collect_classification(cfg, cfg.cls_n)
        for r in cls_rows:
            img = r.get("image")
            label = r.get("label")
            src = r.get("source")
            if not img or label is None:
                continue
            cid = ensure_capture_for_file(img, sensor_type="rgb")
            loc = extract_location_from_row(r)
            if loc and not captures[cid].get("location"):
                captures[cid]["location"] = loc
            cls_cids.add(cid)
            cap = captures[cid]
            cap.setdefault("classifications", [])
            cap["classifications"].append({"label": label, "source": src})

    # --- Paired (series) ---
    paired_json = {"info": {"name": "paired", "version": "v1"}, "captures": [], "series": []}
    if cfg.make_paired:
        paired_rows = mtd.collect_paired_modalities(cfg, max_pairs=30)
        # paired_rows contain rgb/s2_rgb/s1_rgb paths and patch_id/date
        # group by patch_id for series
        by_patch: Dict[str, List[dict]] = {}
        for r in paired_rows:
            pid = r.get("patch_id") or r.get("patch") or "p_unknown"
            by_patch.setdefault(pid, []).append(r)
        for pid, items in by_patch.items():
            # sort by date if present
            items.sort(key=lambda x: x.get("date") or "")
            capture_ids = []
            for it in items:
                rgb = it.get("rgb")
                s2 = it.get("s2_rgb")
                s1 = it.get("s1_rgb")
                # create captures for each modality but unify per-row into one capture
                fname = rgb or s2 or s1
                if not fname:
                    continue
                cid = ensure_capture_for_file(fname, sensor_type="rgb")
                # try to populate location from row metadata
                loc = extract_location_from_row(it)
                if loc and not captures[cid].get("location"):
                    captures[cid]["location"] = loc
                paired_cids.add(cid)
                cap = captures[cid]
                # ensure sensors include any other modalities
                if s2:
                    cap.setdefault("sensors", [])
                    if not any(s.get("file_name") == s2 for s in cap["sensors"]):
                        cap["sensors"].append({"sensor_type": "s2", "file_name": s2})
                if s1:
                    cap.setdefault("sensors", [])
                    if not any(s.get("file_name") == s1 for s in cap["sensors"]):
                        cap["sensors"].append({"sensor_type": "s1", "file_name": s1})
                # add raw modality files (s2a_raw/s2c_raw/s1_raw) into sensors so raw originals are preserved
                try:
                    _add_raw_sensors_from_row(cap, it)
                except Exception:
                    pass
                capture_ids.append(cid)
            if capture_ids:
                paired_json["series"].append({"series_id": f"s_{pid}", "capture_ids": capture_ids, "meta": {"patch_id": pid}})

    # --- VQA-TS ---
    vqa_ts_json = {"info": {"name": "vqa_ts", "version": "v1"}, "questions": [], "captures": [], "series": [], "series_annotations": []}
    if args.count_vqa_ts and args.count_vqa_ts > 0:
        # Use same splitting as original main
        ssl_quota = max(0, args.count_vqa_ts // 2)
        geo_quota = max(0, args.count_vqa_ts - ssl_quota)
        rows_ssl = mtd.collect_vqa_temporal_ssl4eo(cfg, ssl_quota, frames_per_series=args.vqa_ts_frames)
        rows_geo = mtd.collect_vqa_temporal_geollava(cfg, geo_quota, frames_per_series=min(2, args.vqa_ts_frames))
        all_rows = rows_ssl + rows_geo
        # build series map
        series_map: Dict[str, List[dict]] = {}
        for r in all_rows:
            sid = r.get("series_id") or r.get("patch_id") or "s_unknown"
            series_map.setdefault(sid, []).append(r)
        for sid, frames in series_map.items():
            # sort by frame_index
            frames.sort(key=lambda x: int(x.get("frame_index", 0)))
            capture_ids = []
            frame_annotations = []
            for fr in frames:
                rgb = fr.get("rgb")
                if not rgb:
                    continue
                cid = ensure_capture_for_file(rgb, sensor_type="rgb")
                # try to extract location for this frame
                loc = extract_location_from_row(fr)
                if loc and not captures[cid].get("location"):
                    captures[cid]["location"] = loc
                vqa_ts_cids.add(cid)
                # add any raw modality files for this frame (preserve raw originals)
                try:
                    _add_raw_sensors_from_row(captures[cid], fr)
                except Exception:
                    pass
                capture_ids.append(cid)
                vqas = fr.get("vqa_instances") or []
                # translate vqa_instances leaving q_id if present or use question text
                insts = []
                for vi in vqas:
                    # prefer explicit question id fields if present
                    qid_field = vi.get("q_id") or vi.get("question_id") or vi.get("qid")
                    if qid_field:
                        qid_str = str(qid_field)
                        # try to resolve question text from vqa_ts bank or global vqa bank
                        qtext = None
                        # search existing vqa_ts questions
                        for q in vqa_ts_json.get("questions", []):
                            if q.get("q_id") == qid_str:
                                qtext = q.get("question")
                                break
                        # fallback: search vqa questions (non-temporal)
                        if qtext is None:
                            for q in vqa_json.get("questions", []):
                                if q.get("q_id") == qid_str:
                                    qtext = q.get("question")
                                    break
                        insts.append({"q_id": qid_str, "question": qtext, "answers": vi.get("answers", [])})
                    elif vi.get("question"):
                        # create question entry if needed
                        qtxt = vi.get("question")
                        if qtxt:
                            qid = f"q_auto_{abs(hash(qtxt)) % 100000}"
                            if qid not in [q.get("q_id") for q in vqa_ts_json.get("questions", [])]:
                                vqa_ts_json.setdefault("questions", []).append({"q_id": qid, "question": qtxt})
                            insts.append({"q_id": qid, "question": qtxt, "answers": vi.get("answers", [])})
                frame_annotations.append({"capture_id": cid, "vqa_instances": insts})
            if capture_ids:
                vqa_ts_json.setdefault("series", []).append({"series_id": sid, "capture_ids": capture_ids})
                vqa_ts_json.setdefault("series_annotations", []).append({"series_id": sid, "frame_annotations": frame_annotations})

    # finalize captures list for each output (select only used captures per task)
    def select_captures(cids_set):
        # preserve insertion order of captures dict
        return [captures[cid] for cid in captures.keys() if cid in cids_set]

    # write files
    def write_json(p: Path, data):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {p}")

    # caption
    if caption_cids:
        caption_json["captures"] = select_captures(caption_cids)
        write_json(out / "caption.json", caption_json)
    # vqa
    if vqa_cids:
        vqa_json["captures"] = select_captures(vqa_cids)
        write_json(out / "vqa.json", vqa_json)
    # ref
    if ref_cids and (ref_json.get("refs") or []):
        ref_json["captures"] = select_captures(ref_cids)
        write_json(out / "ref.json", ref_json)
    # cls
    if cls_cids:
        cls_json["captures"] = select_captures(cls_cids)
        write_json(out / "cls.json", cls_json)
    # paired (series)
    if paired_json.get("series") and paired_cids:
        paired_json["captures"] = select_captures(paired_cids)
        write_json(out / "paired.json", paired_json)
    # vqa_ts
    if vqa_ts_cids and vqa_ts_json.get("series"):
        vqa_ts_json["captures"] = select_captures(vqa_ts_cids)
        write_json(out / "vqa_ts.json", vqa_ts_json)


if __name__ == "__main__":
    main()
