#!/usr/bin/env python3
"""
Convert Landslide4Sense-2022 HDF5 samples into GeoTIFFs.

For each sample, exports:
- BAND.tif  : Sentinel-2 stack only (B1–B12) as (C,H,W)
- SLOPE.tif : slope raster (single band) [from band 13 if present]
- DEM.tif   : elevation raster (single band) [from band 14 if present]
- MASK.tif  : label mask (from split/mask/mask_*.h5 when available)

Defaults write .tif. You can change extension via --ext (e.g., --ext .tif or .dif).

Example
  python tools/convert_landslide4sense_h5.py \
    --root Landslide4Sense-2022 --split TestData --out out_landslide_s2 --ext .tif

Notes
- Preserves original dtype and values (no rescaling).
- Attempts to detect dataset keys robustly: Sentinel-2, slope, dem.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _lazy_imports():
    try:
        import h5py  # type: ignore
    except Exception as e:
        raise RuntimeError("h5py is required: pip install h5py") from e
    try:
        import numpy as np  # type: ignore
    except Exception as e:
        raise RuntimeError("NumPy is required: pip install numpy") from e
    try:
        import tifffile as tiff  # type: ignore
    except Exception as e:
        raise RuntimeError("tifffile is required: pip install tifffile") from e
    try:
        from PIL import Image  # type: ignore
    except Exception as e:
        raise RuntimeError("Pillow is required for PNG previews: pip install pillow") from e
    return h5py, np, tiff, Image


def _list_datasets(h5) -> List[Tuple[str, object]]:
    out = []
    def rec(g, pref=""):
        for k in g.keys():
            obj = g[k]
            name = f"{pref}{k}"
            try:
                import h5py
                if isinstance(obj, h5py.Dataset):
                    out.append((name, obj))
                else:
                    rec(obj, name + "/")
            except Exception:
                continue
    rec(h5)
    return out


def _pick_s2_dataset(dsets: List[Tuple[str, object]]):
    """Pick the Sentinel-2 dataset: prefer names containing s2/sen2/sentinel.
    Fallback: largest dataset with ndim >= 3.
    """
    cand = []
    for name, ds in dsets:
        n = name.lower()
        try:
            shape = tuple(ds.shape)
        except Exception:
            continue
        if any(x in n for x in ("sen2", "sentinel2", "sentinel_2", "s2")):
            if len(shape) >= 3:
                cand.append((name, ds, shape))
    if not cand:
        for name, ds in dsets:
            try:
                shape = tuple(ds.shape)
            except Exception:
                continue
            if len(shape) >= 3:
                cand.append((name, ds, shape))
    if not cand:
        return None
    # choose by total size desc
    cand.sort(key=lambda x: (x[2][0] if len(x[2])>0 else 0)*(x[2][1] if len(x[2])>1 else 0)*(x[2][2] if len(x[2])>2 else 0), reverse=True)
    return cand[0]


def _pick_single_band(dsets: List[Tuple[str, object]], keywords: Tuple[str, ...]):
    """Pick a 2D dataset by keywords (e.g., ('slope',), ('dem',))."""
    for name, ds in dsets:
        n = name.lower()
        if any(kw in n for kw in keywords):
            try:
                shape = tuple(ds.shape)
                if len(shape) == 2:
                    return name, ds, shape
            except Exception:
                continue
    # fallback: no match
    return None


def _guess_mask_path_for_image(h5_img_path: Path) -> Optional[Path]:
    """Given .../split/img/image_XXX.h5, return .../split/mask/mask_XXX.h5 if exists."""
    try:
        num = h5_img_path.stem.split("image_")[-1]
        mask_dir = h5_img_path.parent.parent / "mask"
        cand = mask_dir / f"mask_{num}.h5"
        return cand if cand.exists() else None
    except Exception:
        return None


def convert_one(h5_path: Path, out_dir: Path, ext: str = ".tif", verbose: bool = False) -> Dict[str, Optional[Path]]:
    h5py, np, tiff, Image = _lazy_imports()
    ensure_dir(out_dir)
    info: Dict[str, Optional[Path]] = {"band": None, "slope": None, "dem": None, "mask": None}

    def _save_rgb_png_from_chw12(chw_arr) -> Optional[Path]:
        try:
            if chw_arr.ndim != 3 or chw_arr.shape[0] < 3:
                return None
            # S2 true color: R=B4(3), G=B3(2), B=B2(1) with 0-based indices
            if chw_arr.shape[0] >= 4:
                r = chw_arr[3]
                g = chw_arr[2]
                b = chw_arr[1]
            else:
                # fallback to first three channels
                r, g, b = chw_arr[0], chw_arr[1], chw_arr[2]
            # per-band min-max to uint8
            def mm(x):
                x = x.astype('float32')
                vmin = float(np.nanmin(x))
                vmax = float(np.nanmax(x))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                    return np.zeros_like(x, dtype='uint8')
                y = (x - vmin) / max(1e-6, (vmax - vmin))
                return (y * 255.0).clip(0, 255).astype('uint8')
            rgb = np.stack([mm(r), mm(g), mm(b)], axis=-1)  # HWC uint8
            out_path = out_dir / "RGB.png"
            Image.fromarray(rgb).save(str(out_path), format="PNG", optimize=True)
            return out_path
        except Exception as e:
            if verbose:
                print(f"Failed to write RGB.png for {h5_path.name}: {e}")
            return None
    try:
        with h5py.File(h5_path, "r") as f:
            dsets = _list_datasets(f)
            # Sentinel-2 multiband
            picked = _pick_s2_dataset(dsets)
            if picked is not None:
                name, ds, shape = picked
                arr = ds[...]
                if arr.ndim < 3:
                    pass
                else:
                    # normalize to (C,H,W)
                    if arr.shape[0] <= 16 and arr.shape[0] < arr.shape[-1]:
                        chw = arr  # (C,H,W)
                    else:
                        chw = np.transpose(arr, (2, 0, 1))  # (H,W,C)->(C,H,W)
                    # Keep only Sentinel-2 bands (B1–B12)
                    chw12 = chw[:12] if chw.shape[0] >= 12 else chw
                    out_path = out_dir / f"BAND{ext}"
                    try:
                        # Write as interleaved samples-per-pixel (H, W, C) to improve QGIS compatibility
                        hwc12 = np.transpose(chw12, (1, 2, 0))
                        tiff.imwrite(str(out_path), hwc12, planarconfig='contig')
                        info["band"] = out_path
                    except Exception as e:
                        if verbose:
                            print(f"Failed to write BAND for {h5_path.name}: {e}")
                    # Write quicklook RGB from B4/B3/B2
                    _ = _save_rgb_png_from_chw12(chw12)
                    # If 14 bands present, derive SLOPE (B13) and DEM (B14)
                    try:
                        C = chw.shape[0]
                        if C >= 14:
                            slope_arr = chw[12]  # B13 (0-based index)
                            dem_arr = chw[13]    # B14
                            slope_path = out_dir / f"SLOPE{ext}"
                            dem_path = out_dir / f"DEM{ext}"
                            try:
                                tiff.imwrite(str(slope_path), slope_arr, photometric='minisblack')
                                info["slope"] = slope_path
                            except Exception as e:
                                if verbose:
                                    print(f"Failed to write derived SLOPE for {h5_path.name}: {e}")
                            try:
                                tiff.imwrite(str(dem_path), dem_arr, photometric='minisblack')
                                info["dem"] = dem_path
                            except Exception as e:
                                if verbose:
                                    print(f"Failed to write derived DEM for {h5_path.name}: {e}")
                    except Exception:
                        pass
            # SLOPE
            picked = _pick_single_band(dsets, ("slope", "slp"))
            if picked is not None:
                name, ds, shape = picked
                arr = ds[...]
                out_path = out_dir / f"SLOPE{ext}"
                try:
                    tiff.imwrite(str(out_path), arr, photometric='minisblack')
                    info["slope"] = out_path
                except Exception as e:
                    if verbose:
                        print(f"Failed to write SLOPE for {h5_path.name}: {e}")
            elif verbose:
                print(f"No SLOPE dataset matched in {h5_path.name}")
            # DEM
            picked = _pick_single_band(dsets, ("dem", "dtm", "elev", "elevation", "height", "srtm"))
            if picked is not None:
                name, ds, shape = picked
                arr = ds[...]
                out_path = out_dir / f"DEM{ext}"
                try:
                    tiff.imwrite(str(out_path), arr, photometric='minisblack')
                    info["dem"] = out_path
                except Exception as e:
                    if verbose:
                        print(f"Failed to write DEM for {h5_path.name}: {e}")
            elif verbose:
                print(f"No DEM dataset matched in {h5_path.name}")
    except Exception as e:
        if verbose:
            print(f"Failed to process {h5_path}: {e}")
        return info

    # Try to load MASK from sibling mask h5
    try:
        mask_h5 = _guess_mask_path_for_image(h5_path)
        if mask_h5 and mask_h5.exists():
            h5py, np, tiff, Image = _lazy_imports()
            with h5py.File(mask_h5, "r") as mf:
                dsets = _list_datasets(mf)
                picked = _pick_single_band(dsets, ("mask", "label", "gt"))
                if picked is not None:
                    name, ds, shape = picked
                    arr = ds[...]
                    out_path = out_dir / f"MASK{ext}"
                    try:
                        tiff.imwrite(str(out_path), arr, photometric='minisblack')
                        info["mask"] = out_path
                    except Exception as e:
                        if verbose:
                            print(f"Failed to write MASK for {h5_path.name}: {e}")
                    # Also write quick-look grayscale PNG for immediate viewing
                    try:
                        m = arr
                        # scale 0/1 masks up to 0/255; otherwise clip to 0..255 then cast
                        try:
                            mmin = float(np.nanmin(m))
                            mmax = float(np.nanmax(m))
                        except Exception:
                            mmin, mmax = 0.0, 1.0
                        if mmax <= 1.0:
                            m8 = (np.clip(m, 0, 1) * 255.0).astype('uint8')
                        else:
                            m8 = np.clip(m, 0, 255).astype('uint8')
                        png_path = out_dir / "mask.png"
                        Image.fromarray(m8, mode="L").save(str(png_path), format="PNG", optimize=True)
                    except Exception as e:
                        if verbose:
                            print(f"Failed to write mask.png for {h5_path.name}: {e}")
        else:
            if verbose:
                print(f"No mask file found for {h5_path.name}")
    except Exception as e:
        if verbose:
            print(f"Failed during MASK stage for {h5_path}: {e}")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="Landslide4Sense-2022", help="Root folder containing TestData/ValidData")
    ap.add_argument("--split", type=str, default="TestData", choices=["TestData", "ValidData", "Both"], help="Which split to convert")
    ap.add_argument("--out", type=str, default="landslide4sense_tif", help="Output root directory")
    ap.add_argument("--ext", type=str, default=".tif", help="Output extension for files (e.g., .tif or .dif)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)
    splits: List[str]
    if args.split == "Both":
        splits = ["TestData", "ValidData"]
    else:
        splits = [args.split]

    converted: List[Dict[str, Optional[Path]]] = []
    for sp in splits:
        img_dir = root / sp / "img"
        if not img_dir.exists():
            if args.verbose:
                print(f"Skip split {sp}, no folder: {img_dir}")
            continue
        for h5_path in sorted(img_dir.glob("*.h5")):
            relname = h5_path.stem  # e.g., image_1
            out_dir = out_root / sp / relname
            info = convert_one(h5_path, out_dir, ext=args.ext, verbose=args.verbose)
            converted.append(info)
            if args.verbose:
                print(f"Converted {h5_path} -> {out_dir}")

    if args.verbose:
        bands = sum(1 for c in converted if c.get("band"))
        slopes = sum(1 for c in converted if c.get("slope"))
        dems = sum(1 for c in converted if c.get("dem"))
        print({"band": bands, "slope": slopes, "dem": dems})


if __name__ == "__main__":
    main()
