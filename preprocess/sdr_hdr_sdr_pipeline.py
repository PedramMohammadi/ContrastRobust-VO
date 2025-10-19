import argparse
import sys
from pathlib import Path
import numpy as np

# ---------- IO helpers ----------
def _read_image(path: str):
    try:
        import imageio.v3 as iio
        return iio.imread(path)
    except Exception:
        from PIL import Image
        return np.array(Image.open(path).convert("RGB"))

def _write_image(path: str, array_uint8: np.ndarray):
    outp = Path(path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v3 as iio
        iio.imwrite(str(outp), array_uint8)
    except Exception:
        from PIL import Image
        Image.fromarray(array_uint8).save(str(outp))

def _ensure_uint(img: np.ndarray, bit_depth: int) -> np.ndarray:
    """Ensure integer code values for ITMO. If float, scale to 8-bit by default."""
    if np.issubdtype(img.dtype, np.integer):
        return img
    f = img.astype(np.float64)
    # assume [0,1] if <=1, otherwise assume already [0,255]
    if f.max() <= 1.0:
        f = np.clip(f, 0.0, 1.0) * 255.0
    return np.round(f).astype(np.uint16 if bit_depth > 8 else np.uint8)

# ---------- Core processing ----------
def _load_modules(itmo_path: str | None, tmo_path: str | None):
    if itmo_path:
        itmo_dir = str(Path(itmo_path).resolve().parent)
        if itmo_dir not in sys.path: sys.path.insert(0, itmo_dir)
    if tmo_path:
        tmo_dir = str(Path(tmo_path).resolve().parent)
        if tmo_dir not in sys.path: sys.path.insert(0, tmo_dir)
    try:
        from proposed_itmo import proposed_itmo
    except Exception as e:
        raise ImportError("Could not import proposed_itmo. Use --itmo-path to point to proposed_itmo.py") from e
    try:
        from proposed_tmo import proposed_tmo
    except Exception as e:
        raise ImportError("Could not import proposed_tmo. Use --tmo-path to point to proposed_tmo.py") from e
    return proposed_itmo, proposed_tmo

def process_one_file(in_path: Path, out_path: Path, *,
                     bit_depth: int, wB: float, wC: float,
                     mapping_curve: str, save_hdr_npy_dir: Path | None,
                     itmo, tmo):
    sdr = _read_image(str(in_path))
    sdr = _ensure_uint(sdr, bit_depth)

    # after reading 'img' as a numpy array
    if sdr.ndim == 2:                 # grayscale → RGB
        sdr = np.repeat(sdr[..., None], 3, axis=2)


    # SDR -> HDR (PQ/BT.2020) via ITMO
    hdr_pq_2020 = itmo(sdr, input_bit_depth=bit_depth, wB=wB, wC=wC)

    # Optionally save intermediate HDR as .npy
    if save_hdr_npy_dir is not None:
        save_hdr_npy_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_hdr_npy_dir / (in_path.stem + "_hdr_pq2020.npy"), hdr_pq_2020)

    # HDR PQ -> SDR (BT.709, gamma-encoded) via TMO
    sdr_out = tmo(hdr_pq_2020, MappingCurve=mapping_curve)

    # Save as 8-bit
    sdr_u8 = np.clip(np.round(sdr_out * 255.0), 0, 255).astype(np.uint8)
    _write_image(str(out_path), sdr_u8)

def main():
    p = argparse.ArgumentParser(description="SDR -> HDR (ProposedITMO) -> SDR (ProposedTMO) pipeline")

    p.add_argument("--input-dir", help="Directory containing SDR frames")
    p.add_argument("--output-dir", help="Directory to write SDR outputs")
    p.add_argument("--glob", default="*.png", help="Glob for input-dir (default: *.png)")
    p.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    p.add_argument("--output-ext", default=".png", help="Output extension for directory mode (default: .png)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

    # Processing params
    p.add_argument("--bit-depth", type=int, default=8, choices=[8,10], help="SDR input bit depth for ITMO")
    p.add_argument("--wB", type=float, default=0.5, help="ITMO brightness weight (default 0.0)")
    p.add_argument("--wC", type=float, default=0.5, help="ITMO contrast weight (default 1.0)")
    p.add_argument("--mapping-curve", default="PiecewiseLinear", choices=["Gamma","PiecewiseLinear"], help="TMO mapping curve")
    p.add_argument("--save-hdr-npy-dir", default=None, help="Optional dir to save intermediate HDR PQ/BT.2020 as .npy")

    # Module paths
    p.add_argument("--itmo-path", default=None, help="Path to proposed_itmo.py if not importable")
    p.add_argument("--tmo-path", default=None, help="Path to proposed_tmo.py if not importable")
    args = p.parse_args()

    itmo, tmo = _load_modules(args.itmo_path, args.tmo_path)

    # Optional HDR save dir
    save_hdr_dir = Path(args.save_hdr_npy_dir) if args.save_hdr_npy_dir else None

    # Directory mode
    in_dir  = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = f"**/{args.glob}" if args.recursive else args.glob
    files = sorted(in_dir.glob(pattern))

    if not files:
        print(f"[warn] No files matched {pattern} in {in_dir}")
        return

    for ip in files:
        rel = ip.relative_to(in_dir) if args.recursive else Path(ip.name)
        out_rel = rel.with_suffix(args.output_ext)
        op = out_dir / out_rel
        if op.exists() and not args.overwrite:
            print(f"[skip] {rel} -> {out_rel} (exists)")
            continue
        try:
            process_one_file(ip, op,
                             bit_depth=args.bit_depth, wB=args.wB, wC=args.wC,
                             mapping_curve=args.mapping_curve,
                             save_hdr_npy_dir=save_hdr_dir,
                             itmo=itmo, tmo=tmo)
            print(f"[ok] {rel} -> {out_rel}")
        except Exception as e:
            print(f"[err] {rel}: {e}")

if __name__ == "__main__":
    main()
