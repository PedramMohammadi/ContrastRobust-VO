#!/usr/bin/env python3
import argparse, os, cv2, numpy as np

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def synthesize_exposures(bgr, gains=(0.5, 1.0, 2.0), gammas=(1.1, 1.0, 0.9)):
    """
    Create 3 exposures from a single LDR image using linear gain + gentle gamma.
    Returns a list of float32 BGR images in [0,1].
    """
    f = bgr.astype(np.float32) / 255.0
    outs = []
    for g, gm in zip(gains, gammas):
        x = np.clip(f * float(g), 0.0, 1.0)
        # apply gamma after gain to adjust mid-tones
        x = np.power(x, float(gm), dtype=np.float32)
        outs.append(np.ascontiguousarray(x))
    return outs

def fuse_mertens(bgr, contrast_w=1.0, sat_w=1.0, exp_w=0.75,
                 gains=(0.5,1.0,2.0), gammas=(1.1,1.0,0.9)):
    exposures = synthesize_exposures(bgr, gains=gains, gammas=gammas)
    mertens = cv2.createMergeMertens(float(contrast_w), float(sat_w), float(exp_w))
    out = mertens.process(exposures)  # float32 [0,1]
    out = np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_seq", required=True, help="Input TUM-style folder (rgb/ + rgb.txt)")
    ap.add_argument("--out_seq", required=True, help="Output folder (rgb/ + rgb.txt)")
    ap.add_argument("--contrast_weight", type=float, default=1.0)
    ap.add_argument("--sat_weight", type=float, default=1.0)
    ap.add_argument("--exp_weight", type=float, default=0.75)
    ap.add_argument("--gains", type=str, default="0.5,1.0,2.0")   # comma-separated
    ap.add_argument("--gammas", type=str, default="1.1,1.0,0.9")  # comma-separated
    ap.add_argument("--max_frames", type=int, default=0, help="Process first N frames (0=all)")
    ap.add_argument("--debug_every", type=int, default=0, help="Save every Nth fused frame to _debug/")
    args = ap.parse_args()

    gains = tuple(float(v) for v in args.gains.split(","))
    gammas = tuple(float(v) for v in args.gammas.split(","))
    if len(gains) != 3 or len(gammas) != 3:
        raise SystemExit("Expect exactly three gains and three gammas (comma-separated).")

    in_rgb = os.path.join(args.in_seq, "rgb")
    in_list = os.path.join(args.in_seq, "rgb.txt")
    if not os.path.isdir(in_rgb) or not os.path.isfile(in_list):
        raise SystemExit(f"Expected {args.in_seq}/rgb and {args.in_seq}/rgb.txt")

    out_rgb = os.path.join(args.out_seq, "rgb")
    ensure_dir(out_rgb)
    out_list = os.path.join(args.out_seq, "rgb.txt")

    # optional debug dir
    dbg_dir = os.path.join(args.out_seq, "_debug")
    if args.debug_every > 0:
        ensure_dir(dbg_dir)

    processed = 0
    with open(in_list, "r") as fin, open(out_list, "w") as fout:
        for line in fin:
            if not line.strip() or line.startswith("#"):
                if line.startswith("#"):
                    fout.write(line)
                continue
            ts, rel = line.strip().split()
            src_path = os.path.join(args.in_seq, rel)
            bgr = cv2.imread(src_path, cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"failed to read {src_path}")
                continue

            fused = fuse_mertens(
                bgr,
                contrast_w=args.contrast_weight,
                sat_w=args.sat_weight,
                exp_w=args.exp_weight,
                gains=gains, gammas=gammas
            )

            # defensive: if near-black, retry with higher exposure weight; if still bad, fall back to CLAHE
            if fused.mean() < 2.0:  # ~dark
                fused_retry = fuse_mertens(
                    bgr,
                    contrast_w=args.contrast_weight,
                    sat_w=args.sat_weight,
                    exp_w=max(args.exp_weight, 1.0),
                    gains=gains, gammas=gammas
                )
                if fused_retry.mean() > fused.mean():
                    fused = fused_retry
            if fused.mean() < 2.0:
                # CLAHE fallback on L channel
                lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
                l, a, c = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                l2 = clahe.apply(l)
                fused = cv2.cvtColor(cv2.merge([l2, a, c]), cv2.COLOR_LAB2BGR)

            out_rel = rel
            out_path = os.path.join(args.out_seq, out_rel)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, fused, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            fout.write(f"{ts} {out_rel}\n")

            processed += 1
            if args.debug_every > 0 and processed % args.debug_every == 0:
                cv2.imwrite(os.path.join(dbg_dir, f"frame_{processed:06d}.png"), fused)

            if args.max_frames and processed >= args.max_frames:
                break

    print(f"Processed {processed} frames. Example mean intensity: {fused.mean():.2f}")
if __name__ == "__main__":
    main()

