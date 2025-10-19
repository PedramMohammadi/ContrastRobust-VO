#!/usr/bin/env python3
import argparse, os, cv2, numpy as np

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def process_img(bgr, gamma=1.0, intensity=0.0, light_adapt=0.8, color_adapt=0.2):
    # OpenCV's TonemapReinhard expects float32 in [0,1]
    f = bgr.astype(np.float32) / 255.0
    tmo = cv2.createTonemapReinhard(gamma=float(gamma),
                                    intensity=float(intensity),
                                    light_adapt=float(light_adapt),
                                    color_adapt=float(color_adapt))
    out = tmo.process(f)          # float32 [0,1]
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_seq", required=True, help="Input TUM-style folder (rgb/ + rgb.txt)")
    ap.add_argument("--out_seq", required=True, help="Output folder (rgb/ + rgb.txt)")
    ap.add_argument("--gamma", type=float, default=1.2)
    ap.add_argument("--intensity", type=float, default=0.0)
    ap.add_argument("--light_adapt", type=float, default=0.8)
    ap.add_argument("--color_adapt", type=float, default=0.2)
    args = ap.parse_args()

    in_rgb = os.path.join(args.in_seq, "rgb")
    in_list = os.path.join(args.in_seq, "rgb.txt")
    if not os.path.isdir(in_rgb) or not os.path.isfile(in_list):
        raise SystemExit(f"Expected {args.in_seq}/rgb and {args.in_seq}/rgb.txt")

    out_rgb = os.path.join(args.out_seq, "rgb")
    ensure_dir(out_rgb)
    out_list = os.path.join(args.out_seq, "rgb.txt")

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
            bgrp = process_img(bgr, args.gamma, args.intensity, args.light_adapt, args.color_adapt)
            out_rel = rel
            out_path = os.path.join(args.out_seq, out_rel)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, bgrp, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            fout.write(f"{ts} {out_rel}\n")

if __name__ == "__main__":
    main()
