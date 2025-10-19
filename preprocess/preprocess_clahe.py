#!/usr/bin/env python3
import argparse, os, cv2, numpy as np

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def process_img(bgr, clipLimit=3.0, tileGridSize=8):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clipLimit), tileGridSize=(int(tileGridSize), int(tileGridSize)))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_seq", required=True, help="Input TUM-style folder (contains rgb/ and rgb.txt)")
    ap.add_argument("--out_seq", required=True, help="Output folder to write processed rgb/ + rgb.txt")
    ap.add_argument("--clipLimit", type=float, default=3.0)
    ap.add_argument("--tileGridSize", type=int, default=8)
    args = ap.parse_args()

    in_rgb = os.path.join(args.in_seq, "rgb")
    in_list = os.path.join(args.in_seq, "rgb.txt")
    if not os.path.isdir(in_rgb) or not os.path.isfile(in_list):
        raise SystemExit(f"Expected {args.in_seq}/rgb and {args.in_seq}/rgb.txt")

    out_rgb = os.path.join(args.out_seq, "rgb")
    ensure_dir(out_rgb)

    # write header
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
                print(f"[WARN] failed to read {src_path}")
                continue
            bgrp = process_img(bgr, args.clipLimit, args.tileGridSize)

            # keep same filename
            out_rel = rel
            out_path = os.path.join(args.out_seq, out_rel)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, bgrp, [cv2.IMWRITE_PNG_COMPRESSION, 3])

            fout.write(f"{ts} {out_rel}\n")

if __name__ == "__main__":
    main()
