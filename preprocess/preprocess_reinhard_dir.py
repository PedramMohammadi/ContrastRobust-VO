#!/usr/bin/env python3
import argparse, os, sys
import numpy as np, cv2

def to_uint8(img):
    if img is None: return None
    if img.dtype == np.uint8: return img
    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        return cv2.convertScaleAbs(img, alpha=255.0/float(info.max))
    return (np.clip(img,0.0,1.0)*255.0).astype(np.uint8)

def main():
    ap = argparse.ArgumentParser(description="Reinhard global TMO: directory→directory (keeps filenames)")
    ap.add_argument("--in_dir",  required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--gamma", type=float, default=1.2)
    ap.add_argument("--intensity", type=float, default=0.0)
    ap.add_argument("--light_adapt", type=float, default=0.8)
    ap.add_argument("--color_adapt", type=float, default=0.2)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(args.in_dir)
                    if f.lower().endswith((".png",".jpg",".jpeg",".bmp",".tif",".tiff"))])
    if not files:
        print("No images in", args.in_dir, file=sys.stderr)

    tonemap = cv2.createTonemapReinhard(gamma=float(args.gamma),
                                        intensity=float(args.intensity),
                                        light_adapt=float(args.light_adapt),
                                        color_adapt=float(args.color_adapt))

    for i,name in enumerate(files,1):
        src = os.path.join(args.in_dir, name)
        img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
        if img is None:
            print("cannot read:", src, file=sys.stderr); continue
        img8 = to_uint8(img)
        # work in BGR float32 0..1 (Reinhard expects 3 channels)
        if img8.ndim == 2: bgr = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
        else:              bgr = img8
        f = bgr.astype(np.float32)/255.0
        out = tonemap.process(f)     # float32 0..1
        out8 = (np.clip(out,0.0,1.0)*255.0).astype(np.uint8)
        # for ORB-SLAM3, grayscale is fine (and closer to baseline)
        gray = cv2.cvtColor(out8, cv2.COLOR_BGR2GRAY)
        # safeguard: if too dark, fallback to CLAHE
        if float(gray.mean()) < 2.0:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            gray = clahe.apply(to_uint8(img if img.ndim==2 else cv2.cvtColor(img8,cv2.COLOR_BGR2GRAY)))
        ok = cv2.imwrite(os.path.join(args.out_dir, name), gray)
        if not ok:
            print("failed to write:", name, file=sys.stderr)
        if i % 200 == 0:
            print(f"[{i}/{len(files)}] {name}")
    print(f"Wrote {len(files)} images → {args.out_dir}")
if __name__ == "__main__":
    main()
