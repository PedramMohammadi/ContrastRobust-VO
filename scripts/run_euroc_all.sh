#!/usr/bin/env bash
set -euo pipefail

# ---------- CONFIG ----------
ROOT="${ROOT:-$HOME/hdr-aware-vo}"
EUROC_ROOT="${EUROC_ROOT:-$ROOT/data_win/EuRoC}"
ORB="${ORB:-$HOME/vo_ws/ORB_SLAM3}"
RES="${RES:-$ROOT/results/trajectories}"

# Space-separated list. Leave empty to auto-detect sequences present under EUROC_ROOT.
EUROC_SEQS="${EUROC_SEQS:-}"

# Evaluation params
TDIFF="${TDIFF:-0.02}"          # timestamp association slack [s]
RPE_DELTA_FRAMES="${RPE_DELTA_FRAMES:-5}"

# ---------- ENV ----------
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"

# [Patch] Run headless to avoid GUI/viewer thread joins blocking shutdowns in batch runs.
unset DISPLAY

# ---------- CHECKS ----------
need() { command -v "$1" >/dev/null 2>&1 || { echo "[MISS] $1 not found"; exit 1; }; }
need python3; need awk
need "$ORB/Examples/Monocular/mono_euroc" || { echo "[MISS] ORB-SLAM3 mono_euroc"; exit 1; }
need evo_ape; need evo_rpe; need evo_res

# CLAHE and Reinhard algorithms
CLAHE_PY="$ROOT/src/hdr_preproc/preprocess_dir.py"
REIN_PY="$ROOT/src/hdr_preproc/preprocess_reinhard_dir.py"

[[ -f "$CLAHE_PY" ]] || { echo "[MISS] $CLAHE_PY"; exit 1; }
[[ -f "$REIN_PY"  ]] || { echo "[MISS] $REIN_PY";  exit 1; }

# Proposed algorithm directories
PROPOSED_PIPE="$ROOT/src/sdr_hdr_sdr_pipeline.py"
PROPOSED_ITMO="$ROOT/src/proposed_itmo.py"
PROPOSED_TMO="$ROOT/src/proposed_tmo.py"

[[ -f "$PROPOSED_PIPE" && -f "$PROPOSED_ITMO" && -f "$PROPOSED_TMO" ]] || {
  echo "[MISS] proposed SDR→HDR→SDR scripts under $ROOT/src/"; exit 1; }

mkdir -p "$RES"

# ---------- HELPERS ----------
mk_times() { # $1 = SEQ_ROOT (e.g., .../EuRoC/MH_01_easy)
  local seq="$1"
  local csv="$seq/mav0/cam0/data.csv"
  local times="$seq/mav0/cam0_times.txt"
  if [[ ! -f "$csv" ]]; then echo "[SKIP] no cam0/data.csv in $seq"; return; fi
  # Write integer nanoseconds from the first CSV column
  tail -n +2 "$csv" | cut -d, -f1 > "$times"
}

preprocess_variant() { # $1=variant $2=SEQ_ROOT -> sets RUN_ROOT
  local variant="$1"; local seq="$2"
  case "$variant" in
    proposed)
      RUN_ROOT="${seq}_proposed"
      mkdir -p "$RUN_ROOT/mav0/cam0/data"
      cp -n "$seq/mav0/cam0/data.csv"  "$RUN_ROOT/mav0/cam0/data.csv"
      cp -n "$seq/mav0/cam0_times.txt" "$RUN_ROOT/mav0/cam0_times.txt"

      python3 "$PROPOSED_PIPE" \
        --input-dir  "$seq/mav0/cam0/data" \
        --output-dir "$RUN_ROOT/mav0/cam0/data" \
        --bit-depth 8 --wB 0 --wC 1 --recursive --overwrite \
        --itmo-path  "$PROPOSED_ITMO" \
        --tmo-path   "$PROPOSED_TMO"
      ;;
    baseline)
      RUN_ROOT="$seq"
      ;;
    clahe)
      RUN_ROOT="${seq}_clahe"
      mkdir -p "$RUN_ROOT/mav0/cam0/data"
      cp -n "$seq/mav0/cam0/data.csv"  "$RUN_ROOT/mav0/cam0/data.csv"
      cp -n "$seq/mav0/cam0_times.txt" "$RUN_ROOT/mav0/cam0_times.txt"
      python3 "$CLAHE_PY" --in_dir "$seq/mav0/cam0/data" --out_dir "$RUN_ROOT/mav0/cam0/data" \
        --method clahe --clipLimit 3.0 --tileGridSize 8
      ;;
    mertens)
      RUN_ROOT="${seq}_mertens"
      mkdir -p "$RUN_ROOT/mav0/cam0/data"
      cp -n "$seq/mav0/cam0/data.csv"  "$RUN_ROOT/mav0/cam0/data.csv"
      cp -n "$seq/mav0/cam0_times.txt" "$RUN_ROOT/mav0/cam0_times.txt"
      python3 "$CLAHE_PY" --in_dir "$seq/mav0/cam0/data" --out_dir "$RUN_ROOT/mav0/cam0/data" \
        --method mertens --contrast_weight 1.0 --sat_weight 1.0 --exp_weight 0.75 \
        --gains 0.5,1.0,2.0 --gammas 1.1,1.0,0.9
      ;;
    reinhard)
      RUN_ROOT="${seq}_reinhard"
      mkdir -p "$RUN_ROOT/mav0/cam0/data"
      cp -n "$seq/mav0/cam0/data.csv"  "$RUN_ROOT/mav0/cam0/data.csv"
      cp -n "$seq/mav0/cam0_times.txt" "$RUN_ROOT/mav0/cam0_times.txt"
      python3 "$REIN_PY" --in_dir "$seq/mav0/cam0/data" --out_dir "$RUN_ROOT/mav0/cam0/data" \
        --gamma 1.2 --intensity 0.0 --light_adapt 0.8 --color_adapt 0.2
      ;;
  esac
}

norm_to_seconds() { # $1=infile -> writes *_sec.txt (TUM 8 cols)
  python3 - "$1" <<'PY'
import sys, pathlib
src = pathlib.Path(sys.argv[1])
dst = src.with_name(src.stem + "_sec.txt")
out = []
for ln in src.read_text().splitlines():
    if not ln or ln.startswith("#"): continue
    p = ln.split()
    if len(p) < 8: continue
    t = float(p[0])
    if t > 1e12: t /= 1e9
    elif t > 1e6: t /= 1e6
    p[0] = f"{t:.9f}"
    out.append(" ".join(p[:8]))
dst.write_text("\n".join(out) + "\n")
print(f"[OK] {src.name} -> {dst.name} ({len(out)} rows)")
PY
}

make_gt_tum() { # $1=SEQ_ROOT -> writes $RES/euroc_${SEQNAME}_gt.tum
  local seq="$1"
  local name; name="$(basename "$seq")"
  local gt_csv=""
  if [[ -f "$seq/mav0/state_groundtruth_estimate0/data.csv" ]]; then
    gt_csv="$seq/mav0/state_groundtruth_estimate0/data.csv"
  elif [[ -f "$seq/mav0/vicon0/data.csv" ]]; then
    gt_csv="$seq/mav0/vicon0/data.csv"
  else
    echo "[WARN] No GT CSV for $name"; return
  fi
  python3 - "$gt_csv" "$RES/euroc_${name}_gt.tum" <<'PY'
import sys, csv, re, pathlib
src, dst = sys.argv[1], sys.argv[2]
with open(src, newline='') as f, open(dst, "w") as g:
    r = csv.reader(f); H0 = next(r)
    H = [re.sub(r'\s|\[.*?\]', '', h.lower()) for h in H0]
    def idx(sub):
        for i,h in enumerate(H):
            if sub in h: return i
        raise KeyError(sub)
    t  = next(i for i,h in enumerate(H) if 'timestamp' in h)
    px,py,pz = idx('p_rs_r_x'), idx('p_rs_r_y'), idx('p_rs_r_z')
    qw,qx,qy,qz = idx('q_rs_w'), idx('q_rs_x'), idx('q_rs_y'), idx('q_rs_z')
    n=0
    for row in r:
        if not row or not row[t]: continue
        ts = int(row[t]) / 1e9
        print(f"{ts:.9f} {row[px]} {row[py]} {row[pz]} {row[qx]} {row[qy]} {row[qz]} {row[qw]}", file=g)
        n+=1
print(f"[OK] Wrote {dst} ({n} poses)")
PY
}

eval_seq() { # $1=SEQNAME
  local name="$1"
  local GT="$RES/euroc_${name}_gt.tum"
  [[ -f "$GT" ]] || { echo "[SKIP] no GT for $name"; return; }

  # APE (Sim3, m)
  for v in proposed baseline clahe mertens reinhard; do
    local est="$RES/euroc_${name}_${v}_keyframes_sec.txt"
    [[ -f "$est" ]] || { echo "[SKIP] missing $est"; continue; }
    evo_ape tum "$GT" "$est" -a --correct_scale -r trans_part --t_max_diff "$TDIFF" \
      --save_results "$RES/euroc_${name}_${v}_ape.zip" >/dev/null
  done

  # [Patch] Pre-delete table to prevent evo_res interactive overwrite prompt.
  rm -f "$RES/euroc_${name}_ape_table.csv"
  evo_res "$RES"/euroc_"${name}"_*_ape.zip --use_filenames --save_table "$RES/euroc_${name}_ape_table.csv" >/dev/null || true

  # RPE (Δ=frames, m)
  for v in proposed baseline clahe mertens reinhard; do
    local est="$RES/euroc_${name}_${v}_keyframes_sec.txt"
    [[ -f "$est" ]] || continue
    evo_rpe tum "$GT" "$est" -a -r trans_part --delta "$RPE_DELTA_FRAMES" -u f --all_pairs --t_max_diff "$TDIFF" \
      --save_results "$RES/euroc_${name}_${v}_rpe5.zip" >/dev/null
  done

  # [Patch] Pre-delete table to prevent evo_res interactive overwrite prompt.
  rm -f "$RES/euroc_${name}_rpe5_table.csv"
  evo_res "$RES"/euroc_"${name}"_*_rpe5.zip --use_filenames --save_table "$RES/euroc_${name}_rpe5_table.csv" >/dev/null || true
}

# ---------- DISCOVER SEQUENCES ----------
if [[ -z "$EUROC_SEQS" ]]; then
  mapfile -t found < <(find -L "$EUROC_ROOT" -maxdepth 1 -mindepth 1 -type d -printf "%f\n" | sort)
  EUROC_SEQS="${found[*]}"
fi

echo "[INFO] Sequences: $EUROC_SEQS"
echo "[INFO] Results -> $RES"
mkdir -p "$RES"

# ---------- MAIN LOOP ----------
for SEQNAME in $EUROC_SEQS; do
  SEQ="$EUROC_ROOT/$SEQNAME"
  [[ -f "$SEQ/mav0/cam0/data.csv" ]] || { echo "[SKIP] $SEQNAME (no cam0/data.csv)"; continue; }

  echo "=== $SEQNAME ==="
  mk_times "$SEQ"

  # Run all variants
  for VAR in proposed baseline clahe mertens reinhard; do
    echo "--- $SEQNAME :: $VAR ---"
    preprocess_variant "$VAR" "$SEQ"
    # Run ORB-SLAM3
    ( cd "$ORB" && ./Examples/Monocular/mono_euroc Vocabulary/ORBvoc.txt Examples/Monocular/EuRoC.yaml "$RUN_ROOT" "$RUN_ROOT/mav0/cam0_times.txt" )
    # Save & normalize
    cp "$ORB/KeyFrameTrajectory.txt" "$RES/euroc_${SEQNAME}_${VAR}_keyframes.txt"
    norm_to_seconds "$RES/euroc_${SEQNAME}_${VAR}_keyframes.txt"
  done

  # GT + eval for this sequence
  make_gt_tum "$SEQ"
  eval_seq "$SEQNAME"

  # Optional Windows mirror
  if [[ -n "${WIN_RESULTS:-}" ]]; then
    rsync -avh --delete "$RES"/ "$WIN_RESULTS"/ >/dev/null
  fi
done

# ---------- GLOBAL SUMMARIES ----------
# [Patch] Pre-delete to avoid evo_res overwrite prompt for global tables.
( cd "$RES" && ls euroc_*_ape.zip  >/dev/null 2>&1 && rm -f euroc_all_ape_table.csv  && evo_res euroc_*_ape.zip  --use_filenames --save_table euroc_all_ape_table.csv  || true )
( cd "$RES" && ls euroc_*_rpe5.zip >/dev/null 2>&1 && rm -f euroc_all_rpe5_table.csv && evo_res euroc_*_rpe5.zip --use_filenames --save_table euroc_all_rpe5_table.csv || true )

# Tidy combined CSV (sequence, variant, metric, rmse, ...)
python3 - <<'PY'
import pandas as pd, glob, os, re, pathlib
RES = os.environ.get("RES", os.path.expanduser("~/hdr-aware-vo/results/trajectories"))
def load(path, metric):
    if not os.path.exists(path): return None
    with open(path, 'r', errors='ignore') as f:
        lines = f.readlines()
    hdr = None
    for i,l in enumerate(lines):
        s=l.strip()
        if s.startswith('rmse,') or s.startswith(',rmse'): hdr=i
    import io
    df = pd.read_csv(io.StringIO(''.join(lines[hdr:]))) if hdr is not None else pd.read_csv(path)
    first = df.columns[0]; df = df.rename(columns={first:'name'})
    df['metric']=metric
    return df
ape = load(os.path.join(RES,'euroc_all_ape_table.csv'), 'APE_Sim3_trans_m')
rpe = load(os.path.join(RES,'euroc_all_rpe5_table.csv'), 'RPE5_trans_m')
dfs=[d for d in [ape,rpe] if d is not None]
if dfs:
    tidy = pd.concat(dfs, ignore_index=True)
    seqs=[]; vars_=[]
    for n in tidy['name'].astype(str):
        b=os.path.basename(n); s=b.replace('.zip',''); parts=s.split('_')
        # euroc_<SEQ>_<variant>_(ape|rpe5)
        if parts[-1] in ('ape','rpe5'):
            vars_.append(parts[-2]); seqs.append('_'.join(parts[1:-2]))
        else:
            vars_.append(parts[-1]); seqs.append('_'.join(parts[1:-1]))
    tidy.insert(0,'sequence',seqs); tidy.insert(1,'variant',vars_)
    keep=['sequence','variant','metric','rmse','mean','median','std','min','max','sse','name']
    for c in keep:
        if c not in tidy.columns: tidy[c]=pd.NA
    tidy[keep].to_csv(os.path.join(RES,'euroc_summary_tidy.csv'), index=False)
    print("[OK] Wrote euroc_summary_tidy.csv")
else:
    print("[WARN] No summary tables found")
PY

echo "[DONE] EuRoC batch complete. Results in: $RES"
