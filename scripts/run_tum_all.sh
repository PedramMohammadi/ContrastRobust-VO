#!/usr/bin/env bash
# Batch TUM RGB-D runs: baseline, clahe, reinhard, mertens
# - Produces KeyFrameTrajectory.txt for each run
# - Evaluates APE (Sim3, meters, translation-only) and RPE (delta=5 frames, all pairs)
# - Writes per-sequence CSV tables and a combined summary

set -u  # don't use -e (Pangolin/WSL may exit non-zero after saving trajectories)
set -o pipefail

ROOT="$HOME/hdr-aware-vo"
DATA="$ROOT/data_win/TUM_RGBD"
ORB="$HOME/vo_ws/ORB_SLAM3"
PY="$ROOT/src/hdr_preproc"
RES="$ROOT/results/trajectories"

mkdir -p "$RES"
: "${LD_LIBRARY_PATH:=}"
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export PATH="$HOME/.local/bin:$PATH"

# Sequences I downloaded
SEQS=(
  freiburg1_desk
  freiburg1_room
  freiburg1_xyz
  freiburg2_desk
  freiburg2_desk_with_person
  freiburg2_large_with_loop
  freiburg3_long_office_household
  freiburg3_nostructure_notexture_near_withloop
  freiburg3_structure_texture_near
  freiburg3_walking_xyz
)

yaml_for_seq() {
  local s="$1"
  case "$s" in
    freiburg1_*) echo "$ORB/Examples/Monocular/TUM1.yaml" ;;
    freiburg2_*) echo "$ORB/Examples/Monocular/TUM2.yaml" ;;
    freiburg3_*) echo "$ORB/Examples/Monocular/TUM3.yaml" ;;
    *)           echo "$ORB/Examples/Monocular/TUM1.yaml" ;;
  esac
}

run_orb_tum() {
  local seq_dir="$1" yaml="$2"
  pushd "$ORB" >/dev/null
  ./Examples/Monocular/mono_tum Vocabulary/ORBvoc.txt "$yaml" "$seq_dir" || true
  popd >/dev/null
}

# -------- batch over sequences --------
for s in "${SEQS[@]}"; do
  echo "================  $s  ================"
  SEQDIR="$DATA/$s"
  GT="$SEQDIR/groundtruth.txt"
  YAML="$(yaml_for_seq "$s")"

  # Preprocessing outputs (keep TUM structure)
  OUT_CLAHE="$DATA/${s}_clahe"
  OUT_REINH="$DATA/${s}_reinhard"
  OUT_MERT="$DATA/${s}_mertens"
  OUT_PROPOSED="$DATA/${s}_proposed"

  # paths to proposed algorithm's scripts
  PROPOSED_PIPE="$ROOT/src/sdr_hdr_sdr_pipeline.py"
  PROPOSED_ITMO="$ROOT/src/proposed_itmo.py"
  PROPOSED_TMO="$ROOT/src/proposed_tmo.py"

  [[ -f "$PROPOSED_PIPE" && -f "$PROPOSED_ITMO" && -f "$PROPOSED_TMO" ]] || {
  echo "[ERR] proposed scripts missing under $ROOT/src/"; exit 1; }

  # Ensure rgb.txt exists (harmless if it already does)
  if [ ! -f "$SEQDIR/rgb.txt" ]; then
    echo "# timestamp filename" > "$SEQDIR/rgb.txt"
    for f in $(ls "$SEQDIR"/rgb/*.png 2>/dev/null | sort); do
      base=$(basename "$f" .png); echo "$base rgb/$(basename "$f")" >> "$SEQDIR/rgb.txt"
    done
  fi

  # ---- Proposed (SDR->HDR->SDR) ----
  echo "[Proposed] $s"
  # make sure the proposed tree exists first
  mkdir -p "$OUT_PROPOSED/rgb"
  # give ORB-SLAM3 its index file in the proposed root
  cp -f "$SEQDIR/rgb.txt" "$OUT_PROPOSED/rgb.txt"
  # keep filenames (timestamps align with GT)
  python3 "$PROPOSED_PIPE" \
    --input-dir  "$SEQDIR/rgb" \
    --output-dir "$OUT_PROPOSED/rgb" \
    --bit-depth 8 --wB 0 --wC 1 --recursive --overwrite \
    --itmo-path  "$PROPOSED_ITMO" \
    --tmo-path   "$PROPOSED_TMO"
  run_orb_tum "$OUT_PROPOSED" "$YAML"
  cp "$ORB/KeyFrameTrajectory.txt" "$RES/${s}_proposed_C_keyframes.txt" 2>/dev/null || true

  # ---- Baseline ----
  echo "[Baseline] $s"
  run_orb_tum "$SEQDIR" "$YAML"
  cp "$ORB/KeyFrameTrajectory.txt" "$RES/${s}_baseline_keyframes.txt" 2>/dev/null || true

  # ---- CLAHE ----
  echo "[CLAHE] $s"
  python3 "$PY/preprocess_clahe.py" --in_seq "$SEQDIR" --out_seq "$OUT_CLAHE" --clipLimit 3.0 --tileGridSize 8
  run_orb_tum "$OUT_CLAHE" "$YAML"
  cp "$ORB/KeyFrameTrajectory.txt" "$RES/${s}_clahe_keyframes.txt" 2>/dev/null || true

  # ---- Reinhard ----
  echo "[Reinhard] $s"
  python3 "$PY/preprocess_reinhard.py" --in_seq "$SEQDIR" --out_seq "$OUT_REINH" --gamma 1.2 --intensity 0.0 --light_adapt 0.8 --color_adapt 0.2
  run_orb_tum "$OUT_REINH" "$YAML"
  cp "$ORB/KeyFrameTrajectory.txt" "$RES/${s}_reinhard_keyframes.txt" 2>/dev/null || true

  # ---- Mertens (exposure fusion, with safe defaults & fallback inside the script) ----
  echo "[Mertens] $s"
  python3 "$PY/preprocess_mertens.py" \
    --in_seq "$SEQDIR" --out_seq "$OUT_MERT" \
    --contrast_weight 1.0 --sat_weight 1.0 --exp_weight 0.75 \
    --gains 0.5,1.0,2.0 --gammas 1.1,1.0,0.9
  run_orb_tum "$OUT_MERT" "$YAML"
  cp "$ORB/KeyFrameTrajectory.txt" "$RES/${s}_mertens_keyframes.txt" 2>/dev/null || true

  # ---- Evaluation (only if ground truth is present) ----
  if [ -f "$GT" ]; then
    echo "[Eval] $s  (GT found)"
    # APE: Sim(3) alignment, translation part (meters)
    for m in proposed baseline clahe reinhard mertens; do
      evo_ape tum "$GT" "$RES/${s}_${m}_keyframes.txt" \
        -a --correct_scale -r trans_part \
        --save_results "$RES/${s}_${m}_ape.zip" || true
    done
    evo_res "$RES/${s}_baseline_ape.zip" "$RES/${s}_clahe_ape.zip" "$RES/${s}_reinhard_ape.zip" "$RES/${s}_mertens_ape.zip" "$RES/${s}_proposed_BC_ape.zip" "$RES/${s}_proposed_B_ape.zip" "$RES/${s}_proposed_C_ape.zip" \
      --use_filenames --save_table "$RES/${s}_ape_table.csv" || true

    # RPE: translation part, delta=5 frames, all pairs (meters)
    for m in proposed baseline clahe reinhard mertens; do
      evo_rpe tum "$GT" "$RES/${s}_${m}_keyframes.txt" \
        -a -r trans_part --delta 5 -u f --all_pairs \
        --save_results "$RES/${s}_${m}_rpe5.zip" || true
    done
    evo_res "$RES/${s}_baseline_rpe5.zip" "$RES/${s}_clahe_rpe5.zip" "$RES/${s}_reinhard_rpe5.zip" "$RES/${s}_mertens_rpe5.zip" "$RES/${s}_proposed_BC_rpe5.zip" "$RES/${s}_proposed_B_rpe5.zip" "$RES/${s}_proposed_C_rpe5.zip" \
      --use_filenames --save_table "$RES/${s}_rpe5_table.csv" || true
  else
    echo "[WARN] $s has no groundtruth.txt — skipping evaluation, trajectories only."
  fi
done

# Combined TUM summary tables (only zips that exist)
echo "[Summary] building combined CSVs…"
APE_ZIPS=$(ls "$RES"/freiburg*_ape.zip 2>/dev/null | xargs -r echo)
RPE_ZIPS=$(ls "$RES"/freiburg*_rpe5.zip 2>/dev/null | xargs -r echo)
[ -n "$APE_ZIPS" ] && evo_res $APE_ZIPS --use_filenames --save_table "$RES/tum_all_ape_table.csv" || true
[ -n "$RPE_ZIPS" ] && evo_res $RPE_ZIPS --use_filenames --save_table "$RES/tum_all_rpe5_table.csv" || true

echo "[DONE] Results in: $RES"
