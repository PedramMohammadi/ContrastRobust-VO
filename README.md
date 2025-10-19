# ContrastRobust-VO

**Goal.** Improve Visual Odometry (VO) robustness under extreme contrast by inserting an HDR pre-processing stage, demonstrating lower drift and fewer tracking failures on public datasets (EuRoC, TUM).

**What’s in this repo**
- One-command scripts to reproduce VO runs with **ORB-SLAM3 (mono)**.
- Pre-processing operators: **Proposed method**, **CLAHE**, **Reinhard**, **Exposure Fusion**, and **No-preprocessing baseline**.
- Results as CSVs + plots

## Highlights — does preprocessing help?

Across EuRoC & TUM, the proposed method improves both long-term drift (**APE**) and short-term consistency (**RPE**), while classical baselines (Mertens/CLAHE) are competitive on some metrics.

| Dataset (scope) | Metric | Median Δ vs Baseline | Win-rate |
|---|---|---:|---:|
| **All (pooled)** | **APE** | **−8.8%** | 59% |
| **All (pooled)** | **RPE (Δ=5)** | **−6.2%** | 50% |
| **TUM** | **APE** | **−10.6%** | **77%** |
| **TUM** | **RPE (Δ=5)** | **−12.3%** | 50% |
| **EuRoC** | **APE** | **−8.8%** | 59% |
| **EuRoC** | **RPE (Δ=5)** | **−6.7%** | 45% |

*Also notable:* **Exposure Fusion (Mertens)** is a strong classic baseline for **APE**, while **CLAHE** often leads on **RPE** (short-term pose error).

## Method overview (pre-processing operators)

Each method outputs an SDR image with **preserved filenames & timestamps** so VO sees consistent timing.

- **Baseline (No pre-processing)** — identity pass-through  

- **Contrast Limited Adaptive Histogram Equalization (CLAHE)** — local histogram equalization with clip limit to protect highlights; good at boosting local contrast while avoiding over-amplification of noise.  

- **Reinhard Global Tone Mapping** — classic photographic tone curve mapping HDR→SDR; here used in SDR space as a gentle global compression/expansion.  

- **Exposure Fusion** — multi-scale fusion (contrast/saturation/well-exposedness weights); robust detail recovery without explicit HDR radiance maps.  

- **Proposed method** — Converts SDR to **HDR (PQ / BT.2020)** via **iTMO**, then converts the HDR back to SDR using a **piecewise curve** tuned for highlight/near-black detail; aims to improve keypoints & matches in extreme contrast.  

> Implementation notes: all operators preserve file names and timestamps; directory helpers are provided for batch conversion.

## Datasets & VO engine

- **Datasets:** EuRoC MAV (mono; stereo optional later), TUM RGB-D (mono RGB)
- **VO Engine:** ORB-SLAM3 (mono); OpenVSLAM optionally as fallback
- **Evaluator:** `evo` APE (Sim3 translation) and RPE with Δ=5 frames

## Reproducibility & fair comparison

- Same ORB-SLAM3 build and YAMLs across variants

- Preprocessors keep filenames/timestamps; no changes to sequence timing

- Evaluator settings fixed: APE (Sim3 translation) and RPE Δ=5 

- Seeds/configs logged alongside results

## Roadmap / future directions

- Parse VO logs for time-to-failure, # relocalizations, inliers/frame

- Adaptive iTMO/TMO (curve & gamut tuned per sequence/frame)

- Broaden VO backends (VINS-Mono, DSO/SVO, OpenVSLAM) and stereo/VIO

- Add harder datasets (night/low-light, motion blur) and real robot runs

- Real-time path (GPU kernels, end-to-end latency budget)

- Lightweight learned enhancer that mimics iTMO but is differentiable and adaptive

## Repository layout
```text
ContrastRobust-VO/
├─ README.md
├─ LICENSE
├─ scripts/
│  ├─ run_euroc_all.sh
│  └─ run_tum_all.sh
├─ preprocess/
│  ├─ proposed_itmo.py
│  ├─ proposed_tmo.py
│  ├─ sdr_hdr_sdr_pipeline.py
│  ├─ preprocess_clahe.py
│  ├─ preprocess_mertens.py
│  ├─ preprocess_reinhard.py
│  └─ preprocess_reinhard_dir.py
├─ results/
│  ├─ Results.md
│  ├─ euroc_all_ape_table.csv
│  ├─ euroc_all_rpe5_table.csv
│  ├─ euroc_summary_tidy.csv
│  ├─ tum_all_ape_table.csv
│  ├─ tum_all_rpe5_table.csv
```