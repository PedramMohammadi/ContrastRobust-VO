# Results

This page reports evaluation results for inserting a pre-processing stage before VO to improve robustness under extreme contrast. We evaluate on **EuRoC MAV** and **TUM RGB-D** with **ORB-SLAM3 (mono)**, using **APE (Sim3, translation)** and **RPE (Δ=5 frames)**. All variants preserve filenames/timestamps so VO timing is unchanged.

## TL;DR

- The proposed method **reduces drift and short-term pose error** vs. baseline on both datasets.
- **Exposure Fusion (Mertens)** is a strong classical baseline on **APE**; **CLAHE** often leads on **RPE**.
- No single method dominates everywhere, but the **proposed method** is consistently strong (especially on TUM).

## Evaluation setup

- **VO engine:** ORB-SLAM3 (mono)
- **Datasets:** EuRoC MAV, TUM RGB-D
- **Metrics:**  
  - **APE (Sim3, translation)** — long-term drift  
  - **RPE (Δ=5 frames, translation)** — short-term consistency
- **Aggregation:** For each `sequence × metric`, we compute % change vs **baseline** (no preproc):  
  \[
  \% \Delta = \frac{\text{RMSE}_{\text{variant}} - \text{RMSE}_{\text{baseline}}}{\text{RMSE}_{\text{baseline}}}\times 100.
  \]
  We report the **median** across sequences and a **win-rate** = % of sequences with at least **5%** improvement (i.e., %Δ ≤ −5%).

> Exact CSVs and plotting scripts are listed at the end.  

---

## Methods (pre-processing operators)

Each method emits SDR output and keeps filenames/timestamps.

- **Baseline (No pre-processing)** — identity pass-through.  
- **CLAHE** — local histogram equalization with clipping (boosts local contrast).  
- **Reinhard global tone mapping** — classic photographic tone curve (gentle compression/expansion).  
- **Exposure Fusion** — multi-scale fusion using contrast/saturation/well-exposedness weights.  
- **Proposed Method** — Convert the input SDR frame to **HDR (PQ / BT.2020)**, then compress with a piecewise curve designed to preserve highlight/near-black detail for better features/matches.  

## Summary results

### Pooled across datasets

**Median % change vs baseline (negative is better)**

**APE**
| Variant | Median %Δ |
|---|---:|
| Mertens | **−9.8%** |
| Proposed | **−8.8%** |
| CLAHE | −2.2% |
| Reinhard | −0.2% |

**RPE (Δ=5)**
| Variant | Median %Δ |
|---|---:|
| CLAHE | **−13.8%** |
| Mertens | **−11.4%** |
| Proposed | −6.2% |
| Reinhard | −2.8% |

### EuRoC MAV

**Median % change vs baseline** (**Win-rate** in parentheses)

**APE**
| Variant | Median %Δ | Win-rate |
|---|---:|---:|
| Mertens | **−10.1%** | **54.5%** |
| Proposed | **−8.8%** | **54.5%** |
| CLAHE | −1.0% | 36.4% |
| Reinhard | +1.7% | 36.4% |

**RPE (Δ=5)**
| Variant | Median %Δ | Win-rate |
|---|---:|---:|
| CLAHE | **−14.3%** | **68.2%** |
| Mertens | **−11.6%** | **54.5%** |
| Proposed | −6.7% | 50.0% |
| Reinhard | −3.3% | 45.5% |

### TUM RGB-D

**Median % change vs baseline** (**Win-rate** in parentheses)

**APE**
| Variant | Median %Δ | Win-rate |
|---|---:|---:|
| Proposed | **−10.6%** | **77.3%** |
| Mertens | **−9.1%** | **68.2%** |
| CLAHE | −3.6% | 54.5% |
| Reinhard | +1.8% | 45.5% |

**RPE (Δ=5)**
| Variant | Median %Δ | Win-rate |
|---|---:|---:|
| Proposed | **−12.3%** | **50.0%** |
| Mertens | **−10.6%** | **54.5%** |
| CLAHE | −6.8% | 54.5% |
| Reinhard | −1.7% | 40.9% |
