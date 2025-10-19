import numpy as np

def _histogram(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Computes image histogram using bin centers (not edges).
    This constructs edges as midpoints between adjacent centers; the extreme bins
    use the same half-width as the nearest interior gap.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    c = np.asarray(centers, dtype=np.float64).ravel()
    if c.size == 0:
        return np.zeros(0, dtype=np.float64)
    if c.size == 1:
        w = 0.5
        edges = np.array([c[0] - w, c[0] + w], dtype=np.float64)
    else:
        half = np.diff(c) / 2.0
        edges = np.concatenate(([c[0] - half[0]], c[:-1] + half, [c[-1] + half[-1]]))
    # numpy.histogram: rightmost edge inclusive, others left-inclusive/right-exclusive by default
    counts, _ = np.histogram(x, bins=edges)
    return counts.astype(np.float64)


def proposed_itmo(
    input_frame: np.ndarray,
    input_bit_depth: int = 8,
    wB: float = 0.5,
    wC: float = 0.5,
) -> np.ndarray:
    """
    This is a Python implementation of the simplified version of the algorithm presented in my journal paper: 
    "a Perception-Based Inverse Tone Mapping Operator for High Dynamic Range Video Applications," 
    in IEEE Transactions on Circuits and Systems for Video Technology, vol. 31, no. 5, pp. 1711-1723, 
    May 2021, doi: 10.1109/TCSVT.2020.3014679.

    Inputs
      input_frame: H×W×3 array with integer code values (e.g., uint8/uint16) or float.
      input_bit_depth: 8 or 10. Determines quantization normalization.
      wB, wC: brightness/contrast weights.

    Output
      imgOut: H×W×3 float64 HDR image in BT.2020 primaries, *light domain*, in nits
              clamped to [0.01, 1000].
    """
    # --- Input & types ---
    img_double = np.asarray(input_frame, dtype=np.float64)

    # PQ constants (SMPTE ST 2084)
    l_max_PQ = 10000.0
    m_PQ = 78.8438
    n_PQ = 0.1593
    c1_PQ = 0.8359
    c2_PQ = 18.8516
    c3_PQ = 18.6875

    # Representation min/max
    maxValueY = ((256.0 * 1.0) * (2.0 ** (input_bit_depth - 8))) - 1.0
    minValueY = (255.0 * 0.0) * (2.0 ** (input_bit_depth - 8))

    # Minimum and maximum brightness levels of the SDR and HDR displays:
    disp_Min = 0.1
    disp_Max = 100.0
    hDR_Min_PQ = 0.0215
    hDR_Max_PQ = 0.7518

    # Clamp input to the nominal code range (same scalar limits for all channels)
    img_double = np.clip(img_double, minValueY, maxValueY)

    # Full scaling and quantization (bit-depth downshift to 8-bit, then /255)
    scale = 2.0 ** (input_bit_depth - 8)
    img_normalized = (img_double / scale) / 255.0

    # Simple gamma to linear light (approximate inverse of 2.2 OETF)
    img_light = np.power(img_normalized, 2.2, dtype=np.float64)

    # Split channels
    red_SDR = img_light[..., 0]
    green_SDR = img_light[..., 1]
    blue_SDR = img_light[..., 2]

    # BT.709 luma
    y_SDR = 0.2126 * red_SDR + 0.7152 * green_SDR + 0.0722 * blue_SDR

    # Map to display light range
    red_SDR_Light = red_SDR * (disp_Max - disp_Min) + disp_Min
    green_SDR_Light = green_SDR * (disp_Max - disp_Min) + disp_Min
    blue_SDR_Light = blue_SDR * (disp_Max - disp_Min) + disp_Min
    L_disp = y_SDR * (disp_Max - disp_Min) + disp_Min

    # Forward PQ (OETF)
    def _to_pq(L):
        t = np.power(np.maximum(L / l_max_PQ, 0.0), n_PQ, dtype=np.float64)  # (L/Lmax)^n
        num = c2_PQ * t + c1_PQ
        den = c3_PQ * t + 1.0
        return np.power(num / den, m_PQ, dtype=np.float64)

    l_disp_PQ = _to_pq(L_disp)
    red_SDR_PQ = _to_pq(red_SDR_Light)
    green_SDR_PQ = _to_pq(green_SDR_Light)
    blue_SDR_PQ = _to_pq(blue_SDR_Light)

    # PQ-domain ranges
    xmin = 0.0623  # PQ(disp_Min)
    xmax = 0.5081  # PQ(disp_Max)
    ymin = 0.0215  # PQ(HDR_Min)
    ymax = 0.7518  # PQ(HDR_Max)

    # Histogram of PQ-ed luminance using bin *centers* (2^bitdepth bins)
    num_bins = int(2 ** input_bit_depth)
    centers = np.linspace(xmin, xmax, num_bins, dtype=np.float64)
    larray_disp_PQ = l_disp_PQ.reshape(-1)
    n_light = _histogram(larray_disp_PQ, centers)

    # Segment thresholds & parameters
    x1 = 0.1233
    x2 = 0.3975
    threshold_low = 35
    threshold_high = 192

    n1 = 35.0
    n2 = 157.0
    n3 = 64.0
    delta1 = 0.0610
    delta2 = 0.2742
    delta3 = 0.1106
    r = 0.2845

    # Probabilities in three brightness areas
    total = np.sum(n_light)
    
    # Avoid divide-by-zero if for some reason histogram is empty
    if total == 0:
        p1 = np.zeros(threshold_low, dtype=np.float64)
        p2 = np.zeros(threshold_high - threshold_low, dtype=np.float64)
        p3 = np.zeros(n_light.size - threshold_high, dtype=np.float64)
    else:
        p1 = n_light[:threshold_low] / total
        p2 = n_light[threshold_low:threshold_high] / total
        p3 = n_light[threshold_high:] / total

    p1 = np.sum(p1)
    p2 = np.sum(p2)
    p3 = np.sum(p3)
    p_1 = (1.0 - p1) / 2.0
    p_2 = (1.0 - p2) / 2.0
    p_3 = (1.0 - p3) / 2.0

    # Brightness/contrast maximization constants
    a1 = p_1 * 0.3073
    a2 = p_2 * 11.5866
    a3 = p_3 * 13.2019
    b1 = p_1 * 3.2207
    b2 = p_2 * 40.7965
    b3 = p_3 * 28.9940

    fc = 0.0809
    fb = 13.6971
    w1 = wC / fc
    w2 = wB / fb

    # Solve for piecewise-linear slopes in PQ domain
    denom1 = (w1 * delta1 * delta1 * p_1 + w2 * a1 - 2.0 * w2 * xmin * b1 + w2 * n1 * xmin * xmin + w2 * n2 * delta1 * delta1)
    denom2 = (w1 * delta2 * delta2 * p_2 + w2 * a2 - 2.0 * w2 * x1 * b2 + w2 * n2 * x1 * x1)
    denom3 = (w1 * delta3 * delta3 * p_3 + w2 * a3 - 2.0 * w2 * xmax * b3 + w2 * n3 * xmax * xmax)

    a = (w2 * n2 * delta1 * x1 - w2 * delta1 * b2) / denom1
    b = (w2 * n1 * (ymin - xmin) * xmin - w2 * (ymin - xmin) * b1 - w2 * n2 * (ymin - xmin) * delta1) / denom1
    c = (-delta1) / (2.0 * denom1)

    d = (w2 * n2 * delta1 * x1 - w2 * delta1 * b2) / denom2
    e = (w2 * n2 * (ymin - xmin) * x1 - w2 * (ymin - xmin) * b2) / denom2
    f = (-delta2) / (2.0 * denom2)

    g = (w2 * n3 * (ymax - xmax) * xmax - w2 * (ymax - xmax) * b3) / denom3
    h = (-delta3) / (2.0 * denom3)

    # Lambda and region gains
    ad = 1.0 - a * d
    x1 = ((a * e + b) / ad)  # + ((a * f + c) / ad) * lambda  (filled after lambda)
    x2 = ((b * d + e) / ad)  # + ((c * d + f) / ad) * lambda
    c1lam = (a * f + c) / ad
    c2lam = (c * d + f) / ad
    # lambda:
    lam_num = r - (x1 * delta1 + x2 * delta2 + g * delta3)
    lam_den = (c1lam * delta1 + c2lam * delta2 + h * delta3)
    lam = lam_num / lam_den
    x1 = x1 + c1lam * lam
    x2 = x2 + c2lam * lam
    X3 = g + h * lam

    # Expansion function (piecewise linear in PQ)
    s1 = x1 + 1.0
    s2 = x2 + 1.0
    s3 = X3 + 1.0
    a1 = ymin - s1 * xmin
    a2 = (s1 - s2) * x1 + a1
    a3 = ymax - s3 * xmax

    # Apply piecewise expansion
    lhdr_PQ = s3 * l_disp_PQ + a3
    mask2 = l_disp_PQ < x2
    lhdr_PQ = np.where(mask2, s2 * l_disp_PQ + a2, lhdr_PQ)
    mask1 = l_disp_PQ < x1
    lhdr_PQ = np.where(mask1, s1 * l_disp_PQ + a1, lhdr_PQ)

    # Clamp to HDR PQ range
    lhdr_PQ = np.clip(lhdr_PQ, ymin, ymax)

    # Ratio in PQ
    l_ratio_PQ = lhdr_PQ / l_disp_PQ

    # Scale R, g, b in PQ domain
    red_HDR_PQ = l_ratio_PQ * red_SDR_PQ
    blue_HDR_PQ = l_ratio_PQ * blue_SDR_PQ
    green_HDR_PQ = l_ratio_PQ * green_SDR_PQ

    # Clamp to [hDR_Min_PQ, hDR_Max_PQ]
    red_HDR_PQ = np.clip(red_HDR_PQ, hDR_Min_PQ, hDR_Max_PQ)
    green_HDR_PQ = np.clip(green_HDR_PQ, hDR_Min_PQ, hDR_Max_PQ)
    blue_HDR_PQ = np.clip(blue_HDR_PQ, hDR_Min_PQ, hDR_Max_PQ)    

    # Gamut conversion: BT.709 -> BT.2020 in normalized domain
    # Then clamp to [0,1] and scale back to light domain.
    imgOut_norm2020_r = 0.6274039 * red_HDR_PQ + 0.32928304 * green_HDR_PQ + 0.04331307 * blue_HDR_PQ
    imgOut_norm2020_g = 0.06909729 * red_HDR_PQ + 0.9195040 * green_HDR_PQ + 0.01136232 * blue_HDR_PQ
    imgOut_norm2020_b = 0.01639144 * red_HDR_PQ + 0.08801331 * green_HDR_PQ + 0.89559525 * blue_HDR_PQ

    imgOut_norm2020_r = np.clip(imgOut_norm2020_r, 0.0, 1.0)
    imgOut_norm2020_g = np.clip(imgOut_norm2020_g, 0.0, 1.0)
    imgOut_norm2020_b = np.clip(imgOut_norm2020_b, 0.0, 1.0)

    # Stack and scale to nits
    imgOut = np.stack([imgOut_norm2020_r, imgOut_norm2020_g, imgOut_norm2020_b], axis=-1)
    return imgOut.astype(np.float64)
