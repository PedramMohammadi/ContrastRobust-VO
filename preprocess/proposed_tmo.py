
import numpy as np
from typing import Literal

_L_MAX_PQ = 10000.0
_M_PQ = 78.8438
_N_PQ = 0.1593
_c1_PQ = 0.8359
_c2_PQ = 18.8516
_c3_PQ = 18.6875

def _from_pq(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    P_root = np.power(np.maximum(P, 0.0), 1.0 / _M_PQ, dtype=np.float64)
    num = np.maximum(P_root - _c1_PQ, 0.0)
    den = (_c2_PQ - _c3_PQ * P_root)
    return _L_MAX_PQ * np.power(np.maximum(num / den, 0.0), 1.0 / _N_PQ, dtype=np.float64)

def proposed_tmo(
    imgIn_HDR: np.ndarray,
    mappingCurve: Literal["PiecewiseLinear","Gamma"]="Gamma",
    hDRmin_PQ: float = 0.0215, #PQ(0.01)
    hDRmax_PQ: float = 0.7518, #PQ(1000)
    sDRmin_PQ: float = 0.0623, #PQ(0.1)
    sDRmax_PQ: float = 0.5081, #PQ(100)
    sDRmin: float = 0.1,
    sDRmax: float = 100.0,
    gammaSDR: float = 2.2,
    alpha: float = 1.0,
) -> np.ndarray:

    """
    This is a Python implementation of my proposed HDR --> SDR conversion algorithm.

    Inputs
        imgIn_HDR: Input HDR frame 
        mappingCurve: Type of the mapping curve for HDR to SDR conversion,
        hDRmin_PQ: PQ equivalent of 0.01 nits
        hDRmax_PQ: PQ equivalent of 1000 nits
        sDRmin_PQ: PQ equivalent of 0.1 nits
        sDRmax_PQ: PQ equivalent of 100 nits
        sDRmin: Minimum brightness value of the input SDR frame (0.1 nits)
        sDRmax: Maximum brightness value of the input SDR frame (100 nits)
        gammaSDR: Gamma value
        alpha: Saturation factor in color correction (Higher mean more saturated colors)

    Output
        imgOut_SDR: Generated SDR frame
    """
    
    img = np.asarray(imgIn_HDR, dtype=np.float64)

    imgIn_HDR_Normalized = img
    imgIn_HDR_PQ = np.clip(imgIn_HDR_Normalized, hDRmin_PQ, hDRmax_PQ)

    red_HDR_PQ   = imgIn_HDR_PQ[..., 0]
    green_HDR_PQ = imgIn_HDR_PQ[..., 1]
    blue_HDR_PQ  = imgIn_HDR_PQ[..., 2]

    #Luminance channel for BT.2020 gamut
    y_HDR_PQ = 0.2627 * red_HDR_PQ + 0.6779 * green_HDR_PQ + 0.0593 * blue_HDR_PQ

    # Mapping curve
    if mappingCurve == "PiecewiseLinear":
        x1, x2 = 0.1310, 0.4597
        s1, s2, s3 = 0.5570, 0.8339, 0.3787
        a1, a2, a3 = 0.0503, 0.0141, 0.2233

        y_SDR_PQ = s3 * y_HDR_PQ + a3
        mask2 = y_HDR_PQ < x2
        y_SDR_PQ = np.where(mask2, s2 * y_HDR_PQ + a2, y_SDR_PQ)
        mask1 = y_HDR_PQ < x1
        y_SDR_PQ = np.where(mask1, s1 * y_HDR_PQ + a1, y_SDR_PQ)
    else:
        c1, c2 = 0.6012, 0.5902
        y_SDR_PQ = c1 * np.power(y_HDR_PQ, c2, dtype=np.float64)

    # Color adjustment
    ratio = np.divide(y_SDR_PQ, y_HDR_PQ, out=np.zeros_like(y_SDR_PQ), where=(y_HDR_PQ != 0))
    scale = alpha * ratio
    red_SDR_PQ   = scale * red_HDR_PQ
    green_SDR_PQ = scale * green_HDR_PQ
    blue_SDR_PQ  = scale * blue_HDR_PQ

    red_SDR_PQ   = np.clip(red_SDR_PQ,   sDRmin_PQ, sDRmax_PQ)
    green_SDR_PQ = np.clip(green_SDR_PQ, sDRmin_PQ, sDRmax_PQ)
    blue_SDR_PQ  = np.clip(blue_SDR_PQ,  sDRmin_PQ, sDRmax_PQ)

    red_SDR   = _from_pq(red_SDR_PQ)
    green_SDR = _from_pq(green_SDR_PQ)
    blue_SDR  = _from_pq(blue_SDR_PQ)

    img_SDR = np.stack([red_SDR, green_SDR, blue_SDR], axis=-1)
    img_SDR = np.clip(img_SDR, sDRmin, sDRmax)

    img_SDR_Normalized = (img_SDR - sDRmin) / (sDRmax - sDRmin)

    r2020 = img_SDR_Normalized[..., 0]
    g2020 = img_SDR_Normalized[..., 1]
    b2020 = img_SDR_Normalized[..., 2]

    # BT.2020 to BT.709 conversion
    red_SDR_709   =  1.6605 * r2020 - 0.5876 * g2020 - 0.0728 * b2020
    green_SDR_709 = -0.1246 * r2020 + 1.1329 * g2020 - 0.0083 * b2020
    blue_SDR_709  = -0.0182 * r2020 - 0.1006 * g2020 + 1.1187 * b2020

    imgOut_709 = np.stack([red_SDR_709, green_SDR_709, blue_SDR_709], axis=-1)
    imgOut_709 = np.clip(imgOut_709, 0.0, 1.0)

    imgOut_SDR = np.power(imgOut_709, 1.0 / gammaSDR, dtype=np.float64)
    return imgOut_SDR.astype(np.float64)
