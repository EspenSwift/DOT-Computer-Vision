import numpy as np

def _moving_average_1d(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    k = int(k)
    if x.size < 2:
        return x
    pad = k // 2
    left = np.repeat(x[:1], pad)
    right = np.repeat(x[-1:], pad)
    y = np.convolve(np.concatenate([left, x, right]), np.ones(k)/k, mode="valid")
    return y

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(float); b = b.astype(float)
    if a.size == 0 or b.size == 0:
        return 0.0
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 1e-12:
        return 0.0
    return num / den

def _compute_stream_consistency(
    Rx: np.ndarray,
    Tx: np.ndarray,
    tau_R: float,
    tau_T: float,
    window_steps: int,
    smooth: int,
):
    """
    Compute sign-match ratio, cosine similarity, and combined score for one stream.

    Returns: dict with
        rho_sign, rho_cos, score, usable_steps, dRx, dTx
    """
    if smooth and smooth > 1:
        Rx_s = _moving_average_1d(Rx, smooth)
        Tx_s = _moving_average_1d(Tx, smooth)
    else:
        Rx_s, Tx_s = Rx, Tx

    dRx = np.diff(Rx_s)
    dTx = np.diff(Tx_s)

    W = min(window_steps, dRx.size)
    if W <= 0:
        return {
            "rho_sign": 0.0,
            "rho_cos": 0.0,
            "score": 0.0,
            "usable_steps": 0,
            "dRx": dRx,
            "dTx": dTx,
        }

    idx = np.arange(0, W, dtype=int)
    dRx_sel = dRx[idx]
    dTx_sel = dTx[idx]

    mask = (np.abs(dRx_sel) >= tau_R) & (np.abs(dTx_sel) >= tau_T)
    if not np.any(mask):
        return {
            "rho_sign": 0.0,
            "rho_cos": 0.0,
            "score": 0.0,
            "usable_steps": 0,
            "dRx": dRx,
            "dTx": dTx,
        }

    dRx_u = dRx_sel[mask]
    dTx_u = dTx_sel[mask]

    sign_match = np.sign(dRx_u) == np.sign(dTx_u)
    rho_sign = float(np.sum(sign_match)) / float(sign_match.size)
    rho_cos = _cosine_similarity(dRx_u, dTx_u)

    # You can tweak weights if needed
    w_sign = 1.0
    w_cos = 1.0
    score = w_sign * rho_sign + w_cos * rho_cos

    return {
        "rho_sign": rho_sign,
        "rho_cos": rho_cos,
        "score": score,
        "usable_steps": int(np.sum(mask)),
        "dRx": dRx,
        "dTx": dTx,
    }

def choose_best_stream(
    Rx1: np.ndarray,
    Tx1: np.ndarray,
    Rx2: np.ndarray,
    Tx2: np.ndarray,
    tau_R: float = 0.2,
    tau_T: float = 1e-3,
    window_steps: int = 200,
    smooth: int = 3,
):
    """
    Decide whether stream 1 (Pose1) or stream 2 (Pose2) is more self-consistent.

    Returns:
        best_idx      : 0 for Pose1, 1 for Pose2
        metrics       : dict with detailed metrics:
                        {
                          "stream1": {rho_sign, rho_cos, score, usable_steps},
                          "stream2": {rho_sign, rho_cos, score, usable_steps}
                        }
    """
    Rx1 = np.asarray(Rx1, dtype=float)
    Tx1 = np.asarray(Tx1, dtype=float)
    Rx2 = np.asarray(Rx2, dtype=float)
    Tx2 = np.asarray(Tx2, dtype=float)

    m1 = _compute_stream_consistency(
        Rx=Rx1, Tx=Tx1,
        tau_R=tau_R, tau_T=tau_T,
        window_steps=window_steps,
        smooth=smooth,
    )

    m2 = _compute_stream_consistency(
        Rx=Rx2, Tx=Tx2,
        tau_R=tau_R, tau_T=tau_T,
        window_steps=window_steps,
        smooth=smooth,
    )

    S1 = m1["score"]
    S2 = m2["score"]

    if S1 > S2:
        best = 0
    elif S2 > S1:
        best = 1
    else:
        # Perfect tie: arbitrary but documented
        best = 0

    metrics = {
        "stream1": {
            "rho_sign": m1["rho_sign"],
            "rho_cos":  m1["rho_cos"],
            "score":    m1["score"],
            "usable_steps": m1["usable_steps"],
        },
        "stream2": {
            "rho_sign": m2["rho_sign"],
            "rho_cos":  m2["rho_cos"],
            "score":    m2["score"],
            "usable_steps": m2["usable_steps"],
        },
    }

    return best, metrics
