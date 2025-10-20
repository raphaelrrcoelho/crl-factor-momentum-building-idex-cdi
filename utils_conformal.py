"""
Conformal utilities:
- Weighted finite-sample quantile (Romano et al., with optional weights)
- Non-crossing enforcement (vectorized + scalar)
- A simple CQR-style calibrator that returns an apply() function

Note:
Your production CQR calibrator for the online heads lives in
crl_factor_bandit_conformal.CQRCalibrator. This module provides
lightweight, notebook-friendly pieces without changing the training API.
"""

from __future__ import annotations
from typing import Tuple, Callable, Optional
import numpy as np

# ==== Horizon scaling (single source of truth) ====

import numpy as np

_HS_MODES = {"none", "per_step", "per_sqrt_time", "per_sigma"}

def horizon_scale_factor(h: int, mode: str = "none", sigma_map: dict | None = None) -> float:
    """
    Returns the multiplicative factor to make metrics comparable across horizons.
      - none:          1
      - per_step:      1/h
      - per_sqrt_time: 1/sqrt(h)
      - per_sigma:     1/sigma[h]  (requires sigma_map: {h: float})
    """
    if mode not in _HS_MODES:
        raise ValueError(f"horizon_scale_mode must be one of {_HS_MODES}, got {mode}")
    h = int(h)
    if h <= 0: 
        return 1.0
    if mode == "none":
        return 1.0
    if mode == "per_step":
        return 1.0 / float(h)
    if mode == "per_sqrt_time":
        return 1.0 / float(np.sqrt(h))
    # per_sigma
    if not sigma_map or h not in sigma_map or sigma_map[h] <= 0:
        raise ValueError("per_sigma requires positive sigma_map entries for each horizon.")
    return 1.0 / float(sigma_map[h])

def apply_horizon_scaling(df, mode: str = "none", sigma_map: dict | None = None,
                          h_col: str = "h", cols: list[str] = ("width","loss")):
    """
    Adds scaled versions of columns, e.g., width_scaled, loss_scaled.
    """
    if mode == "none":
        for c in cols:
            df[f"{c}_scaled"] = df[c].astype(float)
        return df
    df = df.copy()
    factors = df[h_col].astype(int).map(lambda hh: horizon_scale_factor(int(hh), mode, sigma_map))
    for c in cols:
        df[f"{c}_scaled"] = df[c].astype(float) * factors.values
    return df


def weighted_finite_sample_quantile(scores: np.ndarray, weights: Optional[np.ndarray], q: float) -> float:
    """
    Finite-sample quantile with optional non-negative weights.
    If weights is None, uses uniform weights.
    """
    s = np.asarray(scores, dtype=float)
    if s.size == 0:
        return 0.0
    if weights is None:
        w = np.ones_like(s, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != s.shape:
            w = np.broadcast_to(w, s.shape).astype(float)
    order = np.argsort(s)
    s, w = s[order], np.clip(w[order], 0.0, np.inf)
    cw = np.cumsum(w)
    total = cw[-1]
    if total <= 0:
        return float(s[-1])
    tau = float(q) * total
    j = int(np.searchsorted(cw, tau, side="right"))
    j = max(0, min(j, s.size - 1))
    return float(s[j])


def finite_sample_quantile(scores: np.ndarray, q: float) -> float:
    """
    Romano et al. finite-sample correction: k = ceil((n+1)q)/n.
    Returns the kth order statistic of scores.
    """
    s = np.asarray(scores, dtype=float)
    n = s.shape[0]
    if n == 0:
        return 0.0
    k = int(np.ceil((n + 1) * float(q))) - 1  # 0-based
    k = min(max(k, 0), n - 1)
    return float(np.partition(s, k)[k])


def enforce_monotonic(q_lo: np.ndarray, q_hi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Element-wise clamp to guarantee q_lo <= q_hi (vectorized, broadcast OK).
    """
    lo = np.minimum(q_lo, q_hi)
    hi = np.maximum(q_lo, q_hi)
    return lo, hi


def enforce_monotonic_scalar(q_lo: float, q_hi: float) -> Tuple[float, float]:
    """Scalar version of enforce_monotonic."""
    return (q_lo, q_hi) if q_lo <= q_hi else (q_hi, q_lo)


def conformalize_cqr(
    q_lo_hat: np.ndarray,
    q_hi_hat: np.ndarray,
    y_calib: np.ndarray,
    q: float,
    weights: Optional[np.ndarray] = None,
) -> Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Build a CQR adjuster using a CALIBRATION set.

    Inputs
    ------
    q_lo_hat, q_hi_hat : arrays
        Predicted (uncalibrated) lower/upper quantiles on a calibration set.
    y_calib : array
        Realized targets on the calibration set.
    q : float
        Target marginal coverage level (e.g., q = 0.90 → 90% total coverage).
    weights : array or None
        Optional non-negative weights for calibration residuals.

    Returns
    -------
    apply(lo_pred, hi_pred) -> (lo_adj, hi_adj) with monotonicity guaranteed.
    """
    q_lo_hat = np.asarray(q_lo_hat, dtype=float)
    q_hi_hat = np.asarray(q_hi_hat, dtype=float)
    y_calib  = np.asarray(y_calib,  dtype=float)

    if q_lo_hat.shape != q_hi_hat.shape or q_lo_hat.shape != y_calib.shape:
        raise ValueError("q_lo_hat, q_hi_hat, and y_calib must share the same shape.")

    alpha = 1.0 - float(q)
    s_lo = q_lo_hat - y_calib   # (l_pred - y)^+
    s_hi = y_calib - q_hi_hat   # (y - u_pred)^+

    # One-sided finite-sample adjustments (weighted)
    q_lo_adj = weighted_finite_sample_quantile(np.maximum(s_lo, 0.0), weights, 1.0 - alpha)
    q_hi_adj = weighted_finite_sample_quantile(np.maximum(s_hi, 0.0), weights, 1.0 - alpha)

    def apply(lo_pred, hi_pred):
        lo_pred = np.asarray(lo_pred, dtype=float)
        hi_pred = np.asarray(hi_pred, dtype=float)
        lo = lo_pred - q_lo_adj
        hi = hi_pred + q_hi_adj
        return enforce_monotonic(lo, hi)

    return apply

def coldstart_clamp_bounds(l, u, *,
                           min_width=0.002,   # 20bps
                           max_width=0.050):  # 5%
    """
    Symmetric clamp to stabilize raw intervals during calibrator warm-up.
    Ensures u>=l, enforces min/max width around the midpoint.
    """
    l = float(l); u = float(u)
    if not np.isfinite(l) or not np.isfinite(u):
        return l, u
    if u < l:
        l, u = u, l
    mid = 0.5 * (l + u)
    width = max(u - l, 0.0)

    # Enforce minimum width
    width = max(width, float(min_width))
    # Enforce maximum width
    width = min(width, float(max_width))

    l_new = mid - 0.5 * width
    u_new = mid + 0.5 * width
    return l_new, u_new


# ---------- small helpers useful in diagnostics / notebooks ---------- #

def interval_score(y: np.ndarray, l: np.ndarray, u: np.ndarray, alpha: float) -> np.ndarray:
    """
    Symmetric interval score (Gneiting & Raftery); lower is better.
    Mirrors crl_factor_bandit_conformal.interval_score API for convenience.
    """
    y = np.asarray(y, dtype=float); l = np.asarray(l, dtype=float); u = np.asarray(u, dtype=float)
    width = np.maximum(0.0, u - l)
    under = np.maximum(0.0, l - y)
    over  = np.maximum(0.0, y - u)
    return width + (2.0/alpha)*under + (2.0/alpha)*over


def coverage_rate(y: np.ndarray, l: np.ndarray, u: np.ndarray) -> float:
    """
    Fraction of y covered by [l,u].
    """
    y = np.asarray(y, dtype=float); l = np.asarray(l, dtype=float); u = np.asarray(u, dtype=float)
    return float(np.mean((l <= y) & (y <= u)))
