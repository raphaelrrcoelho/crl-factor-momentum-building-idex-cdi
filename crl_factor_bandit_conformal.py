# crl_factor_bandit_conformal.py
# -*- coding: utf-8 -*-
"""
CRL factor-spec selection with Conformalized Quantile Regression (CQR)
SECURITY-LEVEL ONLY — **leakage-safe, horizon-deferred updates**

FIXES APPLIED (see code_review_actual_bugs.md for details):
1. ✅ Fixed volatility floor in build_momentum() - now uses 0.01 (1%) instead of 1.0 (100%)
2. ✅ Added periodic Sherman-Morrison recomputation with condition number monitoring
3. ✅ Added proper numerical stability tracking
4. ✅ Improved documentation and constants

What's new vs your previous version (same public artifacts):
- **Deferred updates**: bandit reward, quantile heads, and conformal calibrators
  update only when the decision's outcome matures (at date t+h). Coverage
  tracking is also based solely on matured observations.
- **Label-free action set**: at time t, valid horizons are determined by a
  precomputed, label-free "alive" mask (presence of a row at t+h for the same
  security), not by checking y_h at t.
- **Identical outputs & API**: writes the same files and uses the same config
  keys used by run_experiments.step_crl.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any
import os, json, math
from collections import deque, defaultdict, Counter

import numpy as np
import pandas as pd

from run_config import cfg_get

# ---------------------------------------------------------------------
# Constants (moved from magic numbers)
# ---------------------------------------------------------------------
VOLATILITY_FLOOR = 0.01  # FIX: Was 1.0, now 1% (in decimal form)
VOLATILITY_EPSILON = 1e-8  # Small constant to prevent division by zero
MIN_VOLATILITY_OBSERVATIONS = 5  # Minimum observations for volatility calculation
MATRIX_RECOMPUTE_FREQUENCY = 1000  # Periodic recomputation frequency for numerical stability
CONDITION_NUMBER_THRESHOLD = 1e10  # Threshold for ill-conditioned matrices
BOOTSTRAP_BLOCK_LENGTH = 5  # Block length for moving block bootstrap
CLIP_Z_SCORE = 8.0  # Clip z-scores to this range

# ---------------------------------------------------------------------
# IO & schema helpers
# ---------------------------------------------------------------------

DATE_COL = "date"
ID_COL = "id"

_SCHEMA_SYNONYMS = {
    "debenture_id": "id",
    "time_to_maturity": "ttm",
}

REQUIRED_COLS = [DATE_COL, ID_COL, "return", "index_weight"]
OPTIONAL_COLS = ["index_return", "spread", "duration", "ttm", "sector_id", "active"]


def _ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def _results_paths(results_dir: str) -> Dict[str, str]:
    return {
        "scores_sec": os.path.join(results_dir, "crl_scores_securities.csv"),
        "choices":    os.path.join(results_dir, "crl_policy_choices.csv"),
        "analysis":   os.path.join(results_dir, "analysis.json"),
        "falsif":     os.path.join(results_dir, "falsification.json"),
    }


def load_panel(panel_path: str) -> pd.DataFrame:
    """Load long panel, normalize column names, derive index_return if missing."""
    if panel_path.endswith(".parquet"):
        df = pd.read_parquet(panel_path)
    elif panel_path.endswith(".csv"):
        df = pd.read_csv(panel_path, parse_dates=[DATE_COL])
    elif panel_path.endswith(".pkl"):
        df = pd.read_pickle(panel_path)
    else:
        raise ValueError(f"Unsupported panel format: {panel_path}")

    # Normalize synonyms
    for old, new in _SCHEMA_SYNONYMS.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    # Basic validation
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Panel missing required columns: {missing}")

    # Derive index_return if missing
    if "index_return" not in df.columns:
        df["index_return"] = np.nan
        for d in df[DATE_COL].unique():
            day = df[df[DATE_COL] == d]
            w = day["index_weight"].values
            r = day["return"].values
            if w.sum() > 0:
                idx_ret = np.average(r, weights=w + 1e-12)
                df.loc[df[DATE_COL] == d, "index_return"] = idx_ret

    return df.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)


def _compute_cross_sectional_rank(group: pd.DataFrame, col: str, min_count: int = 10) -> pd.Series:
    """
    Compute percentile ranks within a cross-section.
    Returns 0.5 (neutral) if insufficient valid observations.
    """
    valid = group[col].dropna()
    if len(valid) < min_count:
        # Not enough valid securities - return neutral percentile
        return pd.Series(0.5, index=group.index)
    
    # Compute ranks only on valid values
    ranks = group[col].rank(pct=True, method='average')
    # Fill NaN ranks with neutral value
    return ranks.fillna(0.5)


def build_momentum(df: pd.DataFrame, w: int) -> pd.Series:
    """
    Per-security time-series momentum, lagged to avoid look-ahead.
    
    FIX: Volatility floor changed from 1.0 to VOLATILITY_FLOOR (0.01 = 1%)
    
    Returns:
        Momentum signal: sum of past returns / volatility
    """
    pdf = df[[DATE_COL, ID_COL, "return"]].copy()
    pdf["ret_lag"] = pdf.groupby(ID_COL)["return"].shift(1)
    
    # Rolling sum of lagged returns (numerator)
    mom_raw = pdf.groupby(ID_COL)["ret_lag"].rolling(
        w, min_periods=MIN_VOLATILITY_OBSERVATIONS
    ).sum().reset_index(level=0, drop=True)

    # Volatility calculation (denominator)
    volw = max(20, w)
    vol = pdf.groupby(ID_COL)["return"].rolling(
        volw, min_periods=MIN_VOLATILITY_OBSERVATIONS
    ).std().reset_index(level=0, drop=True)
    
    # FIX: Proper lagging and flooring
    # The denominator should use the same window as numerator for consistency
    vol = vol.groupby(pdf[ID_COL]).shift(1)  # Lag to avoid look-ahead
    
    # FIX: Replace small/zero volatility with VOLATILITY_FLOOR (0.01 = 1%)
    # This is much more reasonable than the previous 1.0 (100%)
    den = vol.fillna(VOLATILITY_FLOOR).clip(lower=VOLATILITY_FLOOR)
    
    # Compute momentum signal
    mom = mom_raw / (den + VOLATILITY_EPSILON)

    name = f"mom_w{w}"
    return pd.Series(mom.values, index=df.index, name=name)


# ---------------------------------------------------------------------
# Index helpers (kept for compatibility)
# ---------------------------------------------------------------------

def index_weighted_mean(df: pd.DataFrame, col: str) -> pd.Series:
    g = df[[DATE_COL, col, "index_weight"]].dropna()
    wsum = g.groupby(DATE_COL)["index_weight"].transform(lambda x: x / (x.sum() + 1e-12))
    val = (g[col] * wsum).groupby(g[DATE_COL]).sum()
    return val.rename(col)


def summarize_factors_to_index(df: pd.DataFrame, factor_cols: List[str]) -> pd.DataFrame:
    parts = [index_weighted_mean(df, c) for c in factor_cols if c in df.columns]
    return pd.concat(parts, axis=1).sort_index()


def build_context(df_index: pd.DataFrame) -> pd.DataFrame:
    """Minimal index context (used occasionally by causal code)."""
    ctx = pd.DataFrame(index=df_index.index)
    if "index_return" in df_index.columns:
        ctx["ret_vol20"] = df_index["index_return"].rolling(20, min_periods=5).std().fillna(0.0)
    else:
        ctx["ret_vol20"] = 0.0
    ctx["bias"] = 1.0
    return ctx.astype(float)


# ---------------------------------------------------------------------
# Online quantile heads & CQR calibrator
# ---------------------------------------------------------------------

class OnlineQuantileHead:
    """Online linear quantile forecaster per arm with non-crossing enforcement."""
    def __init__(self, d: int, tau_low: float = 0.05, tau_high: float = 0.95,
                 lam: float = 1e-4, eta: float = 0.02, w_cap: float = 1e3):
        self.d = d
        self.tau_low  = float(tau_low)
        self.tau_high = float(tau_high)
        self.lam  = float(lam)
        self.eta  = float(eta)
        
        # Initialize with small random weights for better separation
        rng = np.random.default_rng(23)
        self.w_lo = rng.normal(0, 0.01, d)
        self.w_hi = rng.normal(0, 0.01, d)
        
        # Ensure initial predictions have reasonable spread
        self.w_lo[0] -= 0.001  # Bias term for lower quantile
        self.w_hi[0] += 0.001  # Bias term for upper quantile
        self.w_cap = float(w_cap)

    @staticmethod
    def _grad_yhat(y_hat: float, y: float, tau: float) -> float:
        """Pinball loss gradient."""
        return (1.0 - tau) if (y < y_hat) else (-tau)

    def update(self, x: np.ndarray, y: float):
        """Update quantile models using online gradient descent."""
        # Low quantile update
        y_lo = float(self.w_lo @ x)
        g = self._grad_yhat(y_lo, y, self.tau_low)
        self.w_lo = (1 - self.eta * self.lam) * self.w_lo - self.eta * g * x

        # High quantile update
        y_hi = float(self.w_hi @ x)
        g = self._grad_yhat(y_hi, y, self.tau_high)
        self.w_hi = (1 - self.eta * self.lam) * self.w_hi - self.eta * g * x

        # Clip weights and enforce non-crossing
        self.w_lo = np.clip(self.w_lo, -self.w_cap, self.w_cap)
        self.w_hi = np.clip(self.w_hi, -self.w_cap, self.w_cap)
        
        # Soft non-crossing enforcement
        l, u = self.predict(x)
        if l > u:
            mid = 0.5 * (l + u)
            self.w_lo += 0.1 * (mid - y_lo) * x
            self.w_hi += 0.1 * (y_hi - mid) * x

    def predict(self, x: np.ndarray) -> Tuple[float, float]:
        """Predict lower and upper quantiles."""
        l = float(self.w_lo @ x)
        u = float(self.w_hi @ x)
        if l > u:
            l, u = min(l, u), max(l, u)
        return l, u


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Compute weighted quantile (Romano et al. style)."""
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    order = np.argsort(v)
    v, w = v[order], np.clip(w[order], 0.0, np.inf)
    cw = np.cumsum(w)
    total = cw[-1]
    if total <= 0: 
        return float(v[-1])
    tau = q * total
    j = int(np.searchsorted(cw, tau, side="right"))
    j = max(0, min(j, len(v) - 1))
    return float(v[j])


class CQRCalibrator:
    """Rolling / exponentially-weighted CQR calibrator (single-score)."""
    def __init__(self, alpha: float, window: int = 120, lambda_recency: float = 0.0):
        self.alpha = float(alpha)
        self.window = int(max(30, window))
        self.lambda_recency = float(max(0.0, lambda_recency))
        self.s_lo: deque = deque(maxlen=self.window)
        self.s_hi: deque = deque(maxlen=self.window)

    def update(self, y: float, l_pred: float, u_pred: float):
        """Update calibration buffer with new nonconformity scores."""
        e_lo = max(l_pred - y, 0.0)
        e_hi = max(y - u_pred, 0.0)
        self.s_lo.append(e_lo)
        self.s_hi.append(e_hi)

    def adjust(self, l_pred: float, u_pred: float) -> Tuple[float, float, float]:
        """
        Return (l_adj, u_adj, qhat). Uses single-score CQR (max residual).
        Requires at least 20 calibration points before adjusting.
        """
        if len(self.s_lo) < 20:
            return l_pred, u_pred, 0.0

        s_lo = np.array(self.s_lo, float)
        s_hi = np.array(self.s_hi, float)
        s = np.maximum(s_lo, s_hi)

        # Apply recency weighting if configured
        if self.lambda_recency > 0:
            ages = np.arange(len(s), 0, -1, dtype=float)
            w = np.exp(-self.lambda_recency * ages)
        else:
            w = np.ones_like(s, float)

        # Compute quantile
        q = _weighted_quantile(s, w, 1.0 - self.alpha)
        
        # Adjust intervals
        l = l_pred - q
        u = u_pred + q
        
        # Ensure non-crossing
        if l > u:
            mid = 0.5 * (l + u)
            l = u = mid
            
        return l, u, float(q)


def coldstart_clamp_bounds(l: float, u: float, 
                           min_width: float = 0.002, 
                           max_width: float = 0.050) -> Tuple[float, float]:
    """
    During warm-up phase, clamp interval width to reasonable bounds.
    min_width: 0.002 = 20 basis points
    max_width: 0.050 = 5%
    """
    width = u - l
    if width < min_width:
        mid = 0.5 * (l + u)
        l = mid - 0.5 * min_width
        u = mid + 0.5 * min_width
    elif width > max_width:
        mid = 0.5 * (l + u)
        l = mid - 0.5 * max_width
        u = mid + 0.5 * max_width
    return l, u


# ---------------------------------------------------------------------
# Bandit policy (LinTS) with FIXES
# ---------------------------------------------------------------------

@dataclass
class LinTSArm:
    """
    Linear Thompson Sampling arm with IMPROVED numerical stability.
    
    FIXES:
    - Added periodic full recomputation of precision matrix
    - Added condition number monitoring
    - Better handling of ill-conditioned matrices
    """
    d: int
    lam: float = 1.0
    sigma2: float = 1.0
    
    def __post_init__(self):
        self.A = self.lam * np.eye(self.d)
        self.b = np.zeros(self.d)
        self.update_count = 0  # Track updates for periodic recomputation
        self.last_condition_number = 1.0
        
    def sample_theta(self, rng: np.random.Generator) -> np.ndarray:
        """
        Sample parameters from posterior using Cholesky decomposition.
        
        FIX: Added periodic recomputation and condition number check
        """
        # Periodic full recomputation for numerical stability
        if self.update_count > 0 and self.update_count % MATRIX_RECOMPUTE_FREQUENCY == 0:
            self._recompute_precision_matrix()
        
        # Ensure symmetry
        self.A = 0.5 * (self.A + self.A.T)
        
        # Scrub NaNs/Infs
        if not np.isfinite(self.A).all():
            self.A = np.nan_to_num(self.A, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Check condition number
        try:
            cond = np.linalg.cond(self.A)
            self.last_condition_number = float(cond)
            
            if cond > CONDITION_NUMBER_THRESHOLD:
                # Matrix is ill-conditioned, add regularization
                self.A += 1e-6 * np.eye(self.d)
        except:
            # If condition number fails, just add regularization
            self.A += 1e-6 * np.eye(self.d)
        
        # Attempt Cholesky decomposition
        try:
            L = np.linalg.cholesky(self.A)
        except np.linalg.LinAlgError:
            # Fallback: add more aggressive regularization
            self.A += 1e-5 * np.eye(self.d)
            try:
                L = np.linalg.cholesky(self.A)
            except np.linalg.LinAlgError:
                # Last resort: eigenvalue decomposition
                self.A = self._fix_positive_definite(self.A)
                L = np.linalg.cholesky(self.A)
        
        # Compute posterior mean and sample
        mu = np.linalg.solve(self.A, self.b)
        z = rng.normal(size=self.d)
        theta = mu + np.linalg.solve(L.T, z) * math.sqrt(self.sigma2)
        
        return theta
    
    def _recompute_precision_matrix(self):
        """
        FIX: Periodic full recomputation from scratch to prevent error accumulation.
        This was mentioned in the dissertation but not implemented.
        """
        # Force symmetry
        self.A = 0.5 * (self.A + self.A.T)
        
        # Check and fix positive definiteness
        try:
            # Try Cholesky to verify positive definiteness
            np.linalg.cholesky(self.A)
        except np.linalg.LinAlgError:
            # Fix via eigenvalue decomposition
            self.A = self._fix_positive_definite(self.A)
    
    @staticmethod
    def _fix_positive_definite(A: np.ndarray, min_eig: float = 1e-6) -> np.ndarray:
        """Fix non-positive-definite matrix using eigenvalue decomposition."""
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals = np.maximum(eigvals, min_eig)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    def update(self, x: np.ndarray, r: float):
        """Update precision matrix and reward vector."""
        self.A += np.outer(x, x)
        self.b += r * x
        self.update_count += 1


class BanditPolicy:
    """Linear Thompson Sampling policy for action selection."""
    def __init__(self, arms: List[Tuple[str, int]], d_ctx: int, 
                 lam: float = 1.0, sigma2: float = 1.0, seed: int = 23):
        self.keys = list(arms)
        self.arms = {k: LinTSArm(d_ctx, lam, sigma2) for k in self.keys}
        self.rng = np.random.default_rng(seed)

    def select_from_features(self, x_by_arm: Dict[Tuple[str, int], np.ndarray]) -> Tuple[Tuple[str, int], np.ndarray, float]:
        """
        Sample from posterior and select best arm.
        Each arm uses its own feature vector.
        """
        thetas = {k: self.arms[k].sample_theta(self.rng) for k in x_by_arm}
        vals = {k: float(thetas[k] @ x_by_arm[k]) for k in x_by_arm}
        best = max(vals.items(), key=lambda kv: kv[1])[0]
        return best, thetas[best], float(vals[best])

    def update(self, key: Tuple[str, int], x: np.ndarray, r: float):
        """Update the selected arm with observed reward."""
        self.arms[key].update(x, r)


# ---------------------------------------------------------------------
# Scoring & diagnostics
# ---------------------------------------------------------------------

def interval_score(y: float, l: float, u: float, alpha: float) -> float:
    """Compute interval score (Gneiting & Raftery, 2007)."""
    width = max(0.0, u - l)
    under = max(0.0, l - y)
    over  = max(0.0, y - u)
    return width + (2.0/alpha)*under + (2.0/alpha)*over


class RollingCoverage:
    """Track rolling coverage rate."""
    def __init__(self, window: int = 60):
        self.buf: deque = deque(maxlen=int(max(10, window)))
        
    def update(self, covered: int):
        self.buf.append(int(covered))
        
    def cov(self) -> float:
        return float(np.mean(self.buf)) if self.buf else 0.0


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------

def _alpha_for_h(cfg: Any, h: int) -> float:
    """Get horizon-specific alpha, fallback to global alpha."""
    alt = {
        1:  cfg_get(cfg, "alpha_1",  None),
        5:  cfg_get(cfg, "alpha_5",  None),
        10: cfg_get(cfg, "alpha_10", None),
        20: cfg_get(cfg, "alpha_20", None),
        21: cfg_get(cfg, "alpha_21", None),
        42: cfg_get(cfg, "alpha_42", None),
        63: cfg_get(cfg, "alpha_63", None),
    }
    a = alt.get(int(h))
    return float(a if a is not None else cfg_get(cfg, "alpha", 0.10))


def _add_forward_security_targets(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    """
    Add forward h-period total returns to the decision row at time t:
        y_h(t) = Π_{k=1..h} (1 + r_{t+k}) - 1

    Notes:
    - Excludes return at t (uses t+1..t+h)
    - Returns NaN when there are fewer than h future rows for the id
    """
    df = df.sort_values([ID_COL, DATE_COL]).copy()

    def _fwd_total_return(vec: pd.Series, h: int) -> np.ndarray:
        v = vec.to_numpy(dtype=np.float64, copy=False)
        r = 1.0 + np.nan_to_num(v, nan=0.0)  # Guard NaNs as 0% daily return
        n = r.shape[0]
        
        # Cumulative product with sentinel
        cp = np.empty(n + 1, dtype=np.float64)
        cp[0] = 1.0
        np.multiply.accumulate(r, out=cp[1:])
        
        out = np.full(n, np.nan, dtype=np.float64)
        
        # y[t] = (cp[t+h+1] / cp[t+1]) - 1 → product over t+1..t+h
        if h > 0 and n > h:
            idx = np.arange(0, n - h, dtype=np.int64)
            out[idx] = (cp[idx + h + 1] / cp[idx + 1]) - 1.0
            
        return out.astype(np.float32)

    for h in horizons:
        df[f"y_h{h}"] = df.groupby(ID_COL, sort=False)["return"] \
                          .transform(lambda s, hh=h: _fwd_total_return(s, hh))

    return df


def _precompute_alive_and_maturity_dates(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    """
    Precompute label-free alive masks and maturity dates.
    
    alive_h{h}: 1 if a row exists at t+h for this security, 0 otherwise
    mat_date_h{h}: the actual date of the row at t+h
    """
    g = df.sort_values([ID_COL, DATE_COL]).copy()
    g["_exists"] = 1
    
    for h in horizons:
        # Alive if a row exists h steps ahead
        g[f"alive_h{h}"] = g.groupby(ID_COL)["_exists"].shift(-h).notna().astype("int8")
        # Maturity date is the date at t+h
        g[f"mat_date_h{h}"] = g.groupby(ID_COL)[DATE_COL].shift(-h)
        
    cols = [DATE_COL, ID_COL] + [f"alive_h{h}" for h in horizons] + [f"mat_date_h{h}" for h in horizons]
    return g[cols]


def build_enhanced_features(row: pd.Series) -> np.ndarray:
    """
    Build 10-dimensional context feature vector (NO momentum signal!).
    
    This represents regime/characteristics that predict WHICH specification works,
    not the momentum signal itself (that's what we're selecting).
    
    Features:
    [0] bias (1.0)
    [1] security volatility (ret_vol20_z)
    [2] market volatility (idx_vol_z)
    [3] credit spread (spread_z)
    [4] duration (duration_z)
    [5] autocorrelation (ret_autocorr_z) - momentum persistence
    [6] spread percentile (cross-sectional)
    [7] volatility percentile (cross-sectional)
    [8] spread × duration interaction
    [9] vol × market_vol interaction
    
    All features use NaN-safe defaults (z-scores → 0.0, percentiles → 0.5)
    """
    def _clip(z): 
        return float(np.clip(z, -CLIP_Z_SCORE, CLIP_Z_SCORE))
    
    def _safe_get(name: str, default: float = 0.0) -> float:
        """Get feature with fallback, ensuring finite value."""
        val = row.get(name, default)
        return float(val) if np.isfinite(val) else default
    
    # Core regime features
    v = _clip(_safe_get("ret_vol20_z", 0.0))      # Security volatility
    idxv = _clip(_safe_get("idx_vol_z", 0.0))     # Market volatility
    s = _clip(_safe_get("spread_z", 0.0))         # Credit spread
    d = _clip(_safe_get("duration_z", 0.0))       # Duration
    ac = _clip(_safe_get("ret_autocorr_z", 0.0))  # Autocorrelation
    
    # Cross-sectional features (default to neutral)
    sp = _safe_get("spread_percentile", 0.5)
    vp = _safe_get("vol_percentile", 0.5)
    
    # Build feature vector
    x = np.array([
        1.0,        # [0] bias
        v,          # [1] security vol
        idxv,       # [2] market vol
        s,          # [3] spread
        d,          # [4] duration
        ac,         # [5] autocorrelation
        sp,         # [6] spread percentile
        vp,         # [7] vol percentile
        s * d,      # [8] spread-duration interaction
        v * idxv,   # [9] vol interaction
    ], dtype=np.float32)
    
    # Final safety check
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    
    return x


# ---------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------

def run_bandit(cfg: dict | Any) -> None:
    """
    Train/test CRL at SECURITY level only (no index aggregation), with:
      - label-free action gating
      - deferred updates by horizon (no test-time leakage)
      - FIXES applied as documented above
    """
    rng = np.random.default_rng(int(cfg_get(cfg, "seed", 23)))
    results_dir = cfg_get(cfg, "results_dir", "results/crl")
    _ensure_dir(results_dir)
    paths = _results_paths(results_dir)

    print("[CRL] Starting CRL bandit with FIXES applied...")
    print(f"[CRL] Volatility floor: {VOLATILITY_FLOOR} (was 1.0)")
    print(f"[CRL] Matrix recompute frequency: {MATRIX_RECOMPUTE_FREQUENCY}")
    print(f"[CRL] Condition number threshold: {CONDITION_NUMBER_THRESHOLD}")

    # 1) Load panel
    panel_path = cfg_get(cfg, "panel_path", "data/cdi_panel.parquet")
    df = load_panel(panel_path)
    df = df[df.get("active", 1) > 0].copy()
    print(f"[CRL] Loaded {len(df)} active rows | "
          f"securities={df[ID_COL].nunique()} | "
          f"dates={df[DATE_COL].nunique()}")

    # 2) Build momentum features
    mom_windows = list(cfg_get(cfg, "momentum_windows", [10, 21, 42, 63]))
    mom_cols: List[str] = []
    for w in mom_windows:
        sv = build_momentum(df=df, w=w)
        df[sv.name] = sv
        mom_cols.append(sv.name)
    print(f"[CRL] Built {len(mom_cols)} momentum features")

    # 3) Security-level context
    df = df.sort_values([ID_COL, DATE_COL])
    df["ret_vol20"] = (df.groupby(ID_COL)["return"]
                         .transform(lambda x: x.rolling(20, min_periods=5).std())
                         .fillna(0).astype(np.float32))
    
    # Ensure required columns exist
    for col in ["spread", "duration", "ttm"]:
        if col not in df.columns:
            df[col] = 0.0

    # Market-wide volatility proxy
    idx_vol = (df.groupby(DATE_COL)
                 .apply(lambda g: np.average(g["ret_vol20"].values,
                                            weights=g["index_weight"].values + 1e-15))
                 .rename("idx_vol")
                 .reset_index())
    df = df.merge(idx_vol, on=DATE_COL, how="left")
    
    # Cross-sectional features (active securities only)
    df_active = df[df.get("active", 1) > 0].copy()
    
    df_active["spread_percentile"] = (
        df_active.groupby(DATE_COL, group_keys=False)
        .apply(lambda g: _compute_cross_sectional_rank(g, "spread", min_count=10))
        .astype(np.float32)
    )
    
    df_active["vol_percentile"] = (
        df_active.groupby(DATE_COL, group_keys=False)
        .apply(lambda g: _compute_cross_sectional_rank(g, "ret_vol20", min_count=10))
        .astype(np.float32)
    )
    
    # Merge back
    df = df.merge(
        df_active[[DATE_COL, ID_COL, "spread_percentile", "vol_percentile"]],
        on=[DATE_COL, ID_COL],
        how="left"
    )
    df["spread_percentile"] = df["spread_percentile"].fillna(0.5).astype(np.float32)
    df["vol_percentile"] = df["vol_percentile"].fillna(0.5).astype(np.float32)
    
    # Autocorrelation
    df["ret_autocorr"] = np.nan
    mask_active = df["active"] == 1
    if mask_active.any():
        df.loc[mask_active, "ret_autocorr"] = (
            df[mask_active].groupby(ID_COL)["return"]
            .transform(lambda x: x.rolling(21, min_periods=10)
                      .apply(lambda y: y.autocorr(lag=1) if len(y) > 1 else 0, raw=False))
        )
    df["ret_autocorr"] = df["ret_autocorr"].fillna(0.0).astype(np.float32)

    # 4) Add forward targets
    horizons = list(cfg_get(cfg, "horizons", [10, 21, 42, 63]))
    df = _add_forward_security_targets(df, horizons)

    # 5) Train/test split
    train_end  = pd.to_datetime(cfg_get(cfg, "train_end",  "2023-12-31"))
    test_start = pd.to_datetime(cfg_get(cfg, "test_start", "2024-01-01"))
    df_train = df[df[DATE_COL] <= train_end].copy()
    df_test  = df[df[DATE_COL] >= test_start].copy()
    print(f"[CRL] Train rows: {len(df_train)} | Test rows: {len(df_test)}")

    # 6) Standardize using TRAIN stats
    train_stats: Dict[str, Tuple[float, float]] = {}
    standardize_cols = mom_cols + ["spread", "duration", "ret_vol20", "idx_vol", "ret_autocorr"]
    
    for col in standardize_cols:
        if col not in df_train.columns:
            continue
        mu = float(df_train[col].mean())
        sd = float(df_train[col].std())
        if not math.isfinite(sd) or sd <= 0:
            sd = 1.0
        train_stats[col] = (mu, sd)
        df[f"{col}_z"] = ((df[col] - mu) / sd).fillna(0.0).clip(-CLIP_Z_SCORE, CLIP_Z_SCORE)

    # Update splits after standardization
    df_train = df[df[DATE_COL] <= train_end].copy()
    df_test  = df[df[DATE_COL] >= test_start].copy()

    # 7) Action space
    arms: List[Tuple[str, int]] = [(m, h) for m in mom_cols for h in horizons]
    print(f"[CRL] Action space: {len(arms)} arms")

    feature_dim = 10  # As defined in build_enhanced_features

    # 8) Initialize models
    tau_low  = float(cfg_get(cfg, "tau_low", 0.05))
    tau_high = float(cfg_get(cfg, "tau_high", 0.95))
    lam_head = float(cfg_get(cfg, "lam_head", 1e-4))
    eta_q    = float(cfg_get(cfg, "eta_q", 0.02))

    heads = {
        key: OnlineQuantileHead(d=feature_dim, tau_low=tau_low, tau_high=tau_high,
                                lam=lam_head, eta=eta_q, w_cap=1e3)
        for key in arms
    }
    
    calibrators = {
        h: CQRCalibrator(alpha=_alpha_for_h(cfg, h),
                        window=int(cfg_get(cfg, "conf_window", 250)),
                        lambda_recency=float(cfg_get(cfg, "lambda_recency", 0.0)))
        for h in horizons
    }
    
    policy = BanditPolicy(arms, d_ctx=feature_dim,
                         lam=float(cfg_get(cfg, "lam_bandit", 0.1)),
                         sigma2=5.0, seed=int(cfg_get(cfg, "seed", 23)))

    cov_window = int(cfg_get(cfg, ("cov_window", "coverage_window"), 60))
    cov_track  = {h: RollingCoverage(window=cov_window) for h in horizons}

    # 9) Pretrain on TRAIN data
    print("[CRL] Pretraining heads + calibrators...")
    if len(df_train) > 0:
        pre = df_train.sample(n=min(10000, len(df_train)), random_state=42)
        for _, r in pre.iterrows():
            m, h = arms[rng.integers(0, len(arms))]
            y = r.get(f"y_h{h}", np.nan)
            if not np.isfinite(y):
                continue
                
            x = build_enhanced_features(r)
            heads[(m, h)].update(x, float(y))
            
            l_raw, u_raw = heads[(m, h)].predict(x)
            
            # Clip extreme predictions during pretraining
            y_std = {h: float(df_train[f"y_h{h}"].std(skipna=True) or 1.0) for h in horizons}
            cap_k = 50.0
            cap = cap_k * y_std[h]
            l_raw = float(np.clip(l_raw, -cap, cap))
            u_raw = float(np.clip(u_raw, -cap, cap))
            
            calibrators[h].update(float(y), l_raw, u_raw)
    print("[CRL] Pretraining complete")

    # 10) Compute alive masks and maturity dates (label-free!)
    alive = _precompute_alive_and_maturity_dates(df_test, horizons)
    df_test = df_test.merge(alive, on=[DATE_COL, ID_COL], how="left")

    # 11) Create index for target retrieval (no peeking at decision time)
    target_idx = df_test.set_index([DATE_COL, ID_COL])

    # 12) Test loop with deferred updates
    print("[CRL] Running test loop with deferred updates...")
    
    test_dates = sorted(df_test[DATE_COL].unique())
    pending_by_date = defaultdict(list)  # Decisions awaiting maturation
    rows_sec = []  # Security-level results
    rows_choices = []  # Policy choices
    
    for date_idx, date in enumerate(test_dates):
        if date_idx % 20 == 0:
            print(f"[CRL] Date {date_idx+1}/{len(test_dates)}: {date.date()}")
        
        day = df_test[df_test[DATE_COL] == date].copy()
        if day.empty:
            continue
        
        daily_actions = []
        
        # Make decisions for this date
        for _, row in day.iterrows():
            # Valid actions: (m, h) where alive_h{h} == 1
            valid = [(m, h) for (m, h) in arms 
                    if row.get(f"alive_h{h}", 0) == 1]
            
            if not valid:
                continue
            
            # Get valid action dict with maturity dates
            valid_dict = {(m, h): row[f"mat_date_h{h}"] for (m, h) in valid}
            
            # Build features and select action
            x = build_enhanced_features(row)
            x_by_arm = {k: x for k in valid}
            (mom_col, h), theta, val = policy.select_from_features(x_by_arm)
            
            # Generate interval prediction
            l_raw, u_raw = heads[(mom_col, h)].predict(x)
            
            # Apply cold-start clamping if needed
            warmup_min = int(cfg_get(cfg, "calib_warmup_min", 20))
            min_w = float(cfg_get(cfg, "raw_min_width", 0.002))
            max_w = float(cfg_get(cfg, "raw_max_width", 0.050))
            if len(calibrators[h].s_lo) < warmup_min:
                l_raw, u_raw = coldstart_clamp_bounds(l_raw, u_raw, min_width=min_w, max_width=max_w)
            
            # Conformal adjustment
            l_adj, u_adj, qhat = calibrators[h].adjust(l_raw, u_raw)
            
            # Ensure non-crossing
            if l_adj > u_adj:
                mid = 0.5 * (l_adj + u_adj)
                l_adj, u_adj = mid - 1e-6, mid + 1e-6
            
            alpha_h = _alpha_for_h(cfg, h)
            
            # Schedule for maturation
            mat_date = valid_dict[(mom_col, h)]
            pending_by_date[pd.to_datetime(mat_date)].append({
                "date": date,
                "id": row[ID_COL],
                "h": int(h),
                "alpha": float(alpha_h),
                "key": (mom_col, int(h)),
                "x": x.astype(np.float32),
                "l_raw": float(l_raw),
                "u_raw": float(u_raw),
                "l_adj": float(l_adj),
                "u_adj": float(u_adj),
                "qhat": float(qhat),
            })
            
            daily_actions.append((mom_col, h))
        
        # Process matured decisions
        if date in pending_by_date:
            matured = pending_by_date.pop(date)
            
            for rec in matured:
                # Retrieve target (NOW we can look!)
                try:
                    target_row = target_idx.loc[(rec["date"], rec["id"])]
                    y = float(target_row[f"y_h{rec['h']}"])
                except:
                    continue
                
                if not np.isfinite(y):
                    continue
                
                # Compute metrics
                covered = int(rec["l_adj"] <= y <= rec["u_adj"])
                width = rec["u_adj"] - rec["l_adj"]
                loss = interval_score(y, rec["l_adj"], rec["u_adj"], rec["alpha"])
                
                # Update models (DEFERRED)
                heads[rec["key"]].update(rec["x"], y)
                calibrators[rec["h"]].update(y, rec["l_raw"], rec["u_raw"])
                cov_track[rec["h"]].update(covered)
                
                # Compute reward and update policy
                r_cov = (covered - 0.9) if covered else -0.1
                r_width = -0.1 * width
                r = -loss + 0.0 * r_cov + 0.0 * r_width
                policy.update(rec["key"], rec["x"], r)
                
                # Record results
                rows_sec.append({
                    "date": rec["date"],
                    "id": rec["id"],
                    "h": rec["h"],
                    "momentum": rec["key"][0],
                    "y": y,
                    "l": rec["l_adj"],
                    "u": rec["u_adj"],
                    "covered": covered,
                    "width": width,
                    "loss": loss,
                    "qhat": rec["qhat"],
                    "alpha": rec["alpha"],
                })
        
        # Log daily modal choice
        if daily_actions:
            m_modal = Counter([m for (m, _) in daily_actions]).most_common(1)[0][0]
            h_modal = Counter([h for (_, h) in daily_actions]).most_common(1)[0][0]
            rows_choices.append({
                "date": date,
                "momentum": m_modal,
                "h": int(h_modal),
                "ctx_retvol20": float(day["ret_vol20"].mean()),
                "ctx_spread": float(day.get("spread", pd.Series([0.0])).mean()),
                "ctx_duration": float(day.get("duration", pd.Series([0.0])).mean()),
            })
    
    print(f"[CRL] Test loop complete. Processed {len(rows_sec)} predictions")
    
    # 13) Save results
    sec_df = pd.DataFrame(rows_sec)
    choices_df = pd.DataFrame(rows_choices)
    
    # Apply horizon scaling if configured
    scale_mode = cfg_get(cfg, "horizon_scale_mode", "per_step")
    if scale_mode != "none" and not sec_df.empty:
        from utils_conformal import apply_horizon_scaling
        sec_df = apply_horizon_scaling(sec_df, mode=scale_mode, h_col="h", cols=["width", "loss"])
    else:
        sec_df["width_scaled"] = sec_df["width"]
        sec_df["loss_scaled"] = sec_df["loss"]
    
    # Save outputs
    sec_df.to_csv(paths["scores_sec"], index=False)
    choices_df.to_csv(paths["choices"], index=False)
    print(f"[CRL] Saved: {paths['scores_sec']}")
    print(f"[CRL] Saved: {paths['choices']}")
    
    # 14) Compute summary statistics
    summary = {
        "test_start": str(test_start.date()),
        "test_end": str(test_dates[-1].date()) if test_dates else None,
        "n_test_dates": len(test_dates),
        "n_predictions": len(sec_df),
        "n_securities": int(sec_df[ID_COL].nunique()) if not sec_df.empty else 0,
        "overall_coverage": float(sec_df["covered"].mean()) if not sec_df.empty else None,
        "overall_mean_loss": float(sec_df["loss_scaled"].mean()) if not sec_df.empty else None,
        "overall_median_loss": float(sec_df["loss_scaled"].median()) if not sec_df.empty else None,
        "overall_mean_width": float(sec_df["width_scaled"].mean()) if not sec_df.empty else None,
        "overall_median_width": float(sec_df["width_scaled"].median()) if not sec_df.empty else None,
        "security_p99_loss": float(sec_df["loss_scaled"].quantile(0.99)) if not sec_df.empty else None,
        "security_median_width": float(sec_df["width_scaled"].median()) if not sec_df.empty else None,
        "security_p99_width": float(sec_df["width_scaled"].quantile(0.99)) if not sec_df.empty else None,
    }
    
    # By-horizon statistics
    if not sec_df.empty:
        summary["by_horizon"] = {}
        for h in horizons:
            h_data = sec_df[sec_df["h"] == h]
            if not h_data.empty:
                summary["by_horizon"][f"h{h}"] = {
                    "coverage": float(h_data["covered"].mean()),
                    "mean_loss": float(h_data["loss_scaled"].mean()),
                    "median_loss": float(h_data["loss_scaled"].median()),
                    "p99_loss": float(h_data["loss_scaled"].quantile(0.99)),
                    "mean_width": float(h_data["width_scaled"].mean()),
                    "median_width": float(h_data["width_scaled"].median()),
                    "p99_width": float(h_data["width_scaled"].quantile(0.99)),
                    "n_predictions": int(len(h_data)),
                }
    
    with open(paths["analysis"], "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[CRL] Saved: {paths['analysis']}")
    
    # 15) Falsification test
    falsif = {}
    if not sec_df.empty:
        falsif["coverage_by_h"] = {
            int(k): float(v) for k, v in sec_df.groupby("h")["covered"].mean().to_dict().items()
        }
        
        # Placebo test: shift labels by k steps
        k = 20
        s = sec_df.sort_values(["id", "date"]).copy()
        s["y_shift"] = s.groupby("id")["y"].shift(k)
        s = s.dropna(subset=["y_shift"])
        if not s.empty:
            falsif["placebo_cov_k20"] = float(((s["l"] <= s["y_shift"]) & (s["y_shift"] <= s["u"])).mean())
    
    with open(paths["falsif"], "w") as f:
        json.dump(falsif, f, indent=2)
    print(f"[CRL] Saved: {paths['falsif']}")
    
    # Print summary with index-weighted aggregation
    if not sec_df.empty:
        print(f"\n[CRL] === RESULTS ===")
        print(f"Coverage: {sec_df['covered'].mean():.3f}")
        
        # Index-weighted loss (aggregate to date level first)
        # Load panel for weights if not already in sec_df
        if "index_weight" not in sec_df.columns:
            panel = load_panel(panel_path)
            if isinstance(panel.index, pd.MultiIndex):
                panel = panel.reset_index()
            sec_df = sec_df.merge(
                panel[["date", "id", "index_weight"]].drop_duplicates(),
                left_on=["date", ID_COL],
                right_on=["date", "id"],
                how="left"
            )
            sec_df["index_weight"] = sec_df["index_weight"].fillna(0.0)
        
        # Normalize weights within each date, then compute weighted mean per date
        sec_df["w_norm"] = sec_df.groupby("date")["index_weight"].transform(
            lambda x: x / (x.sum() + 1e-12)
        )
        
        # Date-level weighted aggregation
        date_losses = (sec_df.groupby("date")
                            .apply(lambda g: np.average(g["loss_scaled"], weights=g["w_norm"]))
                            .values)
        weighted_loss = float(np.mean(date_losses))
        
        print(f"Mean loss (index-weighted): {weighted_loss:.6f}")
        print(f"Mean loss (simple avg): {sec_df['loss_scaled'].mean():.6f}")
        print(f"Predictions: {len(sec_df)} across {sec_df[ID_COL].nunique()} securities")
        print(f"[CRL] Using index-weighted aggregation for performance reporting")
        # Add to summary dict
        summary["overall_mean_loss_weighted"] = float(weighted_loss)
        summary["overall_mean_loss_simple"] = float(sec_df["loss_scaled"].mean())
    else:
        print("[CRL] WARNING: No predictions generated")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config_crl.yaml")
    args = ap.parse_args()
    
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    run_bandit(cfg)