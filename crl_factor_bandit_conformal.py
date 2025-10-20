# crl_factor_bandit_conformal.py
# -*- coding: utf-8 -*-
"""
CRL factor-spec selection with Conformalized Quantile Regression (CQR)
SECURITY-LEVEL ONLY — **leakage-safe, horizon-deferred updates**

What’s new vs your previous version (same public artifacts):
- **Deferred updates**: bandit reward, quantile heads, and conformal calibrators
  update only when the decision’s outcome matures (at date t+h). Coverage
  tracking is also based solely on matured observations.
- **Label-free action set**: at time t, valid horizons are determined by a
  precomputed, label-free "alive" mask (presence of a row at t+h for the same
  security), not by checking y_h at t.
- **Identical outputs & API**: writes the same files and uses the same config
  keys used by run_experiments.step_crl.

This replaces the immediate test-time updates and y_h-gated action set that
were inflating/flattening results. See your original code around the test loop
for the behaviors being fixed.  # (context)  :contentReference[oaicite:4]{index=4}
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
    if not os.path.exists(panel_path):
        raise FileNotFoundError(panel_path)
    if panel_path.endswith(".pkl"):
        df = pd.read_pickle(panel_path)
    elif panel_path.endswith(".parquet"):
        df = pd.read_parquet(panel_path)
    else:
        df = pd.read_csv(panel_path)

    # flatten MultiIndex
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    df.columns = [str(c).strip() for c in df.columns]
    for src, dst in _SCHEMA_SYNONYMS.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    if DATE_COL not in df.columns:
        raise KeyError("Panel must contain 'date'.")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    if ID_COL not in df.columns:
        raise KeyError("Panel must contain 'id' (rename debenture_id→id upstream).")

    # Required core fields
    missing = [c for c in ["return", "index_weight"] if c not in df.columns]
    if missing:
        raise KeyError(f"Panel missing required columns: {missing}")

    # If index_return missing, derive once (weight-avg per date)
    if "index_return" not in df.columns:
        tmp = (df.groupby(DATE_COL)
                 .apply(lambda g: np.average(g["return"].values,
                                             weights=g["index_weight"].values + 1e-15))
                 .rename("index_return")
                 .reset_index())
        df = df.merge(tmp, on=DATE_COL, how="left")

    return df.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)

# ---------------------------------------------------------------------
# Factor construction
# ---------------------------------------------------------------------

def _compute_cross_sectional_rank(group: pd.DataFrame, col: str, min_count: int = 10) -> pd.Series:
    """
    Compute percentile rank within a date, handling NaNs and requiring minimum count.
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
    """Per-security time-series momentum, lagged to avoid look-ahead."""
    pdf = df[[DATE_COL, ID_COL, "return"]].copy()
    pdf["ret_lag"] = pdf.groupby(ID_COL)["return"].shift(1)
    # rolling sum of lagged returns
    mom_raw = pdf.groupby(ID_COL)["ret_lag"].rolling(w, min_periods=5).sum().reset_index(level=0, drop=True)

    volw = max(20, w)
    vol = pdf.groupby(ID_COL)["return"].rolling(volw, min_periods=5).std().reset_index(level=0, drop=True)
    vol = vol.groupby(pdf[ID_COL]).shift(1)  # lag std
    den = vol.replace(0, np.nan).fillna(1.0)  # clamp tiny/NaN to 1
    mom = mom_raw / (den + 1e-8)

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
        # With small random initialization:
        rng = np.random.default_rng(23)  # Use a fixed seed for reproducibility
        self.w_lo = rng.normal(0, 0.01, d)  # Small random weights
        self.w_hi = rng.normal(0, 0.01, d)

        # Ensure initial predictions have reasonable spread
        self.w_lo[0] -= 0.001  # Bias term for lower quantile
        self.w_hi[0] += 0.001  # Bias term for upper quantile
        self.w_cap = float(w_cap)

    @staticmethod
    def _grad_yhat(y_hat: float, y: float, tau: float) -> float:
        return (1.0 - tau) if (y < y_hat) else (-tau)

    def update(self, x: np.ndarray, y: float):
        # low quantile
        y_lo = float(self.w_lo @ x)
        g = self._grad_yhat(y_lo, y, self.tau_low)
        self.w_lo = (1 - self.eta * self.lam) * self.w_lo - self.eta * g * x

        # high quantile
        y_hi = float(self.w_hi @ x)
        g = self._grad_yhat(y_hi, y, self.tau_high)
        self.w_hi = (1 - self.eta * self.lam) * self.w_hi - self.eta * g * x

        # clip & enforce non-crossing softly
        self.w_lo = np.clip(self.w_lo, -self.w_cap, self.w_cap)
        self.w_hi = np.clip(self.w_hi, -self.w_cap, self.w_cap)
        l, u = self.predict(x)
        if l > u:
            mid = 0.5 * (l + u)
            self.w_lo += 0.1 * (mid - y_lo) * x
            self.w_hi += 0.1 * (y_hi - mid) * x

    def predict(self, x: np.ndarray) -> Tuple[float, float]:
        l = float(self.w_lo @ x)
        u = float(self.w_hi @ x)
        if l > u:
            l, u = min(l, u), max(l, u)
        return l, u


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    order = np.argsort(v)
    v, w = v[order], np.clip(w[order], 0.0, np.inf)
    cw = np.cumsum(w); total = cw[-1]
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
        e_lo = max(l_pred - y, 0.0)
        e_hi = max(y - u_pred, 0.0)
        self.s_lo.append(e_lo)
        self.s_hi.append(e_hi)

    def adjust(self, l_pred: float, u_pred: float) -> Tuple[float, float, float]:
        """Return (l_adj, u_adj, qhat). Uses single-score CQR (max residual)."""
        if len(self.s_lo) < 20:
            return l_pred, u_pred, 0.0

        s_lo = np.array(self.s_lo, float)
        s_hi = np.array(self.s_hi, float)
        s = np.maximum(s_lo, s_hi)

        if self.lambda_recency > 0:
            ages = np.arange(len(s), 0, -1, dtype=float)
            w = np.exp(-self.lambda_recency * ages)
        else:
            w = np.ones_like(s, float)

        q = _weighted_quantile(s, w, 1.0 - self.alpha)
        l = l_pred - q
        u = u_pred + q
        if l > u:
            mid = 0.5 * (l + u); l = u = mid
        return l, u, float(q)

# ---------------------------------------------------------------------
# Bandit policy (LinTS)
# ---------------------------------------------------------------------

@dataclass
class LinTSArm:
    d: int
    lam: float = 1.0
    sigma2: float = 1.0
    def __post_init__(self):
        self.A = self.lam * np.eye(self.d)
        self.b = np.zeros(self.d)
    def sample_theta(self, rng: np.random.Generator) -> np.ndarray:
        # before Cholesky in LinTSArm.sample_theta(...)
        self.A = 0.5 * (self.A + self.A.T)                      # re-symmetrize
        if not np.isfinite(self.A).all():                       # scrub NaNs/Infs
            self.A = np.nan_to_num(self.A, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            L = np.linalg.cholesky(self.A)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(self.A + 1e-6*np.eye(self.d))
        mu = np.linalg.solve(self.A, self.b)
        z = rng.normal(size=self.d)
        theta = mu + np.linalg.solve(L.T, z) * math.sqrt(self.sigma2)
        return theta
    def update(self, x: np.ndarray, r: float):
        self.A += np.outer(x, x)
        self.b += r * x

class BanditPolicy:
    def __init__(self, arms: List[Tuple[str, int]], d_ctx: int, lam: float = 1.0, sigma2: float = 1.0, seed: int = 23):
        self.keys = list(arms)
        self.arms = {k: LinTSArm(d_ctx, lam, sigma2) for k in self.keys}
        self.rng = np.random.default_rng(seed)

    def select_from_features(self, x_by_arm: Dict[Tuple[str, int], np.ndarray]) -> Tuple[Tuple[str, int], np.ndarray, float]:
        """Arm-specific selection: each arm scored with its own feature vector."""
        thetas = {k: self.arms[k].sample_theta(self.rng) for k in x_by_arm}
        vals = {k: float(thetas[k] @ x_by_arm[k]) for k in x_by_arm}
        best = max(vals.items(), key=lambda kv: kv[1])[0]
        return best, thetas[best], float(vals[best])

    def update(self, key: Tuple[str, int], x: np.ndarray, r: float):
        self.arms[key].update(x, r)

# ---------------------------------------------------------------------
# Scoring & diagnostics
# ---------------------------------------------------------------------

def interval_score(y: float, l: float, u: float, alpha: float) -> float:
    width = max(0.0, u - l)
    under = max(0.0, l - y)
    over  = max(0.0, y - u)
    return width + (2.0/alpha)*under + (2.0/alpha)*over

class RollingCoverage:
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
    Add forward *h-period total returns* to the decision row at time t:
        y_h(t) = Π_{k=1..h} (1 + r_{t+k}) - 1

    Notes
    -----
    - Excludes return at t (uses t+1..t+h).
    - Returns NaN when there are fewer than h future *rows* for the id,
      which is correct: those decisions should not be scheduled to mature.
    """
    df = df.sort_values([ID_COL, DATE_COL]).copy()

    def _fwd_total_return(vec: pd.Series, h: int) -> np.ndarray:
        v = vec.to_numpy(dtype=np.float64, copy=False)
        r = 1.0 + np.nan_to_num(v, nan=0.0)         # guard NaNs as 0% daily return
        n = r.shape[0]
        # cumulative product with a 1 sentinel (cp[k] = Π_{i=0..k-1} r[i])
        cp = np.empty(n + 1, dtype=np.float64)
        cp[0] = 1.0
        np.multiply.accumulate(r, out=cp[1:])
        out = np.full(n, np.nan, dtype=np.float64)
        # y[t] = (cp[t+h+1] / cp[t+1]) - 1  → product over t+1..t+h
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
    Precompute, label-free, whether a security observed at (id,t) will also be
    observed h steps ahead (alive mask), and the actual maturity DATE for that
    h-step ahead observation (to schedule deferred updates).
    This uses groupby-shifts on a constant 'exists' flag (NOT the label y_h). 
    """
    g = df.sort_values([ID_COL, DATE_COL]).copy()
    g["_exists"] = 1
    for h in horizons:
        # alive if a row exists h steps ahead in the same id group
        g[f"alive_h{h}"] = g.groupby(ID_COL)["_exists"].shift(-h).notna().astype("int8")
        # maturity date is simply the date of the row at t+h (None if not alive)
        g[f"mat_date_h{h}"] = g.groupby(ID_COL)[DATE_COL].shift(-h)
    cols = [DATE_COL, ID_COL] + [f"alive_h{h}" for h in horizons] + [f"mat_date_h{h}" for h in horizons]
    return g[cols]

# Public feature builder (exported for reuse, e.g., OPE)
def build_enhanced_features(row: pd.Series) -> np.ndarray:
    """
    Context feature vector for bandit (NO momentum signal - that's what we're selecting!).
    Returns regime/characteristic features that predict WHICH momentum specification works.
    
    Features (10-D):
    - Bias term (1.0)
    - Volatility regime: ret_vol20_z, idx_vol_z
    - Credit characteristics: spread_z, duration_z
    - Autocorrelation: ret_autocorr_z (momentum persistence)
    - Cross-sectional position: spread_percentile, vol_percentile
    - Interactions: spread*duration, vol*idx_vol, vol*spread
    
    All features use NaN-safe defaults:
    - z-scores default to 0.0 (mean)
    - percentiles default to 0.5 (median/neutral)
    """
    def _clip(z): return float(np.clip(z, -8.0, 8.0))
    
    def _safe_get(name: str, default: float = 0.0) -> float:
        """Get feature with fallback, ensuring finite value."""
        val = row.get(name, default)
        return float(val) if np.isfinite(val) else default
    
    # Core regime features (NO momentum signal!)
    v = _clip(_safe_get("ret_vol20_z", 0.0))      # Security volatility
    idxv = _clip(_safe_get("idx_vol_z", 0.0))     # Market volatility
    s = _clip(_safe_get("spread_z", 0.0))         # Credit spread
    d = _clip(_safe_get("duration_z", 0.0))       # Duration
    ac = _clip(_safe_get("ret_autocorr_z", 0.0))  # Autocorrelation (regime persistence)
    
    # Cross-sectional features (default to neutral if missing)
    sp = _safe_get("spread_percentile", 0.5)  # Spread percentile [0,1]
    vp = _safe_get("vol_percentile", 0.5)     # Vol percentile [0,1]
    
    # Feature vector: regime indicators + cross-sectional + interactions
    x = np.array([
        1.0,        # [0] bias
        v,          # [1] security vol (idiosyncratic risk)
        idxv,       # [2] market vol (systematic risk)
        s,          # [3] credit spread (default risk)
        d,          # [4] duration (rate sensitivity)
        ac,         # [5] autocorrelation (momentum regime)
        sp,         # [6] spread percentile (relative credit)
        vp,         # [7] vol percentile (relative volatility)
        s * d,      # [8] credit-duration interaction
        v * idxv,   # [9] idio vs systematic vol interaction
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
      - **deferred** updates by horizon (no test-time leakage)
    """
    rng = np.random.default_rng(int(cfg_get(cfg, "seed", 23)))
    results_dir = cfg_get(cfg, "results_dir", "results/crl")
    _ensure_dir(results_dir)
    paths = _results_paths(results_dir)

    # 1) Load panel and keep active securities
    panel_path = (cfg_get(cfg, "panel_path", None)
                  or "data/cdi_processed.pkl")
    df = load_panel(panel_path)  # :contentReference[oaicite:7]{index=7}
    df = df[df.get("active", 1) > 0].copy()
    print(f"[CRL] Loaded {len(df)} active security-date rows | ids={df[ID_COL].nunique()} dates={df[DATE_COL].nunique()}")

    # 2) Momentum features per security
    mom_windows = list(cfg_get(cfg, "momentum_windows", [10, 21, 42, 63]))
    mom_cols: List[str] = []
    for w in mom_windows:
        sv = build_momentum(df=df, w=w)
        df[sv.name] = sv; mom_cols.append(sv.name)
    print(f"[CRL] Momentum features: {len(mom_cols)}")

    # 3) Security-level context
    df = df.sort_values([ID_COL, DATE_COL])
    df["ret_vol20"] = (df.groupby(ID_COL)["return"]
                         .transform(lambda x: x.rolling(20, min_periods=5).std())
                         .fillna(0).astype(np.float32))
    for col in ["spread", "duration", "ttm"]:
        if col not in df.columns:
            df[col] = 0.0

    # Market-wide realized vol proxy per date
    idx_vol = (df.groupby(DATE_COL)
                 .apply(lambda g: np.average(g["ret_vol20"].values,
                                             weights=g["index_weight"].values + 1e-15))
                 .rename("idx_vol")
                 .reset_index())
    df = df.merge(idx_vol, on=DATE_COL, how="left")
    
    # Filter to active securities for cross-sectional computations
    df_active = df[df.get("active", 1) > 0].copy()

    # Spread percentile (only among active securities with valid spreads)
    df_active["spread_percentile"] = (
        df_active.groupby(DATE_COL, group_keys=False)
        .apply(lambda g: _compute_cross_sectional_rank(g, "spread", min_count=10))
        .astype(np.float32)
    )

    # Volatility percentile (relative vol position)
    df_active["vol_percentile"] = (
        df_active.groupby(DATE_COL, group_keys=False)
        .apply(lambda g: _compute_cross_sectional_rank(g, "ret_vol20", min_count=10))
        .astype(np.float32)
    )

    # Merge back to main dataframe
    df = df.merge(
        df_active[[DATE_COL, ID_COL, "spread_percentile", "vol_percentile"]],
        on=[DATE_COL, ID_COL],
        how="left"
    )

    # Fill NaN percentiles with neutral value for inactive/missing
    df["spread_percentile"] = df["spread_percentile"].fillna(0.5).astype(np.float32)
    df["vol_percentile"] = df["vol_percentile"].fillna(0.5).astype(np.float32)

    # Autocorrelation (active-only, NaN-safe)
    df["ret_autocorr"] = np.nan
    mask_active = df["active"] == 1
    if mask_active.any():
        df.loc[mask_active, "ret_autocorr"] = (
            df[mask_active].groupby(ID_COL)["return"]
            .transform(lambda x: x.rolling(21, min_periods=10)
                       .apply(lambda y: y.autocorr(lag=1) if len(y) > 1 else 0, raw=False))
        )
    df["ret_autocorr"] = df["ret_autocorr"].fillna(0.0).astype(np.float32)

    # 4) Targets (remain stored at decision rows; we do NOT use them to pick actions)
    horizons = list(cfg_get(cfg, "horizons", [10, 21, 42, 63]))
    df = _add_forward_security_targets(df, horizons)

    # 5) Train/Test split   
    train_end  = pd.to_datetime(cfg_get(cfg, "train_end",  "2023-12-31"))
    test_start = pd.to_datetime(cfg_get(cfg, "test_start", "2024-01-01"))
    df_train = df[df[DATE_COL] <= train_end].copy()
    df_test  = df[df[DATE_COL] >= test_start].copy()
    print(f"[CRL] Train rows: {len(df_train)} | Test rows: {len(df_test)}")

    # 6) Standardize using TRAIN stats (creates _z columns used by feature builder)
    train_stats: Dict[str, Tuple[float, float]] = {}
    # Include new features in standardization
    standardize_cols = mom_cols + ["spread", "duration", "ret_vol20", "idx_vol", "ret_autocorr"]
    for col in standardize_cols:
        if col not in df_train.columns:
            continue
        mu = float(df_train[col].mean())
        sd = float(df_train[col].std())
        if not math.isfinite(sd) or sd <= 0:
            sd = 1.0
        train_stats[col] = (mu, sd)
        df[f"{col}_z"] = ((df[col] - mu) / sd).fillna(0.0).clip(-8, 8)


    # 7) Action space
    arms: List[Tuple[str, int]] = [(m, h) for m in mom_cols for h in horizons]
    print(f"[CRL] Action space: {len(arms)} arms")

    feature_dim = 10

    # 8) Models
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
    lam_cov    = float(cfg_get(cfg, "lambda_cov", 0.0))
    lam_width  = float(cfg_get(cfg, "lambda_width", 0.0))

    # 9) Pretraining on TRAIN (OK: uses only train)
    print("[CRL] Pretraining heads + calibrators…")
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
            y_std = {h: float(df_train[f"y_h{h}"].std(skipna=True) or 1.0) for h in horizons}
            cap_k = 50.0
            cap = cap_k * y_std[h]
            l_raw = float(np.clip(l_raw, -cap, cap))
            u_raw = float(np.clip(u_raw, -cap, cap))
            calibrators[h].update(float(y), l_raw, u_raw)
    print("[CRL] Pretraining done.")

    # 10) Label-free alive masks & maturity dates (for deferred updates)
    alive = _precompute_alive_and_maturity_dates(df_test, horizons)
    df_test = df_test.merge(alive, on=[DATE_COL, ID_COL], how="left")

    # 11) Index for retrieving matured targets later (no peeking at decision time)
    #     We only read y_h at the moment of maturation using (date_decision, id).
    targets_idx = (df[[DATE_COL, ID_COL] + [f"y_h{h}" for h in horizons]]
                      .set_index([DATE_COL, ID_COL])
                      .sort_index())

    # 12) Online evaluation on TEST with **deferred updates**
    rows_sec: List[Dict[str, Any]] = []
    rows_choices: List[Dict[str, Any]] = []

    test_dates = sorted(df_test[DATE_COL].unique())
    max_h = max(horizons)
    if len(test_dates) > max_h:
        test_dates = test_dates[:-max_h]  # ensure all scheduled events will mature inside the loop

    print(f"[CRL] Processing {len(test_dates)} test dates…")

    # Pending updates: map maturity_date → list of pending decisions to finalize
    pending_by_date: Dict[pd.Timestamp, List[Dict[str, Any]]] = defaultdict(list)

    def _apply_matured_updates(current_date: pd.Timestamp):
        """Apply all matured updates scheduled for current_date (if any)."""
        matured = pending_by_date.pop(current_date, [])
        for item in matured:
            h = int(item["h"])
            alpha_h = float(item["alpha"])
            # Retrieve realized label aligned at decision time (no test-time use before now)
            try:
                y = float(targets_idx.loc[(item["date"], item["id"]), f"y_h{h}"])
            except KeyError:
                # If somehow missing, skip (shouldn't happen when alive_h was 1)
                continue
            l_adj, u_adj = float(item["l_adj"]), float(item["u_adj"])
            covered = int(l_adj <= y <= u_adj)
            width   = max(0.0, u_adj - l_adj)
            loss    = interval_score(y, l_adj, u_adj, alpha_h)

            from utils_conformal import horizon_scale_factor
            hs_mode = cfg_get(cfg, "horizon_scale_mode", "none")
            scale   = horizon_scale_factor(h, hs_mode)
            width_s = width * scale
            loss_s  = loss  * scale

            # Reward uses **pre-update** rolling coverage (matured-only)
            cov_shortfall = max(0.0, (1.0 - alpha_h) - cov_track[h].cov())

            reward = -(loss_s + lam_width * width_s + lam_cov * cov_shortfall)

            # Update models/calibrators strictly AFTER observing y
            policy.update(item["key"], item["x"], reward)
            heads[item["key"]].update(item["x"], y)
            calibrators[h].update(y, item["l_raw"], item["u_raw"])

            # Update coverage tracker AFTER computing reward
            cov_track[h].update(covered)

            # Log finalized (matured) prediction at the security level
            rows_sec.append({
                "date": item["date"],  # decision date
                "id": item["id"],
                "momentum": item["key"][0],
                "h": h,
                "y": y,
                "l": l_adj,
                "u": u_adj,
                "width": width,
                "covered": covered,
                "loss": loss,
                "width_scaled": width_s,
                "loss_scaled": loss_s,
                "loss_scale_mode": hs_mode,
                "alpha": alpha_h,
                "qhat": float(item["qhat"]),
            })

    for di, date in enumerate(test_dates, 1):
        # 1) Apply any matured updates scheduled for **today**
        _apply_matured_updates(date)

        # 2) Make decisions for today (no label access; calibrators/heads are at matured state)
        day = df_test[df_test[DATE_COL] == date]
        if day.empty:
            continue

        daily_actions = []

        for _, row in day.iterrows():
            # Build label-free valid horizon set: those with alive_h==1
            valid = []
            for h in horizons:
                alive_flag = int(row.get(f"alive_h{h}", 0))
                mat_date = row.get(f"mat_date_h{h}", pd.NaT)
                if alive_flag == 1 and pd.notna(mat_date):
                    valid.append((h, pd.to_datetime(mat_date)))
            if not valid:
                continue  # no action that will produce a matured outcome

            # Build per-momentum features once
            x_by_mom: Dict[str, np.ndarray] = {m: build_enhanced_features(row) for m in mom_cols}

            # Candidate arms (mom,h) only for horizons that are alive today
            valid_x_by_arm = {(m, h): x_by_mom[m] for m in mom_cols for (h, _) in valid}

            # Select best action
            (mom_col, h), theta, _ = policy.select_from_features(valid_x_by_arm)

            # Raw quantiles & conformal adjustment (uses calibrator state based on matured data only)
            x = x_by_mom[mom_col]
            l_raw, u_raw = heads[(mom_col, h)].predict(x)
            
            from utils_conformal import coldstart_clamp_bounds
            warmup_min = int(cfg_get(cfg, "calib_warmup_min", 20))
            min_w = float(cfg_get(cfg, "raw_min_width", 0.002))   # 20 bps default
            max_w = float(cfg_get(cfg, "raw_max_width", 0.050))   # 5% default
            if len(calibrators[h].s_lo) < warmup_min:
                l_raw, u_raw = coldstart_clamp_bounds(l_raw, u_raw, min_width=min_w, max_width=max_w)

            l_adj, u_adj, qhat = calibrators[h].adjust(l_raw, u_raw)

            if l_adj > u_adj:
                mid = 0.5 * (l_adj + u_adj)
                l_adj, u_adj = mid - 1e-6, mid + 1e-6

            alpha_h = _alpha_for_h(cfg, h)

            # Schedule this decision to be finalized at its maturity date
            # (we never use y now; we will look it up at maturation time)
            mat_date = dict(valid)[h]  # lookup maturity date for chosen h
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

        # Log modal daily choice (for interpretation)
        if daily_actions:
            m_modal = Counter([m for (m, _) in daily_actions]).most_common(1)[0][0]
            h_modal = Counter([h for (_, h) in daily_actions]).most_common(1)[0][0]
            rows_choices.append({
                "date": date,
                "momentum": m_modal, "h": int(h_modal),
                "ctx_spread": float(day.get("spread", pd.Series([0.0])).mean()),
                "ctx_retvol20": float(day.get("idx_vol", pd.Series([0.0])).iloc[0] if not day.empty else 0.0),
            })

        if (di % 20) == 0:
            print(f"[CRL] {di}/{len(test_dates)} dates processed… (pending={sum(len(v) for v in pending_by_date.values())})")

    # After the loop, there should be no pending left because we trimmed the last max_h dates,
    # but finalize defensively in case of edge-cases:
    for d in sorted(list(pending_by_date.keys())):
        _apply_matured_updates(d)

    # 13) Save artifacts
    print(f"[CRL] Saving {len(rows_sec)} matured security-level predictions…")

    sec_df = pd.DataFrame(rows_sec)
    if not sec_df.empty and "date" in sec_df.columns:
        sec_df = sec_df.sort_values(["date", "id"])
    sec_df.to_csv(paths["scores_sec"], index=False)

    choices_df = pd.DataFrame(rows_choices)
    if not choices_df.empty and "date" in choices_df.columns:
        choices_df = choices_df.sort_values("date")
    choices_df.to_csv(paths["choices"], index=False)

    # Summary statistics (matches your prior schema)
    summary = {
        "n_security_predictions": int(len(sec_df)),
        "n_unique_securities": int(sec_df[ID_COL].nunique()) if not sec_df.empty else 0,
        "n_unique_dates": int(sec_df["date"].nunique()) if not sec_df.empty else 0,
        "arms": int(len(arms)),
        "horizons": [int(h) for h in horizons],
        "security_coverage": float(sec_df["covered"].mean()) if not sec_df.empty else None,
        "security_mean_loss": float(sec_df["loss_scaled"].mean()) if not sec_df.empty else None,
        "security_median_loss": float(sec_df["loss_scaled"].median()) if not sec_df.empty else None,
        "security_p99_loss": float(sec_df["loss_scaled"].quantile(0.99)) if not sec_df.empty else None,
        "security_mean_width": float(sec_df["width_scaled"].mean()) if not sec_df.empty else None,
        "security_median_width": float(sec_df["width_scaled"].median()) if not sec_df.empty else None,
        "security_p99_width": float(sec_df["width_scaled"].quantile(0.99)) if not sec_df.empty else None,
    }
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

    # 14) Falsification: placebo k-step shift using matured predictions
    falsif = {}
    if not sec_df.empty:
        falsif["coverage_by_h"] = {
            int(k): float(v) for k, v in sec_df.groupby("h")["covered"].mean().to_dict().items()
        }
        # Placebo: compare intervals to labels shifted by k across each id
        k = 20
        s = sec_df.sort_values(["id", "date"]).copy()
        s["y_shift"] = s.groupby("id")["y"].shift(k)
        s = s.dropna(subset=["y_shift"])
        if not s.empty:
            falsif["placebo_cov_k20"] = float(((s["l"] <= s["y_shift"]) & (s["y_shift"] <= s["u"])).mean())

    with open(paths["falsif"], "w") as f:
        json.dump(falsif, f, indent=2)

    print("[CRL] Done.")
    if not sec_df.empty:
        print(f"[CRL] Security coverage: {sec_df['covered'].mean():.3f} | mean loss (scaled): {sec_df['loss_scaled'].mean():.6f}")
        print(f"[CRL] Predictions: {len(sec_df)} across {sec_df[ID_COL].nunique()} securities")
