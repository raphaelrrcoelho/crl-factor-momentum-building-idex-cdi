# causal_effects.py — SECURITY-LEVEL, horizon-aware, leakage-safe (aligned, full features)
# -*- coding: utf-8 -*-
"""
Causal analysis for the CRL momentum project (security level)
=============================================================

This **full** version preserves all core features you had (OOF value function,
CATE-style diagnostic, counterfactual fixed policies with HAC/bootstrapped
uncertainty), fixes the NaN crash, and adds an option to run the value
function in **pointwise (debenture-level)** mode by default.

Key features retained & fixes
-----------------------------
- SECURITY‑level preparation (parity with CRL/Baselines) and horizon aware.
- **Value function V^π with cross‑fitted OOF** using `PurgedTimeSeriesSplit`.
  *New:* `causal_value_pointwise: true` (default) trains at the **debenture**
  level with per‑row sample weights and then aggregates predictions **by date**
  for metrics. Set it to `false` to use legacy per‑date aggregation.
- **NaN‑safe**: explicit `dropna` on `y_h{h}` and `index_weight` before any
  aggregation or learning; guards on finiteness.
- **Counterfactual fixed policies** on the same dates as the learned policy
  with top‑fraction selection by momentum, Newey–West SE + moving block
  bootstrap CIs.
- **Causal learning diagnostic** (alignment between momentum spreads and chosen
  horizons), unchanged but hardened.
- Outputs compatible with your report: writes `causal_effects_summary.csv`.

"""
from __future__ import annotations

import os
import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone

# Project-local utilities (present in your repo)
from crl_factor_bandit_conformal import load_panel
from baselines_crl import prepare_security_data
from ablation_study import _newey_west_se, _mbb_mean_ci
from utils_timesplit import PurgedTimeSeriesSplit  # purged+embargoed cross-fitting

DATE_COL = "date"
ID_COL   = "id"

# ----------------------------- IO utils ----------------------------- #

def _ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

# ----------------------------- Config getter ----------------------------- #

def cfg_get(obj: Any, key: str | tuple | list, default=None):
    """Resilient getter for dicts/objects with dotted/tuple keys."""
    def _one(o, k):
        if isinstance(k, str):
            if "." in k:
                cur = o
                for part in k.split("."):
                    if isinstance(cur, dict) and part in cur:
                        cur = cur[part]
                    elif hasattr(cur, part):
                        cur = getattr(cur, part)
                    else:
                        return None
                return cur
            if isinstance(o, dict) and k in o:
                return o[k]
            if hasattr(o, k):
                return getattr(o, k)
        if isinstance(o, dict) and k in o:
            return o[k]
        return None
    if isinstance(key, (tuple, list)):
        for k in key:
            v = _one(obj, k)
            if v is not None:
                return v
        return default
    v = _one(obj, key)
    return default if v is None else v

# ----------------------------- Value function (OOF) ----------------------------- #

def compute_learned_policy_from_security_logs(results_dir: str,
                                              panel_path: str,
                                              scale: str = "per_step") -> Dict[str, Any]:
    """
    Read <results_dir>/crl_scores_securities.csv and compute index-weighted
    per-date returns of the *actual* CRL actions, by horizon and overall.
    scale ∈ {"raw","per_step"}: report mean returns either raw or divided by h.
    """
    path = os.path.join(results_dir, "crl_scores_securities.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, parse_dates=["date"])
    if df.empty or not set(["date","id","h","y"]).issubset(df.columns):
        return {}

    # Merge index weights from the original panel
    panel = load_panel(panel_path)
    w = panel[["date","id","index_weight"]].drop_duplicates()
    df = df.merge(w, on=["date","id"], how="left")
    df["index_weight"] = df["index_weight"].fillna(1.0)

    # Index-weight within (date,h)
    df["w_norm"] = df.groupby(["date","h"])["index_weight"].transform(lambda s: s / (s.sum() + 1e-12))

    def _series_for_h(h: int) -> pd.Series:
        g = df[df["h"] == h]
        if g.empty:
            return pd.Series(dtype=float)
        s = g.groupby("date").apply(lambda x: float(np.sum(x["w_norm"].values * x["y"].values)))
        if scale == "per_step":
            s = s / float(h)
        return s.sort_index().astype(float)

    res: Dict[str, Any] = {}
    byh = {}
    all_vals = []
    for h in sorted(df["h"].unique()):
        s = _series_for_h(int(h))
        if s.size == 0:
            continue
        nw = float(_newey_west_se(s.values))
        lo, hi = _mbb_mean_ci(s.values, block_len=5, B=2000)
        byh[f"h{int(h)}"] = {
            "mean_return": float(np.mean(s.values)),
            "nw_se": nw,
            "mbb_ci_lo": float(lo),
            "mbb_ci_hi": float(hi),
            "n": int(s.size),
            "scale": scale,
        }
        all_vals.append(s.values)

    if all_vals:
        y = np.concatenate(all_vals)
        nw = float(_newey_west_se(y)); lo, hi = _mbb_mean_ci(y, block_len=5, B=2000)
        res["overall"] = {
            "mean_return": float(np.mean(y)),
            "nw_se": nw,
            "mbb_ci_lo": float(lo),
            "mbb_ci_hi": float(hi),
            "n": int(y.size),
            "scale": scale,
        }
    res["by_h"] = byh
    return res

# Pointwise (security-level) builder for value function
def _build_pointwise_for_value(df: pd.DataFrame, choices: pd.DataFrame, h: int, top_frac: float = 0.20):
    """
    Build (X, y, dates, w) at the SECURITY level for horizon h, using the policy's
    selection rule per date (top `top_frac` by chosen momentum). `w` are per-row
    index weights normalized within each date.
    """
    ch = choices[["date", "momentum", "h"]].drop_duplicates().rename(columns={"momentum": "mom_sel", "h": "h_sel"})
    df_h = df[df.columns.intersection(
        [DATE_COL, ID_COL, f"y_h{h}", "index_weight"] + [c for c in df.columns if c.startswith("mom_")]
    )].copy()
    df_h = df_h.merge(ch, on="date", how="inner")

    # ensure selection momentum exists
    colz = df_h["mom_sel"].astype(str) + "_z"
    mask = colz.isin(df_h.columns)
    if not mask.all():
        colr = df_h["mom_sel"].astype(str)
        mask2 = colr.isin(df_h.columns)
        df_h.loc[~mask & mask2, "mom_sel"] = df_h.loc[~mask & mask2, "mom_sel"]

    # Features for heterogeneity (context, not momentum itself)
    feats = ["ret_vol20_z", "idx_vol_z", "spread_z", "duration_z", "ret_autocorr_z",
             "spread_percentile", "vol_percentile", "index_weight"]
    feats = [f for f in feats if f in df_h.columns]

    X_list, y_list, date_list, w_list = [], [], [], []
    
    for d, day in df_h.groupby(DATE_COL):
        if day.empty or f"y_h{h}" not in day.columns:
            continue
        mname = str(day["mom_sel"].iloc[0])
        mcol = f"{mname}_z" if f"{mname}_z" in day.columns else mname
        if mcol not in day.columns:
            continue

        q = 1.0 - float(top_frac)
        thr = day[mcol].quantile(q)
        sel = day[day[mcol] >= thr]

        # NaN-safe selection
        sel = sel.dropna(subset=[f"y_h{h}", "index_weight"] + feats)
        if sel.empty:
            continue

        w = sel["index_weight"].to_numpy(dtype=np.float32)
        w_norm = w / (w.sum() + 1e-12)
        y_vals = sel[f"y_h{h}"].to_numpy(dtype=np.float32)
        X_vals = sel[feats].to_numpy(dtype=np.float32)

        if not (np.isfinite(y_vals).all() and np.isfinite(X_vals).all() and np.isfinite(w_norm).all()):
            continue

        X_list.append(X_vals)
        y_list.append(y_vals)
        w_list.append(w_norm)
        date_list.extend([d] * len(sel))

    if not X_list:
        return (np.empty((0, len(feats)), dtype=np.float32), 
                np.empty((0,), dtype=np.float32), 
                pd.Series([], dtype="datetime64[ns]"),
                np.empty((0,), dtype=np.float32))

    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list).astype(np.float32)
    w = np.concatenate(w_list).astype(np.float32)
    dates = pd.Series(date_list, dtype="datetime64[ns]")
    
    return X, y, dates, w


def _build_Xy_for_value(df: pd.DataFrame, choices: pd.DataFrame, h: int):
    """
    Build (X, y, dates) for horizon h at the DATE level (legacy mode).
    Aggregates selected securities to a single index-weighted target per date.
    NaN-safe: drops rows missing y_h or index_weight before aggregation.
    """
    ch = choices[["date", "momentum", "h"]].drop_duplicates().rename(columns={"momentum": "mom_sel", "h": "h_sel"})
    df_h = df[df.columns.intersection(
        [DATE_COL, ID_COL, f"y_h{h}", "index_weight"] + [c for c in df.columns if c.startswith("mom_")]
    )].copy()
    df_h = df_h.merge(ch, on="date", how="inner")

    # ensure selection momentum exists
    colz = df_h["mom_sel"].astype(str) + "_z"
    mask = colz.isin(df_h.columns)
    if not mask.all():
        colr = df_h["mom_sel"].astype(str)
        mask2 = colr.isin(df_h.columns)
        df_h.loc[~mask & mask2, "mom_sel"] = df_h.loc[~mask & mask2, "mom_sel"]

    dates = df_h[DATE_COL].drop_duplicates().sort_values()
    X_list, y_list = [], []
    # Use context features (not momentum) for heterogeneous treatment effects
    feats = ["ret_vol20_z", "idx_vol_z", "spread_z", "duration_z", "ret_autocorr_z",
             "spread_percentile", "vol_percentile", "index_weight"]
    feats = [f for f in feats if f in df_h.columns]

    for d in dates:
        day = df_h[df_h[DATE_COL] == d]
        if day.empty or f"y_h{h}" not in day.columns:
            continue
        mname = str(day["mom_sel"].iloc[0])
        mcol = f"{mname}_z" if f"{mname}_z" in day.columns else mname
        if mcol not in day.columns:
            continue

        q = 1.0 - float(cfg_get({}, "cf_top_fraction", 0.20))
        thr = day[mcol].quantile(q)
        sel = day[day[mcol] >= thr]

        # NaN-safe selection
        sel = sel.dropna(subset=[f"y_h{h}", "index_weight"])
        if sel.empty:
            continue

        w = sel["index_weight"].to_numpy(dtype=np.float32)
        w = w / (w.sum() + 1e-12)
        y_vals = sel[f"y_h{h}"].to_numpy(dtype=np.float32)
        y_idx = float(np.sum(w * y_vals))

        x_mean = sel[feats].mean(numeric_only=True)
        if not (np.isfinite(y_idx) and np.isfinite(x_mean).all()):
            continue

        X_list.append(x_mean.to_numpy(dtype=np.float32))
        y_list.append(y_idx)

    if not X_list:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32), pd.Series([], dtype="datetime64[ns]")

    X = np.vstack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    return X, y, dates.reset_index(drop=True)


def estimate_policy_value_function(df: pd.DataFrame,
                                   policy_choices: pd.DataFrame,
                                   cfg: Any) -> Dict[str, Any]:
    """
    Estimate V^π by horizon using purged, cross-fitted OOF predictions.
    Horizon-aware: purge/embargo >= h (already implemented).
    NEW: value_scale ∈ {none, per_step} — scales the label for fair cross-h comparison.
    """
    horizons = list(cfg_get(cfg, "horizons", [10, 21, 42, 63]))
    pointwise = bool(cfg_get(cfg, "causal_value_pointwise", True))
    scale_mode = str(cfg_get(cfg, ("value_scale", "horizon_norm_value"), "per_step")).lower()

    def _scale_y(y: np.ndarray, h: int) -> np.ndarray:
        if scale_mode == "per_step":
            return y / float(max(1, h))
        return y

    out: Dict[str, Any] = {}

    for h in horizons:
        top_frac = float(cfg_get(cfg, "cf_top_fraction", 0.20))
        if pointwise:
            X, y, dates, w = _build_pointwise_for_value(df, policy_choices, h, top_frac=top_frac)
            if y.size == 0: 
                continue
            y = _scale_y(y, h)  # scale BEFORE fitting
            dates_df = dates.to_frame(name="date")
        else:
            X, y, dates = _build_Xy_for_value(df, policy_choices, h)
            if y.size == 0:
                continue
            y = _scale_y(y, h)
            dates_df = dates.to_frame()

        base_ps = int(cfg_get(cfg, "max_purge_steps", 21))
        ps_eff  = int(max(base_ps, int(h)))
        n_splits_v = int(cfg_get(cfg, "n_splits_value", 3))
        splitter = PurgedTimeSeriesSplit(
            n_splits=n_splits_v, purge_steps=ps_eff, embargo_steps=ps_eff, date_col="date"
        )

        base = RandomForestRegressor(
            n_estimators=400, max_depth=12, min_samples_leaf=10,
            n_jobs=-1, random_state=42
        )

        oof = np.full(y.shape, np.nan, dtype=np.float32)
        for fold_i, (tr_idx, te_idx) in enumerate(splitter.split(dates_df), 1):
            if te_idx.size == 0 or tr_idx.size < 100:
                continue
            model = clone(base)
            if pointwise:
                model.fit(X[tr_idx], y[tr_idx], sample_weight=w[tr_idx])
            else:
                model.fit(X[tr_idx], y[tr_idx])
            oof[te_idx] = model.predict(X[te_idx]).astype(np.float32)

        mask_nan = ~np.isfinite(oof)
        if mask_nan.all():
            if y.size >= 1:
                if pointwise:
                    base.fit(X, y, sample_weight=w)
                else:
                    base.fit(X, y)
                oof = base.predict(X).astype(np.float32)
            else:
                continue

        if pointwise:
            df_eval = pd.DataFrame({"date": dates.values, "y": y, "oof": oof, "w": w})
            per_day = df_eval.groupby("date").apply(
                lambda g: pd.Series({
                    "y":  np.average(g["y"].values,  weights=g["w"].values),
                    "oof": np.average(g["oof"].values, weights=g["w"].values),
                })
            )
            yy = per_day["y"].to_numpy(dtype=np.float32)
            pp = per_day["oof"].to_numpy(dtype=np.float32)
            mask = np.isfinite(yy) & np.isfinite(pp)
            out[f"h{h}"] = {
                "mean_y": float(np.nanmean(yy[mask])) if mask.any() else float("nan"),
                "mean_oof": float(np.nanmean(pp[mask])) if mask.any() else float("nan"),
                "oof_corr": float(np.corrcoef(yy[mask], pp[mask])[0,1]) if mask.sum() > 2 else np.nan,
                "n_dates": int(mask.sum()),
                "n_oof": int(np.isfinite(pp).sum()),
                "fallback_full_fit": bool(mask_nan.all()),
                "scale": scale_mode,
            }
        else:
            out[f"h{h}"] = {
                "mean_y": float(np.nanmean(y)),
                "mean_oof": float(np.nanmean(oof)),
                "oof_corr": float(np.corrcoef(y[~mask_nan], oof[~mask_nan])[0,1]) if (~mask_nan).any() else np.nan,
                "n_dates": int(len(dates)),
                "n_oof": int((~mask_nan).sum()),
                "fallback_full_fit": bool(mask_nan.all()),
                "scale": scale_mode,
            }

    return out


# ----------------------------- Counterfactual fixed policies ----------------------------- #

def _pick_momentum_columns(mom_cols: List[str]) -> Dict[str, Optional[str]]:
    """
    Choose canonical short/medium/long momentum columns from what's available.
    """
    if not mom_cols:
        return dict(always_short=None, always_medium=None, always_long=None)

    def _closest(target: int) -> Optional[str]:
        # pick momentum whose window number is closest to 'target'
        wins = []
        for c in mom_cols:
            m = [int(x) for x in re.findall(r"(\d+)", c)]
            wins.append((c, m[0] if m else 0))
        wins = sorted(wins, key=lambda t: abs(t[1] - target))
        return wins[0][0] if wins else None

    return {
        "always_short": _closest(10),
        "always_medium": _closest(21),
        "always_long": _closest(42),
    }


def _index_weighted_forward_return(day: pd.DataFrame,
                                   mom_for_selection: str,
                                   h: int,
                                   top_frac: float = 0.20,
                                   min_sel: int = 5) -> Optional[float]:
    """
    Compute index-weighted forward return for horizon h on a single date,
    after selecting the TOP 'top_frac' by the given momentum column (_z).
    """
    if day.empty or f"y_h{h}" not in day.columns:
        return None

    # Selection by momentum (normalized column; if missing, fallback to raw)
    colz = f"{mom_for_selection}_z" if f"{mom_for_selection}_z" in day.columns else mom_for_selection
    if colz not in day.columns:
        return None

    # Drop NaNs before thresholding
    dd = day.dropna(subset=[colz, f"y_h{h}", "index_weight"]).copy()
    if dd.empty:
        return None

    thr = float(dd[colz].quantile(1.0 - float(top_frac)))
    cset = dd[dd[colz] >= thr].copy()
    if cset.shape[0] < int(min_sel):
        # fallback: take all valid if too few selected
        cset = dd

    w = cset["index_weight"].to_numpy(dtype=np.float64)
    w = w / (w.sum() + 1e-12)
    y = cset[f"y_h{h}"].to_numpy(dtype=np.float64)
    val = float(np.sum(w * y))
    return val if np.isfinite(val) else None


def compute_counterfactual_outcomes(df: pd.DataFrame,
                                    policy_choices: pd.DataFrame,
                                    mom_cols: List[str],
                                    cfg: Any) -> Dict[str, Any]:
    """
    Compare learned policy to fixed momentum policies on the SAME dates as the policy acted.
    Selection: top 'cf_top_fraction' by momentum each date.
    Horizon handling & scaling are configurable:
      - cf_h_mode: "fixed" (use cf_fixed_h) or "policy" (use the learned policy's h per date)
      - cf_return_scale: "per_step" or "raw"
    Also reports Newey–West SE and MBB CIs across the per-date series.
    """
    # Dates & per-date chosen momentum and horizon from the learned policy
    choose = policy_choices[["date", "momentum", "h"]].drop_duplicates().sort_values("date")
    if choose.empty:
        return {}

    # Configurable selection intensity (default: top 20%)
    top_frac = float(cfg_get(cfg, "cf_top_fraction", 0.20))
    min_sel  = int(cfg_get(cfg, "cf_min_selection", 5))

    # Horizon handling
    h_mode   = str(cfg_get(cfg, "cf_h_mode", "policy")).lower()      # "policy" or "fixed"
    fixed_h  = int(cfg_get(cfg, "cf_fixed_h", 21))
    # Scaling
    scale    = str(cfg_get(cfg, "cf_return_scale", "per_step")).lower()  # "per_step" or "raw"
    block_len = int(cfg_get(cfg, "cf_mbb_block", 5))
    B_boot    = int(cfg_get(cfg, "cf_mbb_B", 2000))

    def _h_eval(h_pol: int) -> int:
        return fixed_h if h_mode == "fixed" else int(h_pol)

    def _scale_val(v: float, h_eval: int) -> float:
        return float(v) / float(h_eval) if scale == "per_step" else float(v)

    def _metrics_from(vals: List[float]) -> Dict[str, Any]:
        if not vals:
            return {}
        arr = np.asarray(vals, dtype=np.float32)
        if arr.size == 0:
            return {}
        nw = _newey_west_se(arr)
        lo, hi = _mbb_mean_ci(arr, block_len=block_len, B=B_boot)
        sd = arr.std(ddof=1) if arr.size > 1 else 0.0
        return {
            "mean_return": float(arr.mean()),
            "std_return": float(sd),
            "nw_se": float(nw),
            "mbb_ci_lo": float(lo),
            "mbb_ci_hi": float(hi),
            "sharpe": float((arr.mean() / (sd + 1e-18)) * math.sqrt(252.0)),
            "n_obs": int(arr.size),
            "scale": scale,
            "h_mode": h_mode,
        }

    # Learned policy (per date)
    lp_vals: List[float] = []
    for _, ch in choose.iterrows():
        d = ch["date"]; h_eval = int(_h_eval(int(ch["h"]))); mom = str(ch["momentum"])
        day = df[df[DATE_COL] == d]
        val = _index_weighted_forward_return(day, mom_for_selection=mom, h=h_eval,
                                             top_frac=top_frac, min_sel=min_sel)
        if val is not None:
            lp_vals.append(_scale_val(val, h_eval))
    results: Dict[str, Any] = {}
    met = _metrics_from(lp_vals)
    if met:
        results["learned_policy"] = met

    # Fixed policies: same dates, fixed momentum column & horizon as per mode
    picks = _pick_momentum_columns(mom_cols)
    for name, mom in picks.items():
        if mom is None:
            continue
        vals: List[float] = []
        for _, ch in choose.iterrows():
            d = ch["date"]; h_eval = int(_h_eval(int(ch["h"])));
            day = df[df[DATE_COL] == d]
            val = _index_weighted_forward_return(day, mom_for_selection=mom, h=h_eval,
                                                 top_frac=top_frac, min_sel=min_sel)
            if val is not None:
                vals.append(_scale_val(val, h_eval))
        met = _metrics_from(vals)
        if met:
            results[name] = met

    return results


# ----------------------------- Causal diagnostic ----------------------------- #

def validate_causal_learning(df: pd.DataFrame, choices: pd.DataFrame, cfg: Any) -> Dict[str, Any]:
    """
    Heuristic diagnostic: when the policy chooses horizons associated with
    higher (recent) momentum concentration spreads, do realized spreads align?
    Computes rolling spreads and alignment rate.
    """
    # Build daily spread (long-short by momentum) for a couple of candidate columns
    def _day_spread(g: pd.DataFrame, mom_col: str = "mom_w21_z") -> Optional[float]:
        if mom_col not in g.columns or "index_weight" not in g.columns:
            return None
        thr = g[mom_col].quantile(0.8)
        lo = g[mom_col].quantile(0.2)
        a = g[g[mom_col] >= thr]
        b = g[g[mom_col] <= lo]
        if a.empty or b.empty:
            return None
        # Index-weighted average y_h at a fixed reference horizon (21 by default)
        href = int(cfg_get(cfg, "cf_fixed_h", 21))
        if f"y_h{href}" not in g.columns:
            return None
        wa = a["index_weight"].values; wa = wa / (wa.sum() + 1e-12)
        wb = b["index_weight"].values; wb = wb / (wb.sum() + 1e-12)
        return float(np.sum(wa * a[f"y_h{href}"].values) - np.sum(wb * b[f"y_h{href}"].values))

    res: Dict[str, Any] = {}
    for mom_col in [c for c in ["mom_w10_z","mom_w21_z","mom_w42_z"] if c in df.columns]:
        tmp = df.groupby(DATE_COL).apply(lambda g: _day_spread(g, mom_col)).dropna()
        if tmp.empty:
            continue
        # Alignment: spread sign vs policy's h (e.g., higher h when medium-term spreads are positive)
        ch = choices.set_index("date")["h"]
        common = tmp.index.intersection(ch.index)
        if len(common) == 0:
            continue
        spread = tmp.loc[common].astype(float)
        h = ch.loc[common].astype(int)
        align = np.sign(spread.values) == np.sign(h.values - np.median(h.values))
        res[mom_col] = {
            "mean_spread": float(spread.mean()),
            "mean_alignment": float(align.mean()),
            "n": int(len(common)),
        }
    return res


# ----------------------------- Orchestration ----------------------------- #

def run_causal_effects(cfg: dict | Any):
    """
    End-to-end causal analysis. Assumes CRL step has already produced
    <results_dir>/crl_policy_choices.csv. Integrates with run_experiments.
    """
    results_dir = cfg_get(cfg, "results_dir", "results/crl")
    _ensure_dir(results_dir)
    print("[CAUSAL] Starting causal effects…")

    # 1) Load panel and prepare security-level features (parity with CRL/Baselines)
    panel_path = cfg_get(cfg, ("panel_path", "data.panel_path"), "data/cdi_processed.pkl")
    panel = load_panel(panel_path)
    df, mom_cols = prepare_security_data(panel, cfg)  # creates _z columns & y_h
    print(f"[CAUSAL] Obs={len(df)} | ids={df[ID_COL].nunique()} | dates={df[DATE_COL].nunique()}")

    # 2) Load learned policy choices (produced by CRL)
    choices_path = os.path.join(results_dir, "crl_policy_choices.csv")
    if not os.path.exists(choices_path):
        raise FileNotFoundError(f"Policy choices not found: {choices_path}")
    choices = pd.read_csv(choices_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    print(f"[CAUSAL] Policy decisions loaded: {len(choices)} rows")

    learned_perf = compute_learned_policy_from_security_logs(
        results_dir, panel_path,
        scale=str(cfg_get(cfg, "cf_return_scale", "per_step")).lower(),
    )

    # 3) Value function (OOF) by horizon
    print("[CAUSAL] Estimating policy value function (OOF)…")
    value_res = estimate_policy_value_function(df, choices, cfg)

    # 4) Counterfactual fixed policies (same dates)
    print("[CAUSAL] Computing counterfactual outcomes…")
    cf_res = compute_counterfactual_outcomes(df, choices, mom_cols, cfg)

    # 5) Simple diagnostic on causal alignment
    print("[CAUSAL] Validating causal alignment…")
    align_res = validate_causal_learning(df, choices, cfg)

    # 6) Save summary CSV for the report
    rows: List[Dict[str, Any]] = []
    for pol, met in (cf_res or {}).items():
        rows.append({
            "policy": pol,
            "mean_return": met.get("mean_return", np.nan),
            "std_return": met.get("std_return", np.nan),
            "nw_se": met.get("nw_se", np.nan),
            "mbb_ci_lo": met.get("mbb_ci_lo", np.nan),
            "mbb_ci_hi": met.get("mbb_ci_hi", np.nan),
            "sharpe": met.get("sharpe", np.nan),
            "n_obs": met.get("n_obs", 0),
            "h_mode": met.get("h_mode"),
            "scale": met.get("scale"),
        })
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(results_dir, "causal_effects_summary.csv"), index=False)

    # 7) Save JSON payload (optional, for debugging)
    summary = {
        "learned_policy_from_security_logs": learned_perf,   # <-- add this
        "learned_policy_outperformance": float(
            cf_res.get("learned_policy", {}).get("mean_return", 0.0) -
            max([v.get("mean_return", -np.inf) for k, v in cf_res.items() if k != "learned_policy"], default=-np.inf)
        ) if cf_res else 0.0,
        "value_function": value_res,
        "alignment": align_res,
    }
    with open(os.path.join(results_dir, "causal_effects_meta.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("[CAUSAL] Done. Wrote causal_effects_summary.csv and causal_effects_meta.json")

    return {"summary_csv": os.path.join(results_dir, "causal_effects_summary.csv"),
            "meta_json": os.path.join(results_dir, "causal_effects_meta.json")}


# ----------------------------- CLI ----------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import argparse, yaml
    p = argparse.ArgumentParser(description="Causal effects for CRL (security level)")
    p.add_argument("--config", type=str, default="config_crl.yaml")
    args = p.parse_args()

    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    run_causal_effects(cfg)