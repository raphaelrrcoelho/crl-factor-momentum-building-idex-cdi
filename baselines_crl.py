# baselines_crl.py — LEAKAGE-SAFE BASELINES (deferred updates, label‑free gating)
# -*- coding: utf-8 -*-
"""
Baselines that COPY the CRL evaluation protocol but with fixed action rules.
Key methodological fixes vs your previous version:
- Deferred updates by horizon: heads and conformal calibrators update only
  when the chosen decision's outcome has matured (at t+h).
- Label-free action gating: valid horizons at date t depend on whether the
  (id, t+h) row exists (alive), NOT on whether y_h is finite.

Outputs under <results_dir>:
  - random_scores_securities.csv
  - traditional_scores_securities.csv
  - best_fixed_scores_securities.csv
  - all_baseline_scores_securities.csv
  - baseline_summary_securities.csv

Also exports:
  - prepare_security_data(panel, cfg) → (panel_prep, train_stats, mom_cols)
  - build_enhanced_features(row, mom_col, train_stats=None)  # re-export for OPE

Aligned with pipeline: run_experiments.step_baselines(...) calls run_all_baselines(...).
"""

from __future__ import annotations

import os
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

from run_config import cfg_get  # config helper :contentReference[oaicite:9]{index=9}

# Reuse CRL utilities (feature parity + I/O contracts) :contentReference[oaicite:10]{index=10}
from crl_factor_bandit_conformal import (
    load_panel, build_momentum,
    OnlineQuantileHead, CQRCalibrator, interval_score,
    _ensure_dir, _alpha_for_h, build_enhanced_features,
    _compute_cross_sectional_rank,
)

DATE_COL = "date"
ID_COL   = "id"

# --------------------------- public helpers (for OPE) --------------------------- #


def prepare_security_data(panel: pd.DataFrame, cfg: dict | Any) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build security-level features with TRAIN-only normalization and forward targets.
    Forward targets are **h-step total returns**:
        y_h(t) = Π_{k=1..h} (1 + r_{t+k}) - 1
    """
    df = panel.copy()
    df = df[df.get("active", 1) > 0].sort_values([ID_COL, DATE_COL]).reset_index(drop=True)

    # Momentum features
    mom_windows = list(cfg_get(cfg, "momentum_windows", [10, 21, 42, 63]))
    mom_cols: List[str] = []
    for w in mom_windows:
        # Vol-adjusted only (standard)
        sv = build_momentum(df=df, w=int(w))
        df[sv.name] = sv
        mom_cols.append(sv.name)

    # Security context
    df["ret_vol20"] = (df.groupby(ID_COL)["return"].transform(lambda x: x.rolling(20, min_periods=5).std())
                       .fillna(0).astype(np.float32))
    for col in ["spread", "duration"]:
        if col not in df.columns:
            df[col] = 0.0

    # Market context (index-weighted realized vol proxy)
    idx_vol = (df.groupby(DATE_COL, group_keys=False)
                 .apply(lambda g: np.average(g["ret_vol20"].values,
                                             weights=g.get("index_weight", pd.Series(np.ones(len(g)))) + 1e-15))
                 .rename("idx_vol").reset_index())
    df = df.merge(idx_vol, on=DATE_COL, how="left")

    # ========================================================================
    # Cross-sectional features (NaN-safe, active-only) - SAME AS CRL
    # ========================================================================
    
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

    df = df.merge(
        df_active[[DATE_COL, ID_COL, "spread_percentile", "vol_percentile"]],
        on=[DATE_COL, ID_COL],
        how="left"
    )

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

    # ---- Forward targets: TRUE h-step total returns (exclude t; use t+1..t+h) ----
    horizons = list(cfg_get(cfg, "horizons", [10, 21, 42, 63]))
    def _fwd_total_return(vec: pd.Series, h: int) -> np.ndarray:
        v = vec.to_numpy(dtype=np.float64, copy=False)
        r = 1.0 + np.nan_to_num(v, nan=0.0)
        n = r.shape[0]
        cp = np.empty(n + 1, dtype=np.float64); cp[0] = 1.0
        np.multiply.accumulate(r, out=cp[1:])
        out = np.full(n, np.nan, dtype=np.float64)
        if h > 0 and n > h:
            idx = np.arange(0, n - h, dtype=np.int64)
            out[idx] = (cp[idx + h + 1] / cp[idx + 1]) - 1.0
        return out.astype(np.float32)
    for h in horizons:
        df[f"y_h{h}"] = df.groupby(ID_COL, sort=False)["return"] \
                          .transform(lambda s, hh=h: _fwd_total_return(s, hh))

    # TRAIN/TEST split for normalization
    train_end = pd.to_datetime(cfg_get(cfg, "train_end", "2023-12-31"))
    df_train  = df[df[DATE_COL] <= train_end]

    # TRAIN-only standardization (z-cols used by CRL/OPE builders)
    feature_cols = mom_cols + ["spread", "duration", "ret_vol20", "idx_vol", "ret_autocorr"]
    for col in feature_cols:
        if col not in df.columns:
            continue
        mu = float(df_train[col].mean()) if col in df_train.columns else 0.0
        sd = float(df_train[col].std())  if col in df_train.columns else 1.0
        if not math.isfinite(sd) or sd <= 0: sd = 1.0
        df[f"{col}_z"] = ((df[col] - mu) / sd).fillna(0.0).clip(-8, 8)

    return df.sort_values([DATE_COL, ID_COL]).reset_index(drop=True), mom_cols

# --------------------------- internal helpers --------------------------- #

def _precompute_alive_and_maturity_dates(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    """
    Label-free 'alive' mask and maturity dates for each horizon:
      alive_h{h} = 1{row (id,t+h) exists}; mat_date_h{h} = date at t+h (or NaT).
    """
    g = df.sort_values([ID_COL, DATE_COL]).copy()
    g["_exists"] = 1
    for h in horizons:
        g[f"alive_h{h}"]   = g.groupby(ID_COL)["_exists"].shift(-h).notna().astype("int8")
        g[f"mat_date_h{h}"] = g.groupby(ID_COL)[DATE_COL].shift(-h)
    cols = [DATE_COL, ID_COL] + [f"alive_h{h}" for h in horizons] + [f"mat_date_h{h}" for h in horizons]
    return g[cols]


def _alpha_for(h: int, cfg: dict | Any) -> float:
    return float(_alpha_for_h(cfg, h))

# --- cold-start clamp + σ(y_h) helpers --------------------------------
from typing import Tuple, Dict, Iterable

def build_y_std_map(df_train: pd.DataFrame, horizons: Iterable[int]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for h in horizons:
        col = f"y_h{h}"
        if col in df_train.columns:
            std = float(df_train[col].std(skipna=True))
            out[h] = std if (np.isfinite(std) and std > 0) else 1.0
        else:
            out[h] = 1.0
    return out

def predict_adjust_with_coldstart_clamp(head,
                                        calibrator,
                                        x,
                                        h: int,
                                        y_std_map: Dict[int, float],
                                        warmup_min: int = 20,
                                        cap_k: float = 50.0):
    l_raw, u_raw = head.predict(x)
    l_c, u_c = float(l_raw), float(u_raw)
    if len(calibrator.s_lo) < int(warmup_min):  # still returning raw bounds
        cap = float(cap_k) * float(y_std_map.get(h, 1.0))
        l_c = float(np.clip(l_c, -cap, cap))
        u_c = float(np.clip(u_c, -cap, cap))
    l_adj, u_adj, qhat = calibrator.adjust(l_c, u_c)
    return float(l_adj), float(u_adj), float(qhat), l_c, u_c
# ----------------------------------------------------------------------



# --------------------------- core runner --------------------------- #

def run_all_baselines(
        panel_path: str, train_end_date: str, test_start_date: str,
        config: dict | Any,
) -> pd.DataFrame:
    """
    Run leakage-safe baselines with ALL fixed (momentum, horizon) combinations + Random.
    
    Outputs:
    - <results_dir>/random_scores_securities.csv
    - <results_dir>/fixed_<mom>_h<h>_scores_securities.csv (for each combination)
    - <results_dir>/all_baseline_scores_securities.csv (combined)
    - <results_dir>/baseline_summary_securities.csv (sorted by performance)
    """
    results_dir = cfg_get(config, ("results_dir", "crl.results_dir"), "results/crl")
    _ensure_dir(results_dir)

    print("[BASE] Loading panel…")
    panel = load_panel(panel_path)

    print("[BASE] Preparing security-level features…")
    df, mom_cols = prepare_security_data(panel, config)
    horizons = list(cfg_get(config, "horizons", [10, 21, 42, 63]))

    # Train/Test split (dates come from pipeline args)
    train_end  = pd.to_datetime(train_end_date)
    test_start = pd.to_datetime(test_start_date)
    df_train   = df[df[DATE_COL] <= train_end].copy()
    df_test    = df[df[DATE_COL] >= test_start].copy()

    y_std_map = build_y_std_map(df_train, horizons)
    warmup_min = int(cfg_get(config, "calib_warmup_min", 20))
    cap_k      = float(cfg_get(config, "raw_bound_cap_k", 50.0))

    calib_audit_log: List[Dict[str, Any]] = []

    # Action space
    arms: List[Tuple[str, int]] = [(m, h) for m in mom_cols for h in horizons]
    feature_dim = 10

    # Hyperparams (parity with CRL)
    tau_low  = float(cfg_get(config, "tau_low", 0.05))
    tau_high = float(cfg_get(config, "tau_high", 0.95))
    lam_head = float(cfg_get(config, "lam_head", 1e-4))
    eta_q    = float(cfg_get(config, "eta_q", 0.02))
    seed     = int(cfg_get(config, "seed", 23))

    # Precompute alive masks & maturity dates (label-free gating)
    live = _precompute_alive_and_maturity_dates(df_test, horizons)
    df_test = df_test.merge(live, on=[DATE_COL, ID_COL], how="left")

    # Targets index for maturity lookup (no label access at decision time)
    targets_idx = (df[[DATE_COL, ID_COL] + [f"y_h{h}" for h in horizons]]
                     .set_index([DATE_COL, ID_COL]).sort_index())

    # ==================== DEFINE ALL BASELINES ==================== #
    
    # Optional: allow config to limit which momentum specs to run
    baseline_mom_subset = cfg_get(config, "baseline_momentum_subset", None)
    if baseline_mom_subset:
        baseline_moms = [m for m in mom_cols if m in baseline_mom_subset]
    else:
        baseline_moms = mom_cols

    # Build baseline list: Random + all fixed (mom, h) combinations
    baselines: List[Tuple[str, Optional[Tuple[str,int]]]] = [
        ("random", None),  # Uniform random baseline
    ]
    
    # Add all fixed policies
    for mom in baseline_moms:
        for h in horizons:
            baseline_name = f"fixed_{mom}_h{h}"
            baselines.append((baseline_name, (mom, h)))
    
    print(f"\n[BASE] Running {len(baselines)} baselines:")
    print(f"  - 1 random (uniform over valid actions)")
    print(f"  - {len(baseline_moms) * len(horizons)} fixed policies ({len(baseline_moms)} momentum × {len(horizons)} horizons)")

    all_rows = {}  # baseline_name → DataFrame of matured rows

    # ==================== RUN EACH BASELINE ==================== #
    
    for baseline_idx, (base_name, fixed_arm) in enumerate(baselines, 1):
        print(f"\n[BASE] [{baseline_idx}/{len(baselines)}] Running {base_name} (deferred updates)…")

        # Initialize online heads & calibrators (parity with CRL)
        heads = {
            key: OnlineQuantileHead(d=feature_dim, tau_low=tau_low, tau_high=tau_high,
                                    lam=lam_head, eta=eta_q, w_cap=1e3)
            for key in arms
        }
        calibrators = {
            h: CQRCalibrator(alpha=_alpha_for(h, config),
                             window=int(cfg_get(config, "conf_window", 250)),
                             lambda_recency=float(cfg_get(config, "lambda_recency", 0.0)))
            for h in horizons
        }

        # Pretrain on TRAIN (OK to use labels here)
        rng = np.random.default_rng(seed)
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
                calibrators[h].update(float(y), l_raw, u_raw)

        # Test dates (trim last max(h) so every pending update matures)
        test_dates = sorted(df_test[DATE_COL].unique())
        max_h = max(horizons)
        if len(test_dates) > max_h:
            test_dates = test_dates[:-max_h]

        # Pending updates scheduled by maturity date
        pending_by_date: Dict[pd.Timestamp, List[Dict[str, Any]]] = defaultdict(list)
        out_rows: List[Dict[str, Any]] = []
        eval_rng = np.random.default_rng(seed)

        def _apply_matured_updates(current_date: pd.Timestamp):
            matured = pending_by_date.pop(current_date, [])
            for item in matured:
                h = int(item["h"])
                key = item["key"]
                # Lookup realized label (aligned at decision time)
                try:
                    y = float(targets_idx.loc[(item["date"], item["id"]), f"y_h{h}"])
                except KeyError:
                    continue  # should not happen if alive_h was 1
                # Score
                l_adj, u_adj = float(item["l_adj"]), float(item["u_adj"])
                alpha_h = float(item["alpha"])
                covered = int(l_adj <= y <= u_adj)
                width   = max(0.0, u_adj - l_adj)
                loss    = interval_score(y, l_adj, u_adj, alpha_h)
                # Unified horizon scaling (scalar)
                from utils_conformal import horizon_scale_factor
                hs_mode = cfg_get(config, "horizon_scale_mode", "none")
                scale   = horizon_scale_factor(h, hs_mode)
                width_s = width * scale
                loss_s  = loss  * scale
                # Updates AFTER observing y (no leakage)
                heads[key].update(item["x"], y)
                calibrators[h].update(y, item["l_raw"], item["u_raw"])
                # Log matured prediction
                row = {
                    "date": item["date"],               # decision date
                    "baseline": base_name,
                    "id": item["id"],
                    "momentum": key[0],
                    "h": h,
                    "y": y,
                    "l": l_adj,
                    "u": u_adj,
                    "width": width,
                    "covered": covered,
                    "loss": loss,
                    "alpha": alpha_h,
                    "qhat": float(item["qhat"]),
                    "width_scaled": width_s,
                    "loss_scaled": loss_s,
                    "loss_scale_mode": hs_mode,
                }
                # OPE aid: support size for Random (row-wise propensity info)
                if "support_size" in item:
                    row["support_size"] = int(item["support_size"])
                out_rows.append(row)

        # Main test loop (decisions; no y access)
        for di, date in enumerate(test_dates, 1):
            _apply_matured_updates(date)

            day = df_test[df_test[DATE_COL] == date]
            if day.empty:
                continue

            for _, row in day.iterrows():
                # Label-free valid horizons & maturity dates
                valid = []
                for h in horizons:
                    alive = int(row.get(f"alive_h{h}", 0))
                    mat_d = row.get(f"mat_date_h{h}", pd.NaT)
                    if alive == 1 and pd.notna(mat_d):
                        valid.append((h, pd.to_datetime(mat_d)))
                if not valid:
                    continue

                # Build per-momentum features (once per row)
                x_by_mom = {m: build_enhanced_features(row) for m in baseline_moms}

                # Choose arm per baseline
                if fixed_arm is None:
                    # Random over **valid** (mom,h)
                    valid_arms = [(m, h) for m in baseline_moms for (h, _) in valid]
                    if not valid_arms:
                        continue
                    mom_col, h = valid_arms[eval_rng.integers(0, len(valid_arms))]
                    support_size = len(valid_arms)  # for IPS propensities later
                else:
                    # Fixed policy
                    mom_col, h = fixed_arm
                    # If chosen h not alive today, fallback to first available horizon
                    if h not in [vh for (vh, _) in valid]:
                        h = valid[0][0]
                    support_size = None  # fixed policy doesn't need IPS

                # Predict raw quantiles & conformalize with calibrator state (matured-only)
                x = x_by_mom[mom_col]
                l_adj, u_adj, qhat, l_raw_c, u_raw_c = predict_adjust_with_coldstart_clamp(
                    heads[(mom_col, h)], calibrators[h], x, h, y_std_map,
                    warmup_min=warmup_min, cap_k=cap_k
                )

                if l_adj > u_adj:
                    mid = 0.5 * (l_adj + u_adj)
                    l_adj, u_adj = mid - 1e-6, mid + 1e-6

                # --- Calibrator audit (baselines) ---
                clamp_applied = (len(calibrators[h].s_lo) < warmup_min)
                calib_audit_log.append({
                    "date": date,
                    "baseline": base_name,
                    "h": int(h),
                    "alpha": float(_alpha_for(h, config)),
                    "buffer_len": int(len(calibrators[h].s_lo)),
                    "warmup_min": int(warmup_min),
                    "qhat": float(qhat),
                    "clamp_applied": bool(clamp_applied),
                })

                # Schedule maturity update (we do NOT read y now)
                alpha_h = _alpha_for(h, config)
                mat_date = dict(valid)[h]
                payload = {
                    "date": date,
                    "id": row[ID_COL],
                    "h": int(h),
                    "alpha": float(alpha_h),
                    "key": (mom_col, int(h)),
                    "x": x.astype(np.float32),
                    "l_raw": float(l_raw_c),
                    "u_raw": float(u_raw_c),
                    "l_adj": float(l_adj),
                    "u_adj": float(u_adj),
                    "qhat": float(qhat),
                }
                if support_size is not None:
                    payload["support_size"] = int(support_size)
                pending_by_date[pd.to_datetime(mat_date)].append(payload)

            if (di % 20) == 0:
                pend = sum(len(v) for v in pending_by_date.values())
                print(f"[BASE-{base_name}] {di}/{len(test_dates)} dates processed… (pending={pend})")

        # Finalize any remaining maturities (should be none due to trimming)
        for d in sorted(list(pending_by_date.keys())):
            _apply_matured_updates(d)

        # Persist per-baseline results
        out = pd.DataFrame(out_rows).sort_values(["date", "id"])
        out.to_csv(os.path.join(results_dir, f"{base_name}_scores_securities.csv"), index=False)
        print(f"[BASE-{base_name}] Saved {len(out)} matured predictions.")

        all_rows[base_name] = out

    # ==================== COMBINED SUMMARY ==================== #
    
    # Save calibrator audit (all baselines combined)
    if calib_audit_log:
        audit_df = pd.DataFrame(calib_audit_log).sort_values(["baseline","h","date"])
        audit_df.to_csv(os.path.join(results_dir, "calibrator_audit_baselines.csv"), index=False)

    combined = pd.concat(all_rows.values(), ignore_index=True) if all_rows else pd.DataFrame()
    combined.to_csv(os.path.join(results_dir, "all_baseline_scores_securities.csv"), index=False)

    if combined.empty:
        summary = pd.DataFrame(columns=["baseline","mean_loss","cov_rate","mean_width",
                                        "mean_loss_scaled","mean_width_scaled","n_predictions"])
    else:
        # Load panel for index weights
        panel = load_panel(panel_path)
        if isinstance(panel.index, pd.MultiIndex):
            panel = panel.reset_index()
        weights_df = panel[["date", "id", "index_weight"]].drop_duplicates()

        # Merge weights into combined
        combined = combined.merge(weights_df, on=["date", "id"], how="left")
        combined["index_weight"] = combined["index_weight"].fillna(0.0)

        # Compute index-weighted loss for each baseline BEFORE groupby
        # Step 1: Normalize weights per (date, h) within each baseline
        combined["w_norm"] = combined.groupby(["baseline", "date", "h"])["index_weight"].transform(
            lambda x: x / (x.sum() + 1e-12)
        )

        # Step 2: Compute weighted mean per (baseline, date, h)
        weighted_by_date = (combined.groupby(["baseline", "date", "h"])
                                    .apply(lambda g: np.average(g["loss_scaled"], weights=g["w_norm"]))
                                    .reset_index()
                                    .rename(columns={0: "loss_weighted"}))

        # Step 3: Average across dates per baseline (this is our final weighted loss)
        weighted_summary = (weighted_by_date.groupby("baseline")
                                           .agg(mean_loss_weighted=("loss_weighted", "mean"))
                                           .reset_index())

        # Also compute simple averages for comparison
        simple_summary = (combined.groupby("baseline")
                                  .agg(
                                      mean_loss_simple=("loss_scaled", "mean"),
                                      cov_rate=("covered", "mean"),
                                      mean_width_scaled=("width_scaled", "mean"),
                                      n_predictions=("loss", "count")
                                  )
                                  .reset_index())

        # Merge weighted and simple summaries
        summary = simple_summary.merge(weighted_summary, on="baseline", how="left")

        # Reorder columns and use weighted as primary
        summary = summary[["baseline", "mean_loss_weighted", "mean_loss_simple", 
                           "cov_rate", "mean_width_scaled", "n_predictions"]]
        summary = summary.rename(columns={"mean_loss_weighted": "mean_loss_scaled"})
        summary = summary.sort_values("mean_loss_scaled")
    summary.to_csv(os.path.join(results_dir, "baseline_summary_securities.csv"), index=False)

    # ==================== ENHANCED REPORTING ==================== #
    
    print("\n" + "="*70)
    print("BASELINE SUMMARY (sorted by INDEX-WEIGHTED loss, ↓ better)")
    print("="*70)
    print(summary[["baseline", "mean_loss_scaled", "mean_loss_simple", "cov_rate", 
                   "mean_width_scaled", "n_predictions"]].to_string(index=False))
    
    # Identify best performers
    if not summary.empty:
        random_perf = summary[summary["baseline"] == "random"]
        fixed_only = summary[summary["baseline"].str.startswith("fixed_")]
        
        if not random_perf.empty:
            print(f"\n[BASE] Random baseline: loss_scaled={random_perf.iloc[0]['mean_loss_scaled']:.6f} (weighted)")
        
        if not fixed_only.empty:
            best_fixed = fixed_only.iloc[0]
            worst_fixed = fixed_only.iloc[-1]
            print(f"[BASE] Best fixed policy: {best_fixed['baseline']} (loss_scaled={best_fixed['mean_loss_scaled']:.6f}, weighted)")
            print(f"[BASE] Worst fixed policy: {worst_fixed['baseline']} (loss_scaled={worst_fixed['mean_loss_scaled']:.6f}, weighted)")
            print(f"[BASE] Range across fixed policies: {worst_fixed['mean_loss_scaled'] - best_fixed['mean_loss_scaled']:.6f}")
            
            # Show the difference between weighted and simple
            print(f"\n[BASE] Aggregation method impact:")
            print(f"  Best fixed (weighted): {best_fixed['mean_loss_scaled']:.6f}")
            print(f"  Best fixed (simple):   {best_fixed['mean_loss_simple']:.6f}")
            print(f"  Ratio: {best_fixed['mean_loss_scaled'] / best_fixed['mean_loss_simple']:.2f}x")   
    
    print("="*70)

    return summary

# --------------------------- CLI (optional) --------------------------- #

if __name__ == "__main__":  # pragma: no cover
    import argparse, yaml
    ap = argparse.ArgumentParser(description="Leakage-safe baselines for CRL momentum project")
    ap.add_argument("--config", type=str, default="config_crl.yaml")
    args = ap.parse_args()

    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    panel_path = cfg_get(cfg, ("panel_path", "data.panel_path"), "data/cdi_processed.pkl")
    train_end  = cfg_get(cfg, "train_end", "2023-12-31")
    test_start = cfg_get(cfg, "test_start", "2024-01-01")
    run_all_baselines(panel_path, train_end, test_start, cfg)
