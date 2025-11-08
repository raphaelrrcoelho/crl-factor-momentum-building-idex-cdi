# generate_report.py — Dissertation‑ready report generator (security→index, rich diagnostics)
# -*- coding: utf-8 -*-
"""
Creates a complete Chapter‑5‑grade set of tables, figures, and a Markdown summary
for the CRL momentum project — aligned with the leakage‑safe pipeline.

Outputs under <results_dir>/report/:
  Tables
    - performance_table.csv
    - by_horizon_table.csv
    - per_security_table.csv
    - security_leaderboard_top.csv
    - security_leaderboard_bottom.csv
    - policy_interpretation.csv       (regime selection probs)
    - ope_dr_summary.csv (copy/transform)
    - ablation_overview.csv (copy link) / ablation_diff_vs_base.csv )

  Figures (PNG, headless‑safe Matplotlib/Agg)
    - baselines_bar.png
    - rolling_cov_60d.png
    - loss_timeseries.png
    - width_timeseries.png
    - calibration_gap_by_h.png
    - coverage_heatmap_date_h.png
    - width_box_by_h.png
    - security_coverage_hist.png
    - security_loss_box_by_h.png
    - policy_evolution.png
    - policy_interpretation.png
    - ope_dr.png
    - causal_comparison.png
    - ablation_overview.png
    - ablation_by_h.png

  Text & metadata
    - chapter5_summary.md              ← sectioned, paste‑ready prose + figure callouts
    - falsification.json               (copied)
    - manifest.json

Design decisions
- Start from SECURITY‑LEVEL predictions produced by CRL and baselines, aggregate to index
  for time‑series figures / performance summaries. Uses the same panel loader as CRL.
- Keeps `generate_full_report(results_dir, out_dir)` for pipeline compatibility.
- Reads ablation outputs created by the new ablation module.
"""

from __future__ import annotations

import os
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Headless backend to avoid GUI deps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from crl_factor_bandit_conformal import (
    load_panel
)


# ============================= Generic IO helpers ============================= #

def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def _read_csv_safe(path: str, parse_dates: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        print(f"[WARN] Missing file: {path}")
        return None
    try:
        return pd.read_csv(path, parse_dates=parse_dates)
    except Exception as e:
        print(f"[WARN] Failed reading {path}: {e}")
        return None

def _write_json(path: str, obj: dict):
    try:
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, default=str)
    except Exception as e:
        print(f"[WARN] Could not write JSON {path}: {e}")

def _maybe_float(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except Exception:
        return default

def _resolve_panel_path(results_dir: str) -> str:
    """
    Locate panel path via <results_dir>/config_resolved.yaml, fallback to default.
    Mirrors the logic used in OPE to stay consistent with training.
    """
    cfg_path = os.path.join(results_dir, "config_resolved.yaml")
    if os.path.exists(cfg_path):
        try:
            import yaml
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            # try plain key
            if isinstance(cfg, dict) and "panel_path" in cfg and isinstance(cfg["panel_path"], str):
                return cfg["panel_path"]
            # dotted path "data.panel_path"
            cur = cfg
            for part in "data.panel_path".split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    cur = None; break
            if isinstance(cur, str):
                return cur
        except Exception:
            pass
    return "data/cdi_processed.pkl"


# ======================== Security→Index aggregation ========================= #

def _aggregate_security_to_index(df_sec: pd.DataFrame, panel_path: str) -> pd.DataFrame:
    """
    Aggregate security-level predictions to index level using the **same** panel weights
    used by the CRL runner (load_panel).
    """
    if df_sec is None or df_sec.empty:
        return pd.DataFrame()

    # Merge weights from the original panel
    try:
        from crl_factor_bandit_conformal import load_panel   # same loader as training
        panel = load_panel(panel_path)
        w = panel[["date", "id", "index_weight"]].drop_duplicates()
        dfw = df_sec.merge(w, on=["date", "id"], how="left")
        dfw["index_weight"] = dfw["index_weight"].fillna(1.0)
    except Exception as e:
        print(f"[WARN] Could not load panel for weights ({e}); using uniform weights.")
        dfw = df_sec.copy()
        dfw["index_weight"] = 1.0

    # Normalize per (date,h)
    dfw["w_norm"] = (
        dfw.groupby(["date", "h"])["index_weight"]
           .transform(lambda s: s / (s.sum() + 1e-12))
    )

    agg = {
        "y":      lambda x: np.average(x, weights=dfw.loc[x.index, "w_norm"]),
        "l":      lambda x: np.average(x, weights=dfw.loc[x.index, "w_norm"]),
        "u":      lambda x: np.average(x, weights=dfw.loc[x.index, "w_norm"]),
        "width_scaled":  lambda x: np.average(x, weights=dfw.loc[x.index, "w_norm"]),
        "loss_scaled":   lambda x: np.average(x, weights=dfw.loc[x.index, "w_norm"]),
        "covered":lambda x: np.average(x, weights=dfw.loc[x.index, "w_norm"]),
    }
    # include scaled columns
    for col in ["width_scaled","loss_scaled"]:
        if col in dfw.columns:
            agg[col] = lambda x, c=col: np.average(x, weights=dfw.loc[x.index, "w_norm"])
    if "alpha" in dfw.columns:
        agg["alpha"] = "first"

    out = (dfw.groupby(["date", "h"]).agg(agg)
             .reset_index()
             .sort_values(["date", "h"])
             .reset_index(drop=True))
    return out


# ============================== Plot primitives ============================== #

def _savefig(fig, path: str):
    ensure_dir(os.path.dirname(path))
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_rolling_coverage(scores_idx: pd.DataFrame, out_dir: str, window: int = 60, nominal: float = 0.90):
    if scores_idx is None or scores_idx.empty:
        return
    df = scores_idx.sort_values(["date", "h"]).copy()
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for h, g in df.groupby("h"):
        s = (g.set_index("date")["covered"].rolling(window, min_periods=max(10, window//3)).mean())
        ax.plot(s.index, s.values, label=f"h={int(h)}", linewidth=1.2)
    ax.axhline(nominal, linestyle="--", alpha=0.6, label=f"Target {nominal:.2f}")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Rolling coverage (index-weighted average across securities) (window={window})")
    ax.legend(); ax.grid(True, alpha=0.35)
    _savefig(fig, os.path.join(out_dir, f"rolling_cov_{window}d.png"))

def plot_timeseries_metric(scores_idx: pd.DataFrame, col: str, out_path: str, ylabel: str):
    if scores_idx is None or scores_idx.empty or col not in scores_idx:
        return
    df = scores_idx.sort_values(["date", "h"]).copy()
    fig, ax = plt.subplots(figsize=(10.2, 3.2))
    for h, g in df.groupby("h"):
        ax.plot(g["date"], g[col], label=f"h={int(h)}", linewidth=1.2)
    ax.set_title(f"Index-weighted average of security-level {col} by horizon")
    ax.set_ylabel(ylabel); ax.grid(True, alpha=0.35); ax.legend()
    _savefig(fig, out_path)

def plot_calibration_gap(scores_idx: pd.DataFrame, out_path: str):
    if scores_idx is None or scores_idx.empty:
        return
    # realized coverage minus 1-α
    if "alpha" in scores_idx.columns:
        target = (1.0 - scores_idx.groupby("h")["alpha"].first()).to_dict()
    else:
        # default if alpha not logged: 0.10
        target = {int(h): 0.90 for h in scores_idx["h"].unique()}
    byh = (scores_idx.groupby("h")
                      .agg(real_cov=("covered", "mean"))
                      .reset_index())
    byh["target"] = byh["h"].map(lambda h: target.get(int(h), 0.90))
    byh["gap"] = byh["real_cov"] - byh["target"]

    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.bar(byh["h"].astype(str), byh["gap"].values)
    ax.axhline(0.0, alpha=0.6)
    ax.set_title("Calibration gap by horizon (index-weighted avg of security-level coverage) (realized − target)")
    ax.set_ylabel("Gap")
    ax.grid(True, axis="y", alpha=0.35)
    _savefig(fig, out_path)


def plot_baseline_heatmap(summary_df, out_dir):
    """Heatmap of all fixed policies: momentum × horizon"""
    # Fix: Look for the correct column name ('model' not 'baseline')
    if 'baseline' in summary_df.columns:
        fixed = summary_df[summary_df["baseline"].str.startswith("fixed_")]
        name_col = "baseline"
    elif 'model' in summary_df.columns:
        fixed = summary_df[summary_df["model"].str.startswith("fixed_")]
        name_col = "model"
    else:
        print("[WARN] No baseline/model column found for heatmap")
        return
    
    if fixed.empty:
        print("[WARN] No fixed policies found for heatmap")
        return
    
    # Parse baseline names: "fixed_mom_w21v_h10" → mom=mom_w21v, h=10
    fixed = fixed.copy()
    fixed["momentum"] = fixed[name_col].str.extract(r"fixed_(.+)_h\d+")[0]
    fixed["h"] = fixed[name_col].str.extract(r"h(\d+)")[0].astype(int)
    
    pivot = fixed.pivot(index="momentum", columns="h", values="mean_loss")
    
    if pivot.empty:
        print("[WARN] Empty pivot table for heatmap")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r")
    
    # Labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Momentum")
    ax.set_title("Fixed Policy Performance Landscape (scaled loss)")
    
    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", 
                   color="white" if val > pivot.values.mean() else "black")
    
    plt.colorbar(im, ax=ax, label="Loss (↓ better)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "baseline_heatmap.png"), dpi=200)
    plt.close()

def plot_heatmap_coverage(scores_idx: pd.DataFrame, out_path: str):
    if scores_idx is None or scores_idx.empty:
        return
    piv = (scores_idx.pivot(index="date", columns="h", values="covered")
                      .sort_index())
    if piv.empty:
        return
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    im = ax.imshow(piv.values, aspect="auto")
    ax.set_yticks(np.linspace(0, len(piv.index)-1, 6, dtype=int))
    ax.set_yticklabels([str(p.date()) for p in piv.index[::max(1, len(piv)//5)]])
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([str(int(h)) for h in piv.columns])
    ax.set_xlabel("Horizon"); ax.set_title("Coverage heatmap (date × horizon)")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Coverage")
    _savefig(fig, out_path)

def plot_box_by_h(scores_sec: pd.DataFrame, value_col: str, out_path: str, title: str, ylabel: str):
    if scores_sec is None or scores_sec.empty or value_col not in scores_sec:
        return
    data = [scores_sec[scores_sec["h"] == h][value_col].dropna().values for h in sorted(scores_sec["h"].unique())]
    labels = [str(int(h)) for h in sorted(scores_sec["h"].unique())]
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_title(title); ax.set_ylabel(ylabel); ax.set_xlabel("Horizon")
    ax.grid(True, axis="y", alpha=0.35)
    _savefig(fig, out_path)

def plot_hist_per_security(security_tbl: pd.DataFrame, col: str, out_path: str, title: str, bins: int = 30):
    if security_tbl is None or security_tbl.empty or col not in security_tbl:
        return
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.hist(security_tbl[col].dropna().values, bins=bins)
    ax.set_title(title); ax.set_xlabel(col); ax.set_ylabel("Count")
    ax.grid(True, alpha=0.35)
    _savefig(fig, out_path)

def plot_bar(df: pd.DataFrame, x_col: str, y_col: str, out_path: str, title: str, rotate_xticks: int = 20):
    if df is None or df.empty or (x_col not in df or y_col not in df):
        return
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    d = df.copy().sort_values(y_col)
    x = np.arange(len(d))
    y = d[y_col].astype(float).values

    # Optional error bars: prefer NW SE; else 1/2 CI width
    yerr = None
    if "nw_se" in d.columns and d["nw_se"].notna().any():
        yerr = d["nw_se"].astype(float).values
    elif {"ci_lo","ci_hi"}.issubset(d.columns):
        yerr = ((d["ci_hi"] - d["ci_lo"]).astype(float).values) * 0.5

    if yerr is not None:
        ax.bar(x, y, yerr=yerr, capsize=3)
    else:
        ax.bar(x, y)

    ax.set_title(title); ax.set_ylabel(y_col)
    ax.set_xticks(x); ax.set_xticklabels(d[x_col].astype(str).tolist(), rotation=rotate_xticks, ha="right")
    ax.grid(True, axis="y", alpha=0.35)
    _savefig(fig, out_path)



# ========================== High‑level report builders ========================= #

import numpy as np

def _newey_west_se(x: np.ndarray, max_lags: int = 10) -> float:
    """
    Newey-West (HAC) SE for the mean of a (weakly) dependent series x.
    Bartlett kernel, population-variance style.
    """
    x = np.asarray(x, float)
    x = x - x.mean()
    T = len(x)
    if T <= 1:
        return np.nan
    gamma0 = np.dot(x, x) / T
    var = gamma0
    for lag in range(1, min(max_lags, T-1) + 1):
        gamma = np.dot(x[lag:], x[:-lag]) / T
        weight = 1.0 - lag / (max_lags + 1.0)
        var += 2.0 * weight * gamma
    return np.sqrt(var / T)

def _mbb_mean_ci(series: np.ndarray, block: int = 20, B: int = 1000, alpha: float = 0.10):
    """
    Moving Block Bootstrap (non-overlapping blocks, circular wrap).
    Returns (mean, lo, hi) for 1-alpha CI of the mean.
    """
    rng = np.random.default_rng(12345)
    x = np.asarray(series, float)
    T = len(x)
    if T == 0:
        return np.nan, np.nan, np.nan
    # Build circular blocks
    x2 = np.concatenate([x, x[:block-1]])
    nb = int(np.ceil(T / block))
    boot = np.empty(B)
    for b in range(B):
        idx0 = rng.integers(0, T, size=nb)
        samp = np.concatenate([x2[i:i+block] for i in idx0])[:T]
        boot[b] = samp.mean()
    m = x.mean()
    lo = np.quantile(boot, alpha/2)
    hi = np.quantile(boot, 1 - alpha/2)
    return float(m), float(lo), float(hi)

# ============================================================================
# STATISTICAL TESTS ADDITIONS FOR generate_report.py
# ============================================================================
# Add these functions to your generate_report.py file
#
# These address issues #5, #11, #13, and #16 from the code review:
# - Statistical significance test for 21.7% improvement
# - Empirical coverage statistics
# - Observation count verification
# ============================================================================

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import json
import os


def diebold_mariano_test(series1: np.ndarray, series2: np.ndarray, 
                         h: int = 1, max_lags: int = 10) -> Dict[str, float]:
    """
    Diebold-Mariano test for comparing two forecast methods.
    Tests H0: Equal forecast accuracy vs H1: Different accuracy
    
    Parameters
    ----------
    series1 : np.ndarray
        Loss series from method 1 (e.g., CRL)
    series2 : np.ndarray
        Loss series from method 2 (e.g., Best Fixed)
    h : int
        Forecast horizon (for small-sample adjustment)
    max_lags : int
        Maximum lags for Newey-West HAC covariance
        
    Returns
    -------
    dict with keys: diff_mean, dm_stat, p_value, nw_se
    """
    from scipy import stats
    
    # Difference in losses
    d = series1 - series2
    n = len(d)
    
    if n < 2:
        return {"diff_mean": float(np.mean(d)), "dm_stat": np.nan, 
                "p_value": np.nan, "nw_se": np.nan}
    
    # Mean difference
    d_bar = np.mean(d)
    
    # Newey-West HAC standard error
    d_centered = d - d_bar
    gamma0 = np.dot(d_centered, d_centered) / n
    var = gamma0
    
    for lag in range(1, min(max_lags, n-1) + 1):
        gamma = np.dot(d_centered[lag:], d_centered[:-lag]) / n
        weight = 1.0 - lag / (max_lags + 1.0)
        var += 2.0 * weight * gamma
    
    # Small-sample adjustment for multi-step forecasts (Harvey et al. 1997)
    adjustment = np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
    se = np.sqrt(var / n)
    
    # DM statistic
    dm_stat = d_bar / (se * adjustment) if se > 0 else np.nan
    
    # Two-sided p-value (t-distribution with n-1 df)
    if np.isfinite(dm_stat):
        p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df=n-1))
    else:
        p_value = np.nan
    
    return {
        "diff_mean": float(d_bar),
        "dm_stat": float(dm_stat) if np.isfinite(dm_stat) else np.nan,
        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
        "nw_se": float(se) if np.isfinite(se) else np.nan,
        "n_obs": int(n)
    }


def compute_statistical_significance(results_dir: str, 
                                     crl_scores: pd.DataFrame,
                                     baseline_scores: pd.DataFrame,
                                     out_dir: str,
                                     panel_path: str) -> Dict[str, any]:
    """
    Compute statistical significance of CRL vs baselines using Diebold-Mariano test.
    Addresses Issue #11 from code review.
    
    Parameters
    ----------
    results_dir : str
        Results directory
    crl_scores : pd.DataFrame
        CRL security-level scores with columns: date, h, loss_scaled
    baseline_scores : pd.DataFrame
        Baseline security-level scores with columns: date, h, baseline, loss_scaled
    out_dir : str
        Output directory for results
        
    Returns
    -------
    Dict with test results
    """
    if crl_scores is None or crl_scores.empty:
        print("[STAT] No CRL scores available")
        return {}
    
    if baseline_scores is None or baseline_scores.empty:
        print("[STAT] No baseline scores available")
        return {}
    
    # Aggregate to date level using INDEX WEIGHTING (match performance table)
    # First aggregate security → index (weighted), then to date
    crl_sec = crl_scores.copy()
    if "index_weight" not in crl_sec.columns:
        # Load panel for weights
        panel = load_panel(panel_path)
        if isinstance(panel.index, pd.MultiIndex):
            panel = panel.reset_index()
        crl_sec = crl_sec.merge(
            panel[["date", "id", "index_weight"]].drop_duplicates(),
            on=["date", "id"],
            how="left"
        )
        crl_sec["index_weight"] = crl_sec["index_weight"].fillna(1.0)

    # Normalize weights per (date, h)
    crl_sec["w_norm"] = crl_sec.groupby(["date", "h"])["index_weight"].transform(
        lambda x: x / (x.sum() + 1e-12)
    )

    # Index-weighted aggregation per date
    crl_daily = (crl_sec.groupby("date")
                        .apply(lambda g: np.average(g["loss_scaled"], weights=g["w_norm"]))
                        .rename("crl_loss"))
    
    results = []
    
    # Test against each baseline
    for baseline_name, baseline_data in baseline_scores.groupby("baseline"):
        # Merge weights into baseline
        base_sec = baseline_data.copy()
        if "index_weight" not in base_sec.columns:
            panel = load_panel(panel_path)
            if isinstance(panel.index, pd.MultiIndex):
                panel = panel.reset_index()
            base_sec = base_sec.merge(
                panel[["date", "id", "index_weight"]].drop_duplicates(),
                on=["date", "id"],
                how="left"
            )
            base_sec["index_weight"] = base_sec["index_weight"].fillna(1.0)
        
        # Normalize weights per (date, h)
        base_sec["w_norm"] = base_sec.groupby(["date", "h"])["index_weight"].transform(
            lambda x: x / (x.sum() + 1e-12)
        )
        
        # Aggregate baseline to date level using index weighting
        baseline_daily = (base_sec.groupby("date")
                                  .apply(lambda g: np.average(g["loss_scaled"], weights=g["w_norm"]))
                                  .sort_index()
                                  .rename(f"{baseline_name}_loss"))
        
        # Merge on common dates
        merged = pd.concat([crl_daily, baseline_daily], axis=1).dropna()
        
        if len(merged) < 10:
            print(f"[STAT] Insufficient data for {baseline_name}: {len(merged)} observations")
            continue
        
        # Run Diebold-Mariano test
        dm_result = diebold_mariano_test(
            merged["crl_loss"].values,
            merged[f"{baseline_name}_loss"].values,
            h=1,  # Daily predictions
            max_lags=10
        )
        
        # Calculate improvement percentage
        crl_mean = merged["crl_loss"].mean()
        baseline_mean = merged[f"{baseline_name}_loss"].mean()
        improvement_pct = (baseline_mean - crl_mean) / baseline_mean * 100
        
        results.append({
            "baseline": baseline_name,
            "crl_loss": float(crl_mean),
            "baseline_loss": float(baseline_mean),
            "improvement_pct": float(improvement_pct),
            "dm_statistic": dm_result["dm_stat"],
            "p_value": dm_result["p_value"],
            "nw_se": dm_result["nw_se"],
            "n_dates": dm_result["n_obs"],
            "significant_5pct": dm_result["p_value"] < 0.05 if np.isfinite(dm_result["p_value"]) else False,
            "significant_1pct": dm_result["p_value"] < 0.01 if np.isfinite(dm_result["p_value"]) else False,
        })
    
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(out_dir, "statistical_significance.csv"), index=False)
        
        # Print summary
        print("\n" + "="*70)
        print("STATISTICAL SIGNIFICANCE TESTS (Diebold-Mariano)")
        print("INDEX-WEIGHTED AGGREGATION")
        print("="*70)
        for _, row in results_df.iterrows():
            sig_marker = "***" if row["significant_1pct"] else ("**" if row["significant_5pct"] else "")
            print(f"\nCRL vs {row['baseline']}:")
            print(f"  Improvement: {row['improvement_pct']:.2f}% {sig_marker}")
            print(f"  DM statistic: {row['dm_statistic']:.3f}")
            print(f"  p-value: {row['p_value']:.4f}")
            print(f"  CRL loss: {row['crl_loss']:.6f}")
            print(f"  {row['baseline']} loss: {row['baseline_loss']:.6f}")
        print("="*70)
        print("** = significant at 5% level, *** = significant at 1% level")
        print("="*70 + "\n")
    
    return {"results": results, "results_df": results_df if results else None}


def compute_coverage_statistics(results_dir: str, 
                                crl_scores: pd.DataFrame,
                                out_dir: str,
                                nominal_coverage: float = 0.90) -> Dict:
    """
    Compute comprehensive coverage statistics.
    Addresses Issues #5 and #16 from code review.
    
    Parameters
    ----------
    results_dir : str
        Results directory
    crl_scores : pd.DataFrame
        CRL security-level scores with columns: date, h, covered
    out_dir : str
        Output directory
    nominal_coverage : float
        Nominal coverage level (default 0.90 for alpha=0.10)
        
    Returns
    -------
    Dict with coverage statistics
    """
    if crl_scores is None or crl_scores.empty or "covered" not in crl_scores.columns:
        print("[COV] No coverage data available")
        return {}
    
    # Overall coverage
    overall_cov = crl_scores["covered"].mean()
    
    # Coverage by horizon
    by_horizon = (crl_scores.groupby("h")
                            .agg(
                                coverage=("covered", "mean"),
                                n_predictions=("covered", "count"),
                                min_coverage=("covered", lambda x: x.min()),
                                max_coverage=("covered", lambda x: x.max())
                            )
                            .reset_index())
    
    # Coverage over time (rolling 60-day window)
    crl_daily = (crl_scores.groupby("date")
                           .agg(coverage=("covered", "mean"))
                           .sort_index())
    
    rolling_cov = crl_daily["coverage"].rolling(window=60, min_periods=10).mean()
    
    # Coverage distribution (histogram bins)
    cov_by_date = crl_scores.groupby("date")["covered"].mean()
    hist, bins = np.histogram(cov_by_date.values, bins=20, range=(0, 1))
    
    # Statistical test: Is coverage significantly different from nominal?
    from scipy import stats
    n = len(cov_by_date)
    coverage_test = stats.binomtest(
        int(overall_cov * len(crl_scores)),
        len(crl_scores),
        nominal_coverage,
        alternative='two-sided'
    )
    
    results = {
        "overall_coverage": float(overall_cov),
        "nominal_coverage": float(nominal_coverage),
        "coverage_deviation": float(overall_cov - nominal_coverage),
        "coverage_deviation_pct": float((overall_cov - nominal_coverage) / nominal_coverage * 100),
        "coverage_test": str(coverage_test),
        "n_predictions": int(len(crl_scores)),
        "by_horizon": by_horizon.to_dict('records'),
        "min_date_coverage": float(cov_by_date.min()),
        "max_date_coverage": float(cov_by_date.max()),
        "std_date_coverage": float(cov_by_date.std()),
    }
    
    # Save results
    with open(os.path.join(out_dir, "coverage_statistics.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    by_horizon.to_csv(os.path.join(out_dir, "coverage_by_horizon.csv"), index=False)
    
    # Print summary
    print("\n" + "="*70)
    print("COVERAGE STATISTICS")
    print("="*70)
    print(f"Overall coverage: {overall_cov:.3f} (nominal: {nominal_coverage:.3f})")
    print(f"Deviation: {results['coverage_deviation']:.3f} ({results['coverage_deviation_pct']:.2f}%)")
    print(f"Coverage test: {coverage_test}")
    print(f"\nCoverage by horizon:")
    print(by_horizon.to_string(index=False))
    print(f"\nDate-level coverage range: [{results['min_date_coverage']:.3f}, {results['max_date_coverage']:.3f}]")
    print(f"Date-level coverage std: {results['std_date_coverage']:.3f}")
    print("="*70 + "\n")
    
    return results


def verify_observation_counts(results_dir: str,
                              crl_scores: pd.DataFrame,
                              out_dir: str) -> Dict:
    """
    Verify and report observation counts.
    Addresses Issue #13 from code review.
    
    Parameters
    ----------
    results_dir : str
        Results directory
    crl_scores : pd.DataFrame
        CRL security-level scores
    out_dir : str
        Output directory
        
    Returns
    -------
    Dict with observation statistics
    """
    if crl_scores is None or crl_scores.empty:
        print("[OBS] No data available")
        return {}
    
    # Basic counts
    n_total = len(crl_scores)
    n_securities = crl_scores["id"].nunique()
    n_dates = crl_scores["date"].nunique()
    avg_per_security = n_total / n_securities if n_securities > 0 else 0
    avg_per_date = n_total / n_dates if n_dates > 0 else 0
    
    # By horizon
    by_horizon = (crl_scores.groupby("h")
                            .agg(
                                n_predictions=("id", "count"),
                                n_securities=("id", "nunique"),
                                n_dates=("date", "nunique")
                            )
                            .reset_index())
    
    # Date range
    date_min = crl_scores["date"].min()
    date_max = crl_scores["date"].max()
    
    # Securities with predictions
    securities_by_count = (crl_scores.groupby("id")
                                     .size()
                                     .describe())
    
    results = {
        "total_predictions": int(n_total),
        "unique_securities": int(n_securities),
        "unique_dates": int(n_dates),
        "avg_predictions_per_security": float(avg_per_security),
        "avg_predictions_per_date": float(avg_per_date),
        "date_range": {
            "start": str(date_min.date()) if hasattr(date_min, 'date') else str(date_min),
            "end": str(date_max.date()) if hasattr(date_max, 'date') else str(date_max)
        },
        "by_horizon": by_horizon.to_dict('records'),
        "predictions_per_security": {
            "min": int(securities_by_count["min"]),
            "25pct": int(securities_by_count["25%"]),
            "median": int(securities_by_count["50%"]),
            "75pct": int(securities_by_count["75%"]),
            "max": int(securities_by_count["max"]),
            "mean": float(securities_by_count["mean"]),
            "std": float(securities_by_count["std"])
        }
    }
    
    # Save results
    with open(os.path.join(out_dir, "observation_counts.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    by_horizon.to_csv(os.path.join(out_dir, "observations_by_horizon.csv"), index=False)
    
    # Print summary
    print("\n" + "="*70)
    print("OBSERVATION COUNT VERIFICATION")
    print("="*70)
    print(f"Total predictions: {n_total:,}")
    print(f"Unique securities: {n_securities}")
    print(f"Unique dates: {n_dates}")
    print(f"Average predictions per security: {avg_per_security:.1f}")
    print(f"Average predictions per date: {avg_per_date:.1f}")
    print(f"Date range: {results['date_range']['start']} to {results['date_range']['end']}")
    print(f"\nPredictions per security distribution:")
    print(f"  Min: {results['predictions_per_security']['min']}")
    print(f"  Median: {results['predictions_per_security']['median']}")
    print(f"  Max: {results['predictions_per_security']['max']}")
    print(f"  Mean: {results['predictions_per_security']['mean']:.1f}")
    print(f"\nBy horizon:")
    print(by_horizon.to_string(index=False))
    print("="*70 + "\n")
    
    return results


def _build_performance_table(results_dir: str, out_dir: str, panel_path: str
                             ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Combine learned policy (security→index) with baseline summaries (security).
    Returns (perf_table, crl_scores_sec, crl_scores_idx).
    """
    print("[REPORT] Building performance table…")
    ensure_dir(out_dir)

    # CRL security-level scores → index aggregation (time series by date × h)
    crl_sec = _read_csv_safe(os.path.join(results_dir, "crl_scores_securities.csv"), parse_dates=["date"])
    crl_idx = _aggregate_security_to_index(crl_sec, panel_path) if crl_sec is not None else None

    rows: List[Dict] = []

    def _series_and_stats(g_idx: pd.DataFrame, label: str) -> Dict:
        # Choose loss column: prefer scaled for cross-h comparability
        loss_col = "loss_scaled" if "loss_scaled" in g_idx.columns else "loss"
        if loss_col not in g_idx:
            return {}
        # date-level series (average across h per date)
        s = (g_idx.groupby("date")[loss_col]
                    .mean()
                    .sort_index())
        if s.empty:
            return {}
        nw = _newey_west_se(s.values, max_lags=10)
        m, lo, hi = _mbb_mean_ci(s.values, block=20, B=1000, alpha=0.10)
        return dict(model=label,
                    mean_loss=float(m),
                    nw_se=float(nw),
                    ci_lo=float(lo),
                    ci_hi=float(hi))

    if crl_idx is not None and not crl_idx.empty:
        rows.append(_series_and_stats(crl_idx, "CRL"))
    else:
        print("[WARN] CRL security-level scores not found.")

    # Baselines: read all security-level rows, aggregate to index like CRL, then compute same stats
    all_base = _read_csv_safe(os.path.join(results_dir, "all_baseline_scores_securities.csv"), parse_dates=["date"])
    if all_base is not None and not all_base.empty:
        for b, g in all_base.groupby("baseline"):
            g_idx = _aggregate_security_to_index(g, panel_path)
            stats = _series_and_stats(g_idx, str(b))
            if stats: rows.append(stats)
    else:
        print("[WARN] Baseline security-level scores not found.")

    perf = pd.DataFrame([r for r in rows if r]) if rows else None
    
    if perf is not None and not perf.empty:
        perf.to_csv(os.path.join(out_dir, "performance_table.csv"), index=False)
        
        # For bar plot: show CRL, Random, and best/worst fixed (to avoid clutter)
        perf_sorted = perf.sort_values("mean_loss")
        fixed_only = perf_sorted[perf_sorted["model"].str.startswith("fixed_")]
        
        if len(fixed_only) > 0:
            # Select: CRL, Random, Best Fixed, Worst Fixed
            plot_subset = []
            
            # CRL
            crl_row = perf_sorted[perf_sorted["model"] == "CRL"]
            if not crl_row.empty:
                plot_subset.append(crl_row.iloc[0])
            
            # Random
            random_row = perf_sorted[perf_sorted["model"] == "random"]
            if not random_row.empty:
                plot_subset.append(random_row.iloc[0])
            
            # Best 3 fixed
            for i in range(min(3, len(fixed_only))):
                plot_subset.append(fixed_only.iloc[i])
            
            # Worst 3 fixed
            for i in range(max(0, len(fixed_only)-3), len(fixed_only)):
                if i < 3:  # Don't duplicate if < 6 total
                    continue
                plot_subset.append(fixed_only.iloc[i])
            
            plot_df = pd.DataFrame(plot_subset)
        else:
            plot_df = perf_sorted
        
        plot_bar(plot_df, "model", "mean_loss",
                 os.path.join(out_dir, "baselines_bar.png"),
                 "Mean interval score: CRL vs. selected baselines (↓ better)")
    
    return perf, crl_sec, crl_idx

def _baseline_analysis(results_dir: str, out_dir: str, perf: Optional[pd.DataFrame]):
    """
    Generate detailed baseline comparison: heatmap + best-fixed analysis.
    """
    if perf is None or perf.empty:
        return
    
    # Generate heatmap of all fixed policies
    plot_baseline_heatmap(perf, out_dir)
    
    # Determine which column name is used
    col_name = "baseline" if "baseline" in perf.columns else "model"
    
    # Analysis: CRL vs best-in-hindsight
    crl_perf = perf[perf[col_name] == "CRL"]
    random_perf = perf[perf[col_name] == "random"]
    fixed_only = perf[perf[col_name].str.startswith("fixed_")].sort_values("mean_loss")
    
    if not crl_perf.empty and not fixed_only.empty:
        crl_loss = crl_perf.iloc[0]["mean_loss"]
        best_fixed = fixed_only.iloc[0]
        worst_fixed = fixed_only.iloc[-1]
        
        # Calculate improvements
        improvement_vs_best = (best_fixed["mean_loss"] - crl_loss) / best_fixed["mean_loss"] * 100
        improvement_vs_worst = (worst_fixed["mean_loss"] - crl_loss) / worst_fixed["mean_loss"] * 100
        
        if not random_perf.empty:
            random_loss = random_perf.iloc[0]["mean_loss"]
            improvement_vs_random = (random_loss - crl_loss) / random_loss * 100
        else:
            random_loss = np.nan
            improvement_vs_random = np.nan
        
        # Save comparison table
        comparison = pd.DataFrame([
            {"policy": "CRL (learned)", "mean_loss": crl_loss, "improvement_vs_best_%": 0.0},
            {"policy": f"Best Fixed ({best_fixed[col_name]})", "mean_loss": best_fixed["mean_loss"], 
             "improvement_vs_best_%": -improvement_vs_best},
            {"policy": f"Worst Fixed ({worst_fixed[col_name]})", "mean_loss": worst_fixed["mean_loss"], 
             "improvement_vs_best_%": -(improvement_vs_worst - improvement_vs_best)},
            {"policy": "Random (uniform)", "mean_loss": random_loss, 
             "improvement_vs_best_%": -(improvement_vs_random - improvement_vs_best) if not np.isnan(random_loss) else np.nan},
        ])
        comparison.to_csv(os.path.join(out_dir, "baseline_comparison.csv"), index=False)
        
        # Print summary
        print(f"\n[REPORT] Baseline Comparison:")
        print(f"  CRL loss: {crl_loss:.6f}")
        print(f"  Best fixed: {best_fixed[col_name]} (loss={best_fixed['mean_loss']:.6f})")
        print(f"  Improvement: {improvement_vs_best:.2f}% better than best-in-hindsight fixed policy")
        print(f"  Worst fixed: {worst_fixed[col_name]} (loss={worst_fixed['mean_loss']:.6f})")
        print(f"  Fixed policy range: {worst_fixed['mean_loss'] - best_fixed['mean_loss']:.6f}")

def _build_by_horizon_table(scores_idx: pd.DataFrame, out_dir: str) -> Optional[pd.DataFrame]:
    """
    Horizon‑level summary (index‑weighted): mean loss, coverage, width by h.
    """
    if scores_idx is None or scores_idx.empty:
        return None
    byh = (scores_idx.groupby("h")
                    .agg(mean_loss=("loss_scaled","mean"),
                         coverage=("covered","mean"),
                         mean_width=("width_scaled","mean"),
                         n_days=("date","nunique"))
                    .reset_index())
    byh.to_csv(os.path.join(out_dir, "by_horizon_table.csv"), index=False)
    return byh


def _build_per_security_table(scores_sec: pd.DataFrame, out_dir: str) -> Optional[pd.DataFrame]:
    """
    Security‑level diagnostics (over whole test window): mean loss, coverage, width.
    Also produce top/bottom leaderboards for coverage and loss.
    """
    if scores_sec is None or scores_sec.empty:
        return None
    sec = (scores_sec.groupby("id")
                      .agg(mean_loss=("loss_scaled","mean"),
                           coverage=("covered","mean"),
                           mean_width=("width_scaled","mean"),
                           n_predictions=("loss_scaled","count"))
                      .reset_index()
                      .sort_values("mean_loss"))
    sec.to_csv(os.path.join(out_dir, "per_security_table.csv"), index=False)

    top = sec.nsmallest(30, "mean_loss"); bot = sec.nlargest(30, "mean_loss")
    top.to_csv(os.path.join(out_dir, "security_leaderboard_top.csv"), index=False)
    bot.to_csv(os.path.join(out_dir, "security_leaderboard_bottom.csv"), index=False)

    # visual morsels for "many securities"
    plot_hist_per_security(sec, "coverage", os.path.join(out_dir, "security_coverage_hist.png"),
                           "Cross‑sectional distribution of per‑security coverage")
    plot_box_by_h(scores_sec, "loss_scaled", os.path.join(out_dir, "security_loss_box_by_h.png"),
                  "Per‑security mean scaled loss distribution by horizon", "Mean interval score")
    return sec


def _policy_evolution(results_dir: str, out_dir: str):
    """
    Timeline of selected (momentum, horizon) with legend map.
    """
    choices = _read_csv_safe(os.path.join(results_dir, "crl_policy_choices.csv"), parse_dates=["date"])
    if choices is None or choices.empty or not {"momentum","h","date"}.issubset(choices.columns):
        print("[REPORT] No choices file; skipping policy evolution plot.")
        return

    # Encode categories deterministically
    s = choices["momentum"].astype(str)
    code_map = {k: i for i, k in enumerate(sorted(s.dropna().unique().tolist()))}
    m_code = s.map(code_map).values
    t = choices["date"].values; h = choices["h"].astype(int).values

    fig, axs = plt.subplots(2, 1, figsize=(11.5, 5.2), sharex=True)
    axs[0].plot(t, m_code, linewidth=1.2); axs[0].set_ylabel("Momentum code"); axs[0].grid(True, alpha=0.25)
    axs[0].set_title("Policy evolution: selected specs over time")
    axs[1].plot(t, h, linewidth=1.2); axs[1].set_ylabel("Horizon (steps)"); axs[1].grid(True, alpha=0.25); axs[1].set_xlabel("Date")
    _savefig(fig, os.path.join(out_dir, "policy_evolution.png"))

    # Save legend mapping
    with open(os.path.join(out_dir, "policy_evolution_legend.json"), "w") as f:
        json.dump({"momentum_code_map": code_map}, f, indent=2)


def _policy_interpretation(results_dir: str, out_dir: str, scores_sec: Optional[pd.DataFrame]):
    """
    Group choices by volatility regimes → selection probability for short momentum.
    (Keeps same output contract as the previous module version.)
    """
    choices = _read_csv_safe(os.path.join(results_dir, "crl_policy_choices.csv"), parse_dates=["date"])
    if choices is None or choices.empty:
        return
    df = choices.copy()

    # Preferred: if CRL logged ctx volatility, use it (CRL runner writes ctx_retvol20 / idx_vol).
    def _tertiles(x: pd.Series):
        z = x.dropna().values
        return (np.quantile(z, 1/3), np.quantile(z, 2/3)) if z.size else (0.0, 0.0)

    reg = None
    if "ctx_retvol20" in df.columns and df["ctx_retvol20"].notna().any():
        lo, hi = _tertiles(df["ctx_retvol20"])
        reg = df["ctx_retvol20"].apply(lambda v: "Low" if v < lo else ("Mid" if v < hi else "High"))
    elif scores_sec is not None and not scores_sec.empty and {"date","y"}.issubset(scores_sec.columns):
        # Fallback: build a 60d rolling vol from the index‑level realized return (aggregated from y)
        panel_path = _resolve_panel_path(results_dir)
        idx = _aggregate_security_to_index(scores_sec, panel_path)[["date","y"]].dropna().sort_values("date")
        idx["ret_vol"] = idx["y"].rolling(60, min_periods=10).std()
        df = df.merge(idx[["date","ret_vol"]], on="date", how="left")
        lo, hi = _tertiles(df["ret_vol"])
        reg = df["ret_vol"].apply(lambda v: "Low" if v < lo else ("Mid" if v < hi else "High"))
    else:
        reg = pd.Series(["Mid"]*len(df))

    df["regime"] = reg
    df["mom_short"] = df["momentum"].astype(str).str.contains(r"mom_w5|mom_w10|mom_w21").astype(int)

    probs = (df.groupby("regime")[["mom_short"]]
               .mean()
               .reindex(["Low","Mid","High"])
               .reset_index())
    probs.to_csv(os.path.join(out_dir, "policy_interpretation.csv"), index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    x = np.arange(len(probs["regime"])); w = 0.40
    ax.bar(x + w/2, probs["mom_short"].values, width=w, label="Short momentum (≤21)")
    ax.set_xticks(x); ax.set_xticklabels(probs["regime"].tolist())
    ax.set_ylim(0, 1.0); ax.set_ylabel("Selection probability")
    ax.set_title("Policy preferences by volatility regime"); ax.grid(True, axis="y", alpha=0.35); ax.legend()
    _savefig(fig, os.path.join(out_dir, "policy_interpretation.png"))


def _plot_ope(results_dir: str, out_dir: str):
    """
    Plot DR risk for learned policy and baselines if OPE summary exists.
    (Reads the CSV produced by the OPE step.)
    """
    ope = _read_csv_safe(os.path.join(results_dir, "ope_dr_summary.csv"))
    if ope is None or ope.empty or not set(["policy","DR_risk"]).issubset(ope.columns):
        print("[REPORT] No OPE summary; skipping OPE plot.")
        return
    fig, ax = plt.subplots(figsize=(9.2, 4.1))
    d = ope.copy().sort_values("DR_risk")
    ax.bar(d["policy"].astype(str), d["DR_risk"].astype(float).values)
    ax.set_ylabel("Doubly‑Robust Risk (↓ better)")
    ax.set_title("Off‑Policy Evaluation (DR)")
    ax.set_xticks(range(len(d))); ax.set_xticklabels(d["policy"].astype(str).tolist(), rotation=28, ha="right")
    ax.grid(True, axis="y", alpha=0.35)
    _savefig(fig, os.path.join(out_dir, "ope_dr.png"))


def _plot_causal(results_dir: str, out_dir: str):
    """
    Plot learned vs counterfactual fixed policies (from causal_effects_summary.csv).
    """
    path = os.path.join(results_dir, "causal_effects_summary.csv")
    df = _read_csv_safe(path)
    if df is None or df.empty or not set(["policy","mean_return"]).issubset(df.columns):
        return
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    d = df.sort_values("mean_return").copy()
    highlight = ["green" if "learned" in str(p).lower() else None for p in d["policy"]]
    ax.bar(d["policy"].astype(str), d["mean_return"].astype(float).values)
    ax.set_ylabel("Mean return"); ax.set_title("Learned policy vs counterfactual fixed policies")
    ax.axhline(0, alpha=0.3)
    ax.set_xticks(range(len(d))); ax.set_xticklabels(d["policy"].astype(str).tolist(), rotation=30, ha="right")
    ax.grid(True, axis="y", alpha=0.35)
    _savefig(fig, os.path.join(out_dir, "causal_comparison.png"))


def _plot_ablation(results_dir: str, out_dir: str):
    """
    Read ablation/overview.csv and visualize Δ loss vs BASE (overall and by horizon).
    (Matches the artifacts produced by the upgraded ablation module.)
    """
    abl_dir = os.path.join(results_dir, "ablation")
    over = _read_csv_safe(os.path.join(abl_dir, "overview.csv"))
    if over is None or over.empty:
        return

    # Overall bar with MBB CI whiskers
    ov = over[over["scope"] == "overall"].copy()
    if not ov.empty:
        fig, ax = plt.subplots(figsize=(9.5, 4.0))
        x = np.arange(len(ov))
        y = ov["mean_diff"].values
        lo = ov["mbb_ci_lo"].values; hi = ov["mbb_ci_hi"].values
        err = np.vstack([y - lo, hi - y])
        ax.bar(x, y, yerr=err, capsize=3)
        ax.axhline(0, alpha=0.5)
        ax.set_xticks(x); ax.set_xticklabels(ov["variant"].astype(str).tolist(), rotation=18, ha="right")
        ax.set_ylabel("Δ mean loss vs BASE (↓ better)"); ax.set_title("Ablation overview (overall)")
        ax.grid(True, axis="y", alpha=0.3)
        _savefig(fig, os.path.join(out_dir, "ablation_overview.png"))

    # By horizon heatmap
    bh = over[over["scope"].str.startswith("h=")].copy()
    if not bh.empty:
        bh["h"] = bh["scope"].str.replace("h=","",regex=False).astype(int)
        piv = bh.pivot(index="variant", columns="h", values="mean_diff").sort_index()
        fig, ax = plt.subplots(figsize=(10.5, 5.2))
        im = ax.imshow(piv.values, aspect="auto")
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index.tolist())
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([str(int(h)) for h in piv.columns])
        ax.set_xlabel("Horizon (steps)"); ax.set_title("Ablation: Δ mean loss by horizon (vs BASE)")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i,j]:.3f}", ha="center", va="center", fontsize=7, color="white")
        fig.colorbar(im, ax=ax, shrink=0.9, label="Δ mean loss")
        _savefig(fig, os.path.join(out_dir, "ablation_by_h.png"))


def _copy_falsification(results_dir: str, out_dir: str):
    src = os.path.join(results_dir, "falsification.json")
    if os.path.exists(src):
        try:
            with open(src, "r") as f:
                data = json.load(f)
            _write_json(os.path.join(out_dir, "falsification.json"), data)
        except Exception as e:
            print(f"[WARN] Could not copy falsification.json: {e}")


# =========================== Chapter‑5 markdown writer ========================= #

def _load_analysis_meta(results_dir: str) -> dict:
    """
    Pulls high‑level numbers from analysis.json produced by the CRL runner.
    """
    path = os.path.join(results_dir, "analysis.json")
    if not os.path.exists(path): return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _chapter5_markdown(results_dir: str,
                       out_dir: str,
                       perf: Optional[pd.DataFrame],
                       byh: Optional[pd.DataFrame],
                       sec_tbl: Optional[pd.DataFrame]):
    """
    Generate a structured, paste‑ready Chapter‑5 summary with figure callouts.
    """
    meta = _load_analysis_meta(results_dir)  # includes counts, by‑h stats if available
    md = []
    md.append(f"# Chapter 5 – Results\n")
    md.append(f"**Experiment folder:** `{os.path.abspath(results_dir)}`\n")

    # Data snapshot
    # if meta:
    #     n_rows = meta.get("n_security_predictions")
    #     n_ids  = meta.get("n_unique_securities")
    #     n_dt   = meta.get("n_unique_dates")
    #     md.append("## 5.1 Data snapshot")
    #     md.append(f"- Security–date observations: **{n_rows:,}**")
    #     md.append(f"- Unique securities: **{n_ids:,}**")
    #     md.append(f"- Unique dates: **{n_dt:,}**")
    #     if "horizons" in meta:
    #         md.append(f"- Horizons: **{', '.join(map(str, meta['horizons']))}**")
    #     md.append("")

    # Predictive interval quality
    md.append("## 5.2 Predictive interval quality")
    md.append("- *Figure*: `rolling_cov_60d.png` shows the 60‑day rolling coverage per horizon; the dashed line is the coverage target (1−α).")
    md.append("- *Figure*: `calibration_gap_by_h.png` reports the realized minus target coverage by horizon (bars close to 0 indicate good calibration).")
    md.append("- *Figures*: `loss_timeseries.png` and `width_timeseries.png` plot index‑level loss/width time‑series by horizon.")
    md.append("All coverage/width/loss statistics below are computed at the debenture level and then averaged across the cross-section using index weights (per date × horizon).")

    if byh is not None and not byh.empty:
        md.append("")
        md.append("**Horizon summary (index‑weighted)**")
        md.append(byh.to_markdown(index=False))

    # Cross‑sectional diagnostics
    md.append("\n## 5.3 Cross‑sectional diagnostics")
    md.append("- *Table*: `per_security_table.csv` lists per‑security mean loss, coverage and width; see `security_leaderboard_top.csv` and `security_leaderboard_bottom.csv`.")
    md.append("- *Figures*: `security_coverage_hist.png` (distribution of per‑security coverage) and `security_loss_box_by_h.png` (per‑security mean loss by horizon).")

    # Policy behavior
    md.append("\n## 5.4 Policy behavior and regimes")
    md.append("- *Figure*: `policy_evolution.png` depicts the momentum/horizon choices over time.")
    md.append("- *Figure & Table*: `policy_interpretation.png` and `policy_interpretation.csv` show the policy's short‑momentum preference across volatility regimes.")

    # Baselines and OPE
    md.append("\n## 5.5 Baselines and off-policy evaluation")
    md.append("- *Figure*: `baselines_bar.png` compares CRL against selected baselines (best/worst fixed + random).")
    md.append("- *Figure*: `baseline_heatmap.png` shows the complete performance landscape across all (momentum, horizon) combinations.")
    md.append("- *Table*: `baseline_comparison.csv` quantifies CRL's improvement over best-in-hindsight fixed policy.")
    md.append("- *Table*: `performance_table.csv` contains complete results for all baselines.")
    md.append("- *Figure*: `ope_dr.png` shows DR risk for the learned policy and baselines. Lower is better; DR combines DM and IPS for robustness.")

    # Causal & Counterfactuals
    md.append("\n## 5.6 Causal tests and counterfactuals")
    md.append("- *Figure*: `causal_comparison.png` compares the learned policy to fixed momentum policies on the **same decision dates**.")

    # Ablations
    md.append("\n## 5.7 Ablation study")
    md.append("- *Figures*: `ablation_overview.png` and `ablation_by_h.png` summarize Δ loss vs BASE with bootstrap CIs, overall and by horizon.")

    # Falsification
    md.append("\n## 5.8 Falsification checks")
    md.append("- See `falsification.json` (copied into `report/`) for placebo coverage and per‑horizon realized coverage.")

    # Append overview table
    if perf is not None and not perf.empty:
        md.append("\n## Appendix A — Performance table (all models)")
        md.append(perf.to_markdown(index=False))

    path = os.path.join(out_dir, "chapter5_summary.md")
    ensure_dir(out_dir)
    with open(path, "w") as f:
        f.write("\n".join(md))
    print(f"[REPORT] Wrote Chapter‑5 summary → {path}")


# ============================== Master orchestrator ============================== #

def generate_full_report(results_dir: str, out_dir: Optional[str] = None):
    """
    Build all assets under <results_dir>/report (or a custom out_dir).

    Pipeline integration: `run_experiments.step_report` calls this function.
    """
    out_dir = out_dir or os.path.join(results_dir, "report")
    ensure_dir(out_dir)

    # Resolve panel path for index aggregation from security level
    panel_path = _resolve_panel_path(results_dir)

    # 1) Performance table (+ baselines bar)
    perf, crl_sec, crl_idx = _build_performance_table(results_dir, out_dir, panel_path)

    # 1b) Detailed baseline analysis (heatmap + comparison)
    _baseline_analysis(results_dir, out_dir, perf)

    # 1c) Statistical significance tests
    all_base = _read_csv_safe(os.path.join(results_dir, "all_baseline_scores_securities.csv"), parse_dates=["date"])
    if perf is not None and crl_sec is not None and all_base is not None:
        compute_statistical_significance(results_dir, crl_sec, all_base, out_dir, panel_path)

    # 1d) Coverage statistics  
    if crl_sec is not None:
        compute_coverage_statistics(results_dir, crl_sec, out_dir, nominal_coverage=0.90)

    # 1e) Observation count verification
    if crl_sec is not None:
        verify_observation_counts(results_dir, crl_sec, out_dir)

    # 2) Horizon table + core time-series & calibration diagnostics
    byh = _build_by_horizon_table(crl_idx, out_dir)
    if crl_idx is not None and not crl_idx.empty:
        # Rolling coverage, loss and width timelines, calibration gap & heatmap
        nominal = 1.0 - (crl_idx["alpha"].dropna().iloc[-1] if "alpha" in crl_idx.columns and crl_idx["alpha"].notna().any() else 0.10)
        plot_rolling_coverage(crl_idx, out_dir, window=60, nominal=nominal)
        col_loss  = "loss_scaled"  if "loss_scaled"  in crl_idx.columns else "loss"
        col_width = "width_scaled" if "width_scaled" in crl_idx.columns else "width"
        plot_timeseries_metric(crl_idx, col_loss,  os.path.join(out_dir, "loss_timeseries.png"),  "Interval score (↓ better)")
        plot_timeseries_metric(crl_idx, col_width, os.path.join(out_dir, "width_timeseries.png"), "Mean width")
        plot_calibration_gap(crl_idx, os.path.join(out_dir, "calibration_gap_by_h.png"))
        plot_heatmap_coverage(crl_idx, os.path.join(out_dir, "coverage_heatmap_date_h.png"))

    # 3) Cross-sectional diagnostics (security tables + distribution plots)
    sec_tbl = _build_per_security_table(crl_sec, out_dir)

    # 4) Policy evolution & regime interpretation
    _policy_evolution(results_dir, out_dir)
    _policy_interpretation(results_dir, out_dir, crl_sec)

    # 5) OPE & Causal (if available)
    _plot_ope(results_dir, out_dir)
    _plot_causal(results_dir, out_dir)

    # 6) Ablation visuals (if available)
    _plot_ablation(results_dir, out_dir)

    # 7) Falsification (if available)
    _copy_falsification(results_dir, out_dir)

    # 8) Chapter-5 markdown (paste-ready prose and key tables)
    _chapter5_markdown(results_dir, out_dir, perf, byh, sec_tbl)

    # 9) Manifest
    try:
        assets = sorted([f for f in os.listdir(out_dir) if os.path.isfile(os.path.join(out_dir, f))])
    except Exception:
        assets = []
    _write_json(os.path.join(out_dir, "manifest.json"), {"results_dir": results_dir, "assets": assets})

    print(f"[REPORT] Done. Assets in: {out_dir}")

# ============================== CLI ============================== #

def main():
    ap = argparse.ArgumentParser(description="Generate dissertation‑grade report assets for CRL.")
    ap.add_argument("--results_dir", type=str, required=True, help="Experiment results directory")
    ap.add_argument("--out_dir", type=str, default=None, help="Optional custom output dir (default: <results_dir>/report)")
    args = ap.parse_args()

    generate_full_report(args.results_dir, args.out_dir)


if __name__ == "__main__":
    main()
