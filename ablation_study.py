# ablation_study.py — leakage-safe, pipeline-aligned ablation (security → index)
# -*- coding: utf-8 -*-
"""
Ablation study for CRL factor-spec selection (momentum, CQR)
SECURITY-LEVEL with index-weighted aggregation
====================================================================

What this module does
---------------------
- Launches multiple CRL runs with small parameter tweaks (overrides),
  each under <results_dir>/ablation/<tag>
- Reads **security-level** CRL outputs and aggregates to **index** level
  using panel weights (same loader used by CRL)
- Compares each variant to the BASE CRL using:
    * Moving-Block Bootstrap (MBB) confidence intervals for the
      mean loss-difference series (by horizon and overall)
    * Newey–West (HAC) t-test on the mean loss difference series

Artifacts
---------
<results_dir>/ablation/
  ├─ overview.csv              # one row per variant (+ base), overall & by horizon
  ├─ diff_vs_base.csv          # date-level Δloss series per variant & horizon
  ├─ meta.json                 # parameters, block length, etc.
  └─ <tag>/                    # each variant's full CRL outputs

CLI
---
python ablation_study.py --config config_crl.yaml [--only alpha_low confwin_1000] [--skip-existing]

Pipeline integration
--------------------
- run_experiments.step_ablation(cfg, ...) expects **two** return values:
  (overview_df, diffs_df); this module returns them and still writes files.  # :contentReference[oaicite:4]{index=4}
"""

from __future__ import annotations

import os
import json
import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

# Project modules
from run_config import cfg_get
from crl_factor_bandit_conformal import run_bandit, load_panel  # same loader as CRL  # :contentReference[oaicite:5]{index=5}


# ----------------------------- small utils ----------------------------- #

def _ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def _overrides_to_tag(overrides: Dict[str, Any]) -> str:
    parts = []
    for k, v in sorted(overrides.items()):
        vv = str(v).replace(" ", "")
        parts.append(f"{k}={vv}")
    tag = "__".join(parts)
    tag = tag.replace("/", "-")
    return tag


def _deep_update(d: dict, u: dict) -> dict:
    out = copy.deepcopy(d)
    for k, v in u.items():
        out[k] = v
    return out


# HAC (Newey–West) SE for the mean of a (date-ordered) series
def _newey_west_se(x: np.ndarray, L: Optional[int] = None) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n <= 1:
        return np.nan
    mu = float(np.mean(x))
    eps = x - mu
    if L is None:
        # Andrews-like bandwidth (heuristic)
        L = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    L = max(1, min(L, n - 1))
    gamma0 = np.dot(eps, eps) / n
    var = gamma0
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        cov = np.dot(eps[l:], eps[:-l]) / n
        var += 2.0 * w * cov
    var_mean = var / n
    se = float(np.sqrt(max(var_mean, 1e-18)))
    return se


# Moving Block Bootstrap CI for the mean
def _mbb_mean_ci(x: np.ndarray, block_len: int = 5, B: int = 2000, alpha: float = 0.05,
                 rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
    rng = rng or np.random.default_rng(123)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return (np.nan, np.nan)
    b = max(1, min(block_len, n))
    k = int(np.ceil(n / b))  # blocks per bootstrap path
    x_circ = np.concatenate([x, x[:b - 1]]) if b > 1 else x
    blocks = np.array([x_circ[i:i + b] for i in range(n)])  # n blocks of size b
    means = np.empty(B, dtype=float)
    for i in range(B):
        idx = rng.integers(0, n, size=k)
        path = np.concatenate(blocks[idx])[:n]
        means[i] = float(np.mean(path))
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


# ----------------------------- IO helpers ----------------------------- #

def _aggregate_security_to_index(df_sec: pd.DataFrame, panel_path: str) -> pd.DataFrame:
    """
    Aggregate security-level scores to index level using index weights
    found in the panel (same loader used by CRL).  # :contentReference[oaicite:6]{index=6}

    Returns DataFrame with columns: date, h, y, l, u, width, covered, loss
    """
    if df_sec is None or df_sec.empty:
        return pd.DataFrame()

    try:
        # Load panel for index weights
        panel = load_panel(panel_path)
        weight_df = panel[["date", "id", "index_weight"]].drop_duplicates()
        df_with_w = df_sec.merge(weight_df, on=["date", "id"], how="left")
        df_with_w["index_weight"] = df_with_w["index_weight"].fillna(1.0)
    except Exception as e:
        print(f"[ABL] Could not load panel for weights, using uniform: {e}")
        df_with_w = df_sec.copy()
        df_with_w["index_weight"] = 1.0

    # Normalize per (date, h)
    df_with_w["w_norm"] = (
        df_with_w.groupby(["date", "h"])["index_weight"]
        .transform(lambda x: x / (x.sum() + 1e-12))
    )

    # Weighted aggregation
    agg_dict = {
        "y":      lambda x: np.average(x, weights=df_with_w.loc[x.index, "w_norm"]),
        "l":      lambda x: np.average(x, weights=df_with_w.loc[x.index, "w_norm"]),
        "u":      lambda x: np.average(x, weights=df_with_w.loc[x.index, "w_norm"]),
        "width_scaled":  lambda x: np.average(x, weights=df_with_w.loc[x.index, "w_norm"]),
        "loss_scaled":   lambda x: np.average(x, weights=df_with_w.loc[x.index, "w_norm"]),
        "covered":lambda x: np.average(x, weights=df_with_w.loc[x.index, "w_norm"]),
    }
    df_idx = (df_with_w.groupby(["date", "h"]).agg(agg_dict).reset_index()
              .sort_values(["date", "h"]).reset_index(drop=True))
    return df_idx


def _read_idx_scores(results_dir: str, panel_path: str) -> pd.DataFrame:
    """
    Read CRL **security-level** scores and aggregate to index level.
    Expects: <results_dir>/crl_scores_securities.csv (security-level).  # :contentReference[oaicite:7]{index=7}
    """
    path = os.path.join(results_dir, "crl_scores_securities.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected security-level scores at {path}")

    df_sec = pd.read_csv(path, parse_dates=["date"])
    req = {"date", "id", "h", "y", "l", "u", "width_scaled", "covered", "loss_scaled"}
    missing = req - set(df_sec.columns)
    if missing:
        raise KeyError(f"Scores file missing columns: {missing}")

    return _aggregate_security_to_index(df_sec, panel_path)


# ----------------------------- core ----------------------------- #

@dataclass
class Variant:
    tag: str
    overrides: Dict[str, Any]


def _default_variants(cfg: dict) -> List[Variant]:
    """Curated small set; adjust as needed."""
    return [
        Variant("alpha_low",      {"alpha": 0.05}),
        Variant("alpha_high",     {"alpha": 0.20}),
        Variant("confwin_125",   {"conf_window": 125}),
        Variant("recency_0.02",   {"lambda_recency": 0.02}),
        Variant("lam_bandit_0.5", {"lam_bandit": 0.5}),
    ]


def _run_variant(cfg: dict, base_results: str, v: Variant, skip_existing: bool = False) -> str:
    """
    Run CRL with overrides in <base_results>/ablation/<tag>; return variant results_dir.

    Uses the same entry point as the main runner so we inherit all
    leakage-safe fixes (deferred updates, label-free gating) from CRL.  # :contentReference[oaicite:8]{index=8}
    """
    vdir = os.path.join(base_results, "ablation", v.tag)
    _ensure_dir(vdir)

    # Skip if security-level outputs already exist
    if skip_existing and os.path.exists(os.path.join(vdir, "crl_scores_securities.csv")):
        print(f"[ABL] Skipping {v.tag} (existing scores found)")
        return vdir

    # Shallow overrides; keep same panel & dates
    cfg_v = copy.deepcopy(cfg)
    cfg_v["results_dir"] = vdir
    for k, val in v.overrides.items():
        cfg_v[k] = val

    print(f"[ABL] Running variant: {v.tag} with overrides={v.overrides}")
    run_bandit(cfg_v)  # same API as base run  # :contentReference[oaicite:9]{index=9}
    return vdir


def _compare_variant_to_base(base_scores: pd.DataFrame, var_scores: pd.DataFrame,
                             tag: str, block_len: int = 5, B: int = 2000,
                             alpha: float = 0.05, rng: Optional[np.random.Generator] = None
                             ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (overview_rows_df, diff_rows_df) for this variant.
    Δ is computed on matched (date,h) index series.
    """
    rng = rng or np.random.default_rng(42)

    key = ["date", "h"]
    m = base_scores.merge(var_scores, on=key, suffixes=("_base", "_var"))
    if m.empty:
        raise RuntimeError("No overlapping (date,h) between base and variant.")

    m = m.sort_values(["date", "h"]).reset_index(drop=True)
    m["d_loss_scaled"] = m["loss_scaled_var"] - m["loss_scaled_base"]

    # Overall stats (pool horizons by date)
    series = (m.sort_values("date")
                .groupby("date")["d_loss_scaled"].mean()
                .to_numpy())
    mean_diff = float(np.mean(series))
    se_nw = _newey_west_se(series)
    t_stat = mean_diff / se_nw if (se_nw and np.isfinite(se_nw) and se_nw > 0) else np.nan
    lo, hi = _mbb_mean_ci(series, block_len=block_len, B=B, alpha=alpha, rng=rng)

    rows = [{
        "variant": tag,
        "scope": "overall",
        "mean_loss_base": float(base_scores["loss_scaled"].mean()),
        "mean_loss_var":  float(var_scores["loss_scaled"].mean()),
        "mean_diff": mean_diff,
        "hac_se": se_nw,
        "hac_t": t_stat,
        "mbb_ci_lo": lo,
        "mbb_ci_hi": hi,
        "n": int(series.size),
    }]

    # By horizon
    for h, g in m.groupby("h"):
        s = (g.sort_values("date")["d_loss_scaled"].to_numpy())
        md = float(np.mean(s))
        se = _newey_west_se(s)
        t  = md / se if (se and np.isfinite(se) and se > 0) else np.nan
        lo, hi = _mbb_mean_ci(s, block_len=block_len, B=B, alpha=alpha, rng=rng)
        rows.append({
            "variant": tag, "scope": f"h={int(h)}",
            "mean_loss_base": float(base_scores.loc[base_scores["h"] == h, "loss_scaled"].mean()),
            "mean_loss_var":  float(var_scores.loc[var_scores["h"] == h, "loss_scaled"].mean()),
            "mean_diff": md, "hac_se": se, "hac_t": t,
            "mbb_ci_lo": lo, "mbb_ci_hi": hi,
            "n": int(len(s)),
        })

    # Per-date diff rows for export
    diff_rows = m[["date", "h", "d_loss_scaled"]].copy()
    diff_rows["variant"] = tag
    return pd.DataFrame(rows), diff_rows


def run_ablation(cfg: dict, only: Optional[List[str]] = None, skip_existing: bool = False,
                 block_len: int = 5, B: int = 2000, alpha: float = 0.05
                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main entry point. Returns (overview_df, diffs_df) and writes files under:
      <results_dir>/ablation/{overview.csv, diff_vs_base.csv, meta.json}

    NOTE: This signature **matches the orchestrator's expectations**.  # :contentReference[oaicite:10]{index=10}
    """
    base_results = cfg_get(cfg, ("results_dir", "crl.results_dir"), "results/crl")
    _ensure_dir(os.path.join(base_results, "ablation"))

    panel_path = cfg_get(cfg, ("panel_path", "data.panel_path"), "data/cdi_processed.pkl")

    # Read BASE (aggregate security → index)
    print("[ABL] Reading BASE scores…")
    base_scores = _read_idx_scores(base_results, panel_path)

    # Define variants
    var_list = _default_variants(cfg)
    if only:
        keep = set(only)
        var_list = [v for v in var_list if (v.tag in keep or _overrides_to_tag(v.overrides) in keep)]

    # Run variants
    var_dirs = []
    for v in var_list:
        vdir = _run_variant(cfg, base_results, v, skip_existing=skip_existing)
        var_dirs.append((v, vdir))

    # Compare with BASE
    overview_rows: List[Dict[str, Any]] = []
    diff_rows_all: List[pd.DataFrame] = []
    rng = np.random.default_rng(7)

    # Add BASE row
    overview_rows.append({
        "variant": "base",
        "scope": "overall",
        "mean_loss_base": float(base_scores["loss_scaled"].mean()),
        "mean_loss_var":  float(base_scores["loss_scaled"].mean()),
        "mean_diff": 0.0, "hac_se": 0.0, "hac_t": 0.0,
        "mbb_ci_lo": 0.0, "mbb_ci_hi": 0.0, "n": int(len(base_scores)),
    })
    for h, g in base_scores.groupby("h"):
        overview_rows.append({
            "variant": "base",
            "scope": f"h={int(h)}",
            "mean_loss_base": float(g["loss_scaled"].mean()),
            "mean_loss_var":  float(g["loss_scaled"].mean()),
            "mean_diff": 0.0, "hac_se": 0.0, "hac_t": 0.0,
            "mbb_ci_lo": 0.0, "mbb_ci_hi": 0.0, "n": int(len(g)),
        })

    for v, vdir in var_dirs:
        try:
            var_scores = _read_idx_scores(vdir, panel_path)
        except Exception as e:
            print(f"[ABL] Warning: skipping {v.tag} due to read error: {e}")
            continue

        over, diffs = _compare_variant_to_base(
            base_scores, var_scores, v.tag, block_len=block_len, B=B, alpha=alpha, rng=rng
        )
        overview_rows.extend(over.to_dict("records"))
        diff_rows_all.append(diffs)

    overview = pd.DataFrame(overview_rows)
    diffs = (pd.concat(diff_rows_all, ignore_index=True)
             if diff_rows_all else pd.DataFrame(columns=["date", "h", "d_loss", "variant"]))

    # Write artifacts for the report to pick up  # :contentReference[oaicite:11]{index=11}
    abl_dir = os.path.join(base_results, "ablation")
    overview_path = os.path.join(abl_dir, "overview.csv")
    diffs_path = os.path.join(abl_dir, "diff_vs_base.csv")
    overview.to_csv(overview_path, index=False)
    diffs.to_csv(diffs_path, index=False)

    meta = {
        "block_len": int(block_len),
        "B": int(B),
        "alpha": float(alpha),
        "variants": [dict(tag=v.tag, overrides=v.overrides) for v, _ in var_dirs],
        "base_results": base_results,
        "panel_path": panel_path,
    }
    with open(os.path.join(abl_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[ABL] Wrote: {overview_path} and {diffs_path}")
    return overview, diffs


# ----------------------------- CLI ----------------------------- #

if __name__ == "__main__":  # pragma: no cover
    import argparse
    parser = argparse.ArgumentParser(description="Ablation study for CRL factor selection")
    parser.add_argument("--config", type=str, default="config_crl.yaml")
    parser.add_argument("--only", nargs="*", help="Run only these tags (match by tag or override string).")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs whose scores already exist.")
    parser.add_argument("--block-len", type=int, default=5)
    parser.add_argument("--bootstrap-B", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05, help="Two-sided CI level (0.05 → 95% CI)")
    args = parser.parse_args()

    if yaml is None:
        raise RuntimeError("pyyaml missing; install pyyaml or call run_ablation() programmatically.")
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}

    overview, diffs = run_ablation(cfg, only=args.only, skip_existing=args.skip_existing,
                                   block_len=args.block_len, B=args.bootstrap_B, alpha=args.alpha)
    print(f"[ABL] Done. Rows in overview={len(overview)}, diff rows={len(diffs)}")
