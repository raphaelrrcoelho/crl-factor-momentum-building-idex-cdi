# ope_dr.py — Off-Policy Evaluation (DM / IPS / DR) on RETURNS (policy-invariant)
# -*- coding: utf-8 -*-
"""
Evaluates the learned policy (CRL) off-policy using a behavior log
(Random baseline) by Doubly Robust (DR), Direct Method (DM) and IPS.

Key choices:
- Target is FORWARD RETURN y (policy-invariant), NOT interval score
- Cross-fitted Q-hat with purged+embargoed date splits
- Behavior propensities π_b(x,a) from Random:
    * Prefer logged 'support_size'  → π_b = 1/|A(x)| at decision time
    * Otherwise reconstruct |A(x)| label-free via “alive at t+h” using panel
- Summary writes BOTH 'DR_return' (primary, ↑ better) and a compatibility
  field 'DR_risk = -DR_return' so existing plots continue to work.

Outputs under <results_dir>:
  - ope_dr_rows.csv      (row-wise DM/IPS/DR and internals)
  - ope_dr_summary.csv   (overall + per-horizon)
  - ope_dr_meta.json     (counts, ESS, config)
"""

from __future__ import annotations

import os
import json
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone

from baselines_crl import prepare_security_data  # <-- pull CRL-aligned features
from run_config import cfg_get



# ----------------------------- Safe imports / fallbacks ----------------------------- #

def _fallback_cfg_get(obj: Any, key: str | tuple | list, default=None):
    """
    Minimal resilient getter for configs/dicts/namespaces.
    Tries tuple/list of keys, dotted paths, nested dicts, and attributes.
    """
    def _one(o, k):
        if isinstance(k, str):
            # dotted path
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
            # plain string
            if isinstance(o, dict) and k in o:
                return o[k]
            if hasattr(o, k):
                return getattr(o, k)
        # tuple key
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

try:
    from run_config import cfg_get as _cfg_get
except Exception:
    _cfg_get = _fallback_cfg_get

# Purged splitter (import, else simple fallback)
try:
    from utils_timesplit import PurgedTimeSeriesSplit
except Exception:
    class PurgedTimeSeriesSplit:
        """Very small fallback: chronological K splits by UNIQUE DATES, no purge."""
        def __init__(self, n_splits=3, purge_steps=0, embargo_steps=0, date_col="date", **_):
            self.n_splits = n_splits
            self.date_col = date_col
        def split(self, X: pd.DataFrame) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
            d = sorted(pd.to_datetime(X[self.date_col]).unique())
            k = max(2, int(self.n_splits))
            step = max(1, len(d) // k)
            bins = [d[i*step:(i+1)*step] for i in range(k-1)]
            bins.append(d[(k-1)*step:])
            idx = np.arange(len(X))
            for i in range(k):
                te_dates = set(bins[i])
                te_mask = X[self.date_col].isin(te_dates).to_numpy()
                te_idx = idx[te_mask]
                tr_idx = idx[~te_mask]
                yield tr_idx, te_idx


# ----------------------------- IO Helpers ----------------------------- #

def _ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def _read_df_any(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".pkl", ".pickle"):
        return pd.read_pickle(path)
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    # default to CSV
    return pd.read_csv(path)

def _load_panel(panel_path: str) -> pd.DataFrame:
    df = _read_df_any(panel_path)
    # Normalize schema minimally
    if "date" not in df.columns:
        # common: multiindex to columns or variant name
        if isinstance(df.index, pd.MultiIndex) and "date" in df.index.names:
            df = df.reset_index()
        else:
            raise ValueError("Panel must have a 'date' column.")
    if "id" not in df.columns:
        # common raw name: 'debenture_id'
        if "debenture_id" in df.columns:
            df = df.rename(columns={"debenture_id": "id"})
        else:
            raise ValueError("Panel must have an 'id' column (or 'debenture_id').")
    df["date"] = pd.to_datetime(df["date"])
    return df


# ----------------------------- Features & Actions ----------------------------- #

def _action_key(mom: str, h: int) -> str:
    return f"{str(mom)}|h={int(h)}"

def _make_action_index(actions: Iterable[Tuple[str, int]]) -> Tuple[Dict[str, int], List[Tuple[str,int]]]:
    uniq = sorted({(str(m), int(h)) for (m, h) in actions})
    keys = [_action_key(m, h) for (m, h) in uniq]
    mapping = {k: i for i, k in enumerate(keys)}
    return mapping, uniq

def _one_hot_action(mom: str, h: int, action_to_idx: Dict[str,int], n_actions: int) -> np.ndarray:
    k = _action_key(mom, h)
    oh = np.zeros(n_actions, dtype=np.float32)
    if k in action_to_idx:
        oh[action_to_idx[k]] = 1.0
    return oh


# ----------------------------- Propensity reconstruction ----------------------------- #

def _reconstruct_support_size(row, horizons, date_to_rank, id_ranks):
    """
    Reconstruct support size (number of valid actions) at decision time.
    An action (momentum, h) is valid if a row exists at date t+h for this security.
    
    Parameters
    ----------
    row : pd.Series
        Row with 'id' and 'date'
    horizons : list of int
        All horizons to check
    date_to_rank : dict
        Mapping (id, date) -> chronological rank (int) within that id's time series
    id_ranks : dict
        Mapping id -> set of all ranks where that id appears
    
    Returns
    -------
    int : Number of valid horizons (1 if none or can't determine)
    """
    key = (row["id"], pd.to_datetime(row["date"]))
    r = date_to_rank.get(key, None)
    if r is None:
        return 1  # Conservative: assume at least 1 valid action
    
    id_key = row["id"]
    ranks = id_ranks.get(id_key, set())
    if not ranks:
        return 1
    
    # Count how many horizons h have a row at rank r+h
    valid_count = sum(1 for h in horizons if (r + int(h)) in ranks)
    return max(valid_count, 1)  # At least 1 to avoid division by zero


# ----------------------------- OPE core ----------------------------- #

def _align_behavior_and_eval(results_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Random (behavior) and CRL (evaluation) matured security-level logs.
    Returns (R, E) with minimal columns.
    """
    R_path = os.path.join(results_dir, "random_scores_securities.csv")
    E_path = os.path.join(results_dir, "crl_scores_securities.csv")

    R = _read_df_any(R_path)
    E = _read_df_any(E_path)

    # Normalize schema
    for df in (R, E):
        if "date" not in df.columns:
            raise ValueError("Missing 'date' column in matured logs.")
        df["date"] = pd.to_datetime(df["date"])
        if "id" not in df.columns:
            if "debenture_id" in df.columns:
                df.rename(columns={"debenture_id": "id"}, inplace=True)
            else:
                raise ValueError("Missing 'id' column in matured logs.")
        # Ensure standard column names
        if "horizon" in df.columns and "h" not in df.columns:
            df.rename(columns={"horizon": "h"}, inplace=True)

    # Keep only the essentials up front; keep extras for later merges
    need_R = ["date", "id", "momentum", "h", "y", "support_size", "index_weight"]
    need_E = ["date", "id", "momentum", "h"]
    for c in need_R:
        if c not in R.columns:
            # leave missing; will backfill some later (support_size, index_weight)
            pass
    for c in need_E:
        if c not in E.columns:
            raise ValueError(f"Evaluation log missing '{c}': {E_path}")

    return R, E

def _prepare_training_table(R: pd.DataFrame, E: pd.DataFrame) -> pd.DataFrame:
    """
    Join behavior and evaluation actions on the same context (date,id).
    """
    key = ["date", "id"]
    # suffixes: _b for behavior, _e for evaluation
    cols_R = key + ["momentum", "h", "y", "support_size", "index_weight"]
    cols_E = key + ["momentum", "h"]
    R0 = R[[c for c in cols_R if c in R.columns]].copy()
    E0 = E[[c for c in cols_E if c in E.columns]].copy()

    R0 = R0.rename(columns={"momentum": "mom_b", "h": "h_b"})
    E0 = E0.rename(columns={"momentum": "mom_e", "h": "h_e"})

    Rme = pd.merge(R0, E0, on=["date","id"], how="inner")
    # drop rows with missing target or actions
    Rme = Rme.dropna(subset=["mom_b", "h_b", "mom_e", "h_e", "y"])
    Rme["h_b"] = Rme["h_b"].astype(int)
    Rme["h_e"] = Rme["h_e"].astype(int)
    return Rme

def _merge_panel_features(Rme: pd.DataFrame, panel: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    feats = feature_cols
    keep = ["date", "id"] + [c for c in feats if c in panel.columns]
    P = panel[keep].copy()
    P["date"] = pd.to_datetime(P["date"])
    out = pd.merge(Rme, P, on=["date","id"], how="left")
    return out

def _build_feat_matrix(df: pd.DataFrame,
                       action_col_m: str,
                       action_col_h: str,
                       feature_cols: List[str],
                       action_to_idx: Dict[str,int],
                       n_actions: int) -> np.ndarray:
    """
    Concatenate one-hot(action) with panel features for each row.
    """
    n = len(df)
    X = np.empty((n, n_actions + len(feature_cols)), dtype=np.float32)
    # Fill features block first to vectorize
    if feature_cols:
        X[:, n_actions:] = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    else:
        X[:, n_actions:] = 0.0
    # One-hot per row
    for i, r in enumerate(df.itertuples(index=False)):
        mom = getattr(r, action_col_m)
        h = int(getattr(r, action_col_h))
        X[i, :n_actions] = _one_hot_action(mom, h, action_to_idx, n_actions)
    return X

def _effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    s1 = (w.sum())**2
    s2 = (w**2).sum() + 1e-18
    return float(s1 / s2)

def compute_ope_dr(results_dir: str,
                   cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    OPE on RETURNS (policy-invariant). Uses CRL-aligned standardized features:
    momentum *_z, ret_vol20_z, idx_vol_z, spread_z, duration_z.
    Cross-fitted Q̂ with purged+embargoed date splits (purge = max_h).
    """
    cfg = cfg or {}
    _ensure_dir(results_dir)

    # 1) Load behavior/eval logs (Random & CRL)
    R, E = _align_behavior_and_eval(results_dir)
    print(f"[OPE] Loaded behavior={len(R):,} rows, eval={len(E):,} rows")

    # 2) Load panel and build CRL/Baselines feature table (TRAIN-only standardization)
    panel_path = _cfg_get(cfg, ("panel_path", "data.panel_path"), "data/cdi_processed.pkl")
    # prepare_security_data computes momentum columns (not used as features), 
    # context z-scores, cross-sectional features, and forward labels
    panel_full, _ = prepare_security_data(_load_panel(panel_path), cfg)

    # 3) Build training/eval dataframe on shared contexts
    Rme = _prepare_training_table(R, E)
    if Rme.empty:
        raise ValueError("No overlapping (date,id) between behavior and evaluation logs.")

    # 4) Feature columns: CRL-aligned z-features
    base_z = [c for c in ["ret_vol20_z", "idx_vol_z", "spread_z", "duration_z", "ret_autocorr_z"] 
              if c in panel_full.columns]
    percentile_feats = [c for c in ["spread_percentile", "vol_percentile"] 
                       if c in panel_full.columns]
    feature_cols = base_z + percentile_feats + ["index_weight"]

    # 5) Merge features into the Rme table
    feats = panel_full[["date","id"] + feature_cols].copy()
    feats["date"] = pd.to_datetime(feats["date"])
    Rme = pd.merge(Rme, feats, on=["date","id"], how="left")
    for c in feature_cols:
        if c in Rme.columns:
            Rme[c] = Rme[c].astype(np.float32).fillna(0.0)

    # 6) Action dictionary (union of actions on the overlap)
    actions = set(zip(Rme["mom_b"].astype(str), Rme["h_b"].astype(int))) | \
              set(zip(Rme["mom_e"].astype(str), Rme["h_e"].astype(int)))
    action_to_idx, actions_sorted = _make_action_index(actions)
    n_actions = len(action_to_idx)
    print(f"[OPE] Action space: {n_actions} arms")

    # 7) Propensities π_b from Random (prefer logged support_size; else reconstruct)
    horizons = sorted(set(Rme["h_b"].unique()) | set(Rme["h_e"].unique()))
    if "support_size" in Rme.columns and Rme["support_size"].notna().any():
        pi_b = 1.0 / Rme["support_size"].fillna(1).clip(lower=1).to_numpy(dtype=np.float32)
    else:
        print("[OPE] Reconstructing behavior support_size (alive at t+h) from panel.")
        # Build proper rank structures for alive-at-t+h reconstruction
        panel_dates = panel_full[["id", "date"]].copy()
        panel_dates["date"] = pd.to_datetime(panel_dates["date"])
        panel_dates = panel_dates.sort_values(["id", "date"]).drop_duplicates()
        
        # Create rank mapping: (id, date) -> chronological rank within id
        panel_dates["rank"] = panel_dates.groupby("id").cumcount()
        date_to_rank = dict(zip(
            zip(panel_dates["id"], panel_dates["date"]),
            panel_dates["rank"]
        ))
        
        # Create rank_set: id -> set of all ranks for that id
        id_ranks = panel_dates.groupby("id")["rank"].apply(set).to_dict()
        
        # Now reconstruct support sizes
        sup = Rme.apply(
            lambda r: _reconstruct_support_size(r, horizons, date_to_rank, id_ranks), 
            axis=1
        )
        pi_b = (1.0 / sup.clip(lower=1)).to_numpy(dtype=np.float32)
    pi_b = np.clip(pi_b, 1e-6, 1.0).astype(np.float32)

    # 8) Targets and weights
    y_b = Rme["y"].to_numpy(dtype=np.float32)
    h_b = Rme["h_b"].to_numpy(dtype=np.int32)
    w_idx = Rme["index_weight"].to_numpy(dtype=np.float32) if "index_weight" in Rme.columns else np.ones_like(y_b)

    from utils_conformal import horizon_scale_factor
    hs_mode = _cfg_get(cfg, "horizon_scale_mode", "none")
    # per-row scaling: factor(h_b[i])
    scalars = np.array([horizon_scale_factor(int(hh), hs_mode) for hh in h_b], dtype=np.float32)
    y_b = y_b * scalars

    # 9) Build design matrices for behavior/evaluation actions
    date_df = Rme[["date"]].copy()
    X_b = _build_feat_matrix(Rme, "mom_b", "h_b", feature_cols, action_to_idx, n_actions)
    X_e = _build_feat_matrix(Rme, "mom_e", "h_e", feature_cols, action_to_idx, n_actions)

    # 10) Cross-fitting Q̂ with horizon-aware purge (purge=max_h)
    max_h = int(max(horizons) if horizons else 0)
    ps = int(_cfg_get(cfg, "max_purge_steps", max_h))
    n_splits = int(_cfg_get(cfg, "n_splits_ope", 5))
    splitter = PurgedTimeSeriesSplit(n_splits=n_splits, purge_steps=ps, embargo_steps=ps, date_col="date")

    base = RandomForestRegressor(
        n_estimators=int(_cfg_get(cfg, "q_n_trees", 600)),
        max_depth=int(_cfg_get(cfg, "q_max_depth", 14)),
        min_samples_leaf=int(_cfg_get(cfg, "q_min_leaf", 8)),
        n_jobs=-1, random_state=42
    )

    oof_pred_b = np.full_like(y_b, np.nan, dtype=np.float32)  # Q̂(x, a_b)
    q_ehat      = np.full_like(y_b, np.nan, dtype=np.float32)  # Q̂(x, a_e)
    same_action = (Rme["mom_b"].astype(str) == Rme["mom_e"].astype(str)) & (Rme["h_b"].astype(int) == Rme["h_e"].astype(int))

    n_folds_ok = 0
    for k, (tr_idx, te_idx) in enumerate(splitter.split(date_df), 1):
        if te_idx.size == 0 or tr_idx.size < 200:
            continue
        model = clone(base)
        model.fit(X_b[tr_idx], y_b[tr_idx])
        oof_pred_b[te_idx] = model.predict(X_b[te_idx]).astype(np.float32)
        q_ehat[te_idx]     = model.predict(X_e[te_idx]).astype(np.float32)
        n_folds_ok += 1

    # Fallback for any left-out rows
    if np.isnan(oof_pred_b).any() or np.isnan(q_ehat).any():
        model = clone(base).fit(X_b, y_b)
        mask_nan = np.isnan(oof_pred_b)
        if mask_nan.any():
            oof_pred_b[mask_nan] = model.predict(X_b[mask_nan]).astype(np.float32)
        mask_nan = np.isnan(q_ehat)
        if mask_nan.any():
            q_ehat[mask_nan] = model.predict(X_e[mask_nan]).astype(np.float32)

    # 11) Estimators
    iw  = (same_action.values.astype(np.float32) / np.clip(pi_b, 1e-6, 1.0)).astype(np.float32)
    dm  = q_ehat
    ips = iw * y_b
    dr  = q_ehat + iw * (y_b - oof_pred_b)

    # self-normalized variants (saved but not summarized)
    iw_sn  = iw / (iw.sum() + 1e-12)
    ips_sn = iw_sn * y_b

    # 12) Aggregate (unweighted and index-weighted)
    def _agg(vals: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        v = np.asarray(vals, dtype=np.float64)
        if weights is None:
            return float(np.mean(v))
        w = np.asarray(weights, dtype=np.float64)
        w = w / (w.sum() + 1e-18)
        return float(np.sum(w * v))

    overall = {
        "DM": _agg(dm), "IPS": _agg(ips), "DR": _agg(dr),
        "DM_w": _agg(dm, w_idx), "IPS_w": _agg(ips, w_idx), "DR_w": _agg(dr, w_idx),
        "DM_sn": _agg(q_ehat + ips_sn - iw_sn * oof_pred_b),
        "IPS_sn": _agg(ips_sn),
        "n": int(len(Rme)),
    }

    # By horizon (eval action h_e)
    by_h = []
    for h in sorted(Rme["h_e"].unique()):
        m = (Rme["h_e"] == h).to_numpy()
        if m.sum() == 0:
            continue
        d = {
            "h": int(h),
            "DM": _agg(dm[m]), "IPS": _agg(ips[m]), "DR": _agg(dr[m]),
            "DM_w": _agg(dm[m], w_idx[m]), "IPS_w": _agg(ips[m], w_idx[m]), "DR_w": _agg(dr[m], w_idx[m]),
            "n": int(m.sum())
        }
        by_h.append(d)

    # 13) Save row-wise and summary (same paths/field names as before)
    rows = Rme[["date","id","mom_b","h_b","mom_e","h_e","y"]].copy()
    rows["DM_hat"]    = dm
    rows["IPS_hat"]   = ips
    rows["DR_hat"]    = dr
    rows["DM_hat_w"]  = dm * w_idx
    rows["IPS_hat_w"] = ips * w_idx
    rows["DR_hat_w"]  = dr * w_idx
    rows["q_bhat_oof"] = oof_pred_b
    rows["q_ehat"]     = q_ehat
    rows["same_action"] = same_action.astype(int)
    rows["pi_b"] = pi_b
    rows["iw"]   = iw
    rows["iw_sn"] = iw_sn
    rows_path = os.path.join(results_dir, "ope_dr_rows.csv")
    rows.to_csv(rows_path, index=False)

    summ_rows = []
    dr_overall = overall["DR_w"] if np.isfinite(overall["DR_w"]) else overall["DR"]
    summ_rows.append({"policy": "CRL (overall)", "DR_return": dr_overall, "DR_risk": -dr_overall})
    for s in by_h:
        val = s["DR_w"] if np.isfinite(s["DR_w"]) else s["DR"]
        summ_rows.append({"policy": f"CRL(h={s['h']})", "DR_return": val, "DR_risk": -val})
    summ = pd.DataFrame(summ_rows)
    summ_path = os.path.join(results_dir, "ope_dr_summary.csv")
    summ.to_csv(summ_path, index=False)

    ess_ips = _effective_sample_size(iw)
    meta = {
        "n_rows": len(Rme),
        "n_folds_used": n_splits,
        "ess_ips": ess_ips,
        "feature_cols": feature_cols,
        "n_actions": n_actions,
        "actions": [{"momentum": m, "h": int(h)} for (m, h) in actions_sorted],
        "purge_steps": ps,
        "results_dir": results_dir,
    }
    with open(os.path.join(results_dir, "ope_dr_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print("\n[OPE] Summary (return ↑ better):")
    print(summ[["policy","DR_return"]].to_string(index=False))
    print(f"\n[OPE] Effective sample size (IPS): {ess_ips:,.1f}")
    print(f"[OPE] Wrote: {rows_path}")
    print(f"[OPE] Wrote: {summ_path}")
    return {"rows_path": rows_path, "summary_path": summ_path, "meta": meta}

# ----------------------------- CLI ----------------------------- #

if __name__ == "__main__":  # pragma: no cover
    import argparse, yaml
    p = argparse.ArgumentParser(description="Off-Policy Evaluation (DM/IPS/DR) on returns (policy-invariant)")
    p.add_argument("results_dir", nargs="?", default="results/cdi/crl")
    p.add_argument("--config", type=str, default="config_crl.yaml")
    args = p.parse_args()

    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    compute_ope_dr(args.results_dir, cfg)
