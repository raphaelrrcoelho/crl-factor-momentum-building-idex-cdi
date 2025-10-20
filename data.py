# data.py
# -*- coding: utf-8 -*-
"""
Debenture-level data preparation for CRL momentum experiments
=============================================================
- Reads Idex CDI raw files (XLS/XLSX), normalizes schema
- Computes per-debenture TOTAL returns (no look-ahead)
- Attaches index (benchmark) TOTAL return from index level series
- Adds light fundamental/context features needed by downstream consumers
- Does NOT compute forward targets (kept inside model code)

Outputs (MultiIndex by ['date','debenture_id']) with at minimum:
  'return', 'index_weight', 'index_return', 'spread', 'duration', 'sector_id', 'active'

Downstream:
- prepare_panel_for_crl.py flattens + renames debenture_id→id and writes
  data/cdi_processed.pkl for CRL (which requires ['date','id','return','index_weight','index_return']). 
"""

from __future__ import annotations

import os, glob, warnings
from dataclasses import dataclass
from typing import Optional, List, Tuple, Iterable

import numpy as np
import pandas as pd


# ----------------------------- Basics -----------------------------

TRADING_DAYS_PER_YEAR = 252.0

@dataclass
class UniversePaths:
    data_dir: str = "data"
    universe: str = "cdi"

    @property
    def pattern(self) -> str:
        # Adjust pattern if your raw files differ
        return f"idex_{self.universe.lower()}*.xls*"


def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def _safe_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _read_xls(path: str) -> pd.DataFrame:
    # Pandas auto-selects engine (xlrd/openpyxl) if available
    return pd.read_excel(path)


# ----------------------------- Load & Normalize -----------------------------

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize common Idex columns, coercing types and units to decimals.
    """
    rename = {
        "Data": "date",
        "Debênture": "debenture_id",
        "Debenture": "debenture_id",
        "Duration": "duration",
        "Duração": "duration",
        "Peso no índice (%)": "index_weight",
        "Peso no índice": "index_weight",
        "MTM ponderado (%)": "weighted_mtm",
        "Variação ponderada (%)": "weighted_return",
        "Carrego ponderado (%)": "weighted_carry",
        "Spread de compra (%)": "spread",
        "MID spread (Bps/NTNB)": "spread",
        "Índice (nível)": "index_value",
        "Indice (nível)": "index_value",
        "Índice": "index_level",
        "Indice": "index_level",
        "Setor": "sector",
        "Segmento": "sector",
        "Emissor": "issuer",
        "Rating": "rating",
    }
    df = df.rename(columns=rename)

    # Core identifiers
    if "date" not in df.columns:
        raise ValueError("Missing 'date' column in raw file.")
    df["date"] = _to_dt(df["date"])

    if "debenture_id" not in df.columns:
        # If no explicit id, attempt a fallback (e.g., Ticker)
        if "Ticker" in df.columns:
            df["debenture_id"] = df["Ticker"].astype(str)
        else:
            raise ValueError("Missing 'debenture_id' (or alternative) in raw file.")
    else:
        df["debenture_id"] = df["debenture_id"].astype(str)

    # Percent-to-decimal
    pct_cols = ["index_weight", "weighted_mtm", "weighted_return", "weighted_carry"]
    for c in pct_cols:
        if c in df.columns:
            if df[c].dtype == object:
                df[c] = (df[c].astype(str)
                              .str.replace("%", "", regex=False)
                              .str.replace(",", ".", regex=False))
            df[c] = _safe_float(df[c]) / 100.0

    # Spread: keep in decimal (% as decimal)
    if "spread" in df.columns:
        if df["spread"].dtype == object:
            df["spread"] = (df["spread"].astype(str)
                                      .str.replace("%", "", regex=False)
                                      .str.replace(",", ".", regex=False))
        df["spread"] = _safe_float(df["spread"]) / 100.0

    # Numerics
    for c in ("duration", "index_value", "index_level"):
        if c in df.columns:
            df[c] = _safe_float(df[c])

    # Categories
    for c in ("sector", "issuer", "rating"):
        if c in df.columns:
            df[c] = df[c].astype("category")

    df = df.dropna(subset=["date", "debenture_id"]).sort_values(["date", "debenture_id"])
    return df


def _load_raw_folder(paths: UniversePaths) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(paths.data_dir, paths.pattern)))
    if not files:
        warnings.warn(f"No raw files found at {paths.data_dir}/{paths.pattern}")
        return pd.DataFrame()

    dfs = []
    for fp in files:
        try:
            raw = _read_xls(fp)
            dfs.append(_normalize_columns(raw))
        except Exception as e:
            warnings.warn(f"Failed to read {os.path.basename(fp)}: {e}")

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ----------------------------- Panel construction -----------------------------

def _build_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build complete debenture-level panel:
    - per-asset TOTAL return from weighted components (when available)
    - index return from index_value if present (else computed later)
    - active flag from index_weight
    - sector_id and minimal fundamental context
    """
    if df.empty:
        return df

    out = df.sort_values(["debenture_id", "date"]).copy()

    # Active flag from weight
    if "index_weight" in out.columns:
        out["index_weight"] = _safe_float(out["index_weight"]).fillna(0.0)
        out["active"] = (out["index_weight"] > 0.0).astype("int8")
    else:
        out["index_weight"] = 0.0
        out["active"] = np.int8(0)

    # Forward-fill a few identifying attributes per debenture
    ffill_cols = [c for c in ["sector", "issuer", "spread", "duration"] if c in out.columns]
    out[ffill_cols] = (out.groupby("debenture_id", sort=False)[ffill_cols]
                           .apply(lambda g: g.ffill())
                           .reset_index(level=0, drop=True))

    # Per-asset TOTAL returns from weighted components if present
    if {"weighted_mtm", "weighted_carry", "index_weight"}.issubset(out.columns):
        w = out["index_weight"].to_numpy(dtype=np.float32)
        mtm = _safe_float(out["weighted_mtm"]).to_numpy(dtype=np.float32)
        carry = _safe_float(out["weighted_carry"]).to_numpy(dtype=np.float32)
        weighted_total = mtm# + carry  # already total
        total_ret = np.zeros_like(w, dtype=np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            active = w > 0
            total_ret[active] = weighted_total[active] / np.maximum(w[active], 1e-15)
        out["return"] = np.where(np.isfinite(total_ret), total_ret, 0.0).astype("float32")
    else:
        # If not available, keep/rename any 'return' column if present, else 0
        out["return"] = _safe_float(out.get("return", 0.0)).fillna(0.0).astype("float32")

    # Index (benchmark) TOTAL return from index_value series if available
    if "index_value" in out.columns:
        idx = (out[["date", "index_value"]]
                 .dropna()
                 .drop_duplicates(subset=["date"], keep="first")
                 .set_index("date")["index_value"]
                 .sort_index())
        idx_ret = idx.pct_change().fillna(0.0).rename("index_return").reset_index()
        out = out.merge(idx_ret, on="date", how="left")
    else:
        out["index_return"] = np.nan  # will be filled below

    # If index_return still missing, compute as weight-avg of returns by date
    if out["index_return"].isna().any():
        tmp = (out.groupby("date", sort=False)
                  .apply(lambda g: np.average(
                      _safe_float(g["return"]).to_numpy(dtype=float),
                      weights=_safe_float(g["index_weight"]).to_numpy(dtype=float) + 1e-15))
                  .rename("index_return")
                  .reset_index())
        out = out.drop(columns=["index_return"], errors="ignore")
        out = out.merge(tmp, on="date", how="left")

    out["index_return"] = _safe_float(out["index_return"]).fillna(0.0).astype("float32")

    # Basic context
    out["spread"] = _safe_float(out.get("spread", 0.0)).fillna(0.0).astype("float32")
    out["duration"] = _safe_float(out.get("duration", 0.0)).fillna(0.0).astype("float32")

    # Sector numeric code
    if "sector" in out.columns:
        out["sector_id"] = out["sector"].cat.codes.astype("int16")
    else:
        out["sector_id"] = np.int16(-1)

    # Simple time-to-maturity proxy (time remaining until last observed sample of each id)
    ttm_days = out.groupby("debenture_id", sort=False)["date"].transform(lambda s: (s.max() - s).dt.days)
    out["time_to_maturity"] = (ttm_days / TRADING_DAYS_PER_YEAR).astype("float32")

    # Final shape & types
    keep = ["date", "debenture_id", "return", "index_weight", "index_return",
            "spread", "duration", "sector_id", "active", "time_to_maturity"]
    out = out[keep].sort_values(["date", "debenture_id"])

    # MultiIndex for consumer
    out = out.set_index(["date", "debenture_id"])
    return out


# ----------------------------- Public entry point -----------------------------

def prepare_data(data_dir: str = "data") -> Optional[pd.DataFrame]:
    """
    Main entry: build debenture-level panel for CRL pipeline.
    Returns a MultiIndex DataFrame indexed by ['date','debenture_id'] with
    the required minimal contract and useful context columns.

    The downstream CRL loader expects the flattened file to contain:
      - 'date' (datetime), 'id' (string), 'return' (float),
        'index_weight' (float), and 'index_return' (float).
    """
    paths = UniversePaths(data_dir=data_dir, universe="cdi")
    raw = _load_raw_folder(paths)
    if raw.empty:
        raise RuntimeError(f"No data found under {data_dir} for '{paths.pattern}'")

    panel = _build_panel(raw)

    # Basic validation
    req = ["return", "index_weight", "index_return"]
    missing = [c for c in req if c not in panel.columns]
    if missing:
        raise ValueError(f"Prepared panel missing required columns: {missing}")

    # Sanity: no look-ahead here; forward targets are computed inside CRL/Baselines.
    return panel

"""
Create the flattened CRL panel from debenture-level MultiIndex data.

Writes:
  - data/cdi_panel.parquet   (snappy)
  - data/cdi_processed.pkl   (the path your config uses)

Contract (downstream consumers rely on this):
  Required columns: ['date','id','return','index_weight','index_return']
  Optional-but-expected: ['spread','duration','sector_id','active','ttm']
"""

import os
import pandas as pd
from data import prepare_data  # returns MultiIndex ['date','debenture_id']  (this file)

REQUIRED = ["date", "id", "return", "index_weight", "index_return"]

def create_crl_panel(data_dir: str = "data") -> str:
    print("="*60)
    print("PREPARING PANEL FOR CRL EXPERIMENTS")
    print("="*60)

    panel = prepare_data(data_dir=data_dir)
    if panel is None or panel.empty:
        raise ValueError("Failed to create panel: no rows returned by prepare_data()")

    # Flatten MultiIndex and rename to consumer schema
    flat = panel.reset_index()
    if "debenture_id" in flat.columns:
        flat = flat.rename(columns={"debenture_id": "id"})
    if "time_to_maturity" in flat.columns and "ttm" not in flat.columns:
        flat = flat.rename(columns={"time_to_maturity": "ttm"})

    # ============ ADD DATA QUALITY FILTERS HERE ============
    
    # 1. Remove securities with insufficient observations
    min_obs_per_security = 100
    security_counts = flat.groupby('id').size()
    valid_securities = security_counts[security_counts >= min_obs_per_security].index
    n_before = flat['id'].nunique()
    flat = flat[flat['id'].isin(valid_securities)]
    n_after = flat['id'].nunique()
    print(f"Filtered securities: {n_before} → {n_after} (removed {n_before - n_after} with <{min_obs_per_security} obs)")
    
    # 2. Cap extreme returns (likely data errors)
    # return_cap = 0.30  # 30% daily return cap
    # extreme_count = (flat['return'].abs() > return_cap).sum()
    # if extreme_count > 0:
    #     flat['return'] = flat['return'].clip(-return_cap, return_cap)
    #     print(f"Capped {extreme_count} extreme returns at ±{return_cap:.0%}")
    
    # 3. Fix spread outliers
    # if 'spread' in flat.columns:
    #     spread_cap = 0.10  # 10% max spread
    #     outlier_count = (flat['spread'] > spread_cap).sum()
    #     if outlier_count > 0:
    #         # Option 1: Cap spreads
    #         flat['spread'] = flat['spread'].clip(0, spread_cap)
    #         print(f"Capped {outlier_count} spread outliers at {spread_cap:.1%}")
    
    # 4. Final quality check: remove securities with too many NaN returns
    # missing_threshold = 0.20  # Remove if >20% missing
    # missing_pct = flat.groupby('id')['return'].apply(lambda x: x.isna().mean())
    # high_missing = missing_pct[missing_pct > missing_threshold].index
    # if len(high_missing) > 0:
    #     flat = flat[~flat['id'].isin(high_missing)]
    #     print(f"Removed {len(high_missing)} securities with >{missing_threshold:.0%} missing returns")
    
    # ============ END DATA QUALITY FILTERS ============

    # Enforce required columns; add safe defaults if optional context missing
    for col, default in [
        ("spread", 0.0),
        ("duration", 0.0),
        ("sector_id", -1),
        ("active", 1),
    ]:
        if col not in flat.columns:
            flat[col] = default

    # Validate required contract
    missing = [c for c in REQUIRED if c not in flat.columns]
    if missing:
        raise ValueError(f"Panel missing required columns for CRL: {missing}")

    # Types and ordering
    flat["date"] = pd.to_datetime(flat["date"])
    flat = flat.sort_values(["date", "id"]).reset_index(drop=True)

    # Write to both expected locations/names
    out_parquet = os.path.join(data_dir, "cdi_panel.parquet")
    out_pkl     = os.path.join(data_dir, "cdi_processed.pkl")

    # Ensure folder exists
    os.makedirs(data_dir, exist_ok=True)

    flat.to_parquet(out_parquet, compression="snappy", index=False)
    flat.to_pickle(out_pkl)

    print(f"\nSaved: {out_parquet} and {out_pkl}")
    print(f"Date range: {flat['date'].min().date()} → {flat['date'].max().date()} | "
          f"rows={len(flat)} | ids={flat['id'].nunique()}")
    return out_pkl  # default path used by your config_crl.yaml

if __name__ == "__main__":
    p = prepare_data("data")
    print("✓ Panel prepared:", p.shape)
    path = create_crl_panel("data")
    print(f"\n✓ Panel ready at: {path}")
