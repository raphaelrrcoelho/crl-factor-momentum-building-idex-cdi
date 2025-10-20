"""
Purged + Embargoed time-series splitter (Lopez de Prado):
- Splits by UNIQUE DATES in chronological order (no shuffling)
- Purges a window of *steps* BEFORE each test fold
- Embargoes a window of *steps* AFTER each test fold
- Horizon-aware helper returns purge=embargo=h (in steps)

Compatible with:
- causal_effects.py (cross-fitted S-learner)                ← uses PurgedTimeSeriesSplit
- ope_dr.py (cross-fitted Q-model for DR/IPS/DM)            ← uses PurgedTimeSeriesSplit
"""

from __future__ import annotations
from typing import Iterator, Tuple, Optional
import numpy as np
import pandas as pd


class PurgedTimeSeriesSplit:
    """
    Purged K-fold with embargo at the DATE level (by *steps*, not wall-clock days*).

    Parameters
    ----------
    n_splits : int
        Number of chronological folds (>= 2).
    purge_steps : int, optional
        Steps (unique dates) to drop from TRAIN immediately BEFORE each test window.
    embargo_steps : int, optional
        Steps (unique dates) to drop from TRAIN immediately AFTER each test window.
    date_col : str
        Column name containing a pandas-convertible datetime.

    Notes
    -----
    - For backward compatibility, __init__ also accepts `purge_days` / `embargo_days`
      and maps them to steps. Upstream code passes horizons (h) here, which reflect
      *trading steps*, so interpreting them as steps is correct in our pipeline.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_steps: int = 0,
        embargo_steps: int = 0,
        date_col: str = "date",
        **kwargs,
    ):
        assert n_splits >= 2, "n_splits must be >= 2"
        # Back-compat shims:
        if "purge_days" in kwargs and purge_steps == 0:
            purge_steps = int(kwargs["purge_days"])
        if "embargo_days" in kwargs and embargo_steps == 0:
            embargo_steps = int(kwargs["embargo_days"])

        self.n_splits = int(n_splits)
        self.purge_steps = int(max(0, purge_steps))
        self.embargo_steps = int(max(0, embargo_steps))
        self.date_col = date_col

    def __repr__(self):
        return (f"PurgedTimeSeriesSplit(n_splits={self.n_splits}, "
                f"purge_steps={self.purge_steps}, embargo_steps={self.embargo_steps}, "
                f"date_col='{self.date_col}')")

    def split(self, df: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Yield (train_idx, test_idx) numpy arrays of integer positional indices.
        """
        if self.date_col not in df.columns:
            raise KeyError(f"date_col='{self.date_col}' not in frame.")

        # Normalize to date-only; keep original order
        dates = pd.to_datetime(df[self.date_col]).dt.normalize()
        udates = np.array(sorted(dates.unique()))
        n_u = len(udates)
        if n_u < self.n_splits:
            raise ValueError(f"Not enough unique dates ({n_u}) for n_splits={self.n_splits}.")

        # Evenly allocate test folds across unique dates
        fold_sizes = np.full(self.n_splits, n_u // self.n_splits, dtype=int)
        fold_sizes[: n_u % self.n_splits] += 1

        # Map each row to its date-rank in udates (O(1) lookup)
        date_to_rank = {d: i for i, d in enumerate(udates)}
        ranks = dates.map(date_to_rank).to_numpy()

        start = 0
        for fold_size in fold_sizes:
            test_lo_rank = start
            test_hi_rank = start + fold_size - 1

            # Test set: rows whose rank ∈ [test_lo_rank, test_hi_rank]
            test_mask = (ranks >= test_lo_rank) & (ranks <= test_hi_rank)

            # Train set: everything strictly before (test_lo - purge_steps)
            #            OR strictly after  (test_hi + embargo_steps)
            purge_cut   = max(-1, test_lo_rank - self.purge_steps - 1)
            embargo_cut = test_hi_rank + self.embargo_steps + 1
            train_mask = (ranks <= purge_cut) | (ranks >= embargo_cut)

            test_idx = np.flatnonzero(test_mask)
            train_idx = np.flatnonzero(train_mask)

            if test_idx.size == 0:
                raise RuntimeError("Empty test fold encountered; consider fewer splits.")
            if train_idx.size == 0 or test_idx.size == 0:
                start += fold_size
                print("Empty train fold encountered; adjust purge/embargo.")
                continue
            if np.intersect1d(train_idx, test_idx).size:
                raise RuntimeError("Purged split produced overlapping indices; check logic.")

            yield train_idx, test_idx
            start += fold_size


def horizon_aware_splitter_steps(n_splits: int, h_steps: int, date_col: str = "date") -> PurgedTimeSeriesSplit:
    """
    Convenience: use horizon h as both purge and embargo window (in steps).
    """
    return PurgedTimeSeriesSplit(n_splits=n_splits, purge_steps=h_steps, embargo_steps=h_steps, date_col=date_col)
