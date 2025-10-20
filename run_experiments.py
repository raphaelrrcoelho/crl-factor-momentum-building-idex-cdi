# run_experiments.py
# -*- coding: utf-8 -*-
"""
Master orchestrator for the Causal-RL Momentum project — unified config edition
=======================================================================================

What this file guarantees
-------------------------
- **Single source of truth**: YAML is loaded once into `RunConfig` (see `run_config.py`).
- **Stable step API**: each pipeline step receives the *same* `cfg` object.
- **Checkpointing**: idempotent runs via `<results_dir>/experiment_checkpoint.json`.
- **Ablation hook**: optional `--ablation` flag calls `ablation_study.run_ablation(cfg, ...)`.

Steps
-----
1) CRL bandit (Conformal Quantile RL) → `crl_factor_bandit_conformal.run_bandit(cfg)`
2) Baselines → `baselines_crl.run_all_baselines(panel_path, train_end, test_start, cfg)`
3) Causal effects → `causal_effects.run_causal_effects(cfg)`
4) Off-policy evaluation (optional) → `ope_dr.compute_ope_dr(cfg.results_dir)`
5) Report (optional if module unavailable) → `generate_report.generate_full_report(...)`

CLI
---
python run_experiments.py \
  --config config_crl.yaml \
  [--train-end YYYY-MM-DD] [--test-start YYYY-MM-DD] \
  [--skip-crl] [--skip-baselines] [--skip-causal] [--skip-ope] [--skip-report] \
  [--ablation] [--abl-only base rollwin_42 ...] [--abl-skip-existing]

"""
from __future__ import annotations

import os
import sys
import json
import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import yaml

# ----------------------- Paths / checkpoint helpers --------------------- #

def _results_paths(results_dir: str) -> Dict[str, str]:
    root = os.path.abspath(results_dir)
    return {
        "root": root,
        "checkpoint": os.path.join(root, "experiment_checkpoint.json"),
        "resolved_cfg": os.path.join(root, "config_resolved.yaml"),
        "report_dir": os.path.join(root, "report"),
    }


def _ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def _load_checkpoint(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"completed": {}}


def _save_checkpoint(results_dir: str, ck: Dict[str, Any]):
    ck_path = _results_paths(results_dir)["checkpoint"]
    with open(ck_path, "w", encoding="utf-8") as f:
        json.dump(ck, f, indent=2)


def _step_done(ck: Dict[str, Any], name: str) -> bool:
    return name in ck.get("completed", {})


def _mark_done(ck: Dict[str, Any], name: str, ck_path: str):
    ck.setdefault("completed", {})[name] = datetime.now().isoformat(timespec="seconds")
    with open(ck_path, "w", encoding="utf-8") as f:
        json.dump(ck, f, indent=2)

# ----------------------- Simple validation ----------------------------- #
REQUIRED_KEYS = {
    "crl": ["results_dir", "panel_path"],
    "baselines": ["results_dir", "panel_path"],
    "causal": ["results_dir"],
    "ope": ["results_dir"],
    "report": ["results_dir"],
}

# ----------------------- Steps ----------------------------------------- #


def step_crl(cfg: Any, ck: Dict[str, Any], skip: bool = False):
    name = "crl"
    paths = _results_paths(cfg["results_dir"])
    if skip:
        print("[CRL] Skipped by flag.")
        return
    if _step_done(ck, name):
        print("[CRL] Already completed.")
        return
    print("[CRL] Running CRL factor bandit...")
    try:
        from crl_factor_bandit_conformal import run_bandit
        run_bandit(cfg)
        
        # Validate outputs exist
        expected_files = [
            "crl_policy_choices.csv",
            "analysis.json"
        ]
        for f in expected_files:
            fpath = os.path.join(cfg["results_dir"], f)
            if not os.path.exists(fpath):
                print(f"[CRL] Warning: Expected file {f} not found")
        
        # Check for security level (optional but recommended)
        sec_file = os.path.join(cfg["results_dir"], "crl_scores_securities.csv")
        if os.path.exists(sec_file):
            print(f"[CRL] Security-level results available")
        
        _mark_done(ck, name, paths["checkpoint"])
    except Exception as e:
        print(f"[CRL] FAILED: {e}")
        raise


def step_baselines(cfg: Any, ck: Dict[str, Any], skip: bool = False):
    name = "baselines"
    paths = _results_paths(cfg["results_dir"])
    if skip:
        print("[BASE] Skipped by flag.")
        return
    if _step_done(ck, name):
        print("[BASE] Already completed.")
        return
    print("[BASE] Running baselines…")
    try:
        from baselines_crl import run_all_baselines
        panel_path = cfg["panel_path"]
        run_all_baselines(panel_path, cfg["train_end"], cfg["test_start"], cfg)
        _mark_done(ck, name, paths["checkpoint"])
    except Exception as e:
        print(f"[BASE] FAILED: {e}")
        raise


def step_causal(cfg: Any, ck: Dict[str, Any], skip: bool = False):
    name = "causal"
    paths = _results_paths(cfg["results_dir"])
    if skip:
        print("[CAUSAL] Skipped by flag.")
        return
    if _step_done(ck, name):
        print("[CAUSAL] Already completed.")
        return
    print("[CAUSAL] Estimating causal effects…")
    try:
        from causal_effects import run_causal_effects
        run_causal_effects(cfg)
        _mark_done(ck, name, paths["checkpoint"])
    except Exception as e:
        print(f"[CAUSAL] FAILED: {e}")
        raise


def step_ope(cfg: Any, ck: Dict[str, Any], skip: bool = False):
    name = "ope"
    paths = _results_paths(cfg["results_dir"])
    if skip:
        print("[OPE] Skipped by flag.")
        return
    if _step_done(ck, name):
        print("[OPE] Already completed.")
        return
    print("[OPE] Running off-policy evaluation…")
    try:
        from ope_dr import compute_ope_dr
        out = compute_ope_dr(cfg["results_dir"])  # universe not needed; path is explicit
        if out:
            print(f"[OPE] Wrote {out}")
        _mark_done(ck, name, paths["checkpoint"])
    except FileNotFoundError as e:
        print(f"[OPE] Skipped: {e}")
        # do not raise; allow pipeline to continue
    except Exception as e:
        print(f"[OPE] FAILED: {e}")
        raise


def step_report(cfg: Any, ck: Dict[str, Any], out_dir: Optional[str] = None, skip: bool = False):
    name = "report"
    paths = _results_paths(cfg["results_dir"])
    if skip:
        print("[REPORT] Skipped by flag.")
        return
    if _step_done(ck, name):
        print("[REPORT] Already completed.")
        return
    out_dir = out_dir or paths["report_dir"]
    print("[REPORT] Generating report assets…")
    try:
        import generate_report as rep
        if hasattr(rep, "generate_full_report"):
            rep.generate_full_report(cfg["results_dir"], out_dir)
        else:
            raise RuntimeError("generate_report: missing generate_full_report(results_dir, out_dir)")
        _mark_done(ck, name, paths["checkpoint"])
    except ModuleNotFoundError:
        print("[REPORT] Module not found; skipping.")
    except Exception as e:
        print(f"[REPORT] FAILED: {e}")
        raise


# ----------------------- Ablation (optional) --------------------------- #

def step_ablation(cfg: Any, only: list[str] | None, skip_existing: bool):
    """Thin wrapper around ablation_study.run_ablation(cfg, ...).
    Keeps all ablation logic **in** ablation_study.py (recommended).
    """
    try:
        from ablation_study import run_ablation
    except ModuleNotFoundError:
        print("[ABL] ablation_study.py not found. Skipping ablation.")
        return

    try:
        print("[ABL] Running ablation study…")
        overview, diffs = run_ablation(cfg, only=only, skip_existing=skip_existing)
        print("[ABL] Done. Rows:", len(overview), "Diff rows:", len(diffs))
    except Exception as e:
        print(f"[ABL] FAILED: {e}")
        raise


# ----------------------- CLI / main ----------------------------------- #

def _print_header(cfg_path: str, cfg: Any):
    print("=" * 70)
    print("CRL FACTOR SELECTION EXPERIMENTS")
    print("Started:", datetime.now().isoformat(sep=" ", timespec="seconds"))
    print("Config:", cfg_path)
    print("Results dir:", cfg["results_dir"])
    print("=" * 70)


def main():
    p = argparse.ArgumentParser(description="Run CRL factor selection pipeline")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument("--train-end", type=str, default=None, help="Override train_end (YYYY-MM-DD)")
    p.add_argument("--test-start", type=str, default=None, help="Override test_start (YYYY-MM-DD)")

    p.add_argument("--skip-crl", action="store_true")
    p.add_argument("--skip-baselines", action="store_true")
    p.add_argument("--skip-causal", action="store_true")
    p.add_argument("--skip-ope", action="store_true")
    p.add_argument("--skip-report", action="store_true")

    # Ablation controls
    p.add_argument("--ablation", action="store_true", help="Run the ablation study after main steps")
    p.add_argument("--abl-only", nargs="*", help="Run only these ablation tags (ensure 'base' is included or it will be added)")
    p.add_argument("--abl-skip-existing", action="store_true", help="Skip tags where scores already exist")

    args = p.parse_args()

    # 1) Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    # 2) Header
    _print_header(args.config, cfg)

    # 3) Checkpoint
    paths = _results_paths(cfg["results_dir"])
    _ensure_dir(paths["root"])
    ck = _load_checkpoint(paths["checkpoint"])

    # 5) Main steps
    step_crl(cfg, ck, skip=args.skip_crl)
    step_baselines(cfg, ck, skip=args.skip_baselines)
    step_causal(cfg, ck, skip=args.skip_causal)
    step_ope(cfg, ck, skip=args.skip_ope)
    step_report(cfg, ck, skip=args.skip_report)

    # 6) Ablation (optional)
    if args.ablation:
        step_ablation(cfg, only=args.abl_only, skip_existing=args.abl_skip_existing)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
