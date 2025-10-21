# Causal Reinforcement Learning for Dynamic Factor Specification in Brazilian Debenture Markets

## 📊 Project Overview

This repository implements **adaptive momentum factor selection** for Brazilian corporate bond markets using causal reinforcement learning. Instead of committing to fixed momentum windows and forecast horizons based on historical backtests, this framework learns which specifications work under which market conditions.

### Key Results

Using IDEX CDI data (2019-2024), the adaptive framework achieves:
- **21.7% reduction** in interval score losses vs best fixed specification
- **90% coverage rate** maintained despite market regime shifts
- **Dynamic adaptation**: Short-term momentum (73%) during volatile periods, long-term (68%) during calm markets
- Analysis of **607 securities** across **144,701 security-date predictions**

## 🎯 Research Question

> Can a contextual bandit framework dynamically select optimal momentum specifications (window × horizon) based on prevailing market conditions, outperforming static approaches while maintaining calibrated uncertainty estimates?

## 🏗️ Repository Structure

```
crl-factor-momentum-building-idex-cdi/
│
├── config_crl.yaml                     # Master configuration file
│
├── run_experiments.py                  # Main pipeline orchestrator
│   ├── Step 1: CRL bandit (crl_factor_bandit_conformal)
│   ├── Step 2: Baselines (baselines_crl)
│   ├── Step 3: Causal effects (causal_effects)
│   ├── Step 4: Off-policy evaluation (ope_dr)
│   ├── Step 5: Report generation (generate_report)
│   └── Step 6: Ablation study (ablation_study)
│
├── Core Implementation
│   ├── crl_factor_bandit_conformal.py  # Main CRL with Linear Thompson Sampling
│   ├── baselines_crl.py                # Fixed specification baselines
│   ├── causal_effects.py               # Double ML causal validation
│   ├── ablation_study.py               # Parameter sensitivity analysis
│   └── run_config.py                   # Configuration utilities
│
├── Data Processing
│   ├── data.py                         # IDEX CDI data loader and normalizer and Panel preparation for CRL
│
├── Evaluation & Reporting
│   ├── generate_report.py              # Dissertation-grade figures/tables
│   ├── ope_dr.py                       # Off-policy evaluation
│   └── utils_timesplit.py              # Purged time series cross-validation
│
├── data/                               # Data directory (not included)
│   ├── idex_cdi_*.xlsx                # Raw IDEX CDI files
│   └── cdi_panel.parquet              # Processed panel data
│
├── results/                            # Experiment outputs
│   └── cdi/crl/
│       ├── crl_scores_securities.csv  # Security-level predictions
│       ├── crl_policy_choices.csv     # Policy decisions over time
│       ├── analysis.json              # Performance metrics
│       ├── ablation/                  # Ablation study results
│       └── report/                    # Generated figures and tables
│
```

## Data Preparation

1. Place IDEX CDI Excel files in `data/` directory
2. Run data preparation:
```python
python data.py --universe cdi
```

### Running the Complete Pipeline

```bash
python run_experiments.py --config config_crl.yaml --ablation
```

## 📈 Methodology

### Contextual Bandit Framework

The system treats specification selection as a sequential learning problem:

- **State Space** (10-dimensional): Market volatility, credit spreads, autocorrelation, cross-sectional ranks
- **Action Space**: 16 combinations (4 momentum windows × 4 forecast horizons)
- **Reward**: Negative interval score with regularization for stability
- **Algorithm**: Linear Thompson Sampling for exploration-exploitation balance

### Key Characteristics

1. **Leakage-Safe Implementation**
   - Deferred updates: Models update only when outcomes mature (at t+h)
   - Label-free action gating: Valid horizons determined by data availability, not future returns
   - Horizon-aware purged cross-validation for causal validation

2. **Conformalized Quantile Regression (CQR)**
   - Online quantile models with adaptive learning rates
   - Separate calibration per horizon
   - Recency weighting for non-stationary adaptation

3. **Causal Validation**
   - Double machine learning with temporal controls
   - Heterogeneous treatment effect analysis
   - Counterfactual policy comparisons

## 🔬 Core Modules

### `crl_factor_bandit_conformal.py`
Main CRL implementation with:
- `BanditPolicy`: Linear Thompson Sampling for action selection
- `OnlineQuantileHead`: Dual-head quantile regression (5th/95th percentiles)
- `CQRCalibrator`: Conformal calibration with rolling windows
- `build_enhanced_features()`: 10-dimensional state representation

### `baselines_crl.py`
Fixed specification baselines:
- Traditional (21-day momentum, 21-day horizon)
- Best fixed (selected from training performance)
- Random selection
- All fixed combinations for comparison

### `causal_effects.py`
Causal inference module:
- Out-of-fold value function estimation
- Counterfactual fixed policies
- Treatment effect heterogeneity analysis
- Newey-West standard errors for time series

### `ablation_study.py`
Sensitivity analysis:
- Coverage level variations (α ∈ {0.05, 0.10, 0.20})
- Calibration window sizes (125, 250, 500)
- Recency weighting parameters
- Bandit regularization strength

## 📊 Configuration

The `config_crl.yaml` file controls all experiment parameters.

## 📈 Results & Visualization

The pipeline generates comprehensive analysis outputs:

### Performance Tables
- `performance_table.csv`: CRL vs baselines comparison
- `by_horizon_table.csv`: Horizon-specific metrics
- `per_security_table.csv`: Security-level performance

### Figures (in `results/cdi/crl/report/`)
- `baselines_bar.png`: Performance comparison
- `rolling_cov_60d.png`: Coverage over time
- `policy_evolution.png`: Specification choices over time
- `ablation_overview.png`: Sensitivity analysis

### Statistical Tests
- Newey-West HAC standard errors
- Moving block bootstrap confidence intervals
- Diebold-Mariano predictive accuracy tests

## 🤝 Contributing

While this is a dissertation project, contributions are welcome for:
- Extensions to other asset classes
- Alternative bandit algorithms
- Enhanced causal inference methods
- Performance optimizations
