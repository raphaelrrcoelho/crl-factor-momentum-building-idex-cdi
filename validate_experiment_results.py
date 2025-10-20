import pandas as pd
import numpy as np
from crl_factor_bandit_conformal import load_panel

def validate_experiment_results():
    """Run comprehensive validation checks on experiment results."""
    
    print("="*60)
    print("EXPERIMENT VALIDATION DIAGNOSTICS")
    print("="*60)
    
    # 1. Check panel data quality
    panel = load_panel("data/cdi_processed.pkl")
    
    print("\n1. DATA QUALITY CHECKS")
    print("-"*40)
    
    # Check returns
    ret_stats = panel['return'].describe()
    print(f"Return statistics:\n{ret_stats}")
    
    extreme = panel[panel['return'].abs() > 0.5]
    if len(extreme) > 0:
        print(f"\n⚠️ {len(extreme)} extreme returns (>50%) found")
        print(f"Max return: {panel['return'].max():.2%}")
        print(f"Min return: {panel['return'].min():.2%}")
    
    # Check spreads
    if 'spread' in panel.columns:
        spread_stats = panel['spread'].describe()
        print(f"\nSpread statistics:\n{spread_stats}")
        if panel['spread'].max() < 0.01:
            print("⚠️ Spreads likely in decimal form, not basis points!")
    
    # 2. Check CRL results
    print("\n2. CRL RESULTS VALIDATION")
    print("-"*40)
    
    try:
        crl = pd.read_csv("results/cdi/crl/crl_scores_securities.csv")
        
        # Coverage by sample size
        cov_by_id = crl.groupby('id').agg({
            'covered': 'mean',
            'loss_scaled': 'mean',
            'id': 'count'
        }).rename(columns={'id': 'n_preds'})
        
        # Find suspicious perfect coverage
        perfect = cov_by_id[(cov_by_id['covered'] == 1.0) & (cov_by_id['n_preds'] < 30)]
        if len(perfect) > 0:
            print(f"\n⚠️ {len(perfect)} securities have IMPOSSIBLE perfect coverage:")
            print(perfect[['covered', 'n_preds']])
        
        # Check if horizon scaling is working
        by_h = crl.groupby('h')[['loss_scaled', 'loss']].mean()
        print(f"\nLoss by horizon (should be comparable if scaling works):")
        print(by_h)
        
    except FileNotFoundError:
        print("CRL results not found")
    
    # 3. Check causal effects
    print("\n3. CAUSAL EFFECTS VALIDATION")
    print("-"*40)
    
    try:
        causal = pd.read_csv("results/cdi/crl/causal_effects_summary.csv")
        
        for _, row in causal.iterrows():
            if pd.notna(row['mean_return']) and pd.notna(row['std_return']):
                daily_sharpe = row['mean_return'] / (row['std_return'] + 1e-12)
                annual_sharpe = daily_sharpe * np.sqrt(252)
                
                print(f"\n{row['policy']}:")
                print(f"  Mean return: {row['mean_return']:.6f}")
                print(f"  Std return: {row['std_return']:.6f}")
                print(f"  Daily Sharpe: {daily_sharpe:.4f}")
                print(f"  Annual Sharpe: {annual_sharpe:.4f}")
                
                if annual_sharpe > 5:
                    print(f"  ⚠️ Sharpe > 5 is suspicious!")
                    
                    # Check if returns are already annualized
                    if row['mean_return'] > 0.01:
                        print(f"  ⚠️ Mean return > 1% suggests already annualized")
                        correct_sharpe = row['mean_return'] / row['std_return']
                        print(f"  Corrected Sharpe: {correct_sharpe:.4f}")
    
    except FileNotFoundError:
        print("Causal effects results not found")
    
    # 4. Securities with too few observations
    print("\n4. SAMPLE SIZE ISSUES")  
    print("-"*40)
    
    sec_counts = panel.groupby('id').size()
    thin = sec_counts[sec_counts < 100]
    
    print(f"Securities with <100 obs: {len(thin)}")
    print(f"Securities with <50 obs: {(sec_counts < 50).sum()}")
    print(f"Securities with <30 obs: {(sec_counts < 30).sum()}")
    
    if len(thin) > 0:
        print(f"\n⚠️ {len(thin)} securities have insufficient data and should be excluded")
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    validate_experiment_results()