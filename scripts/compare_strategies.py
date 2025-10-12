"""
Compare heuristic vs RL agent strategies to identify the performance gap.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_ledger(path):
    """Load a parquet ledger file."""
    return pd.read_parquet(path)

def analyze_strategy(ledger_path, label):
    """Analyze charging/discharging patterns from a ledger."""
    df = load_ledger(ledger_path)
    
    # Calculate key metrics
    total_grid_cost = df['grid_cost'].sum() if 'grid_cost' in df else 0
    total_ev_charging = df['ev_power_kw'][df['ev_power_kw'] > 0].sum() if 'ev_power_kw' in df else 0
    total_ev_discharging = df['ev_power_kw'][df['ev_power_kw'] < 0].sum() if 'ev_power_kw' in df else 0
    
    # Find charging during expensive vs cheap periods
    if 'electricity_price' in df:
        price_median = df['electricity_price'].median()
        expensive_charging = df[(df['ev_power_kw'] > 0) & (df['electricity_price'] > price_median)]['ev_power_kw'].sum()
        cheap_charging = df[(df['ev_power_kw'] > 0) & (df['electricity_price'] <= price_median)]['ev_power_kw'].sum()
        expensive_discharging = df[(df['ev_power_kw'] < 0) & (df['electricity_price'] > price_median)]['ev_power_kw'].sum()
        cheap_discharging = df[(df['ev_power_kw'] < 0) & (df['electricity_price'] <= price_median)]['ev_power_kw'].sum()
    else:
        expensive_charging = cheap_charging = expensive_discharging = cheap_discharging = 0
    
    results = {
        'label': label,
        'total_cost': total_grid_cost,
        'total_charging_kwh': total_ev_charging,
        'total_discharging_kwh': abs(total_ev_discharging),
        'expensive_charging_kwh': expensive_charging,
        'cheap_charging_kwh': cheap_charging,
        'expensive_discharging_kwh': abs(expensive_discharging),
        'cheap_discharging_kwh': abs(cheap_discharging),
    }
    
    # Calculate efficiency metrics
    if total_ev_charging > 0:
        results['pct_cheap_charging'] = 100 * cheap_charging / total_ev_charging
    else:
        results['pct_cheap_charging'] = 0
        
    if abs(total_ev_discharging) > 0:
        results['pct_expensive_discharging'] = 100 * abs(expensive_discharging) / abs(total_ev_discharging)
    else:
        results['pct_expensive_discharging'] = 0
    
    return results, df

# Find the latest heuristic and RL runs
print("Looking for heuristic and RL ledgers...")

# Heuristic - find most recent
heuristic_dirs = [d for d in os.listdir('results') if d.startswith('ppo_residential_v2g_')]
heuristic_dirs.sort(reverse=True)

heuristic_ledger = None
rl_ledger = None

for d in heuristic_dirs[:5]:  # Check last 5 runs
    ledger_path = os.path.join('results', d, 'ledgers', 'global.parquet')
    if os.path.exists(ledger_path):
        # Check if it has the heuristic marker (run from heuristic script)
        log_path = os.path.join('results', d, 'baseline.log')
        if os.path.exists(log_path):
            heuristic_ledger = ledger_path
            print(f"Found heuristic ledger: {ledger_path}")
        else:
            # Likely an RL run
            if rl_ledger is None:
                rl_ledger = ledger_path
                print(f"Found RL ledger: {ledger_path}")

if heuristic_ledger and rl_ledger:
    print("\n" + "="*80)
    print("STRATEGY COMPARISON")
    print("="*80)
    
    heur_results, heur_df = analyze_strategy(heuristic_ledger, "Heuristic")
    rl_results, rl_df = analyze_strategy(rl_ledger, "RL Agent")
    
    print(f"\n{'Metric':<35} {'Heuristic':>15} {'RL Agent':>15} {'Difference':>15}")
    print("-"*80)
    print(f"{'Total Grid Cost ($)':<35} {heur_results['total_cost']:>15.2f} {rl_results['total_cost']:>15.2f} {rl_results['total_cost']-heur_results['total_cost']:>15.2f}")
    print(f"{'Total Charging (kWh)':<35} {heur_results['total_charging_kwh']:>15.2f} {rl_results['total_charging_kwh']:>15.2f} {rl_results['total_charging_kwh']-heur_results['total_charging_kwh']:>15.2f}")
    print(f"{'Total Discharging (kWh)':<35} {heur_results['total_discharging_kwh']:>15.2f} {rl_results['total_discharging_kwh']:>15.2f} {rl_results['total_discharging_kwh']-heur_results['total_discharging_kwh']:>15.2f}")
    print()
    print(f"{'% Charging During Cheap Periods':<35} {heur_results['pct_cheap_charging']:>15.1f} {rl_results['pct_cheap_charging']:>15.1f} {rl_results['pct_cheap_charging']-heur_results['pct_cheap_charging']:>15.1f}")
    print(f"{'% Discharging During Expensive':<35} {heur_results['pct_expensive_discharging']:>15.1f} {rl_results['pct_expensive_discharging']:>15.1f} {rl_results['pct_expensive_discharging']-heur_results['pct_expensive_discharging']:>15.1f}")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    # Identify the main differences
    cost_gap = rl_results['total_cost'] - heur_results['total_cost']
    print(f"1. Cost Gap: ${cost_gap:.2f}")
    
    if heur_results['pct_cheap_charging'] > rl_results['pct_cheap_charging'] + 5:
        print(f"2. RL agent charges during expensive periods more often ({heur_results['pct_cheap_charging']:.1f}% vs {rl_results['pct_cheap_charging']:.1f}% cheap)")
    
    if heur_results['pct_expensive_discharging'] > rl_results['pct_expensive_discharging'] + 5:
        print(f"3. Heuristic discharges more during expensive periods ({heur_results['pct_expensive_discharging']:.1f}% vs {rl_results['pct_expensive_discharging']:.1f}%)")
    
    if abs(heur_results['total_discharging_kwh']) > abs(rl_results['total_discharging_kwh']) * 1.5:
        print(f"4. Heuristic uses V2G much more aggressively ({heur_results['total_discharging_kwh']:.1f} kWh vs {rl_results['total_discharging_kwh']:.1f} kWh)")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: EV Power over time
    ax = axes[0, 0]
    if 'ev_power_kw' in heur_df and 'ev_power_kw' in rl_df:
        ax.plot(heur_df.index[:1000], heur_df['ev_power_kw'][:1000], label='Heuristic', alpha=0.7)
        ax.plot(rl_df.index[:1000], rl_df['ev_power_kw'][:1000], label='RL Agent', alpha=0.7)
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('EV Power (kW)')
        ax.set_title('EV Charging/Discharging Pattern (First 1000 steps)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Price vs EV Power correlation
    ax = axes[0, 1]
    if 'electricity_price' in heur_df and 'ev_power_kw' in heur_df:
        ax.scatter(heur_df['electricity_price'], heur_df['ev_power_kw'], alpha=0.3, s=10, label='Heuristic')
        ax.scatter(rl_df['electricity_price'], rl_df['ev_power_kw'], alpha=0.3, s=10, label='RL Agent')
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Electricity Price ($/kWh)')
        ax.set_ylabel('EV Power (kW)')
        ax.set_title('Price vs Charging/Discharging Pattern')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Charging distribution by price quartile
    ax = axes[1, 0]
    if 'electricity_price' in heur_df and 'ev_power_kw' in heur_df:
        quartiles = pd.qcut(heur_df['electricity_price'], 4, labels=['Q1 (Cheap)', 'Q2', 'Q3', 'Q4 (Expensive)'])
        
        heur_charging_by_q = []
        rl_charging_by_q = []
        for q in ['Q1 (Cheap)', 'Q2', 'Q3', 'Q4 (Expensive)']:
            heur_charging_by_q.append(heur_df[quartiles == q]['ev_power_kw'].clip(lower=0).sum())
            rl_charging_by_q.append(rl_df[quartiles == q]['ev_power_kw'].clip(lower=0).sum())
        
        x = np.arange(4)
        width = 0.35
        ax.bar(x - width/2, heur_charging_by_q, width, label='Heuristic', alpha=0.7)
        ax.bar(x + width/2, rl_charging_by_q, width, label='RL Agent', alpha=0.7)
        ax.set_xlabel('Price Quartile')
        ax.set_ylabel('Total Charging (kWh)')
        ax.set_title('Charging by Price Quartile')
        ax.set_xticks(x)
        ax.set_xticklabels(['Q1\n(Cheap)', 'Q2', 'Q3', 'Q4\n(Expensive)'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Discharging distribution by price quartile
    ax = axes[1, 1]
    if 'electricity_price' in heur_df and 'ev_power_kw' in heur_df:
        heur_discharging_by_q = []
        rl_discharging_by_q = []
        for q in ['Q1 (Cheap)', 'Q2', 'Q3', 'Q4 (Expensive)']:
            heur_discharging_by_q.append(abs(heur_df[quartiles == q]['ev_power_kw'].clip(upper=0).sum()))
            rl_discharging_by_q.append(abs(rl_df[quartiles == q]['ev_power_kw'].clip(upper=0).sum()))
        
        x = np.arange(4)
        width = 0.35
        ax.bar(x - width/2, heur_discharging_by_q, width, label='Heuristic', alpha=0.7)
        ax.bar(x + width/2, rl_discharging_by_q, width, label='RL Agent', alpha=0.7)
        ax.set_xlabel('Price Quartile')
        ax.set_ylabel('Total Discharging (kWh)')
        ax.set_title('Discharging (V2G) by Price Quartile')
        ax.set_xticks(x)
        ax.set_xticklabels(['Q1\n(Cheap)', 'Q2', 'Q3', 'Q4\n(Expensive)'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/strategy_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: results/strategy_comparison.png")
    
else:
    print("Could not find both heuristic and RL ledgers for comparison.")
    print(f"Heuristic: {heuristic_ledger}")
    print(f"RL: {rl_ledger}")
