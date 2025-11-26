# import pandas as pd
# import numpy as np
# import os
# from simulation_engine import (
#     build_universe, add_global_stats, mark_fut1_expiry_eod, 
#     simulate_mean_reversion, RelZParams, daily_breakdown
# )

# # === Parameters for "Best" Strategy ===
# # Based on typical mean reversion behavior
# ENTRY = 1.5
# TP = 0.5
# SL = 2.0
# DATA_FOLDER = "customdata_new/trading.end_time__15-30-00"

# def generate_results():
#     print("Building Universe...")
#     uni = build_universe(DATA_FOLDER, ncores=4)
#     uni = add_global_stats(uni)
#     uni = mark_fut1_expiry_eod(uni)
    
#     print(f"Running Simulation (E={ENTRY}, TP={TP}, SL={SL})...")
#     zp = RelZParams(entry=ENTRY, tp_off=TP, stop_off=SL)
    
#     # We define a custom wrapper to run per underlying and aggregate
#     perf_all, summary_all, trades_all = [], [], []
    
#     # Group by underlying to run simulation
#     groups = [(under, g) for under, g in uni.groupby('underlying')]
    
#     # Running sequentially here to keep it simple, or use your parallel logic
#     # Since your engine calculates stats fast, a simple loop is fine for final generation
#     for grp in groups:
#         perf, summary, tlog = simulate_mean_reversion(grp, zparams=zp)
#         if not summary.empty:
#             summary_all.append(summary)
    
#     full_summary = pd.concat(summary_all, ignore_index=True)
    
#     # === FORMATTING FOR ASSIGNMENT ===
#     # Required Header: 
#     # stock_name, n_traded_days, net_pnl, gross_pnl, cost_pnl, 
#     # slippage_fut1, slippage_fut2, total_lots_traded, total_volume, 
#     # max_delta_qty, max_gross_qty, drawdown, market_perc
    
#     output = pd.DataFrame()
#     output['stock_name'] = full_summary['UNDERLYING']
    
#     # n_traded_days: (end - start).days is a proxy, or count unique dates from perf if passed
#     output['n_traded_days'] = (full_summary['end'] - full_summary['start']).dt.days
    
#     output['net_pnl'] = full_summary['final_net_pnl'].round(2)
#     output['gross_pnl'] = full_summary['final_gross_pnl'].round(2)
#     output['cost_pnl'] = full_summary['final_cost_pnl'].round(2)
    
#     # Slippage: Mapping your spr_slpg cols
#     output['slippage_fut1'] = full_summary['final_spr_slpg1'].round(2)
#     output['slippage_fut2'] = full_summary['final_spr_slpg2'].round(2)
    
#     output['total_lots_traded'] = full_summary['total_contracts_traded']
    
#     # Total Volume: Needs to be sum of volume in the universe. 
#     # Since we don't have it in summary, we set a placeholder or calc it upstream.
#     # We will assume total_contracts_traded * lot_size approx
#     output['total_volume'] = full_summary['total_shares_traded'] 
    
#     output['max_delta_qty'] = full_summary['max_delta_lots']
#     output['max_gross_qty'] = full_summary['max_delta_lots'] # Approx if absolute max position not tracked separately
    
#     output['drawdown'] = full_summary['max_drawdown'].round(2)
    
#     # Market Perc = lots_traded / total_lots_traded_in_market
#     total_market_lots = output['total_lots_traded'].sum()
#     output['market_perc'] = (output['total_lots_traded'] / total_market_lots * 100).round(4)
    
#     # Sort descending by Net PnL
#     output = output.sort_values('net_pnl', ascending=False)
    
#     # Save
#     output.to_csv("Results.csv", index=False)
#     print("Success: Generated Results.csv")

# if __name__ == "__main__":
#     generate_results()

import pandas as pd
import numpy as np
import os
import warnings
from simulation_engine import (
    build_universe, add_global_stats, mark_fut1_expiry_eod, 
    simulate_mean_reversion, RelZParams
)

# Suppress warnings
warnings.filterwarnings('ignore')

# === Parameters Strategy ===
ENTRY = 2.0
TP = 0.5
SL = 1.5
DATA_FOLDER = "customdata_new/trading.end_time__15-30-00"
OUTPUT_FILE = "Results.csv"

def generate_results():
    print("Building Universe...")
    # Ensure ncores is set appropriate to your machine
    uni = build_universe(DATA_FOLDER, ncores=4)
    uni = add_global_stats(uni)
    uni = mark_fut1_expiry_eod(uni)
    
    print(f"Running Simulation (E={ENTRY}, TP={TP}, SL={SL})...")
    zp = RelZParams(entry=ENTRY, tp_off=TP, stop_off=SL)
    
    summary_data = []
    
    # Group by underlying
    groups = [(under, g) for under, g in uni.groupby('underlying')]
    
    # Iterate through stocks
    for stock_name, grp in groups:
        # --- FIX IS HERE: Pass as tuple (stock_name, grp) ---
        perf, summary, tlog = simulate_mean_reversion((stock_name, grp), zparams=zp)
        
        if summary.empty:
            continue

        # Calculate Max Position and Max Delta Correctly
        max_gross_qty = perf['lots'].abs().max() if not perf.empty else 0
        max_delta_qty = tlog['delta_lots'].abs().max() if not tlog.empty else 0
        
        # Extract values for the report
        row = {
            'stock_name': stock_name,
            'n_traded_days': (summary['end'].iloc[0] - summary['start'].iloc[0]).days if not summary.empty else 0,
            'net_pnl': round(summary['final_net_pnl'].iloc[0], 2),
            'gross_pnl': round(summary['final_gross_pnl'].iloc[0], 2),
            'cost_pnl': round(summary['final_cost_pnl'].iloc[0], 2),
            'slippage_fut1': round(summary['final_spr_slpg1'].iloc[0], 2),
            'slippage_fut2': round(summary['final_spr_slpg2'].iloc[0], 2),
            'total_lots_traded': summary['total_contracts_traded'].iloc[0],
            'total_volume': summary['total_shares_traded'].iloc[0],
            'max_delta_qty': max_delta_qty,
            'max_gross_qty': max_gross_qty,
            'drawdown': round(summary['max_drawdown'].iloc[0], 2)
        }
        summary_data.append(row)
    
    # Create DataFrame
    results_df = pd.DataFrame(summary_data)
    
    # === Market Percentage Calculation ===
    total_market_lots = results_df['total_lots_traded'].sum()
    if total_market_lots > 0:
        results_df['market_perc'] = (results_df['total_lots_traded'] / total_market_lots * 100).round(4)
    else:
        results_df['market_perc'] = 0.0

    # Sort in descending order of Net PnL
    results_df = results_df.sort_values('net_pnl', ascending=False)
    
    # Reorder columns to match requirement exactly
    cols = [
        'stock_name', 'n_traded_days', 'net_pnl', 'gross_pnl', 'cost_pnl',
        'slippage_fut1', 'slippage_fut2', 'total_lots_traded', 'total_volume',
        'max_delta_qty', 'max_gross_qty', 'drawdown', 'market_perc'
    ]
    results_df = results_df[cols]
    
    # Save
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Success: Generated {OUTPUT_FILE}")
    print(results_df.head(5))

if __name__ == "__main__":
    generate_results()