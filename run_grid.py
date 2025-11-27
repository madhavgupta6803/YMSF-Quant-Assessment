import os
import pandas as pd
from itertools import product
from simulation_engine import (
    build_universe, 
    add_global_stats, 
    mark_fut1_expiry_eod, 
    simulate_mean_reversion, 
    RelZParams
)

# Config
DATA_FOLDER = "customdata_new/trading.end_time__15-30-00"
RESULTS_DIR = "./grid_results"

def run_relz_grid_fixed(uni: pd.DataFrame,
                        entry_list,
                        tp_off_list,
                        stop_off_list,
                        save_dir,
                        prefix="mr_spread"):
    """
    A fixed version of the grid search that correctly loops over stocks.
    """
    # 1. Group the universe by underlying stock ONCE
    groups = [(under, g) for under, g in uni.groupby('underlying')]
    
    all_summary = []

    # 2. Loop over every parameter combination
    for E, TP, SL in product(entry_list, tp_off_list, stop_off_list):
        print(f"Testing Config: Entry={E}, TP={TP}, SL={SL}...")
        
        zp = RelZParams(entry=E, tp_off=TP, stop_off=SL)
        
        # Accumulate results for THIS specific parameter set
        current_summary_list = []
        current_tlog_list = []
        
        for stock_name, grp in groups:
            # Correctly pass as tuple (name, df)
            p, s, t = simulate_mean_reversion((stock_name, grp), zparams=zp)
            
            if not s.empty:
                current_summary_list.append(s)
                current_tlog_list.append(t)
        
        if not current_summary_list:
            continue

        # Combine results for this config
        summary = pd.concat(current_summary_list, ignore_index=True)
        tlog = pd.concat(current_tlog_list, ignore_index=True)

        # Tag the data with the parameters used
        cfg_name = f"E{E}_TP{TP}_SL{SL}"
        summary['cfg'] = cfg_name
        
        # Save individual config result
        if save_dir:
            summary.to_csv(f"{save_dir}/{prefix}_summary_{cfg_name}.csv", index=False)
            # Optional: Save trades if you need deep debugging (uses a lot of disk space)
            # tlog.to_csv(f"{save_dir}/{prefix}_trades_{cfg_name}.csv", index=False)
            
        all_summary.append(summary)

    print("Grid Search Finished.")

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("Loading Universe...")
    # Load data
    uni = build_universe(DATA_FOLDER, ncores=4)
    uni = add_global_stats(uni)
    uni = mark_fut1_expiry_eod(uni)
    
    print(f"Starting Grid Search in {RESULTS_DIR}...")
    
    # Run the fixed grid search logic
    run_relz_grid_fixed(
        uni,
        entry_list=[1.5, 2.0, 2.5],       # Test these Entries
        tp_off_list=[0.5, 1.0],           # Test these Take Profits
        stop_off_list=[1.5, 2.0],         # Test these Stop Losses
        save_dir=RESULTS_DIR,
        prefix="mr_spread_rel"
    )

if __name__ == "__main__":
    main()