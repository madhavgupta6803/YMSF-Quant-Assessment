# run_grid.py
import os
from simulation_engine import build_universe, add_global_stats, mark_fut1_expiry_eod, run_relz_grid

# Config
DATA_FOLDER = "customdata_new/trading.end_time__15-30-00"
RESULTS_DIR = "./grid_results"

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("Loading Universe...")
    uni = build_universe(DATA_FOLDER, ncores=4)
    uni = add_global_stats(uni)
    uni = mark_fut1_expiry_eod(uni)
    
    print("Running Grid Search...")
    # This will generate multiple CSV files in ./grid_results
    # with names like mr_spread_rel_summary_E1.5_TP0.5_SL2.0.csv
    run_relz_grid(
        uni,
        entry_list=[1.5, 2.0, 2.5],       # Test these Entries
        tp_off_list=[0.5, 1.0],           # Test these TPs
        stop_off_list=[1.5, 2.0],         # Test these SLs
        save_dir=RESULTS_DIR,
        prefix="mr_spread_rel"
    )
    print("Grid Search Complete.")

if __name__ == "__main__":
    main()