import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from simulation_engine import build_universe, add_global_stats

# Configuration
DATA_FOLDER = "customdata_new/trading.end_time__15-30-00"
OUTPUT_A = "Problem1_A.pdf"
OUTPUT_B = "Problem1_B.pdf"
OUTPUT_C = "Problem1_C.pdf"

def calculate_dte(df):
    """Approximates Days to Expiry based on the Fut1 Expiry Month."""
    # Note: Your engine parses expiry month/year. We need an approximate DTE.
    # We will assume Expiry is last Thursday of the month found in 'month_fut1'
    # For visualization, simple day difference is sufficient.
    
    # Create a proxy expiry date (End of month)
    # This is an approximation for plotting purposes
    current_year = df['timestamp'].dt.year
    df['approx_expiry'] = pd.to_datetime(
        current_year.astype(str) + '-' + df['month_fut1'].astype(str) + '-1'
    ) + pd.offsets.MonthEnd(0)
    
    df['days_to_expiry'] = (df['approx_expiry'] - df['timestamp']).dt.days
    return df

def generate_plots():
    print("Loading data for Problem 1 Analysis...")
    # 1. Reuse your optimized loader
    uni = build_universe(DATA_FOLDER, ncores=4)
    uni = calculate_dte(uni)
    
    # 2. Add Metrics required for plots
    # Spread % for comparability
    uni['spread_cm_fut1_pct'] = (uni['ltp_FUT1'] - uni['cash_ltp']) / uni['cash_ltp'] * 100
    uni['spread_fut1_fut2_pct'] = (uni['ltp_FUT2'] - uni['ltp_FUT1']) / uni['ltp_FUT1'] * 100
    
    # Volume Ratios (Avoid div/0)
    # Note: Your loader stores cumsum 'ttq'. We need delta for volume ratio? 
    # Or just use the snapshot volume if available. 
    # Based on your loader, 'ttq_SPD_15' is delta volume. Let's use that.
    uni['vol_ratio_cm_fut1'] = 1.0 # Placeholder if CM volume isn't perfectly tracked in universe
    # Using raw cumulative for ratios as a proxy if deltas aren't perfect
    uni['vol_ratio_fut1_fut2'] = uni['ttq_FUT1'] / (uni['ttq_FUT2'].replace(0, 1))

    # --- PLOT A: Spreads vs DTE ---
    print(f"Generating {OUTPUT_A}...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    
    # Sample data to avoid overplotting 10 million points
    plot_data = uni.sample(frac=0.1) if len(uni) > 100000 else uni
    
    sns.scatterplot(data=plot_data, x='days_to_expiry', y='spread_cm_fut1_pct', 
                    ax=axes[0], alpha=0.1, s=10)
    axes[0].set_title("CM-FUT1 Spread % vs Days to Expiry")
    axes[0].invert_xaxis()
    
    sns.scatterplot(data=plot_data, x='days_to_expiry', y='spread_fut1_fut2_pct', 
                    ax=axes[1], alpha=0.1, color='orange', s=10)
    axes[1].set_title("FUT1-FUT2 Spread % vs Days to Expiry")
    axes[1].invert_xaxis()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_A)

    # --- PLOT B: Volume Ratios ---
    print(f"Generating {OUTPUT_B}...")
    fig, axes = plt.subplots(1, 1, figsize=(10, 6))
    
    # Filter extreme outliers
    v_data = plot_data[plot_data['vol_ratio_fut1_fut2'] < 50]
    sns.lineplot(data=v_data, x='days_to_expiry', y='vol_ratio_fut1_fut2', ax=axes)
    axes.set_title("Volume Ratio (FUT1 / FUT2) vs Days to Expiry")
    axes.invert_xaxis()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_B)

    # --- PLOT C: Distribution ---
    print(f"Generating {OUTPUT_C}...")
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    
    sns.histplot(data=uni, x='spread_cm_fut1_pct', bins=100, kde=True, ax=axes[0])
    axes[0].set_title("Distribution of CM-FUT1 Spread %")
    axes[0].set_xlim(-2, 2)
    
    sns.histplot(data=uni, x='spread_fut1_fut2_pct', bins=100, kde=True, ax=axes[1], color='orange')
    axes[1].set_title("Distribution of FUT1-FUT2 Spread %")
    axes[1].set_xlim(-2, 2)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_C)
    print("Problem 1 Analysis Complete.")

if __name__ == "__main__":
    generate_plots()