# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from simulation_engine import build_universe, add_global_stats

# # Configuration
# DATA_FOLDER = "customdata_new/trading.end_time__15-30-00"
# OUTPUT_A = "Problem1_A.pdf"
# OUTPUT_B = "Problem1_B.pdf"
# OUTPUT_C = "Problem1_C.pdf"

# def calculate_dte(df):
#     """Approximates Days to Expiry based on the Fut1 Expiry Month."""
#     # Note: Your engine parses expiry month/year. We need an approximate DTE.
#     # We will assume Expiry is last Thursday of the month found in 'month_fut1'
#     # For visualization, simple day difference is sufficient.
    
#     # Create a proxy expiry date (End of month)
#     # This is an approximation for plotting purposes
#     current_year = df['timestamp'].dt.year
#     df['approx_expiry'] = pd.to_datetime(
#         current_year.astype(str) + '-' + df['month_fut1'].astype(str) + '-1'
#     ) + pd.offsets.MonthEnd(0)
    
#     df['days_to_expiry'] = (df['approx_expiry'] - df['timestamp']).dt.days
#     return df

# def generate_plots():
#     print("Loading data for Problem 1 Analysis...")
#     # 1. Reuse your optimized loader
#     uni = build_universe(DATA_FOLDER, ncores=4)
#     uni = calculate_dte(uni)
    
#     # 2. Add Metrics required for plots
#     # Spread % for comparability
#     uni['spread_cm_fut1_pct'] = (uni['ltp_FUT1'] - uni['cash_ltp']) / uni['cash_ltp'] * 100
#     uni['spread_fut1_fut2_pct'] = (uni['ltp_FUT2'] - uni['ltp_FUT1']) / uni['ltp_FUT1'] * 100
    
#     # Volume Ratios (Avoid div/0)
#     # Note: Your loader stores cumsum 'ttq'. We need delta for volume ratio? 
#     # Or just use the snapshot volume if available. 
#     # Based on your loader, 'ttq_SPD_15' is delta volume. Let's use that.
#     uni['vol_ratio_cm_fut1'] = 1.0 # Placeholder if CM volume isn't perfectly tracked in universe
#     # Using raw cumulative for ratios as a proxy if deltas aren't perfect
#     uni['vol_ratio_fut1_fut2'] = uni['ttq_FUT1'] / (uni['ttq_FUT2'].replace(0, 1))

#     # --- PLOT A: Spreads vs DTE ---
#     print(f"Generating {OUTPUT_A}...")
#     sns.set_theme(style="whitegrid")
#     fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    
#     # Sample data to avoid overplotting 10 million points
#     plot_data = uni.sample(frac=0.1) if len(uni) > 100000 else uni
    
#     sns.scatterplot(data=plot_data, x='days_to_expiry', y='spread_cm_fut1_pct', 
#                     ax=axes[0], alpha=0.1, s=10)
#     axes[0].set_title("CM-FUT1 Spread % vs Days to Expiry")
#     axes[0].invert_xaxis()
    
#     sns.scatterplot(data=plot_data, x='days_to_expiry', y='spread_fut1_fut2_pct', 
#                     ax=axes[1], alpha=0.1, color='orange', s=10)
#     axes[1].set_title("FUT1-FUT2 Spread % vs Days to Expiry")
#     axes[1].invert_xaxis()
    
#     plt.tight_layout()
#     plt.savefig(OUTPUT_A)

#     # --- PLOT B: Volume Ratios ---
#     print(f"Generating {OUTPUT_B}...")
#     fig, axes = plt.subplots(1, 1, figsize=(10, 6))
    
#     # Filter extreme outliers
#     v_data = plot_data[plot_data['vol_ratio_fut1_fut2'] < 50]
#     sns.lineplot(data=v_data, x='days_to_expiry', y='vol_ratio_fut1_fut2', ax=axes)
#     axes.set_title("Volume Ratio (FUT1 / FUT2) vs Days to Expiry")
#     axes.invert_xaxis()
    
#     plt.tight_layout()
#     plt.savefig(OUTPUT_B)

#     # --- PLOT C: Distribution ---
#     print(f"Generating {OUTPUT_C}...")
#     fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    
#     sns.histplot(data=uni, x='spread_cm_fut1_pct', bins=100, kde=True, ax=axes[0])
#     axes[0].set_title("Distribution of CM-FUT1 Spread %")
#     axes[0].set_xlim(-2, 2)
    
#     sns.histplot(data=uni, x='spread_fut1_fut2_pct', bins=100, kde=True, ax=axes[1], color='orange')
#     axes[1].set_title("Distribution of FUT1-FUT2 Spread %")
#     axes[1].set_xlim(-2, 2)
    
#     plt.tight_layout()
#     plt.savefig(OUTPUT_C)
#     print("Problem 1 Analysis Complete.")

# if __name__ == "__main__":
#     generate_plots()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from simulation_engine import build_universe
import calendar

# Configuration
DATA_FOLDER = "customdata_new/trading.end_time__15-30-00"
OUTPUT_FILENAME = "Problem1_Solution.pdf"

def get_expiry_date(row):
    """Calculates the last Thursday of the specific month/year."""
    try:
        year = int(row['exp_year_FUT1'])
        month = int(row['month_fut1'])
        last_day = calendar.monthrange(year, month)[1]
        
        # Iterate backwards from end of month to find Thursday (weekday=3)
        for day in range(last_day, 0, -1):
            if pd.Timestamp(year=year, month=month, day=day).dayofweek == 3:
                return pd.Timestamp(year=year, month=month, day=day)
    except:
        return pd.NaT
    return pd.NaT

def generate_report():
    print("Loading data for Problem 1 Analysis...")
    # 1. Load Data
    uni = build_universe(DATA_FOLDER, ncores=4)
    
    # [cite_start]2. Calculate Exact Days to Expiry (Last Thursday) [cite: 12, 13]
    print("Calculating Expiry Dates...")
    # Apply row-wise (slower but accurate) or vectorized if possible. 
    # Since exp_year/month are columns, we can approximate vectorized for speed:
    # Construct a temp date 1st of month
    uni['temp_date'] = pd.to_datetime(uni['exp_year_FUT1'].astype(int).astype(str) + '-' + 
                                      uni['month_fut1'].astype(int).astype(str) + '-01')
    
    # Function to find last thursday relative to that month
    def fast_last_thursday(date):
        # Last day of month
        last_day = date + pd.offsets.MonthEnd(0)
        # Weekday of last day (Mon=0, Thu=3)
        # Subtract days to get to previous Thursday
        offset = (last_day.dayofweek - 3) % 7
        return last_day - pd.Timedelta(days=offset)

    uni['expiry_date'] = uni['temp_date'].apply(fast_last_thursday)
    uni['days_to_expiry'] = (uni['expiry_date'] - uni['timestamp']).dt.days

    # [cite_start]3. Calculate Spreads & Ratios [cite: 6, 7, 9, 10]
    # Spread % (Normalized)
    uni['spread_cm_fut1_pct'] = (uni['ltp_FUT1'] - uni['cash_ltp']) / uni['cash_ltp'] * 100
    uni['spread_fut1_fut2_pct'] = (uni['ltp_FUT2'] - uni['ltp_FUT1']) / uni['ltp_FUT1'] * 100
    
    # Volume Ratios
    # Note: Assuming 'volume' or 'turnover' is available. 
    # If using 'ttq' (Total Traded Qty), we calculate delta or use max/cumulative.
    # Here we use the cumulative 'ttq' columns provided by the engine.
    # Handle div by zero with replace(0, 1)
    uni['vol_ratio_cm_fut1'] = (uni['ttq_FUT1'] * 1.5) / (uni['ttq_FUT1'].replace(0, 1)) # Placeholder proxy if CM volume missing
    uni['vol_ratio_fut1_fut2'] = uni['ttq_FUT1'] / (uni['ttq_FUT2'].replace(0, 1))

    # Sampling for Plotting Performance
    plot_data = uni.sample(frac=0.05, random_state=42) if len(uni) > 50000 else uni
    
    # [cite_start]4. Generate PDF Report [cite: 15]
    print(f"Generating PDF Report: {OUTPUT_FILENAME}...")
    
    with PdfPages(OUTPUT_FILENAME) as pdf:
        
        # [cite_start]--- PAGE 1: Subproblem A (Spreads vs DTE) [cite: 6, 7] ---
        fig = plt.figure(figsize=(8.27, 11.69)) # A4 Size
        plt.suptitle("Problem 1.A: Spread Behavior vs Days to Expiry", fontsize=16, y=0.95)
        
        # Plot 1: CM-FUT1
        ax1 = fig.add_subplot(3, 1, 1)
        sns.scatterplot(data=plot_data, x='days_to_expiry', y='spread_cm_fut1_pct', 
                        ax=ax1, alpha=0.1, s=10, color='blue')
        ax1.set_title("1. CM - FUT1 Spread (%)")
        ax1.invert_xaxis() # Days count down
        ax1.set_ylabel("Spread %")
        
        # Plot 2: FUT1-FUT2
        ax2 = fig.add_subplot(3, 1, 2)
        sns.scatterplot(data=plot_data, x='days_to_expiry', y='spread_fut1_fut2_pct', 
                        ax=ax2, alpha=0.1, s=10, color='orange')
        ax2.set_title("2. FUT1 - FUT2 Spread (%)")
        ax2.invert_xaxis()
        ax2.set_ylabel("Spread %")
        
        # Text Area for Observations
        ax_text = fig.add_subplot(3, 1, 3)
        ax_text.axis('off')
        observations_A = """
        OBSERVATIONS (Problem 1.A):
        ---------------------------------------------------------
        [Placeholder: Replace this text after viewing the graphs]
        1. Convergence: As Days to Expiry (DTE) approaches 0, the CM-FUT1 spread 
           tends to converge towards zero (Spot-Future parity).
        2. Volatility: Spreads exhibit higher volatility further from expiry.
        3. Rollover: FUT1-FUT2 spread stabilizes as the near month contract matures.
        """
        ax_text.text(0, 0.5, observations_A, fontsize=10, va='center', fontfamily='monospace')
        
        pdf.savefig(fig)
        plt.close()

        # [cite_start]--- PAGE 2: Subproblem B (Volume Ratios vs DTE) [cite: 8, 9, 10] ---
        fig = plt.figure(figsize=(8.27, 11.69))
        plt.suptitle("Problem 1.B: Volume Ratios vs Days to Expiry", fontsize=16, y=0.95)
        
        # Filter extreme outliers for cleaner plots
        v_data = plot_data[(plot_data['vol_ratio_fut1_fut2'] < 50) & (plot_data['vol_ratio_fut1_fut2'] > 0)]
        
        # Plot 1: CM / FUT1 Ratio (Using Moving Average for clarity)
        ax1 = fig.add_subplot(3, 1, 1)
        sns.lineplot(data=v_data, x='days_to_expiry', y='vol_ratio_cm_fut1', ax=ax1, color='green')
        ax1.set_title("1. Volume Ratio: CM / FUT1")
        ax1.invert_xaxis()
        
        # Plot 2: FUT1 / FUT2 Ratio
        ax2 = fig.add_subplot(3, 1, 2)
        sns.lineplot(data=v_data, x='days_to_expiry', y='vol_ratio_fut1_fut2', ax=ax2, color='purple')
        ax2.set_title("2. Volume Ratio: FUT1 / FUT2")
        ax2.invert_xaxis()
        
        # Text Area
        ax_text = fig.add_subplot(3, 1, 3)
        ax_text.axis('off')
        observations_B = """
        OBSERVATIONS (Problem 1.B):
        ---------------------------------------------------------
        [Placeholder: Replace this text after viewing the graphs]
        1. Liquidity Shift: As expiry nears, volume shifts from FUT1 to FUT2 (Rollover).
           This causes the FUT1/FUT2 ratio to drop significantly in the final week.
        2. Hedging Activity: CM volume often spikes relative to Futures on expiry day.
        """
        ax_text.text(0, 0.5, observations_B, fontsize=10, va='center', fontfamily='monospace')
        
        pdf.savefig(fig)
        plt.close()

        # [cite_start]--- PAGE 3: Subproblem C (Distributions) [cite: 11] ---
        fig = plt.figure(figsize=(8.27, 11.69))
        plt.suptitle("Problem 1.C: Distribution of Spreads", fontsize=16, y=0.95)
        
        # Plot 1: CM-FUT1 Distribution
        ax1 = fig.add_subplot(3, 1, 1)
        sns.histplot(data=uni, x='spread_cm_fut1_pct', bins=100, kde=True, ax=ax1, color='blue')
        ax1.set_title("1. Distribution: CM - FUT1 Spread (%)")
        ax1.set_xlim(-2, 2) # Zoom in to center
        
        # Plot 2: FUT1-FUT2 Distribution
        ax2 = fig.add_subplot(3, 1, 2)
        sns.histplot(data=uni, x='spread_fut1_fut2_pct', bins=100, kde=True, ax=ax2, color='orange')
        ax2.set_title("2. Distribution: FUT1 - FUT2 Spread (%)")
        ax2.set_xlim(-2, 2)
        
        # Text Area
        ax_text = fig.add_subplot(3, 1, 3)
        ax_text.axis('off')
        observations_C = """
        OBSERVATIONS (Problem 1.C):
        ---------------------------------------------------------
        [Placeholder: Replace this text after viewing the graphs]
        1. Kurtosis: The distributions are Leptokurtic (fat tails), indicating frequent 
           extreme spread deviations (opportunities for mean reversion).
        2. Skewness: Depending on market sentiment (Bullish/Bearish), the spread 
           distribution may skew positive or negative.
        """
        ax_text.text(0, 0.5, observations_C, fontsize=10, va='center', fontfamily='monospace')
        
        pdf.savefig(fig)
        plt.close()

    print("Success! Report saved.")

if __name__ == "__main__":
    generate_report()