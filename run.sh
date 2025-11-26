#!/bin/bash

echo "--- YMSF QR Assignment Submission ---"

# 1. Install dependencies (if needed, usually pre-installed in environment)
# pip install pandas numpy seaborn matplotlib tqdm

# 2. Run Problem 1 Analysis (PDF Generation)
echo "[1/2] Generating Problem 1 Analysis PDFs..."
python3 problem1_plots.py

# 3. Run Problem 2 Simulation (Results CSV)
echo "[2/2] Running Strategy Simulation & Generating Results.csv..."
python3 problem2_runner.py

echo "--- Done! Check the folder for PDFs and Results.csv ---"