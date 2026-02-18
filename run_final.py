"""
Phase C: Final Training & Test
Train on 2021-2024, test on 2025
"""

import sys
import json
import os
from data_loader import RLDataLoader
from train import run_final_train_test

DATA_PATH = "C:\\Users\\Administrator\\Desktop\\RL Agent\\data\\US100_RL_features.parquet"

print("\n" + "="*80)
print("STARTING PHASE C: FINAL TRAIN & TEST")
print("="*80 + "\n")

# Load data
print("Loading data...")
loader = RLDataLoader(DATA_PATH)
print("[OK] Data loaded\n")

# Load most recent metadata for epsilon calibration
# Prefer walk-forward metadata if available, fallback to burn-in
print("Loading metadata for epsilon calibration...")

# Try walk-forward first (most recent fold)
wf_metadata_path = "models/wf_wf-4_agent_metadata.json"
burnin_metadata_path = "models/burnin_agent_metadata.json"

if os.path.exists(wf_metadata_path):
    metadata_path = wf_metadata_path
    print(f"Using walk-forward metadata: {wf_metadata_path}")
elif os.path.exists(burnin_metadata_path):
    metadata_path = burnin_metadata_path
    print(f"Using burn-in metadata: {burnin_metadata_path}")
else:
    print("[ERROR] No metadata found. Please run burn-in or walk-forward first.")
    sys.exit(1)

try:
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        observed_steps_per_day = metadata['observed_steps_per_day']
    print(f"[OK] Loaded observed_steps_per_day: {observed_steps_per_day:.1f}\n")
except KeyError:
    print("[ERROR] observed_steps_per_day not found in metadata")
    sys.exit(1)

# Run final train & test WITH observed_steps_per_day
run_final_train_test(loader, observed_steps_per_day=observed_steps_per_day)

print("\n" + "="*80)
print("[SUCCESS] FINAL TRAINING & TEST COMPLETE")
print("="*80)
print("\nFinal model saved to: models/final_agent.pth")
print("Log saved to: logs/Final-Train-Test_*.txt")
print("\nTraining pipeline complete! Your agent is ready.")