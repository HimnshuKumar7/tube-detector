"""
run_pipeline.py  —  executes Steps 1-4 in order.

Usage (from the project root, with venv activated):
    python run_pipeline.py

Prerequisites:
    pip install -r requirements.txt
    Place all 70 images in data/uploads/
    Place annotations.csv and labels.csv in data/
"""

import subprocess
import sys
import os

STEPS = [
    ("Step 1 – Split data",           "step1_split_data.py"),
    ("Step 2 – Train YOLOv8-OBB",     "step2_train_obb.py"),
    ("Step 3 – Train YOLOv8-Pose",    "step3_train_pose.py"),
    ("Step 4 – Inference & Metrics",  "step4_inference_metrics.py"),
]

def run(script_name, step_label):
    print("\n" + "═"*70)
    print(f"  {step_label}")
    print("═"*70)
    ret = subprocess.run([sys.executable, script_name], check=False)
    if ret.returncode != 0:
        print(f"\n❌  {script_name} failed (exit {ret.returncode}). Aborting.")
        sys.exit(ret.returncode)
    print(f"\n✅  {step_label} finished.")

for label, script in STEPS:
    run(script, label)

print("\n\n🎉  Full pipeline complete!")
