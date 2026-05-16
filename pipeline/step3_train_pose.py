"""
Step 3: Convert labels.csv → YOLOv8-Pose format and train the pose model.

Pose label format (one line per object):
    class cx cy w h  kp1x kp1y kp1v  kp2x kp2y kp2v
    (all spatial values normalised 0-1; v = visibility flag)

Keypoints:
    kp1 = hinge (joint)
    kp2 = tip   (tab)
"""

import os
import math
import numpy as np
import pandas as pd
import yaml
import shutil

# ── Config ─────────────────────────────────────────────────────────────────────
IMG_W, IMG_H = 640, 480
DATA_ROOT    = "data"
TRAIN_LAB    = os.path.join(DATA_ROOT, "train_labels.csv")
TEST_LAB     = os.path.join(DATA_ROOT, "test_labels.csv")
TRAIN_IMGS   = os.path.join(DATA_ROOT, "train", "images")
TEST_IMGS    = os.path.join(DATA_ROOT, "test",  "images")
TRAIN_LBLS   = os.path.join(DATA_ROOT, "train", "labels_pose")
TEST_LBLS    = os.path.join(DATA_ROOT, "test",  "labels_pose")
YAML_PATH    = os.path.join(DATA_ROOT, "pose_dataset.yaml")
MODEL_SAVE   = "runs/pose"

for d in [TRAIN_LBLS, TEST_LBLS]:
    os.makedirs(d, exist_ok=True)

CLASS_ID = 0

# ── Convert one CSV split → per-image .txt pose label files ───────────────────
def convert_split(csv_path, label_dir):
    df = pd.read_csv(csv_path)
    for img_name, grp in df.groupby("image"):
        stem = os.path.splitext(img_name)[0]
        lines = []
        for _, row in grp.iterrows():
            # Normalised bbox (YOLO format: cx cy w h)
            cx = row["cx_px"]  / IMG_W
            cy = row["cy_px"]  / IMG_H
            bw = row["w_px"]   / IMG_W
            bh = row["h_px"]   / IMG_H

            # Clamp
            cx, cy, bw, bh = [max(0.0, min(1.0, v)) for v in [cx, cy, bw, bh]]

            # Keypoints (normalised)
            hx = max(0.0, min(1.0, row["hinge_x_px"] / IMG_W))
            hy = max(0.0, min(1.0, row["hinge_y_px"] / IMG_H))
            hv = int(row["hinge_vis"])   # 0=not labeled, 1=labeled not visible, 2=visible

            tx = max(0.0, min(1.0, row["tip_x_px"] / IMG_W))
            ty = max(0.0, min(1.0, row["tip_y_px"] / IMG_H))
            tv = int(row["tip_vis"])

            line = (f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} "
                    f"{hx:.6f} {hy:.6f} {hv} "
                    f"{tx:.6f} {ty:.6f} {tv}")
            lines.append(line)

        out_path = os.path.join(label_dir, stem + ".txt")
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")

print("Converting train labels to Pose labels …")
convert_split(TRAIN_LAB, TRAIN_LBLS)
print("Converting test labels to Pose labels …")
convert_split(TEST_LAB, TEST_LBLS)
print("Pose label files written.")

# ── Write dataset YAML ─────────────────────────────────────────────────────────
yaml_content = {
    "path":  os.path.abspath(DATA_ROOT),
    "train": "train/images",
    "val":   "test/images",
    "nc":    1,
    "names": {0: "tube"},
    "kpt_shape": [2, 3],    # 2 keypoints, 3 values each (x, y, visibility)
}
with open(YAML_PATH, "w") as f:
    yaml.dump(yaml_content, f, default_flow_style=False)
print(f"Pose dataset YAML → {YAML_PATH}")

# ── Symlink label dirs ─────────────────────────────────────────────────────────
# IMPORTANT: pose labels go in a separate subdir to avoid collision with OBB labels
for split, lbl_dir in [("train", TRAIN_LBLS), ("test", TEST_LBLS)]:
    yolo_lbl = os.path.join(DATA_ROOT, split, "labels")
    if os.path.islink(yolo_lbl):
        os.remove(yolo_lbl)
    elif os.path.isdir(yolo_lbl) and not os.path.islink(yolo_lbl):
        shutil.rmtree(yolo_lbl)
    os.symlink(os.path.abspath(lbl_dir), yolo_lbl)
print("Pose label symlinks created.")

# ── Train YOLOv8-Pose ──────────────────────────────────────────────────────────
print("\n🚀 Starting YOLOv8-Pose training (35 epochs, CPU) …")
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")   # nano pose

results = model.train(
    data      = YAML_PATH,
    epochs    = 35,
    imgsz     = 640,
    batch     = 4,
    device    = "cpu",
    project   = MODEL_SAVE,
    name      = "tube_pose",
    patience  = 10,
    workers   = 0,           # Windows safe
    exist_ok  = True,
    cache     = False,
    verbose   = True,
)

print("\n✅ Pose training complete.")
print(f"Best weights → {os.path.join(MODEL_SAVE, 'tube_pose', 'weights', 'best.pt')}")
