"""
Step 2: Convert annotations → YOLOv8-OBB format (xyxyxyxy normalised)
        and train the YOLOv8-OBB model for 35 epochs on CPU.

OBB label format (one line per object):
    class_id  x1 y1  x2 y2  x3 y3  x4 y4     (all normalised 0-1)

We derive the 4 corners from (center_x, center_y, bbox_w, bbox_h, bbox_rotation).
bbox_rotation is clockwise degrees.
"""

import os
import math
import numpy as np
import pandas as pd
import yaml
import shutil

# ── Config ─────────────────────────────────────────────────────────────────────
IMG_W, IMG_H = 640, 480      # image size
DATA_ROOT    = "data"
TRAIN_ANN    = os.path.join(DATA_ROOT, "train_annotations.csv")
TEST_ANN     = os.path.join(DATA_ROOT, "test_annotations.csv")
TRAIN_IMGS   = os.path.join(DATA_ROOT, "train", "images")
TEST_IMGS    = os.path.join(DATA_ROOT, "test",  "images")
TRAIN_LBLS   = os.path.join(DATA_ROOT, "train", "labels_obb")
TEST_LBLS    = os.path.join(DATA_ROOT, "test",  "labels_obb")
YAML_PATH    = os.path.join(DATA_ROOT, "obb_dataset.yaml")
MODEL_SAVE   = "runs/obb"

for d in [TRAIN_LBLS, TEST_LBLS]:
    os.makedirs(d, exist_ok=True)

CLASS_ID = 0   # single class: microcentrifuge tube

# ── Geometry helpers ───────────────────────────────────────────────────────────
def rotated_box_corners(cx, cy, w, h, angle_deg_cw):
    """
    Return the 4 corners of a rotated bounding box.
    angle_deg_cw: clockwise rotation in degrees.
    Returns array of shape (4, 2) in pixel coords.
    """
    angle_rad = math.radians(-angle_deg_cw)   # convert CW → CCW for math
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    # Half extents
    hw, hh = w / 2.0, h / 2.0
    # Corners relative to center (top-left, top-right, bottom-right, bottom-left)
    local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=float)
    # Rotation matrix
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated = (R @ local.T).T
    return rotated + np.array([cx, cy])

def corners_to_yolo_obb(corners, img_w=IMG_W, img_h=IMG_H):
    """Normalise corners to 0-1 range in YOLO OBB order."""
    pts = corners.copy()
    pts[:, 0] /= img_w
    pts[:, 1] /= img_h
    return pts.flatten()   # x1 y1 x2 y2 x3 y3 x4 y4

# ── Convert one CSV split → per-image .txt label files ────────────────────────
def convert_split(csv_path, label_dir):
    df = pd.read_csv(csv_path)
    for img_name, grp in df.groupby("image"):
        stem = os.path.splitext(img_name)[0]
        lines = []
        for _, row in grp.iterrows():
            cx   = row["center_x"]
            cy   = row["center_y"]
            bx   = row["bbox_x"]
            by   = row["bbox_y"]
            bw   = row["bbox_w"]
            bh   = row["bbox_h"]
            rot  = row["bbox_rotation"]   # CW degrees

            # Recompute center from bbox_x/y (top-left) + w/h for consistency
            # bbox_x/y is top-left of the *axis-aligned* bounding rect,
            # so the true rotated box center is center_x, center_y
            corners = rotated_box_corners(cx, cy, bw, bh, rot)
            obb = corners_to_yolo_obb(corners)
            # clamp to [0, 1]
            obb = np.clip(obb, 0.0, 1.0)
            coords_str = " ".join(f"{v:.6f}" for v in obb)
            lines.append(f"{CLASS_ID} {coords_str}")

        out_path = os.path.join(label_dir, stem + ".txt")
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")

print("Converting train annotations to OBB labels …")
convert_split(TRAIN_ANN, TRAIN_LBLS)
print("Converting test annotations to OBB labels …")
convert_split(TEST_ANN, TEST_LBLS)
print("Label files written.")

# ── Write dataset YAML ─────────────────────────────────────────────────────────
yaml_content = {
    "path":  os.path.abspath(DATA_ROOT),
    "train": "train/images",
    "val":   "test/images",
    "nc":    1,
    "names": {0: "tube"},
}
with open(YAML_PATH, "w") as f:
    yaml.dump(yaml_content, f, default_flow_style=False)
print(f"Dataset YAML written → {YAML_PATH}")

# ── Symlink label dirs so YOLO finds them ──────────────────────────────────────
# YOLO OBB expects labels alongside images: train/labels, val/labels
for split, lbl_dir in [("train", TRAIN_LBLS), ("test", TEST_LBLS)]:
    yolo_lbl = os.path.join(DATA_ROOT, split, "labels")
    if os.path.islink(yolo_lbl) or os.path.exists(yolo_lbl):
        if os.path.islink(yolo_lbl):
            os.remove(yolo_lbl)
        elif os.path.isdir(yolo_lbl):
            shutil.rmtree(yolo_lbl)
    os.symlink(os.path.abspath(lbl_dir), yolo_lbl)
print("Label symlinks created.")

# ── Train YOLOv8-OBB ──────────────────────────────────────────────────────────
print("\n🚀 Starting YOLOv8-OBB training (35 epochs, CPU) …")
from ultralytics import YOLO

model = YOLO("yolov8n-obb.pt")   # nano OBB – fastest on CPU

results = model.train(
    data      = YAML_PATH,
    epochs    = 35,
    imgsz     = 640,
    batch     = 4,          # small batch for CPU
    device    = "cpu",
    project   = MODEL_SAVE,
    name      = "tube_obb",
    patience  = 10,
    workers   = 0,          # Windows: set workers=0
    exist_ok  = True,
    cache     = False,
    verbose   = True,
)

print("\n✅ OBB training complete.")
print(f"Best weights → {os.path.join(MODEL_SAVE, 'tube_obb', 'weights', 'best.pt')}")
