# Microcentrifuge Tube Detection Pipeline
## YOLOv8-OBB + YOLOv8-Pose — CPU / Windows / VS Code

---

## Folder layout expected BEFORE running

```
project_root/
├── data/
│   ├── annotations.csv        ← 371 rows (ground truth)
│   ├── labels.csv             ← 368 rows (keypoint labels)
│   └── uploads/               ← all 70 .png images go here
│
├── step1_split_data.py
├── step2_train_obb.py
├── step3_train_pose.py
├── step4_inference_metrics.py
├── run_pipeline.py
└── requirements.txt
```

After running the pipeline the structure grows to:

```
project_root/
├── data/
│   ├── train/
│   │   ├── images/            ← 56 training images
│   │   ├── labels_obb/        ← OBB .txt files
│   │   ├── labels_pose/       ← Pose .txt files
│   │   └── labels -> labels_obb  (symlink, swapped in step3)
│   ├── test/
│   │   ├── images/            ← 14 test images
│   │   ├── labels_obb/
│   │   ├── labels_pose/
│   │   └── labels -> …
│   ├── obb_dataset.yaml
│   ├── pose_dataset.yaml
│   ├── train_annotations.csv
│   ├── test_annotations.csv
│   ├── train_labels.csv
│   └── test_labels.csv
│
├── runs/
│   ├── obb/tube_obb/weights/best.pt
│   └── pose/tube_pose/weights/best.pt
│
└── predictions.csv            ← final output
```

---

## Setup (Windows, VS Code terminal)

```powershell
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies (CPU-only PyTorch is the default on Windows)
pip install -r requirements.txt
```

---

## Running the pipeline

```powershell
# Run all 4 steps in sequence
python run_pipeline.py
```

Or run each step individually:

```powershell
python step1_split_data.py       # split 70 → 56/14, partition CSVs
python step2_train_obb.py        # train YOLOv8n-OBB  (35 epochs, CPU)
python step3_train_pose.py       # train YOLOv8n-Pose (35 epochs, CPU)
python step4_inference_metrics.py  # infer + compute metrics → predictions.csv
```

---

## What each step does

### Step 1 — Data split
- Randomly splits 70 images → 56 train / 14 test (seed=42, reproducible)
- Partitions `annotations.csv` and `labels.csv` accordingly
- Copies images into `data/train/images/` and `data/test/images/`

### Step 2 — YOLOv8-OBB training
**Input:** `annotations.csv` columns:
`center_x, center_y, bbox_w, bbox_h, bbox_rotation`

**Conversion formula** — each rotated box → 4 corners:
```
angle_rad = -bbox_rotation_deg * π/180    # CW → CCW
local_corners = [(-w/2,-h/2), (w/2,-h/2), (w/2,h/2), (-w/2,h/2)]
rotated = R(angle_rad) @ local_corners
world   = rotated + [center_x, center_y]
```
Corners are normalised to [0,1] and written as:
`class_id  x1 y1  x2 y2  x3 y3  x4 y4`

**Model:** `yolov8n-obb.pt`, 35 epochs, batch=4, CPU

### Step 3 — YOLOv8-Pose training
**Input:** `labels.csv` columns:
`cx_px, cy_px, w_px, h_px, hinge_x_px, hinge_y_px, tip_x_px, tip_y_px`

Label format:
`class cx cy w h  hx hy hv  tx ty tv`   (all spatial values / image dims)

Keypoints:
- **kp1** = hinge (joint)
- **kp2** = tip (tab)

**Model:** `yolov8n-pose.pt`, 35 epochs, batch=4, CPU

### Step 4 — Inference & metrics

**OBB model output:**
- `center_x, center_y` from `xywhr[0:2]`
- `bbox_w, bbox_h` from `xywhr[2:4]`
- `bbox_rotation` from `xywhr[4]` → degrees
- `bbox_x, bbox_y` = min corner of axis-aligned enclosing rect of the 4 OBB corners

**Pose model output:**
```
angle_deg = atan2(-(tip_y - hinge_y), tip_x - hinge_x) * 180/π  mod 360
```
(Y-axis is flipped because image coordinates increase downward)

**Matching:** each OBB detection is matched to the nearest pose detection
(Euclidean center distance < 60 px).

---

## Output — `predictions.csv`

| Column | Source |
|--------|--------|
| image | image filename |
| center_x | OBB model |
| center_y | OBB model |
| bbox_x | OBB model (top-left of enclosing rect) |
| bbox_y | OBB model (top-left of enclosing rect) |
| bbox_w | OBB model (rotated box width) |
| bbox_h | OBB model (rotated box height) |
| bbox_rotation | OBB model (degrees, 0-360) |
| angle_deg | Pose model via atan2 |

---

## Metrics

| Metric | Description |
|--------|-------------|
| Precision | TP / (TP + FP) — boxes matched at IoU ≥ 0.4 |
| Recall | TP / (TP + FN) |
| F1 | Harmonic mean of Precision & Recall |
| MAE per feature | Mean absolute error for each numeric column |
| Angle MAE | Circular mean absolute error in degrees |

---

## Notes for CPU training
- `workers=0` is required on Windows to avoid DataLoader multiprocessing errors.
- Training 35 epochs with `yolov8n` (nano) takes roughly **2–4 hours per model** on a modern laptop CPU.
- Increase `batch` if you have >16 GB RAM.
- For faster results, use `imgsz=320` instead of 640.
