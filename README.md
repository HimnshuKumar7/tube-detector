# 🔬 Microcentrifuge Tube Detector

**Automated detection of microcentrifuge tube positions and lid orientations**  
**using a dual YOLOv8 pipeline — OBB for localization, Pose for angle estimation**

[🚀 Live Demo](https://tube-detector-byp8dhgnzews7rqquciwhz.streamlit.app/) ·



## 📌 Overview

This project solves the problem of automatically detecting microcentrifuge tubes in overhead RGB images and extracting key parameters — position, bounding box dimensions, rotation, and lid angle — without manual measurement.

Given a 640×480 overhead image containing 3–6 tubes on varied backgrounds (desk, white surface, black surface), the system:

1. Detects each tube as an oriented bounding box (OBB)
2. Localizes two keypoints per tube: the hinge joint and the tip tab
3. Computes the lid rotation angle from the hinge→tip direction vector

---

## 🧠 Technical Approach

### Architecture: Two-Model Pipeline

```
Input Image (640×480 RGB)
        │
        ├──▶  YOLOv8n-OBB  ──▶  center_x, center_y, bbox_w, bbox_h, bbox_rotation
        │
        └──▶  YOLOv8n-Pose ──▶  hinge_xy, tip_xy  ──▶  angle_deg
                                                              │
                              Nearest-center matching (< 60px threshold)
                                                              │
                                                    Final merged output
```

### Model 1 — YOLOv8n-OBB (Oriented Bounding Box)
Detects each tube as a rotated rectangle, giving precise position and orientation even when tubes overlap or are tightly packed.

- **Input:** Raw image
- **Output:** `center_x`, `center_y`, `bbox_w`, `bbox_h`, `bbox_rotation` (degrees)
- **Label format:** `class x1 y1 x2 y2 x3 y3 x4 y4` (4 corners, normalised 0–1)
- **Corner derivation from CSV annotations:**
  ```
  angle_rad = -bbox_rotation_deg × π/180   (CW → CCW)
  corners   = R(angle_rad) @ local_corners + [center_x, center_y]
  ```

### Model 2 — YOLOv8n-Pose (Keypoint Detection)
Detects two keypoints per tube to determine the physical lid orientation.

- **Keypoint 1 ( Hinge):** The joint connecting lid to body
- **Keypoint 2 ( Tip):** The tab on the lid
- **Angle computation:**
  ```
  angle_deg = atan2(-(tip_y − hinge_y), tip_x − hinge_x) × 180/π  mod 360
  ```
  *Y-axis is negated because image coordinates increase downward*

### Matching Strategy
Each OBB detection is paired with the nearest Pose detection by Euclidean center distance, with a 60px threshold to avoid false matches.

---

## 📊 Results

> Evaluated on 14 held-out test images (80/20 split, seed=42) — 74 predicted detections

| Metric | Value |
|--------|-------|
| **Precision** | *(run step4 to fill)* |
| **Recall** | *(run step4 to fill)* |
| **F1 Score** | *(run step4 to fill)* |
| **Angle MAE** | *(circular error in degrees)* |
| **IoU Threshold** | 0.4 |
| **Test images** | 14 |
| **Predicted detections** | 74 |

### Per-Feature MAE (on matched pairs)

| Feature | MAE |
|---------|-----|
| center_x | — |
| center_y | — |
| bbox_x | — |
| bbox_y | — |
| bbox_w | — |
| bbox_h | — |
| bbox_rotation | — |
| angle_deg | — *(circular)* |

---

## 🗂️ Repository Structure

```
tube-detector/
│
├── 📄 app.py                   ← Streamlit web application
├── 📄 inference.py             ← Shared OBB + Pose inference logic
├── 📄 requirements.txt         ← Python dependencies
├── 📄 predictions.csv          ← Model predictions on the test set
│
├── 📁 models/
│   ├── obb_best.pt             ← Trained OBB model weights (~6MB)
│   └── tube_best.pt            ← Trained Pose model weights (~6MB)
│
└── 📁 pipeline/
    ├── run_pipeline.py         ← Run all 4 steps in sequence
    ├── step1_split_data.py     ← 80/20 train/test split
    ├── step2_train_obb.py      ← OBB model training
    ├── step3_train_pose.py     ← Pose model training
    └── step4_inference_metrics.py  ← Inference + evaluation
```

---

## 🚀 Quick Start — Run the App Locally

### Prerequisites
- Python 3.10+
- Windows / Linux / macOS
- 4GB+ RAM

```powershell
# 1. Clone the repository
git clone https://github.com/HimnshuKumar7/tube-detector.git
cd tube-detector

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. Upload any overhead image of microcentrifuge tubes to see live detection results.

---

## 🔧 Retraining With New Data

The pipeline is designed to be easily retrained with additional images. More data directly improves precision and angle accuracy.

### Step 1 — Prepare Your Data

```
data/
├── images/           ← All .png images (any quantity)
├── annotations.csv   ← Bounding box ground truth
└── labels.csv        ← Keypoint ground truth
```

**`annotations.csv` required columns:**

| Column | Type | Description |
|--------|------|-------------|
| `image` | string | Filename e.g. `img001.png` |
| `center_x` | float | Tube lid center X (pixels) |
| `center_y` | float | Tube lid center Y (pixels) |
| `bbox_x` | float | Bounding box top-left X |
| `bbox_y` | float | Bounding box top-left Y |
| `bbox_w` | float | Bounding box width |
| `bbox_h` | float | Bounding box height |
| `bbox_rotation` | float | Rotation in degrees (clockwise) |
| `angle_deg` | float | Lid angle [0, 360), counter-clockwise from +X |

**`labels.csv` required columns:**

| Column | Type | Description |
|--------|------|-------------|
| `image` | string | Filename |
| `cx_px`, `cy_px` | float | Bounding box center (pixels) |
| `w_px`, `h_px` | float | Bounding box dimensions |
| `hinge_x_px`, `hinge_y_px` | float | Hinge keypoint (pixels) |
| `hinge_vis` | int | Visibility: 0=absent, 1=occluded, 2=visible |
| `tip_x_px`, `tip_y_px` | float | Tip keypoint (pixels) |
| `tip_vis` | int | Visibility flag |

### Step 2 — Run the Pipeline

```powershell
# Run all 4 steps automatically
python pipeline/run_pipeline.py
```

Or run steps individually for more control:

```powershell
python pipeline/step1_split_data.py          # Split images 80/20
python pipeline/step2_train_obb.py           # Train OBB model
python pipeline/step3_train_pose.py          # Train Pose model
python pipeline/step4_inference_metrics.py   # Evaluate → predictions.csv
```

### Step 3 — Copy New Weights to `models/`

```powershell
# Windows
copy runs\obb\tube_obb\weights\best.pt   models\obb_best.pt
copy runs\pose\tube_pose\weights\best.pt models\tube_best.pt

# Linux / macOS
cp runs/obb/tube_obb/weights/best.pt   models/obb_best.pt
cp runs/pose/tube_pose/weights/best.pt models/tube_best.pt
```

---

## ⚡ Tips for Better Results

| What to tune | Parameter | Recommendation |
|--------------|-----------|----------------|
| **More data** | — | 200+ images dramatically improve accuracy |
| **Epochs** | `epochs=35` | Increase to 100–150 with larger datasets |
| **Batch size** | `batch=4` | Use 8–16 if you have 16GB+ RAM |
| **Image size** | `imgsz=640` | Try `1280` for higher-resolution images |
| **GPU training** | `device="cpu"` | Change to `"0"` for CUDA GPU |
| **Larger model** | `yolov8n` | Try `yolov8s` (small) for better accuracy |
| **Data augmentation** | automatic | Ultralytics handles mosaic, flip, HSV |
| **Angle accuracy** | — | Ensure tip/hinge labels are precise to ±2px |

> 💡 **Rule of thumb:** The original model used 56 training images and nano weights.  
> With 200+ images and `yolov8s`, you should see F1 improve by ~10–15%.

---

## 🌐 Deployment

The app is deployed on **Streamlit Cloud** and accessible at:

👉 **[tube-detector-byp8dhgnzews7rqquciwhz.streamlit.app](https://tube-detector-byp8dhgnzews7rqquciwhz.streamlit.app)**

To redeploy after retraining:
1. Push updated `models/obb_best.pt` and `models/tube_best.pt` to GitHub
2. Streamlit Cloud auto-redeploys on every push to `main`

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `ultralytics` | YOLOv8 OBB + Pose models |
| `streamlit` | Web application framework |
| `opencv-python-headless` | Image I/O and annotation drawing |
| `torch` | Deep learning backend (CPU build) |
| `pandas` | CSV handling and results export |
| `Pillow` | Image format support |

---

## 📝 Training Details

| Setting | OBB Model | Pose Model |
|---------|-----------|------------|
| Base weights | `yolov8n-obb.pt` | `yolov8n-pose.pt` |
| Epochs | 35 | 35 |
| Batch size | 4 | 4 |
| Image size | 640 | 640 |
| Device | CPU | CPU |
| Train images | 56 | 56 |
| Val images | 14 | 14 |
| Early stopping | patience=10 | patience=10 |

---

## 📐 Coordinate System

- Origin: **top-left** corner of the image
- X increases **rightward**, Y increases **downward**
- Angle **0°** points along the positive X-axis (rightward)
- Angles increase **counter-clockwise**
- `angle_deg` is defined by the **hinge → tip** direction

---


Made with 🔬 for automated lab vision · YOLOv8 · Streamlit
