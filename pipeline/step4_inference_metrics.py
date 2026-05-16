"""
Step 4: Inference on test set using both trained models.
        - YOLOv8-OBB  → center_x, center_y, bbox_x, bbox_y, bbox_w, bbox_h, bbox_rotation
        - YOLOv8-Pose → angle_deg  via  atan2(tip_y - hinge_y, tip_x - hinge_x)

Output CSV columns (matching ground truth):
    image, center_x, center_y, bbox_x, bbox_y, bbox_w, bbox_h, bbox_rotation, angle_deg

Metrics reported:
    Detection  : Precision, Recall, F1
    Per-feature: MAE for each of the 8 numeric columns
    Angle error: circular MAE (handles 0/360 wrap-around)
"""

import os
import math
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_ROOT    = "data"
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test", "images")
TEST_ANN_CSV = os.path.join(DATA_ROOT, "test_annotations.csv")
OBB_WEIGHTS  = r"C:\Zeon\ps1\final\runs\obb\runs\obb\tube_obb\weights\best.pt"
POSE_WEIGHTS = r"C:\Zeon\ps1\final\runs\pose\runs\pose\tube_pose\weights\best.pt"
OUT_CSV      = "predictions.csv"

IMG_W, IMG_H = 640, 480
IOU_THRESH   = 0.4    # IoU threshold for matching predicted→GT box
CONF_THRESH  = 0.25

# ── Load models ────────────────────────────────────────────────────────────────
print("Loading OBB model …")
obb_model  = YOLO(OBB_WEIGHTS)
print("Loading Pose model …")
pose_model = YOLO(POSE_WEIGHTS)

# ── Helper: angle from keypoints ───────────────────────────────────────────────
def keypoints_to_angle(hinge_x, hinge_y, tip_x, tip_y):
    """
    Compute angle in [0, 360) degrees from hinge → tip direction.
    Angle 0 = positive X axis (rightward), increases counter-clockwise.
    """
    dx = tip_x - hinge_x
    dy = -(tip_y - hinge_y)   # flip Y because image Y goes downward
    angle = math.degrees(math.atan2(dy, dx))
    return angle % 360.0

# ── Helper: OBB → bbox_x, bbox_y (top-left of axis-aligned enclosing rect) ────
def obb_corners_to_bbox(corners_px):
    """
    corners_px: (4,2) array of pixel corners.
    Returns bbox_x, bbox_y (top-left), bbox_w, bbox_h of the axis-aligned rect.
    Also returns center_x, center_y.
    """
    cx = corners_px[:, 0].mean()
    cy = corners_px[:, 1].mean()
    x_min, y_min = corners_px[:, 0].min(), corners_px[:, 1].min()
    x_max, y_max = corners_px[:, 0].max(), corners_px[:, 1].max()
    return cx, cy, x_min, y_min, x_max - x_min, y_max - y_min

# ── Helper: IoU for axis-aligned boxes ────────────────────────────────────────
def iou_aabb(a, b):
    """a, b: (x, y, w, h) in pixels."""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0]+a[2], a[1]+a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0]+b[2], b[1]+b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter / union if union > 0 else 0.0

# ── Run OBB inference ──────────────────────────────────────────────────────────
print("\nRunning OBB inference on test images …")
obb_results  = {}   # img_stem → list of dicts

test_images = sorted(os.listdir(TEST_IMG_DIR))
for img_file in test_images:
    img_path = os.path.join(TEST_IMG_DIR, img_file)
    stem     = os.path.splitext(img_file)[0]
    res      = obb_model.predict(img_path, conf=CONF_THRESH, verbose=False)[0]

    detections = []
    if res.obb is not None and len(res.obb) > 0:
        for i in range(len(res.obb)):
            # xywhr: center_x, center_y, width, height, angle (radians, CCW)
            xywhr  = res.obb.xywhr[i].cpu().numpy()   # shape (5,)
            corners = res.obb.xyxyxyxy[i].cpu().numpy().reshape(4, 2)  # (4,2) px

            cx, cy      = float(xywhr[0]), float(xywhr[1])
            bw, bh      = float(xywhr[2]), float(xywhr[3])
            angle_rad   = float(xywhr[4])               # radians, CCW in OpenCV
            bbox_rot_deg = math.degrees(angle_rad) % 360.0

            _, _, bx, by, _, _ = obb_corners_to_bbox(corners)
            # Use OBB width/height (rotated box dims)
            detections.append({
                "center_x":     cx,
                "center_y":     cy,
                "bbox_x":       float(bx),
                "bbox_y":       float(by),
                "bbox_w":       bw,
                "bbox_h":       bh,
                "bbox_rotation": bbox_rot_deg,
                "corners":      corners,
                "conf":         float(res.obb.conf[i].cpu()),
            })
    obb_results[stem] = detections

# ── Run Pose inference ─────────────────────────────────────────────────────────
print("Running Pose inference on test images …")
pose_results = {}   # img_stem → list of (cx,cy,bw,bh, hinge_x, hinge_y, tip_x, tip_y, angle_deg)

for img_file in test_images:
    img_path = os.path.join(TEST_IMG_DIR, img_file)
    stem     = os.path.splitext(img_file)[0]
    res      = pose_model.predict(img_path, conf=CONF_THRESH, verbose=False)[0]

    detections = []
    if res.keypoints is not None and len(res.boxes) > 0:
        for i in range(len(res.boxes)):
            box  = res.boxes.xywh[i].cpu().numpy()  # cx cy w h (pixels)
            kpts = res.keypoints.xy[i].cpu().numpy() # (2, 2)  → [[hx,hy],[tx,ty]]

            hx, hy = float(kpts[0, 0]), float(kpts[0, 1])
            tx, ty = float(kpts[1, 0]), float(kpts[1, 1])
            angle  = keypoints_to_angle(hx, hy, tx, ty)

            detections.append({
                "cx":       float(box[0]),
                "cy":       float(box[1]),
                "bw":       float(box[2]),
                "bh":       float(box[3]),
                "hinge_x":  hx,
                "hinge_y":  hy,
                "tip_x":    tx,
                "tip_y":    ty,
                "angle_deg": angle,
            })
    pose_results[stem] = detections

# ── Match OBB detections with Pose detections (nearest-center) ────────────────
print("Merging OBB + Pose detections …")

def nearest_pose(obb_det, pose_dets, max_dist=60):
    """Find closest pose detection to OBB detection by center distance."""
    if not pose_dets:
        return None
    dists = [math.hypot(obb_det["center_x"] - p["cx"],
                         obb_det["center_y"] - p["cy"])
             for p in pose_dets]
    idx   = int(np.argmin(dists))
    return pose_dets[idx] if dists[idx] < max_dist else None

merged_rows = []
for img_file in test_images:
    stem = os.path.splitext(img_file)[0]
    # Use PNG name for matching GT (annotations use .png)
    img_name_png = stem + ".png"

    for obb_det in obb_results.get(stem, []):
        pose_det = nearest_pose(obb_det, pose_results.get(stem, []))
        angle_deg = pose_det["angle_deg"] if pose_det is not None else float("nan")
        merged_rows.append({
            "image":         img_name_png,
            "center_x":      round(obb_det["center_x"], 2),
            "center_y":      round(obb_det["center_y"], 2),
            "bbox_x":        round(obb_det["bbox_x"],   2),
            "bbox_y":        round(obb_det["bbox_y"],   2),
            "bbox_w":        round(obb_det["bbox_w"],   2),
            "bbox_h":        round(obb_det["bbox_h"],   2),
            "bbox_rotation": round(obb_det["bbox_rotation"], 2),
            "angle_deg":     round(angle_deg, 2) if not math.isnan(angle_deg) else float("nan"),
        })

pred_df = pd.DataFrame(merged_rows, columns=[
    "image","center_x","center_y","bbox_x","bbox_y","bbox_w","bbox_h","bbox_rotation","angle_deg"
])
pred_df.to_csv(OUT_CSV, index=False)
print(f"Predictions saved → {OUT_CSV}  ({len(pred_df)} rows)")

# ══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("              EVALUATION METRICS")
print("="*60)

gt_df = pd.read_csv(TEST_ANN_CSV)

# ── Detection metrics: match predicted → GT per image by IoU ─────────────────
TP = FP = FN = 0

all_matched_pairs = []   # (gt_row, pred_row) for error computation

for img_name, gt_grp in gt_df.groupby("image"):
    stem = os.path.splitext(img_name)[0]
    pred_grp = pred_df[pred_df["image"] == img_name].copy()

    gt_boxes   = gt_grp[["bbox_x","bbox_y","bbox_w","bbox_h"]].values.tolist()
    pred_boxes = pred_grp[["bbox_x","bbox_y","bbox_w","bbox_h"]].values.tolist() if len(pred_grp) else []

    matched_gt   = set()
    matched_pred = set()

    for gi, gb in enumerate(gt_boxes):
        best_iou, best_pi = 0.0, -1
        for pi, pb in enumerate(pred_boxes):
            iou = iou_aabb(gb, pb)
            if iou > best_iou:
                best_iou, best_pi = iou, pi
        if best_iou >= IOU_THRESH and best_pi not in matched_pred:
            matched_gt.add(gi)
            matched_pred.add(best_pi)
            # Store matched pair for feature error
            gt_row   = gt_grp.iloc[gi]
            pred_row = pred_grp.iloc[best_pi]
            all_matched_pairs.append((gt_row, pred_row))

    tp = len(matched_gt)
    fp = len(pred_boxes) - len(matched_pred)
    fn = len(gt_boxes)   - len(matched_gt)
    TP += tp; FP += fp; FN += fn

precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
f1        = (2 * precision * recall / (precision + recall)
             if (precision + recall) > 0 else 0.0)

print(f"\n{'Detection Metrics':}")
print(f"  TP={TP}  FP={FP}  FN={FN}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1        : {f1:.4f}")

# ── Per-feature MAE ──────────────────────────────────────────────────────────
def circular_error(pred_angle, gt_angle):
    """Smallest angular difference, result in [0, 180]."""
    diff = abs(pred_angle - gt_angle) % 360.0
    return min(diff, 360.0 - diff)

features = ["center_x","center_y","bbox_x","bbox_y","bbox_w","bbox_h","bbox_rotation"]
errors   = {f: [] for f in features}
angle_errors = []

for gt_row, pred_row in all_matched_pairs:
    for f in features:
        errors[f].append(abs(float(gt_row[f]) - float(pred_row[f])))
    if not math.isnan(float(pred_row["angle_deg"])):
        angle_errors.append(circular_error(float(pred_row["angle_deg"]),
                                           float(gt_row["angle_deg"])))

print(f"\n{'Per-Feature MAE (on matched pairs)':}")
print(f"  {'Feature':<16}  {'MAE':>10}  {'Samples':>8}")
print(f"  {'-'*40}")
for f in features:
    errs = errors[f]
    mae  = np.mean(errs) if errs else float("nan")
    print(f"  {f:<16}  {mae:>10.3f}  {len(errs):>8}")

angle_mae = np.mean(angle_errors) if angle_errors else float("nan")
print(f"  {'angle_deg':<16}  {angle_mae:>10.3f}  {len(angle_errors):>8}  (circular MAE)")

print(f"\n{'Summary':}")
print(f"  Total GT tubes       : {TP+FN}")
print(f"  Total predicted      : {TP+FP}")
print(f"  Matched (TP)         : {TP}")
print(f"  Unmatched GT (FN)    : {FN}")
print(f"  False positives (FP) : {FP}")
print(f"  Angle MAE            : {angle_mae:.2f}°")
print(f"\nOutput file: {os.path.abspath(OUT_CSV)}")
print("✅ Step 4 complete.")
