"""
inference.py — Shared inference logic for OBB + Pose models.
Used by both step4_inference_metrics.py and app.py.
"""

import math
import numpy as np
import os
os.environ["YOLO_VERBOSE"] = "False"   
from ultralytics import YOLO

CONF_THRESH = 0.25
MAX_DIST    = 60   # pixels: max OBB↔Pose center distance for matching


def load_models(obb_weights: str, pose_weights: str):
    obb_model  = YOLO(obb_weights)
    pose_model = YOLO(pose_weights)
    return obb_model, pose_model


def keypoints_to_angle(hinge_x, hinge_y, tip_x, tip_y) -> float:
    dx = tip_x - hinge_x
    dy = -(tip_y - hinge_y)   # flip Y (image coords go downward)
    return math.degrees(math.atan2(dy, dx)) % 360.0


def _nearest_pose(obb_cx, obb_cy, pose_dets):
    if not pose_dets:
        return None
    dists = [math.hypot(obb_cx - p["cx"], obb_cy - p["cy"]) for p in pose_dets]
    idx = int(np.argmin(dists))
    return pose_dets[idx] if dists[idx] < MAX_DIST else None


def run_inference(image_path: str, obb_model, pose_model,conf_thresh: float = 0.25) -> list[dict]:
    """
    Run OBB + Pose inference on a single image.

    Returns a list of dicts, one per detected tube:
        center_x, center_y, bbox_x, bbox_y, bbox_w, bbox_h,
        bbox_rotation, angle_deg,
        hinge_x, hinge_y, tip_x, tip_y,   ← extras for visualisation
        conf                                ← OBB confidence
    """
    # ── OBB ──────────────────────────────────────────────────────────────────
    obb_res = obb_model.predict(image_path, conf=conf_thresh, verbose=False)[0]
    obb_dets = []

    if obb_res.obb is not None and len(obb_res.obb) > 0:
        for i in range(len(obb_res.obb)):
            xywhr   = obb_res.obb.xywhr[i].cpu().numpy()
            corners = obb_res.obb.xyxyxyxy[i].cpu().numpy().reshape(4, 2)

            cx, cy   = float(xywhr[0]), float(xywhr[1])
            bw, bh   = float(xywhr[2]), float(xywhr[3])
            angle_r  = float(xywhr[4])
            bbox_rot = math.degrees(angle_r) % 360.0

            x_min = float(corners[:, 0].min())
            y_min = float(corners[:, 1].min())

            obb_dets.append({
                "center_x":      round(cx, 2),
                "center_y":      round(cy, 2),
                "bbox_x":        round(x_min, 2),
                "bbox_y":        round(y_min, 2),
                "bbox_w":        round(bw, 2),
                "bbox_h":        round(bh, 2),
                "bbox_rotation": round(bbox_rot, 2),
                "corners":       corners,
                "conf":          round(float(obb_res.obb.conf[i].cpu()), 3),
            })

    # ── Pose ─────────────────────────────────────────────────────────────────
    pose_res = pose_model.predict(image_path, conf=conf_thresh, verbose=False)[0]
    pose_dets = []

    if pose_res.keypoints is not None and len(pose_res.boxes) > 0:
        for i in range(len(pose_res.boxes)):
            box  = pose_res.boxes.xywh[i].cpu().numpy()
            kpts = pose_res.keypoints.xy[i].cpu().numpy()

            hx, hy = float(kpts[0, 0]), float(kpts[0, 1])
            tx, ty = float(kpts[1, 0]), float(kpts[1, 1])

            pose_dets.append({
                "cx":       float(box[0]),
                "cy":       float(box[1]),
                "hinge_x":  round(hx, 2),
                "hinge_y":  round(hy, 2),
                "tip_x":    round(tx, 2),
                "tip_y":    round(ty, 2),
                "angle_deg": round(keypoints_to_angle(hx, hy, tx, ty), 2),
            })

    # ── Merge ─────────────────────────────────────────────────────────────────
    results = []
    for det in obb_dets:
        pose = _nearest_pose(det["center_x"], det["center_y"], pose_dets)
        results.append({
            "center_x":      det["center_x"],
            "center_y":      det["center_y"],
            "bbox_x":        det["bbox_x"],
            "bbox_y":        det["bbox_y"],
            "bbox_w":        det["bbox_w"],
            "bbox_h":        det["bbox_h"],
            "bbox_rotation": det["bbox_rotation"],
            "angle_deg":     pose["angle_deg"] if pose else float("nan"),
            "hinge_x":       pose["hinge_x"]  if pose else float("nan"),
            "hinge_y":       pose["hinge_y"]  if pose else float("nan"),
            "tip_x":         pose["tip_x"]    if pose else float("nan"),
            "tip_y":         pose["tip_y"]    if pose else float("nan"),
            "conf":          det["conf"],
            "corners":       det["corners"],
        })

    return results
