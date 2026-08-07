"""
app.py — Microcentrifuge Tube Detector
        YOLOv8-OBB + YOLOv8-Pose
        No cv2 dependency — uses PIL + numpy only for all drawing.
"""

import math
import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from inference import load_models, run_inference

st.set_page_config(
    page_title="Tube Detector",
    page_icon="🔬",
    layout="wide",
)

# ─── Drawing helpers (PIL-only, no cv2) ───────────────────────────────────────
COLORS_PIL = [
    (0, 200, 255),
    (255, 100, 30),
    (30, 210, 100),
    (200, 100, 255),
    (255, 210, 0),
    (0, 150, 255),
]

def draw_polygon(draw, pts, color, width=2):
    """Draw a closed polygon from an (N,2) array."""
    pts_list = [(int(p[0]), int(p[1])) for p in pts]
    pts_list.append(pts_list[0])          # close the polygon
    draw.line(pts_list, fill=color, width=width)

def draw_arrow(draw, x0, y0, x1, y1, color, width=2, tip_frac=0.2):
    """Draw an arrowed line from (x0,y0) to (x1,y1)."""
    draw.line([(x0, y0), (x1, y1)], fill=color, width=width)
    # arrowhead
    angle = math.atan2(y1 - y0, x1 - x0)
    length = math.hypot(x1 - x0, y1 - y0) * tip_frac
    for side in [math.pi / 6, -math.pi / 6]:
        ax = x1 - length * math.cos(angle - side)
        ay = y1 - length * math.sin(angle - side)
        draw.line([(x1, y1), (int(ax), int(ay))], fill=color, width=width)

def draw_circle(draw, cx, cy, r, color):
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

def draw_label(draw, cx, cy, text, color):
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((cx + 5, cy - 20), text, font=font)
    draw.rectangle(bbox, fill=(0, 0, 0))
    draw.text((cx + 5, cy - 20), text, fill=color, font=font)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")

obb_default  = "models/obb_best.pt"
pose_default = "models/tube_best.pt"

st.sidebar.subheader("Model Paths")
obb_path  = st.sidebar.text_input("OBB weights",  value=obb_default)
pose_path = st.sidebar.text_input("Pose weights", value=pose_default)

st.sidebar.subheader("Detection")
conf_thresh = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)

st.sidebar.subheader("Overlay")
draw_obb   = st.sidebar.checkbox("OBB box",     value=True)
draw_kpts  = st.sidebar.checkbox("Keypoints",   value=True)
draw_arrow_flag = st.sidebar.checkbox("Angle arrow", value=True)
draw_lbl   = st.sidebar.checkbox("Labels",      value=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("🔬 Microcentrifuge Tube Detector")
st.caption(
    "YOLOv8-OBB + YOLOv8-Pose — upload an image to detect tubes and extract parameters"
)
st.divider()

# ─── Load models ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def get_models(op, pp):
    return load_models(op, pp)

obb_ok  = os.path.exists(obb_path)
pose_ok = os.path.exists(pose_path)

if not obb_ok or not pose_ok:
    missing = []
    if not obb_ok:  missing.append(f"`{obb_path}`")
    if not pose_ok: missing.append(f"`{pose_path}`")
    st.warning(
        f"⚠️ Weight file(s) not found: {', '.join(missing)}. "
        "Update the paths in the sidebar."
    )
    st.stop()

try:
    obb_model, pose_model = get_models(obb_path, pose_path)
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

# ─── Upload ───────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload an image", type=["png", "jpg", "jpeg", "bmp", "tiff"]
)

if uploaded is None:
    st.info("Upload an image above to begin detection.")
    st.stop()

# ─── Inference ────────────────────────────────────────────────────────────────
with tempfile.NamedTemporaryFile(
    suffix=os.path.splitext(uploaded.name)[1], delete=False
) as tmp:
    tmp.write(uploaded.read())
    tmp_path = tmp.name

with st.spinner("Running detection…"):
    detections = run_inference(
        tmp_path, obb_model, pose_model, conf_thresh=conf_thresh
    )

# ─── Draw annotations using PIL only ─────────────────────────────────────────
pil_img = Image.open(tmp_path).convert("RGB")
draw    = ImageDraw.Draw(pil_img)

for idx, det in enumerate(detections):
    color = COLORS_PIL[idx % len(COLORS_PIL)]

    if draw_obb and det.get("corners") is not None:
        draw_polygon(draw, det["corners"], color, width=2)

    cx, cy = int(det["center_x"]), int(det["center_y"])
    draw_circle(draw, cx, cy, 4, color)

    hx = det.get("hinge_x")
    hy = det.get("hinge_y")
    tx = det.get("tip_x")
    ty = det.get("tip_y")
    has_kpts = (
        hx is not None and hy is not None and
        tx is not None and ty is not None and
        hx == hx and tx == tx          # nan check
    )

    if draw_kpts and has_kpts:
        draw_circle(draw, int(hx), int(hy), 7, (50, 220, 120))
        draw_circle(draw, int(tx), int(ty), 7, (255, 160, 40))

    if draw_arrow_flag and has_kpts:
        draw_arrow(draw, int(hx), int(hy), int(tx), int(ty),
                   (255, 255, 255), width=2)

    if draw_lbl:
        draw_label(draw, cx, cy, f"T{idx+1}  {det['conf']:.2f}", color)

img_rgb = np.array(pil_img)
n = len(detections)

# ─── Layout ───────────────────────────────────────────────────────────────────
col_img, col_res = st.columns([1.2, 1], gap="large")

with col_img:
    st.subheader("Annotated Image")
    st.image(img_rgb, use_container_width=True)
    st.caption("🟢 Hinge keypoint   🟠 Tip keypoint   → Angle direction")

with col_res:
    st.subheader(f"Results — {n} tube{'s' if n != 1 else ''} detected")

    with_angle = sum(1 for d in detections if d["angle_deg"] == d["angle_deg"])
    avg_conf   = sum(d["conf"] for d in detections) / n if n else 0

    m1, m2, m3 = st.columns(3)
    m1.metric("Tubes",      n)
    m2.metric("With angle", with_angle)
    m3.metric("Avg conf",   f"{avg_conf:.2f}")

    st.divider()

    if n == 0:
        st.warning("No tubes detected. Try lowering the confidence threshold.")
    else:
        for idx, det in enumerate(detections):
            ang     = det["angle_deg"]
            ang_str = f"{ang:.2f}°" if ang == ang else "N/A"

            with st.expander(
                f"Tube {idx+1}  —  conf {det['conf']:.3f}", expanded=True
            ):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("center_x",      f"{det['center_x']:.1f}")
                c2.metric("center_y",      f"{det['center_y']:.1f}")
                c3.metric("bbox_x",        f"{det['bbox_x']:.1f}")
                c4.metric("bbox_y",        f"{det['bbox_y']:.1f}")

                c5, c6, c7, c8 = st.columns(4)
                c5.metric("bbox_w",        f"{det['bbox_w']:.1f}")
                c6.metric("bbox_h",        f"{det['bbox_h']:.1f}")
                c7.metric("bbox_rotation", f"{det['bbox_rotation']:.2f}°")
                c8.metric("angle_deg",     ang_str)

        export_df = pd.DataFrame([{
            "tube":          f"T{i+1}",
            "center_x":     d["center_x"],
            "center_y":     d["center_y"],
            "bbox_x":       d["bbox_x"],
            "bbox_y":       d["bbox_y"],
            "bbox_w":       d["bbox_w"],
            "bbox_h":       d["bbox_h"],
            "bbox_rotation": d["bbox_rotation"],
            "angle_deg":    d["angle_deg"],
            "conf":         d["conf"],
        } for i, d in enumerate(detections)])

        st.download_button(
            "⬇️ Download results as CSV",
            data=export_df.to_csv(index=False).encode(),
            file_name=f"tubes_{os.path.splitext(uploaded.name)[0]}.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ─── Raw table ────────────────────────────────────────────────────────────────
if detections:
    with st.expander("📋 Full results table"):
        cols = [
            "center_x", "center_y", "bbox_x", "bbox_y",
            "bbox_w", "bbox_h", "bbox_rotation", "angle_deg", "conf",
        ]
        df = pd.DataFrame([{c: d[c] for c in cols} for d in detections])
        df.index = [f"Tube {i+1}" for i in range(n)]
        st.dataframe(df.style.format("{:.2f}"), use_container_width=True)
