"""
app.py — Microcentrifuge Tube Detector
        YOLOv8-OBB + YOLOv8-Pose

Run:
    streamlit run app.py
"""

import math, os, tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from inference import load_models, run_inference

st.set_page_config(
    page_title="Tube Detector",
    page_icon="🔬",
    layout="wide",
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")

obb_default  = "models/obb_best.pt"
pose_default = "models/pose_best.pt"

st.sidebar.subheader("Model Paths")
obb_path  = st.sidebar.text_input("OBB weights",  value=obb_default)
pose_path = st.sidebar.text_input("Pose weights", value=pose_default)

st.sidebar.subheader("Detection")
conf_thresh = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)

st.sidebar.subheader("Overlay")
draw_obb   = st.sidebar.checkbox("OBB box",      value=True)
draw_kpts  = st.sidebar.checkbox("Keypoints",    value=True)
draw_arrow = st.sidebar.checkbox("Angle arrow",  value=True)
draw_label = st.sidebar.checkbox("Labels",       value=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("🔬 Microcentrifuge Tube Detector")
st.caption("YOLOv8-OBB + YOLOv8-Pose — upload an image to detect tubes and extract parameters")
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
    st.warning(f"⚠️ Weight file(s) not found: {', '.join(missing)}. Update the paths in the sidebar.")
    st.stop()

try:
    obb_model, pose_model = get_models(obb_path, pose_path)
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

# ─── Upload ───────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "tiff"])

if uploaded is None:
    st.info("Upload an image above to begin detection.")
    st.stop()

# ─── Inference ────────────────────────────────────────────────────────────────
with tempfile.NamedTemporaryFile(suffix=os.path.splitext(uploaded.name)[1], delete=False) as tmp:
    tmp.write(uploaded.read())
    tmp_path = tmp.name

with st.spinner("Running detection…"):
    detections = run_inference(tmp_path, obb_model, pose_model)

# ─── Draw annotations ─────────────────────────────────────────────────────────
COLORS = [
    (0, 200, 255), (255, 100, 30), (30, 210, 100),
    (200, 100, 255), (255, 210, 0), (0, 150, 255),
]

img_bgr = cv2.imread(tmp_path)
if img_bgr is None:
    pil = Image.open(tmp_path).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

for idx, det in enumerate(detections):
    color = COLORS[idx % len(COLORS)]

    if draw_obb and det.get("corners") is not None:
        pts = det["corners"].astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_bgr, [pts], True, color, 2, cv2.LINE_AA)

    cx, cy = int(det["center_x"]), int(det["center_y"])
    cv2.circle(img_bgr, (cx, cy), 4, color, -1, cv2.LINE_AA)

    hx, hy = det.get("hinge_x"), det.get("hinge_y")
    tx, ty = det.get("tip_x"),   det.get("tip_y")
    has_kpts = hx == hx and tx == tx  # nan check

    if draw_kpts and has_kpts:
        cv2.circle(img_bgr, (int(hx), int(hy)), 7, (50, 220, 120), -1, cv2.LINE_AA)
        cv2.circle(img_bgr, (int(tx), int(ty)), 7, (255, 160, 40), -1, cv2.LINE_AA)

    if draw_arrow and has_kpts:
        cv2.arrowedLine(img_bgr, (int(hx), int(hy)), (int(tx), int(ty)),
                        (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.25)

    if draw_label:
        label = f"T{idx+1}  {det['conf']:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_bgr, (cx+5, cy-lh-8), (cx+lw+10, cy-2), (0, 0, 0), -1)
        cv2.putText(img_bgr, label, (cx+7, cy-5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
n = len(detections)

# ─── Layout: image left, results right ────────────────────────────────────────
col_img, col_res = st.columns([1.2, 1], gap="large")

with col_img:
    st.subheader("Annotated Image")
    st.image(img_rgb, use_container_width=True)
    st.caption("🟢 Hinge keypoint   🔵 Tip keypoint   → Angle direction")

with col_res:
    st.subheader(f"Results — {n} tube{'s' if n != 1 else ''} detected")

    # Summary metrics
    with_angle = sum(1 for d in detections if d["angle_deg"] == d["angle_deg"])
    avg_conf   = sum(d["conf"] for d in detections) / n if n else 0

    m1, m2, m3 = st.columns(3)
    m1.metric("Tubes", n)
    m2.metric("With angle", with_angle)
    m3.metric("Avg conf", f"{avg_conf:.2f}")

    st.divider()

    if n == 0:
        st.warning("No tubes detected. Try lowering the confidence threshold.")
    else:
        for idx, det in enumerate(detections):
            ang = det["angle_deg"]
            ang_str = f"{ang:.2f}°" if ang == ang else "N/A"

            with st.expander(f"Tube {idx+1}  —  conf {det['conf']:.3f}", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("center_x",     f"{det['center_x']:.1f}")
                c2.metric("center_y",     f"{det['center_y']:.1f}")
                c3.metric("bbox_x",       f"{det['bbox_x']:.1f}")
                c4.metric("bbox_y",       f"{det['bbox_y']:.1f}")

                c5, c6, c7, c8 = st.columns(4)
                c5.metric("bbox_w",       f"{det['bbox_w']:.1f}")
                c6.metric("bbox_h",       f"{det['bbox_h']:.1f}")
                c7.metric("bbox_rotation",f"{det['bbox_rotation']:.2f}°")
                c8.metric("angle_deg",    ang_str)

        # Download
        export_df = pd.DataFrame([{
            "tube":         f"T{i+1}",
            "center_x":    d["center_x"],
            "center_y":    d["center_y"],
            "bbox_x":      d["bbox_x"],
            "bbox_y":      d["bbox_y"],
            "bbox_w":      d["bbox_w"],
            "bbox_h":      d["bbox_h"],
            "bbox_rotation": d["bbox_rotation"],
            "angle_deg":   d["angle_deg"],
            "conf":        d["conf"],
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
        cols = ["center_x","center_y","bbox_x","bbox_y","bbox_w","bbox_h","bbox_rotation","angle_deg","conf"]
        df = pd.DataFrame([{c: d[c] for c in cols} for d in detections])
        df.index = [f"Tube {i+1}" for i in range(n)]
        st.dataframe(df.style.format("{:.2f}"), use_container_width=True)