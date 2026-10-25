"""Homography Calibration Slider & Visual Inspector.

Interactive Streamlit Application for calibrating multi-camera perspective
homography transformations for retail shopper tracking and floorplan analytics.
"""

from datetime import datetime, timezone
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

# Setup system path
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from core.config_loader import DataLoader, CAMERA_ANOMALY_REGISTRY, sanitize_filename
from core.homography_engine import (
    compute_homography_dlt,
    project_points,
    calculate_foot_point,
    build_transform_matrix,
    apply_calibration_to_pairs,
)
from core.spatial_validator import SpatialValidator
from core.floorplan_renderer import FloorplanRenderer
from core.exporter import CalibrationExporter

st.set_page_config(
    page_title="Homography Calibration Slider",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 26px; font-weight: 700; color: #1e293b; margin-bottom: 2px; }
    .sub-header { font-size: 14px; color: #64748b; margin-bottom: 18px; }
    .metric-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; }
    .badge-critical { background-color: #fee2e2; color: #b91c1c; font-weight: 600; padding: 3px 8px; border-radius: 6px; }
    .badge-strong { background-color: #ffedd5; color: #c2410c; font-weight: 600; padding: 3px 8px; border-radius: 6px; }
    .badge-normal { background-color: #f1f5f9; color: #475569; font-weight: 600; padding: 3px 8px; border-radius: 6px; }
    .badge-calibrated { background-color: #dcfce7; color: #15803d; font-weight: 600; padding: 3px 8px; border-radius: 6px; }

    /* Compact styling for precision nudge buttons in sidebar */
    section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] button {
        padding: 2px 4px !important;
        font-size: 11px !important;
        min-height: 28px !important;
        line-height: 1.2 !important;
        margin-bottom: 2px !important;
    }
</style>
""", unsafe_allow_html=True)


# Clear any stale Streamlit resource caches to guarantee all new methods are loaded
try:
    st.cache_resource.clear()
except Exception:
    pass

loader = DataLoader()
validator = SpatialValidator(loader.regions, loader.scale_px_per_m)
renderer = FloorplanRenderer(loader.regions, loader.floorplan_path, loader.floorplan_size)
exporter = CalibrationExporter(loader.export_data)


def get_camera_overlay_path(cam_id: str) -> Optional[str]:
    """Retrieves overlay path with resilient fallback."""
    if hasattr(loader, "get_overlay_image_path"):
        p = loader.get_overlay_image_path(cam_id)
        if p and os.path.exists(p):
            return p
    fn = sanitize_filename(cam_id) + ".png"
    p = os.path.join(WORKSPACE_DIR, "homography_visual_downloads_20261023", "detect_20261024", "out", "images", fn)
    return p if os.path.exists(p) else None


def get_camera_pairs_path(cam_id: str) -> Optional[str]:
    """Retrieves pairs visualization path with resilient fallback."""
    if hasattr(loader, "get_pairs_image_path"):
        p = loader.get_pairs_image_path(cam_id)
        if p and os.path.exists(p):
            return p
    p_dir = os.path.join(WORKSPACE_DIR, "homography_visual_downloads_20261023", "images")
    fn = sanitize_filename(cam_id) + ".png"
    p = os.path.join(p_dir, fn)
    if os.path.exists(p):
        return p
    if os.path.exists(p_dir):
        for f in os.listdir(p_dir):
            if f.startswith(sanitize_filename(cam_id)) and f.endswith(".png"):
                return os.path.join(p_dir, f)
    return None


def get_camera_crop_bounds(cam_id: str) -> Optional[Tuple[int, int, int, int]]:
    """Retrieves crop bounds with resilient fallback."""
    if hasattr(loader, "get_crop_bounds"):
        cb = loader.get_crop_bounds(cam_id)
        if cb:
            return cb
    cdets = loader.get_baseline_detections_for_camera(cam_id)
    mapped = [d["map_px"] for d in cdets if d.get("map_px")]
    if not mapped:
        return None
    xs = [p[0] for p in mapped]
    ys = [p[1] for p in mapped]
    x0 = max(0, math.floor(min(xs) - 100))
    y0 = max(0, math.floor(min(ys) - 100))
    x1 = min(3750, math.ceil(max(xs) + 100))
    y1 = min(3368, math.ceil(max(ys) + 100))
    return (x0, y0, x1, y1)


def get_camera_frame_size(frame_name: str) -> Tuple[int, int]:
    """Retrieves frame size with resilient fallback."""
    if hasattr(loader, "get_frame_size"):
        return loader.get_frame_size(frame_name)
    p = loader.get_frame_path(frame_name)
    if p and os.path.exists(p):
        try:
            with Image.open(p) as img:
                return img.size
        except Exception:
            pass
    return (640, 360)

# Session state initialization
if "session_calibrations" not in st.session_state:
    st.session_state.session_calibrations = {}
if "session_custom_pairs" not in st.session_state:
    st.session_state.session_custom_pairs = {}

overrides_path = os.path.join(WORKSPACE_DIR, "homography_calibration_overrides.json")
if os.path.exists(overrides_path):
    try:
        with open(overrides_path, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
            for cid, record in saved_data.get("cameras", {}).items():
                params = record.get("parameters", {})
                if cid not in st.session_state.session_calibrations and params:
                    st.session_state.session_calibrations[cid] = {
                        "dx": float(params.get("delta_x_px", 0.0)),
                        "dy": float(params.get("delta_y_px", 0.0)),
                        "sx": float(params.get("scale_x", 1.0)),
                        "sy": float(params.get("scale_y", 1.0)),
                        "theta_deg": float(params.get("rotation_deg", 0.0)),
                        "foot_ratio": float(params.get("foot_ratio", 1.0)),
                    }
                if cid not in st.session_state.session_custom_pairs and record.get("calibrated_pairs"):
                    st.session_state.session_custom_pairs[cid] = record["calibrated_pairs"]
    except Exception:
        pass

# ----------------- SIDEBAR: CAMERA SELECTION -----------------
st.sidebar.markdown("### 📷 Camera Selection")
all_cameras = loader.get_camera_list()

filter_option = st.sidebar.selectbox(
    "Filter by Status",
    [
        "All Cameras (98)",
        "🚨 Critical Anomalies (4)",
        "⚠️ Strong Indication (3)",
        "🔍 Needs Verification (2)",
        "Normal (89)",
        "💾 Session Calibrated",
    ],
)

filtered_cameras = []
for c in all_cameras:
    cid = c["id"]
    status = c["anomaly_status"]
    is_calibrated = (
        cid in st.session_state.session_calibrations
        or cid in st.session_state.get("session_custom_pairs", {})
    )

    if filter_option.startswith("All"):
        filtered_cameras.append(c)
    elif filter_option.startswith("🚨") and "Confirmed" in status:
        filtered_cameras.append(c)
    elif filter_option.startswith("⚠️") and "Strong" in status:
        filtered_cameras.append(c)
    elif filter_option.startswith("🔍") and "Needs" in status:
        filtered_cameras.append(c)
    elif filter_option.startswith("Normal") and status == "Normal":
        filtered_cameras.append(c)
    elif filter_option.startswith("💾") and is_calibrated:
        filtered_cameras.append(c)

if not filtered_cameras:
    filtered_cameras = all_cameras

cam_options = {
    c["id"]: f"{'💾 ' if (c['id'] in st.session_state.session_calibrations or c['id'] in st.session_state.get('session_custom_pairs', {})) else ''}{c['id']} ({c['anomaly_status']})"
    for c in filtered_cameras
}
selected_cam_id = st.sidebar.selectbox("Select Camera", options=list(cam_options.keys()), format_func=lambda x: cam_options[x])

cam_info = next(c for c in all_cameras if c["id"] == selected_cam_id)
anomaly_meta = cam_info.get("anomaly_details")

# Camera Metadata Display
with st.sidebar.expander("ℹ️ Camera Details & Anomaly Record", expanded=True):
    st.markdown(f"**Label:** `{cam_info['label']}`")
    st.markdown(f"**Frame:** `{cam_info['frame']}`")
    st.markdown(f"**Detections:** `{cam_info['detections_count']} persons`")
    st.markdown(f"**Pairs:** `{cam_info['pairs_count']} pairs`")
    
    status_class = "badge-critical" if "Confirmed" in cam_info["anomaly_status"] else ("badge-strong" if "Strong" in cam_info["anomaly_status"] else "badge-normal")
    st.markdown(f"**Status:** <span class='{status_class}'>{cam_info['anomaly_status']}</span>", unsafe_allow_html=True)
    
    if anomaly_meta:
        st.markdown(f"**Intended Aisle:** `{anomaly_meta.get('intended_aisle')}`")
        st.markdown(f"**Recorded Aisle:** `{anomaly_meta.get('recorded_aisle')}`")
        st.caption(anomaly_meta.get("description", ""))

# ----------------- CALIBRATION PARAMETERS (CONTROLS) -----------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Calibration Controls")

# Callbacks for state updates (preventing StreamlitAPIException)
def cb_apply_preset(cid, pdx, pdy):
    st.session_state[f"dx_{cid}"] = float(pdx)
    st.session_state[f"dy_{cid}"] = float(pdy)
    st.session_state[f"sx_{cid}"] = 1.0
    st.session_state[f"sy_{cid}"] = 1.0
    st.session_state[f"theta_{cid}"] = 0.0
    st.session_state[f"foot_{cid}"] = 1.0

def cb_reset_calibration(cid):
    st.session_state[f"dx_{cid}"] = 0.0
    st.session_state[f"dy_{cid}"] = 0.0
    st.session_state[f"sx_{cid}"] = 1.0
    st.session_state[f"sy_{cid}"] = 1.0
    st.session_state[f"theta_{cid}"] = 0.0
    st.session_state[f"foot_{cid}"] = 1.0
    if cid in st.session_state.session_calibrations:
        del st.session_state.session_calibrations[cid]

def cb_nudge(key, delta, min_val, max_val):
    curr = float(st.session_state.get(key, 0.0))
    st.session_state[key] = round(max(min_val, min(max_val, curr + delta)), 2)

def cb_set_val(key, val):
    st.session_state[key] = float(val)

# Slider keys per camera
key_dx = f"dx_{selected_cam_id}"
key_dy = f"dy_{selected_cam_id}"
key_sx = f"sx_{selected_cam_id}"
key_sy = f"sy_{selected_cam_id}"
key_theta = f"theta_{selected_cam_id}"
key_foot = f"foot_{selected_cam_id}"

# Fetch saved session calibration or fall back to empty dict
saved_calib = st.session_state.session_calibrations.get(selected_cam_id) or {}

# Initialize session state keys for current camera if not already set
default_dx = saved_calib.get("dx", 0.0)
default_dy = saved_calib.get("dy", 0.0)
default_sx = saved_calib.get("sx", 1.0)
default_sy = saved_calib.get("sy", 1.0)
default_theta = saved_calib.get("theta_deg", 0.0)
default_foot = saved_calib.get("foot_ratio", 1.0)

if key_dx not in st.session_state:
    st.session_state[key_dx] = float(default_dx)
if key_dy not in st.session_state:
    st.session_state[key_dy] = float(default_dy)
if key_sx not in st.session_state:
    st.session_state[key_sx] = float(default_sx)
if key_sy not in st.session_state:
    st.session_state[key_sy] = float(default_sy)
if key_theta not in st.session_state:
    st.session_state[key_theta] = float(default_theta)
if key_foot not in st.session_state:
    st.session_state[key_foot] = float(default_foot)

# Recommended Preset button with callback
if anomaly_meta and ("recommended_dy" in anomaly_meta or "recommended_dx" in anomaly_meta):
    rec_dy = float(anomaly_meta.get("recommended_dy", 0.0))
    rec_dx = float(anomaly_meta.get("recommended_dx", 0.0))
    st.sidebar.button(
        f"⚡ Apply Recommended Preset (ΔY={rec_dy:.0f}px, ΔX={rec_dx:.0f}px)",
        on_click=cb_apply_preset,
        args=(selected_cam_id, rec_dx, rec_dy),
        use_container_width=True,
    )

val_dx = st.sidebar.slider("Horizontal Translation ΔX (px)", min_value=-500.0, max_value=500.0, step=1.0, help="Pergeseran sumbu X di denah (1 px ≈ 3.2 cm)", key=key_dx)

# Precision Nudge Buttons for ΔX
col_x1, col_x2, col_x3, col_x4, col_x5 = st.sidebar.columns([1, 1, 0.8, 1, 1])
col_x1.button("⏪ -10", key=f"btn_dx_m10_{selected_cam_id}", on_click=cb_nudge, args=(key_dx, -10.0, -500.0, 500.0), use_container_width=True, help="Geser ke kiri -10 px")
col_x2.button("◀ -1", key=f"btn_dx_m1_{selected_cam_id}", on_click=cb_nudge, args=(key_dx, -1.0, -500.0, 500.0), use_container_width=True, help="Geser ke kiri -1 px (presisi)")
col_x3.button("0", key=f"btn_dx_0_{selected_cam_id}", on_click=cb_set_val, args=(key_dx, 0.0), use_container_width=True, help="Reset ΔX ke 0")
col_x4.button("+1 ▶", key=f"btn_dx_p1_{selected_cam_id}", on_click=cb_nudge, args=(key_dx, 1.0, -500.0, 500.0), use_container_width=True, help="Geser ke kanan +1 px (presisi)")
col_x5.button("+10 ⏩", key=f"btn_dx_p10_{selected_cam_id}", on_click=cb_nudge, args=(key_dx, 10.0, -500.0, 500.0), use_container_width=True, help="Geser ke kanan +10 px")

col_x_c1, col_x_c2, col_x_c3, col_x_c4 = st.sidebar.columns(4)
col_x_c1.button("-50", key=f"btn_dx_m50_{selected_cam_id}", on_click=cb_nudge, args=(key_dx, -50.0, -500.0, 500.0), use_container_width=True, help="Lompat -50 px (~1.6 m)")
col_x_c2.button("-5", key=f"btn_dx_m5_{selected_cam_id}", on_click=cb_nudge, args=(key_dx, -5.0, -500.0, 500.0), use_container_width=True, help="Geser -5 px (~16 cm)")
col_x_c3.button("+5", key=f"btn_dx_p5_{selected_cam_id}", on_click=cb_nudge, args=(key_dx, 5.0, -500.0, 500.0), use_container_width=True, help="Geser +5 px (~16 cm)")
col_x_c4.button("+50", key=f"btn_dx_p50_{selected_cam_id}", on_click=cb_nudge, args=(key_dx, 50.0, -500.0, 500.0), use_container_width=True, help="Lompat +50 px (~1.6 m)")

val_dy = st.sidebar.slider("Vertical Translation ΔY (px)", min_value=-500.0, max_value=500.0, step=1.0, help="Pergeseran sumbu Y di denah (1 px ≈ 3.2 cm)", key=key_dy)

# Precision Nudge Buttons for ΔY
col_y1, col_y2, col_y3, col_y4, col_y5 = st.sidebar.columns([1, 1, 0.8, 1, 1])
col_y1.button("⏪ -10", key=f"btn_dy_m10_{selected_cam_id}", on_click=cb_nudge, args=(key_dy, -10.0, -500.0, 500.0), use_container_width=True, help="Geser ke atas -10 px")
col_y2.button("▲ -1", key=f"btn_dy_m1_{selected_cam_id}", on_click=cb_nudge, args=(key_dy, -1.0, -500.0, 500.0), use_container_width=True, help="Geser ke atas -1 px (presisi)")
col_y3.button("0", key=f"btn_dy_0_{selected_cam_id}", on_click=cb_set_val, args=(key_dy, 0.0), use_container_width=True, help="Reset ΔY ke 0")
col_y4.button("+1 ▼", key=f"btn_dy_p1_{selected_cam_id}", on_click=cb_nudge, args=(key_dy, 1.0, -500.0, 500.0), use_container_width=True, help="Geser ke bawah +1 px (presisi)")
col_y5.button("+10 ⏩", key=f"btn_dy_p10_{selected_cam_id}", on_click=cb_nudge, args=(key_dy, 10.0, -500.0, 500.0), use_container_width=True, help="Geser ke bawah +10 px")

col_y_c1, col_y_c2, col_y_c3, col_y_c4 = st.sidebar.columns(4)
col_y_c1.button("-50", key=f"btn_dy_m50_{selected_cam_id}", on_click=cb_nudge, args=(key_dy, -50.0, -500.0, 500.0), use_container_width=True, help="Lompat -50 px (~1.6 m)")
col_y_c2.button("-5", key=f"btn_dy_m5_{selected_cam_id}", on_click=cb_nudge, args=(key_dy, -5.0, -500.0, 500.0), use_container_width=True, help="Geser -5 px (~16 cm)")
col_y_c3.button("+5", key=f"btn_dy_p5_{selected_cam_id}", on_click=cb_nudge, args=(key_dy, 5.0, -500.0, 500.0), use_container_width=True, help="Geser +5 px (~16 cm)")
col_y_c4.button("+50", key=f"btn_dy_p50_{selected_cam_id}", on_click=cb_nudge, args=(key_dy, 50.0, -500.0, 500.0), use_container_width=True, help="Lompat +50 px (~1.6 m)")

with st.sidebar.expander("Advanced Scaling & Rotation"):
    val_sx = st.slider("Scale Sx (Horizontal Stretch)", min_value=0.5, max_value=1.5, step=0.01, key=key_sx)
    c_sx1, c_sx2, c_sx3, c_sx4, c_sx5 = st.columns([1, 1, 0.8, 1, 1])
    c_sx1.button("-0.05", key=f"btn_sx_m05_{selected_cam_id}", on_click=cb_nudge, args=(key_sx, -0.05, 0.5, 1.5), use_container_width=True)
    c_sx2.button("-0.01", key=f"btn_sx_m01_{selected_cam_id}", on_click=cb_nudge, args=(key_sx, -0.01, 0.5, 1.5), use_container_width=True)
    c_sx3.button("1.0", key=f"btn_sx_1_{selected_cam_id}", on_click=cb_set_val, args=(key_sx, 1.0), use_container_width=True)
    c_sx4.button("+0.01", key=f"btn_sx_p01_{selected_cam_id}", on_click=cb_nudge, args=(key_sx, 0.01, 0.5, 1.5), use_container_width=True)
    c_sx5.button("+0.05", key=f"btn_sx_p05_{selected_cam_id}", on_click=cb_nudge, args=(key_sx, 0.05, 0.5, 1.5), use_container_width=True)

    val_sy = st.slider("Scale Sy (Depth Stretch)", min_value=0.5, max_value=1.5, step=0.01, key=key_sy)
    c_sy1, c_sy2, c_sy3, c_sy4, c_sy5 = st.columns([1, 1, 0.8, 1, 1])
    c_sy1.button("-0.05", key=f"btn_sy_m05_{selected_cam_id}", on_click=cb_nudge, args=(key_sy, -0.05, 0.5, 1.5), use_container_width=True)
    c_sy2.button("-0.01", key=f"btn_sy_m01_{selected_cam_id}", on_click=cb_nudge, args=(key_sy, -0.01, 0.5, 1.5), use_container_width=True)
    c_sy3.button("1.0", key=f"btn_sy_1_{selected_cam_id}", on_click=cb_set_val, args=(key_sy, 1.0), use_container_width=True)
    c_sy4.button("+0.01", key=f"btn_sy_p01_{selected_cam_id}", on_click=cb_nudge, args=(key_sy, 0.01, 0.5, 1.5), use_container_width=True)
    c_sy5.button("+0.05", key=f"btn_sy_p05_{selected_cam_id}", on_click=cb_nudge, args=(key_sy, 0.05, 0.5, 1.5), use_container_width=True)

    val_theta = st.slider("Rotation Δθ (deg)", min_value=-25.0, max_value=25.0, step=0.5, key=key_theta)
    c_th1, c_th2, c_th3, c_th4, c_th5 = st.columns([1, 1, 0.8, 1, 1])
    c_th1.button("-5.0°", key=f"btn_th_m5_{selected_cam_id}", on_click=cb_nudge, args=(key_theta, -5.0, -25.0, 25.0), use_container_width=True)
    c_th2.button("-0.5°", key=f"btn_th_m05_{selected_cam_id}", on_click=cb_nudge, args=(key_theta, -0.5, -25.0, 25.0), use_container_width=True)
    c_th3.button("0°", key=f"btn_th_0_{selected_cam_id}", on_click=cb_set_val, args=(key_theta, 0.0), use_container_width=True)
    c_th4.button("+0.5°", key=f"btn_th_p05_{selected_cam_id}", on_click=cb_nudge, args=(key_theta, 0.5, -25.0, 25.0), use_container_width=True)
    c_th5.button("+5.0°", key=f"btn_th_p5_{selected_cam_id}", on_click=cb_nudge, args=(key_theta, 5.0, -25.0, 25.0), use_container_width=True)

    val_foot = st.slider("Foot Ratio α", min_value=0.80, max_value=1.00, step=0.01, help="1.0 = Bottom Edge, 0.92 = Ankle level", key=key_foot)
    c_ft1, c_ft2, c_ft3 = st.columns([1, 1.2, 1])
    c_ft1.button("-0.01", key=f"btn_ft_m01_{selected_cam_id}", on_click=cb_nudge, args=(key_foot, -0.01, 0.80, 1.00), use_container_width=True)
    c_ft2.button("1.0 (Reset)", key=f"btn_ft_1_{selected_cam_id}", on_click=cb_set_val, args=(key_foot, 1.0), use_container_width=True)
    c_ft3.button("+0.01", key=f"btn_ft_p01_{selected_cam_id}", on_click=cb_nudge, args=(key_foot, 0.01, 0.80, 1.00), use_container_width=True)

col_rst, col_save = st.sidebar.columns(2)
col_rst.button("🔄 Reset", on_click=cb_reset_calibration, args=(selected_cam_id,), use_container_width=True)

current_params = {
    "dx": val_dx,
    "dy": val_dy,
    "sx": val_sx,
    "sy": val_sy,
    "theta_deg": val_theta,
    "foot_ratio": val_foot,
}

if col_save.button("💾 Save", use_container_width=True):
    st.session_state.session_calibrations[selected_cam_id] = current_params
    st.sidebar.success(f"Saved calibration for {selected_cam_id}!")

# ----------------- DATA PREPARATION & PROJECTION -----------------
# Load raw and baseline detections
c_baseline_dets = loader.get_baseline_detections_for_camera(selected_cam_id)
cam_raw_meta = loader.cameras_by_id.get(selected_cam_id, {})
cam_pairs = loader.homography_by_cam.get(selected_cam_id, {}).get("pairs", [])

# Check if there are custom edited pairs in session state
has_custom_pairs = bool(
    "session_custom_pairs" in st.session_state
    and selected_cam_id in st.session_state.session_custom_pairs
)

if has_custom_pairs:
    valid_pairs = [
        {"id": p.get("id"), "framePt": list(p.get("framePt", [0, 0])), "mapPt": list(p.get("mapPt", [0, 0]))}
        for p in st.session_state.session_custom_pairs[selected_cam_id]
        if p.get("framePt") is not None and p.get("mapPt") is not None
    ]
else:
    valid_pairs = [
        {"id": p.get("id"), "framePt": list(p.get("framePt", [0, 0])), "mapPt": list(p.get("mapPt", [0, 0]))}
        for p in cam_pairs
        if p.get("framePt") is not None and p.get("mapPt") is not None
    ]

# Determine pivot center for rotation/scale: camera physical pos or median of map points
pivot_center = (1875.0, 1684.0)
if "pos" in cam_raw_meta:
    pivot_center = tuple(cam_raw_meta["pos"])
elif c_baseline_dets:
    xs = [d["map_px"][0] for d in c_baseline_dets if d.get("map_px")]
    ys = [d["map_px"][1] for d in c_baseline_dets if d.get("map_px")]
    if xs and ys:
        pivot_center = (float(np.median(xs)), float(np.median(ys)))

# Build Transformation Matrix T_corr
T_corr = build_transform_matrix(
    dx=val_dx,
    dy=val_dy,
    sx=val_sx,
    sy=val_sy,
    theta_deg=val_theta,
    center=pivot_center,
)

# Baseline homography matrix H_base
H_base = None
H_calib = None
if has_custom_pairs and len(valid_pairs) >= 4:
    # When user customizes pairs, recalculate baseline homography directly from the updated correspondence pairs
    src_f = np.array([p["framePt"] for p in valid_pairs], dtype=float)
    dst_m = np.array([p["mapPt"] for p in valid_pairs], dtype=float)
    H_base, _ = compute_homography_dlt(src_f, dst_m)
elif len(c_baseline_dets) >= 4:
    src_f = np.array([d["foot_native"] for d in c_baseline_dets], dtype=float)
    dst_m = np.array([d["map_px"] for d in c_baseline_dets], dtype=float)
    H_base, _ = compute_homography_dlt(src_f, dst_m)
elif len(valid_pairs) >= 4:
    src_f = np.array([p["framePt"] for p in valid_pairs], dtype=float)
    dst_m = np.array([p["mapPt"] for p in valid_pairs], dtype=float)
    H_base, _ = compute_homography_dlt(src_f, dst_m)

if H_base is not None:
    H_calib = T_corr @ H_base
    H_calib = H_calib / (H_calib[2, 2] if abs(H_calib[2, 2]) > 1e-9 else 1.0)

# Project detections
baseline_pts = []
calibrated_pts = []
detection_table_rows = []

for idx, d in enumerate(c_baseline_dets, start=1):
    bx_map, by_map = d["map_px"]
    baseline_pts.append([bx_map, by_map])

    # Foot-point adjustment if foot_ratio changed
    ltrb = d["bbox_native_ltrb"]
    fx, fy = calculate_foot_point(ltrb[0], ltrb[1], ltrb[2], ltrb[3], foot_ratio=val_foot)

    # Transform
    if H_calib is not None:
        proj = project_points(H_calib, np.array([[fx, fy]]))
        cx_map, cy_map = float(proj[0, 0]), float(proj[0, 1])
    else:
        # Fallback to direct T_corr on baseline map point
        proj = project_points(T_corr, np.array([[bx_map, by_map]]))
        cx_map, cy_map = float(proj[0, 0]), float(proj[0, 1])

    calibrated_pts.append([cx_map, cy_map])

    # Aisle names
    aisle_before = validator.find_aisle_for_point(bx_map, by_map) or "Outside Aisle"
    aisle_after = validator.find_aisle_for_point(cx_map, cy_map) or "Outside Aisle"
    dist_shift = np.hypot(cx_map - bx_map, cy_map - by_map)

    detection_table_rows.append({
        "ID": d.get("n", idx),
        "Conf": f"{d.get('conf', 0.0):.2f}",
        "Foot Native": f"({fx:.1f}, {fy:.1f})",
        "Baseline Map (X, Y)": f"({bx_map:.1f}, {by_map:.1f})",
        "Baseline Aisle": aisle_before,
        "Calibrated Map (X, Y)": f"({cx_map:.1f}, {cy_map:.1f})",
        "Calibrated Aisle": aisle_after,
        "Shift (px / m)": f"{dist_shift:.1f} px ({dist_shift / loader.scale_px_per_m:.2f} m)",
    })

baseline_pts_arr = np.array(baseline_pts, dtype=float) if baseline_pts else np.zeros((0, 2))
calibrated_pts_arr = np.array(calibrated_pts, dtype=float) if calibrated_pts else np.zeros((0, 2))

# Evaluate metrics
intended_target = anomaly_meta.get("intended_aisle") if anomaly_meta else None
eval_metrics = validator.evaluate_points(
    baseline_pts_arr,
    calibrated_pts_arr,
    intended_aisle=intended_target,
    store_bounds=loader.floorplan_size,
)

# ----------------- MAIN UI LAYOUT -----------------
st.markdown("<div class='main-header'>📐 Homography Calibration Slider & Visual Inspector</div>", unsafe_allow_html=True)
st.markdown(f"<div class='sub-header'>Calibrating Camera: <b>{selected_cam_id}</b> | Frame: <code>{cam_info['frame']}</code></div>", unsafe_allow_html=True)

# Metrics Bar
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
col_m1.metric("Persons Detected", f"{len(detection_table_rows)}")
col_m2.metric("Mean Shift", f"{eval_metrics['mean_shift_px']:.1f} px", f"{eval_metrics['mean_shift_m']:.2f} m")

if intended_target:
    col_m3.metric(
        f"Aisle Walkway %",
        f"{eval_metrics['intended_aisle_calibrated_pct']}%",
        f"{eval_metrics['intended_aisle_calibrated_pct'] - eval_metrics['intended_aisle_baseline_pct']:+.1f}%",
        help=f"Persentase orang di lorong {intended_target}",
    )
    col_m4.metric(
        f"Product Shelf %",
        f"{eval_metrics['intended_shelf_calibrated_pct']}%",
        f"{eval_metrics['intended_shelf_calibrated_pct'] - eval_metrics['intended_shelf_baseline_pct']:+.1f}%",
        help=f"Persentase orang di poligon rak produk terkait",
    )
else:
    col_m3.metric("Max Shift", f"{eval_metrics['max_shift_px']:.1f} px", f"{eval_metrics['max_shift_m']:.2f} m")
    col_m4.metric("Pairs Count", f"{len(valid_pairs)}")

col_m5.metric("Out of Bounds", f"{eval_metrics['out_of_bounds_calibrated']}", f"{eval_metrics['out_of_bounds_calibrated'] - eval_metrics['out_of_bounds_baseline']:+d}")

# ----------------- VISUALIZATION PANELS (OVERLAY INSPECTOR) -----------------
st.markdown("### 🖼️ Camera Frame & Floorplan Overlay Inspector")

# Check if active calibration slider has changed from baseline
is_calibrated_active = (
    abs(val_dx) > 1e-3
    or abs(val_dy) > 1e-3
    or abs(val_sx - 1.0) > 1e-3
    or abs(val_sy - 1.0) > 1e-3
    or abs(val_theta) > 1e-3
    or abs(val_foot - 1.0) > 1e-3
)

overlay_path = get_camera_overlay_path(selected_cam_id)
pairs_path = get_camera_pairs_path(selected_cam_id)

col_ctrl1, col_ctrl2 = st.columns([3, 1])
with col_ctrl1:
    view_options = [
        "🖼️ Gambar Overlay Asli Langsung (Baseline)",
        "🎛️ Live Calibrated Overlay (Slider Real-time)",
        "📌 Correspondence Pairs Overlay",
        "🗺️ Denah Vektor Lengkap (Full Store Map)",
    ]
    # Set default: if calibration is active, show calibrated preview, else show original overlay image directly
    selected_view = st.radio(
        "Pilih Mode Tampilan Visual:",
        view_options,
        index=1 if is_calibrated_active else 0,
        horizontal=True,
    )

with col_ctrl2:
    if is_calibrated_active:
        st.markdown(
            f"<div style='margin-top:14px;'><span class='badge-calibrated'>⚡ Slider Aktif (ΔX={val_dx:+.0f}, ΔY={val_dy:+.0f})</span></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='margin-top:14px;'><span class='badge-normal'>ℹ️ Baseline Original (Unmodified)</span></div>",
            unsafe_allow_html=True,
        )

# Render chosen view
if selected_view.startswith("🖼️ Gambar Overlay Asli"):
    if overlay_path and os.path.exists(overlay_path):
        overlay_orig = Image.open(overlay_path)
        st.image(
            overlay_orig,
            use_container_width=True,
            caption=f"Gambar Overlay Asli (Baseline): {selected_cam_id} | File: {os.path.basename(overlay_path)} ({overlay_orig.width}x{overlay_orig.height} px)",
        )
    else:
        st.warning(f"File overlay tidak ditemukan untuk kamera {selected_cam_id}.")

elif selected_view.startswith("🎛️ Live Calibrated Overlay"):
    if overlay_path and os.path.exists(overlay_path):
        overlay_base = Image.open(overlay_path).convert("RGB")
        crop_bounds = get_camera_crop_bounds(selected_cam_id)
        frame_size = get_camera_frame_size(cam_info["frame"])

        calibrated_overlay_img = renderer.render_calibrated_overlay(
            base_overlay_img=overlay_base,
            baseline_dets=c_baseline_dets,
            calibrated_pts=calibrated_pts_arr,
            crop_bounds=crop_bounds,
            frame_size=frame_size,
            val_foot=val_foot,
            header_height=64,
            gutter_width=40,
            current_params=current_params,
        )

        st.image(
            calibrated_overlay_img,
            use_container_width=True,
            caption=(
                f"Overlay Terkalibrasi (Live): 🟢 Baseline Map Pt ➔ 🟣 Vektor Pergeseran ➔ 🔵 Posisi Baru "
                f"(ΔX={val_dx:+.0f}px, ΔY={val_dy:+.0f}px, Sx={val_sx:.2f}, Sy={val_sy:.2f}, θ={val_theta:+.1f}°, Foot={val_foot:.2f})"
                if is_calibrated_active
                else f"Overlay Baseline (Belum ada pergeseran slider). Geser slider di sidebar untuk melihat kalibrasi live!"
            ),
        )
    else:
        st.warning(f"File overlay tidak ditemukan untuk kamera {selected_cam_id}.")

elif selected_view.startswith("📌 Correspondence Pairs"):
    if pairs_path and os.path.exists(pairs_path):
        pairs_img = Image.open(pairs_path)
        st.image(
            pairs_img,
            use_container_width=True,
            caption=f"Correspondence Pairs Visualization: {selected_cam_id} | File: {os.path.basename(pairs_path)} ({pairs_img.width}x{pairs_img.height} px)",
        )
    else:
        st.warning(f"File visualisasi pairs tidak ditemukan untuk kamera {selected_cam_id}.")

elif selected_view.startswith("🗺️ Denah Vektor"):
    map_overlay = renderer.render_overlay(
        camera_meta=cam_raw_meta,
        baseline_pts=baseline_pts_arr,
        calibrated_pts=calibrated_pts_arr,
        pairs=valid_pairs,
        crop_box=None,
    )
    st.image(
        map_overlay,
        use_container_width=True,
        caption="Denah Toko Lengkap (Full Store Vector Map: 3750x3368 px)",
    )

# ----------------- TABS: DETAILS, PAIR EDITOR, EXPORT -----------------
st.markdown("---")
tab_table, tab_pairs, tab_export = st.tabs(["📋 Detection Comparison Table", "📌 Correspondence Pairs Editor", "📤 Export Calibrated Data"])

with tab_table:
    st.markdown("##### Person Detections & Spatial Aisle Classification")
    if detection_table_rows:
        st.dataframe(detection_table_rows, use_container_width=True)
    else:
        st.info("No detections recorded for this camera frame.")

with tab_pairs:
    st.markdown("##### Correspondence Pairs (`pairs`) in Configuration")
    st.caption("These are the ground-plane correspondence points stored in cameras_export.json used to compute homography.")

    if pairs_path and os.path.exists(pairs_path):
        with st.expander("🖼️ Lihat Gambar Visualisasi Pasangan Titik (Pairs Side-by-Side)", expanded=False):
            st.image(Image.open(pairs_path), use_container_width=True, caption=f"Pairs Image: {os.path.basename(pairs_path)}")

    if valid_pairs:
        if has_custom_pairs:
            st.info("✏️ **Pasangan titik kamera ini telah dikustomisasi secara manual.** Matriks homografi & proyeksi denah langsung mengintegrasikan koordinat titik editan ini.")
        else:
            st.caption("ℹ️ Menampilkan pasangan titik default dari konfigurasi.")

        pairs_table = []
        for p in valid_pairs:
            fpt = p.get("framePt", [0, 0])
            mpt = p.get("mapPt", [0, 0])
            # Apply current T_corr to see calibrated mapPt
            m_h = T_corr @ np.array([mpt[0], mpt[1], 1.0], dtype=float)
            calib_mpt = [round(float(m_h[0] / m_h[2]), 1), round(float(m_h[1] / m_h[2]), 1)] if abs(m_h[2]) > 1e-9 else mpt

            pairs_table.append({
                "Pair ID": p.get("id"),
                "Frame Pt (x, y)": f"({fpt[0]}, {fpt[1]})",
                "Baseline Map Pt (X, Y)": f"({mpt[0]}, {mpt[1]})",
                "Calibrated Map Pt (X, Y)": f"({calib_mpt[0]}, {calib_mpt[1]})",
            })
        st.dataframe(pairs_table, use_container_width=True)

        # Single Pair Editor
        st.markdown("###### Edit Individual Pair Point")
        pair_ids = [p["id"] for p in valid_pairs]
        edit_pid = st.selectbox("Select Pair ID to Edit", pair_ids, key=f"sel_pid_{selected_cam_id}")
        target_pair = next(p for p in valid_pairs if p["id"] == edit_pid)

        c_ep1, c_ep2, c_ep3, c_ep4 = st.columns(4)
        ep_fx = c_ep1.number_input(
            "Frame X",
            value=float(target_pair["framePt"][0]),
            step=1.0,
            key=f"ep_fx_{selected_cam_id}_{edit_pid}",
        )
        ep_fy = c_ep2.number_input(
            "Frame Y",
            value=float(target_pair["framePt"][1]),
            step=1.0,
            key=f"ep_fy_{selected_cam_id}_{edit_pid}",
        )
        ep_mx = c_ep3.number_input(
            "Map X",
            value=float(target_pair["mapPt"][0]),
            step=1.0,
            key=f"ep_mx_{selected_cam_id}_{edit_pid}",
        )
        ep_my = c_ep4.number_input(
            "Map Y",
            value=float(target_pair["mapPt"][1]),
            step=1.0,
            key=f"ep_my_{selected_cam_id}_{edit_pid}",
        )

        col_btn_apply, col_btn_reset = st.columns(2)
        if col_btn_apply.button("💾 Apply Point Edit", use_container_width=True, type="primary"):
            if "session_custom_pairs" not in st.session_state:
                st.session_state.session_custom_pairs = {}

            if selected_cam_id not in st.session_state.session_custom_pairs:
                st.session_state.session_custom_pairs[selected_cam_id] = [
                    {"id": p["id"], "framePt": list(p["framePt"]), "mapPt": list(p["mapPt"])}
                    for p in valid_pairs
                ]

            for p in st.session_state.session_custom_pairs[selected_cam_id]:
                if p["id"] == edit_pid:
                    p["framePt"] = [float(ep_fx), float(ep_fy)]
                    p["mapPt"] = [float(ep_mx), float(ep_my)]
                    break

            st.session_state[f"ep_fx_{selected_cam_id}_{edit_pid}"] = float(ep_fx)
            st.session_state[f"ep_fy_{selected_cam_id}_{edit_pid}"] = float(ep_fy)
            st.session_state[f"ep_mx_{selected_cam_id}_{edit_pid}"] = float(ep_mx)
            st.session_state[f"ep_my_{selected_cam_id}_{edit_pid}"] = float(ep_my)

            st.success(f"Berhasil memperbarui Pair #{edit_pid} menjadi Frame: ({ep_fx:.1f}, {ep_fy:.1f}), Map: ({ep_mx:.1f}, {ep_my:.1f})!")
            st.rerun()

        if col_btn_reset.button("↺ Reset Pairs Kamera Ini ke Default", use_container_width=True):
            if "session_custom_pairs" in st.session_state and selected_cam_id in st.session_state.session_custom_pairs:
                del st.session_state.session_custom_pairs[selected_cam_id]
            for p in valid_pairs:
                pid = p["id"]
                st.session_state.pop(f"ep_fx_{selected_cam_id}_{pid}", None)
                st.session_state.pop(f"ep_fy_{selected_cam_id}_{pid}", None)
                st.session_state.pop(f"ep_mx_{selected_cam_id}_{pid}", None)
                st.session_state.pop(f"ep_my_{selected_cam_id}_{pid}", None)
            st.success(f"Pasangan titik untuk {selected_cam_id} telah direset ke default konfigurasi!")
            st.rerun()
    else:
        st.warning("No correspondence pairs configured for this camera.")

with tab_export:
    st.markdown("##### Export Calibrated Data (Dual Formats)")
    st.write(
        "You can export the results in two distinct, independently verifiable formats:"
    )

    # Show list of calibrated cameras in session
    saved_cams = list(st.session_state.session_calibrations.keys())
    if selected_cam_id not in saved_cams and (val_dx != 0 or val_dy != 0 or val_sx != 1.0 or val_sy != 1.0 or val_theta != 0):
        saved_cams.append(selected_cam_id)
    for cc in st.session_state.get("session_custom_pairs", {}).keys():
        if cc not in saved_cams:
            saved_cams.append(cc)

    st.markdown(f"**Cameras with active calibration ({len(saved_cams)}):** `{saved_cams}`")

    col_exp_a, col_exp_b = st.columns(2)

    with col_exp_a:
        st.markdown("### Format A: Overrides & Audit Log (JSON)")
        st.caption("Exports explicit calibration parameters (ΔX, ΔY, Sx, Sy, θ), baseline & calibrated homography matrices, and spatial evaluation metrics.")
        
        # Prepare Format A calibrated pairs
        calib_pairs_curr = apply_calibration_to_pairs(valid_pairs, T_corr)
        
        exp_a_path = os.path.join(WORKSPACE_DIR, "homography_calibration_overrides.json")
        if st.button("💾 Export Format A to Disk", use_container_width=True):
            saved_file_a = exporter.export_format_a_overrides(
                camera_id=selected_cam_id,
                camera_label=cam_info["label"],
                frame=cam_info["frame"],
                mode="slider_transform",
                parameters=current_params,
                baseline_H=H_base,
                calibrated_H=H_calib,
                T_matrix=T_corr,
                metrics=eval_metrics,
                calibrated_pairs=calib_pairs_curr,
                output_path=exp_a_path,
            )
            st.success(f"Format A exported successfully to:\n`{saved_file_a}`")

        # Download button
        with open(exp_a_path, "r", encoding="utf-8") if os.path.exists(exp_a_path) else open(os.devnull) as f:
            content_a = f.read() if os.path.exists(exp_a_path) else "{}"
        st.download_button(
            "⬇️ Download Format A JSON",
            data=content_a,
            file_name="homography_calibration_overrides.json",
            mime="application/json",
            use_container_width=True,
        )

    with col_exp_b:
        st.markdown("### Format B: Production Master Config (JSON)")
        st.caption("Creates a 100% schema-compatible clone of `cameras_export.json` with updated `pairs` for all calibrated cameras. Production pipelines can load this immediately.")
        st.info("💡 **Catatan Kompatibilitas:** Format B memperbarui pasangan titik `mapPt` pada master JSON. Parameter Foot Ratio (α) merupakan parameter internal detector pipeline dan tidak disimpan di schema JSON. Untuk kompatibilitas penuh dengan pipeline produksi, gunakan Foot Ratio = 1.0 (default `bottom_center`).")

        # Build map of updated pairs for all saved cameras
        all_calibrated_pairs = {}
        for cid in saved_cams:
            if "session_custom_pairs" in st.session_state and cid in st.session_state.session_custom_pairs:
                c_pairs = st.session_state.session_custom_pairs[cid]
            else:
                c_pairs = loader.homography_by_cam.get(cid, {}).get("pairs", [])

            c_p = st.session_state.session_calibrations.get(cid, current_params if cid == selected_cam_id else {})
            cid_meta = loader.cameras_by_id.get(cid, {})
            cid_pivot = tuple(cid_meta.get("pos", [1875.0, 1684.0]))
            c_T = build_transform_matrix(
                dx=c_p.get("dx", 0.0),
                dy=c_p.get("dy", 0.0),
                sx=c_p.get("sx", 1.0),
                sy=c_p.get("sy", 1.0),
                theta_deg=c_p.get("theta_deg", 0.0),
                center=cid_pivot,
            )
            all_calibrated_pairs[cid] = apply_calibration_to_pairs(c_pairs, c_T)

        exp_b_path = os.path.join(WORKSPACE_DIR, "cameras_export_calibrated.json")
        if st.button("🚀 Export Format B to Disk", use_container_width=True):
            saved_file_b = exporter.export_format_b_master_compatible(
                calibrated_pairs_by_camera=all_calibrated_pairs,
                output_path=exp_b_path,
            )
            st.success(f"Format B exported successfully to:\n`{saved_file_b}`")

        with open(exp_b_path, "r", encoding="utf-8") if os.path.exists(exp_b_path) else open(os.devnull) as f:
            content_b = f.read() if os.path.exists(exp_b_path) else "{}"
        st.download_button(
            "⬇️ Download Format B JSON",
            data=content_b,
            file_name="cameras_export_calibrated.json",
            mime="application/json",
            use_container_width=True,
        )
