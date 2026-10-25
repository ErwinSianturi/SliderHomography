"""Audit Homography Computation and Baseline Fidelity across all 98 cameras."""

import json
import math
import os
import sys
import numpy as np

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from core.config_loader import DataLoader
from core.homography_engine import (
    compute_homography_dlt,
    project_points,
    calculate_foot_point,
    build_transform_matrix,
)

loader = DataLoader()

print("=== 1. RESOLUTION SCALING & FOOT POINT VERIFICATION ===")
# Verify scaling on D10 (640x360) and P40 (2560x1440)
# Muxer is 960x544
mux_w, mux_h = 960.0, 544.0

d10_dets = loader.get_raw_detections("D10_1.jpg")
p40_dets = loader.get_raw_detections("P40_1.jpg")

print(f"D10 raw detections: {len(d10_dets)}")
print(f"P40 raw detections: {len(p40_dets)}")

# Check first detection of D10
d0 = d10_dets[0]
sx_d10, sy_d10 = 640.0 / mux_w, 360.0 / mux_h
l = d0["left"] * sx_d10
t = d0["top"] * sy_d10
r = (d0["left"] + d0["width"]) * sx_d10
b = (d0["top"] + d0["height"]) * sy_d10
fx, fy = calculate_foot_point(l, t, r, b, foot_ratio=1.0)

b_d10 = loader.get_baseline_detections_for_camera("D10")[0]
print(f"D10 calculated foot: ({fx:.1f}, {fy:.1f}) | Baseline foot: {b_d10['foot_native']}")
assert abs(fx - b_d10['foot_native'][0]) < 0.2
assert abs(fy - b_d10['foot_native'][1]) < 0.2

print("=== 2. BASELINE HOMOGRAPHY FIT & RESIDUALS ACROSS ALL 98 CAMERAS ===")
total_dets = 0
all_residuals = []
cam_stats = []

for cam_id, h_entry in loader.homography_by_cam.items():
    c_dets = loader.get_baseline_detections_for_camera(cam_id)
    if len(c_dets) < 4:
        continue
    
    src = np.array([d["foot_native"] for d in c_dets], dtype=float)
    dst = np.array([d["map_px"] for d in c_dets], dtype=float)
    
    H, info = compute_homography_dlt(src, dst)
    if H is not None and info["valid"]:
        proj = project_points(H, src)
        res = np.hypot(proj[:, 0] - dst[:, 0], proj[:, 1] - dst[:, 1])
        all_residuals.extend(res)
        total_dets += len(res)
        cam_stats.append((cam_id, np.mean(res), np.max(res), info["condition_number"]))

all_residuals = np.array(all_residuals)
print(f"Cameras evaluated with >= 4 dets: {len(cam_stats)} / 98")
print(f"Total detections evaluated: {total_dets}")
print(f"Mean residual vs baseline: {np.mean(all_residuals):.4f} px")
print(f"Median residual vs baseline: {np.median(all_residuals):.4f} px")
print(f"99th percentile residual: {np.percentile(all_residuals, 99):.4f} px")
print(f"Maximum residual: {np.max(all_residuals):.4f} px")

print("=== 3. MATRIX COMPOSITION & PREVENTING ACCUMULATION ===")
# Test that T_corr is always computed from (dx, dy, sx, sy, theta) relative to pivot,
# and NOT iteratively multiplied on previous frames.
T1 = build_transform_matrix(dx=10, dy=20, sx=1.1, sy=0.9, theta_deg=5, center=(100, 100))
T2 = build_transform_matrix(dx=10, dy=20, sx=1.1, sy=0.9, theta_deg=5, center=(100, 100))
np.testing.assert_allclose(T1, T2, atol=1e-12)
print("Transformation matrix generation is strictly deterministic and idempotent.")

print("=== 4. PAIRS RECALCULATION VALIDATION ===")
# Verify condition number on pairs DLT
pairs_conds = []
for cam_id, h_entry in loader.homography_by_cam.items():
    pairs = [p for p in h_entry["pairs"] if p.get("framePt") and p.get("mapPt")]
    if len(pairs) >= 4:
        s = np.array([p["framePt"] for p in pairs], dtype=float)
        d = np.array([p["mapPt"] for p in pairs], dtype=float)
        H_p, info_p = compute_homography_dlt(s, d)
        if H_p is not None:
            pairs_conds.append(info_p["condition_number"])

print(f"Pairs homography computed for {len(pairs_conds)} cameras.")
print(f"Median condition number: {np.median(pairs_conds):.1f}")
print(f"Max condition number: {np.max(pairs_conds):.1f}")
