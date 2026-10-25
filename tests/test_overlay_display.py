"""Test overlay image loading and live calibration rendering."""

import os
import sys
import numpy as np
from PIL import Image

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from core.config_loader import DataLoader
from core.floorplan_renderer import FloorplanRenderer


def test_overlay_image_availability():
    loader = DataLoader()
    cameras = loader.get_camera_list()
    assert len(cameras) == 98, f"Expected 98 cameras, got {len(cameras)}"

    missing_overlays = []
    for c in cameras:
        cid = c["id"]
        overlay_path = loader.get_overlay_image_path(cid)
        if not overlay_path or not os.path.exists(overlay_path):
            missing_overlays.append(cid)

    assert len(missing_overlays) == 0, f"Missing overlay images for: {missing_overlays}"


def test_render_calibrated_overlay():
    loader = DataLoader()
    renderer = FloorplanRenderer(loader.regions, loader.floorplan_path, loader.floorplan_size)

    test_cam = "Shelf M Wow Spagheti, Energen, Sterile Milk (4mm)"
    overlay_path = loader.get_overlay_image_path(test_cam)
    assert overlay_path is not None and os.path.exists(overlay_path)

    base_overlay = Image.open(overlay_path).convert("RGB")
    dets = loader.get_baseline_detections_for_camera(test_cam)
    crop_bounds = loader.get_crop_bounds(test_cam)
    frame_size = loader.get_frame_size(loader.cam_to_frame[test_cam])

    assert crop_bounds is not None
    assert len(dets) > 0

    # Simulate calibrated points shifted by dy=-75
    b_pts = np.array([d["map_px"] for d in dets], dtype=float)
    c_pts = b_pts.copy()
    c_pts[:, 1] -= 75.0

    calibrated_img = renderer.render_calibrated_overlay(
        base_overlay_img=base_overlay,
        baseline_dets=dets,
        calibrated_pts=c_pts,
        crop_bounds=crop_bounds,
        frame_size=frame_size,
        val_foot=1.0,
        current_params={"dx": 0.0, "dy": -75.0, "sx": 1.0, "sy": 1.0, "theta_deg": 0.0},
    )

    assert calibrated_img.size == base_overlay.size
    # Verify image was modified
    base_arr = np.array(base_overlay)
    calib_arr = np.array(calibrated_img)
    diff = np.sum(np.abs(base_arr.astype(int) - calib_arr.astype(int)))
    assert diff > 0, "Calibrated overlay image should contain drawn shift lines/points"


if __name__ == "__main__":
    test_overlay_image_availability()
    test_render_calibrated_overlay()
    print("ALL OVERLAY TESTS PASSED!")
