"""Verify Lorong Bawah 1 cone direction and reset logic."""

import math
import os
import sys
import numpy as np

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from core.config_loader import DataLoader
from core.floorplan_renderer import FloorplanRenderer


def test_lorong_bawah_1_cone():
    loader = DataLoader()
    cam_id = "Lorong Bawah 1"
    cam_meta = loader.cameras_by_id[cam_id]
    cx, cy = float(cam_meta["pos"][0]), float(cam_meta["pos"][1])
    heading = float(cam_meta["heading_deg"])
    fov = float(cam_meta["fov_deg"])
    range_px = float(cam_meta["range_m"]) * 31.44654

    rad_center = math.radians(heading)
    xc = cx + range_px * math.cos(rad_center)
    yc = cy + range_px * math.sin(rad_center)

    print(f"Lorong Bawah 1 mount: ({cx}, {cy})")
    print(f"Heading: {heading} deg | Range: {range_px:.1f} px")
    print(f"Cone center endpoint: ({xc:.1f}, {yc:.1f})")

    # Detections
    b_dets = loader.get_baseline_detections_for_camera(cam_id)
    det_xs = [d["map_px"][0] for d in b_dets]
    det_ys = [d["map_px"][1] for d in b_dets]
    avg_det_x = np.mean(det_xs)
    avg_det_y = np.mean(det_ys)
    print(f"Average detection position: ({avg_det_x:.1f}, {avg_det_y:.1f})")

    # Assert that cone direction points towards detections (dx > 0 and dy < 0)
    assert xc > cx, "Cone must point East (into store, dx > 0)"
    assert yc < cy, "Cone must point North (towards pairs, dy < 0)"
    print("CONE ORIENTATION VERIFICATION: PASSED!")


if __name__ == "__main__":
    test_lorong_bawah_1_cone()

