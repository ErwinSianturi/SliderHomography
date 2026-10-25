"""End-to-End Simulation Test:
Validates that applying calibration to Shelf M Wow Spagheti, Energen, Sterile Milk (4mm)
and exporting Format B (cameras_export_calibrated.json) successfully shifts the detections
back into the Instant Noodles aisle when read by the production homography logic.
"""

import json
import os
import sys
import numpy as np

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from core.config_loader import DataLoader
from core.homography_engine import build_transform_matrix, apply_calibration_to_pairs, compute_homography_dlt, project_points
from core.spatial_validator import SpatialValidator
from core.exporter import CalibrationExporter


def test_end_to_end():
    loader = DataLoader()
    validator = SpatialValidator(loader.regions, loader.scale_px_per_m)
    exporter = CalibrationExporter(loader.export_data)

    target_cam = "Shelf M Wow Spagheti, Energen, Sterile Milk (4mm)"
    baseline_dets = loader.get_baseline_detections_for_camera(target_cam)
    pairs = loader.homography_by_cam[target_cam]["pairs"]

    print("Target Camera:", target_cam)
    print("Baseline Detections Count:", len(baseline_dets))
    b_map_pts = np.array([d["map_px"] for d in baseline_dets])
    print("Baseline Mean Y:", float(np.mean(b_map_pts[:, 1])))

    # Recommended empirical correction: Delta Y = -75.0 px (replaces historical -180 px)
    T = build_transform_matrix(dx=0, dy=-75.0)
    calib_pairs = apply_calibration_to_pairs(pairs, T)

    out_file = os.path.join(WORKSPACE_DIR, "tests", "tmp_cameras_export_calibrated.json")
    exporter.export_format_b_master_compatible({target_cam: calib_pairs}, out_file)

    # Read back using production logic
    with open(out_file, "r", encoding="utf-8") as f:
        dl_new = json.load(f)

    h_entry = next(h for h in dl_new["homography"] if h["cameraId"] == target_cam)
    valid_pairs = [p for p in h_entry["pairs"] if p.get("framePt") and p.get("mapPt")]
    src = np.array([p["framePt"] for p in valid_pairs], dtype=float)
    dst = np.array([p["mapPt"] for p in valid_pairs], dtype=float)
    H_new, info = compute_homography_dlt(src, dst)

    assert info["valid"], "New H must be numerically valid"
    print("H_new Condition Number:", info["condition_number"])

    feet = np.array([d["foot_native"] for d in baseline_dets])
    calib_map_pts = project_points(H_new, feet)
    print("Calibrated Mean Y:", float(np.mean(calib_map_pts[:, 1])))

    eval_res = validator.evaluate_points(b_map_pts, calib_map_pts, intended_aisle="Instant & Import Noodles")
    print(f"Baseline in Instant & Import Noodles: {eval_res['intended_aisle_baseline_count']} / {eval_res['count']} ({eval_res['intended_aisle_baseline_pct']}%)")
    print(f"Calibrated in Instant & Import Noodles: {eval_res['intended_aisle_calibrated_count']} / {eval_res['count']} ({eval_res['intended_aisle_calibrated_pct']}%)")
    print(f"Calibrated in Noodle Shelves: {eval_res['intended_shelf_calibrated_count']} / {eval_res['count']} ({eval_res['intended_shelf_calibrated_pct']}%)")
    print(f"Mean Shift Distance: {eval_res['mean_shift_px']:.1f} px ({eval_res['mean_shift_m']:.2f} m)")

    # Assert that calibrated points successfully increase containment in noodles
    assert eval_res['intended_aisle_calibrated_count'] > eval_res['intended_aisle_baseline_count']

    if os.path.exists(out_file):
        os.remove(out_file)

    print("END-TO-END VALIDATION PASSED!")


if __name__ == "__main__":
    test_end_to_end()
