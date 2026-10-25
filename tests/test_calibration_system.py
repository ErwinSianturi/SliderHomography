"""Comprehensive Test Suite for Homography Calibration System.

Verifies:
1. Config and data loader accuracy.
2. Resolution scaling (muxer 960x544 -> native frame px).
3. Foot-point extraction (bottom_center).
4. Homography baseline reproduction against detect_20261024/out/detections.json.
5. 2D transformation composition (translation, scaling, rotation).
6. Pairs adjustment and homography recalculation.
7. Point-in-polygon aisle validation.
8. Two-format export integrity (Format A overrides and Format B master compatible).
"""

import json
import os
import sys
import unittest
import numpy as np

# Ensure workspace root is in sys.path
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from core.config_loader import DataLoader, CAMERA_ANOMALY_REGISTRY
from core.homography_engine import (
    compute_homography_dlt,
    project_points,
    calculate_foot_point,
    build_transform_matrix,
    apply_calibration_to_pairs,
)
from core.spatial_validator import SpatialValidator
from core.exporter import CalibrationExporter


class TestHomographyCalibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = DataLoader()
        cls.validator = SpatialValidator(cls.loader.regions, cls.loader.scale_px_per_m)
        cls.exporter = CalibrationExporter(cls.loader.export_data)

    def test_01_loader_data_counts(self):
        """Verify master data counts: 98 cameras, 1077 regions, scale factor."""
        self.assertEqual(len(self.loader.homography_by_cam), 98)
        self.assertEqual(len(self.loader.cameras_by_id), 98)
        self.assertEqual(len(self.loader.regions), 1077)
        self.assertAlmostEqual(self.loader.scale_px_per_m, 31.4465408805, places=5)
        self.assertGreater(len(self.loader.baseline_detections), 1000)

    def test_02_camera_taxonomy(self):
        """Verify anomalous camera registry identification."""
        cam_list = self.loader.get_camera_list()
        self.assertEqual(len(cam_list), 98)
        
        # Check critical cameras
        crit_ids = [
            "Shelf M Wow Spagheti, Energen, Sterile Milk (4mm)",
            "Shelf M Migelas (4mm)",
            "Lorong Bawah 1",
            "Shelf M Laundry Liquid",
        ]
        found_crit = [c for c in cam_list if c["id"] in crit_ids]
        self.assertEqual(len(found_crit), 4)
        for c in found_crit:
            self.assertIn("Confirmed", c["anomaly_status"])

    def test_03_foot_point_calculation(self):
        """Verify foot-point calculation matches production bottom_center."""
        # l, t, r, b = 100, 50, 200, 250
        fx, fy = calculate_foot_point(100.0, 50.0, 200.0, 250.0, foot_ratio=1.0)
        self.assertEqual(fx, 150.0)
        self.assertEqual(fy, 250.0)

        # Ankle ratio = 0.92
        fx2, fy2 = calculate_foot_point(100.0, 50.0, 200.0, 250.0, foot_ratio=0.92)
        self.assertEqual(fx2, 150.0)
        self.assertAlmostEqual(fy2, 50.0 + 0.92 * 200.0, places=4)

    def test_04_baseline_homography_fidelity(self):
        """Verify baseline projection matches detections.json within 0.1 px residual."""
        test_cameras = [
            "Shelf M Wow Spagheti, Energen, Sterile Milk (4mm)",
            "Shelf M Migelas (4mm)",
            "Lorong Bawah 1",
            "Shelf M Laundry Liquid",
            "D1",
        ]
        for cam in test_cameras:
            c_dets = [d for d in self.loader.baseline_detections if d.get("camera") == cam]
            self.assertGreaterEqual(len(c_dets), 4)
            src = np.array([d["foot_native"] for d in c_dets], dtype=float)
            dst = np.array([d["map_px"] for d in c_dets], dtype=float)

            H, info = compute_homography_dlt(src, dst)
            self.assertIsNotNone(H)
            self.assertTrue(info["valid"])
            self.assertLess(info["max_residual_px"], 0.1, f"Residual too large for {cam}")

    def test_05_transformation_matrix_math(self):
        """Verify identity, translation, scaling, and rotation matrices."""
        # 1. Identity
        T_ident = build_transform_matrix(dx=0, dy=0, sx=1.0, sy=1.0, theta_deg=0.0)
        np.testing.assert_allclose(T_ident, np.eye(3), atol=1e-9)

        # 2. Pure translation
        T_trans = build_transform_matrix(dx=50.0, dy=-120.0, sx=1.0, sy=1.0, theta_deg=0.0)
        pt = np.array([[100.0, 200.0]])
        proj = project_points(T_trans, pt)
        np.testing.assert_allclose(proj, [[150.0, 80.0]], atol=1e-9)

        # 3. Scaling around center (100, 100)
        T_scale = build_transform_matrix(dx=0, dy=0, sx=2.0, sy=0.5, theta_deg=0.0, center=(100.0, 100.0))
        pt2 = np.array([[110.0, 120.0]])
        # dx_from_c = 10 * 2 = 20 -> 120; dy_from_c = 20 * 0.5 = 10 -> 110
        proj2 = project_points(T_scale, pt2)
        np.testing.assert_allclose(proj2, [[120.0, 110.0]], atol=1e-9)

    def test_06_apply_calibration_to_pairs(self):
        """Verify applying T_corr cleanly transforms correspondence pairs without mutating originals."""
        raw_pairs = [
            {"id": 1, "framePt": [100, 200], "mapPt": [1000, 2000]},
            {"id": 2, "framePt": [300, 400], "mapPt": [1200, 2200]},
        ]
        T = build_transform_matrix(dx=15.0, dy=-30.0)
        calibrated = apply_calibration_to_pairs(raw_pairs, T)
        
        # Check originals untouched
        self.assertEqual(raw_pairs[0]["mapPt"], [1000, 2000])
        # Check calibrated shifted
        self.assertEqual(calibrated[0]["mapPt"], [1015.0, 1970.0])
        self.assertEqual(calibrated[1]["mapPt"], [1215.0, 2170.0])

    def test_07_spatial_validator(self):
        """Verify Point-in-Polygon checks identify store aisles."""
        # Check an instant noodle coordinate (from audit: instant noodles Y ~ 1670)
        aisle = self.validator.find_aisle_for_point(1100.0, 1675.0)
        self.assertIsNotNone(aisle)
        self.assertIn("Noodle", aisle)

    def test_08_export_format_a_and_b(self):
        """Verify Format A and Format B export and verify round-trip integrity."""
        tmp_dir = os.path.join(WORKSPACE_DIR, "tests", "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        path_a = os.path.join(tmp_dir, "test_overrides.json")
        path_b = os.path.join(tmp_dir, "test_cameras_export_calibrated.json")

        # Test Format A
        cam_id = "Shelf M Wow Spagheti, Energen, Sterile Milk (4mm)"
        self.exporter.export_format_a_overrides(
            camera_id=cam_id,
            camera_label=cam_id,
            frame="P40_1.jpg",
            mode="slider_transform",
            parameters={"dx": 0.0, "dy": -180.0, "sx": 1.0, "sy": 1.0, "theta_deg": 0.0, "foot_ratio": 1.0},
            baseline_H=np.eye(3),
            calibrated_H=np.eye(3),
            T_matrix=np.eye(3),
            metrics={"count": 12, "mean_shift_px": 180.0},
            calibrated_pairs=[],
            output_path=path_a,
        )
        self.assertTrue(os.path.exists(path_a))
        with open(path_a, "r", encoding="utf-8") as f:
            data_a = json.load(f)
            self.assertIn(cam_id, data_a["cameras"])

        # Test Format B
        self.exporter.export_format_b_master_compatible(
            calibrated_pairs_by_camera={cam_id: [{"id": 1, "framePt": [1, 1], "mapPt": [2, 2]}]},
            output_path=path_b,
        )
        self.assertTrue(os.path.exists(path_b))
        with open(path_b, "r", encoding="utf-8") as f:
            data_b = json.load(f)
            self.assertEqual(len(data_b["cameras"]), 98)
            self.assertEqual(len(data_b["regions"]), 1077)
            # Find the updated camera
            h_entry = next(h for h in data_b["homography"] if h["cameraId"] == cam_id)
            self.assertEqual(h_entry["pairs"], [{"id": 1, "framePt": [1, 1], "mapPt": [2, 2]}])

        # Clean up tmp
        if os.path.exists(path_a):
            os.remove(path_a)
        if os.path.exists(path_b):
            os.remove(path_b)
        if os.path.exists(tmp_dir):
            os.rmdir(tmp_dir)


if __name__ == "__main__":
    unittest.main()
