"""Comprehensive Test Suite for Camera Examination Status Management.

Verifies:
1. Default initial status is valid and normalized ('Unchecked', 'Flagged', 'Checked').
2. User can select, update, and toggle between 'Checked', 'Flagged', and 'Unchecked' repeatedly.
3. Status changes persist to disk (camera_statuses.json) and survive reload.
4. Filtering by the 3 statuses correctly segments the camera fleet.
5. Removal of legacy automatic severity classifications ('Critical', 'Strong', etc.).
6. Graceful handling of empty, None, and legacy status inputs without errors.
7. End-to-end export compatibility (Format A & Format B).
"""

import json
import os
import shutil
import sys
import unittest
from typing import Dict

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from core.config_loader import (
    DataLoader,
    VALID_CAMERA_STATUSES,
    DEFAULT_CAMERA_STATUS,
    normalize_camera_status,
    load_camera_statuses,
    save_camera_statuses,
)
from core.exporter import CalibrationExporter


class TestCameraStatusSystem(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = os.path.join(WORKSPACE_DIR, "tests", "tmp_status")
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.loader = DataLoader()
        self.exporter = CalibrationExporter(self.loader.export_data)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_allowed_statuses_definition(self):
        """Ensure allowed statuses are strictly 'Unchecked', 'Flagged', 'Checked'."""
        self.assertEqual(set(VALID_CAMERA_STATUSES), {"Unchecked", "Flagged", "Checked"})
        self.assertEqual(DEFAULT_CAMERA_STATUS, "Unchecked")

    def test_02_legacy_and_invalid_status_normalization(self):
        """Verify normalization gracefully handles None, empty, and legacy labels."""
        # Clean defaults
        self.assertEqual(normalize_camera_status(None), "Unchecked")
        self.assertEqual(normalize_camera_status(""), "Unchecked")
        self.assertEqual(normalize_camera_status("unknown_val"), "Unchecked")
        self.assertEqual(normalize_camera_status("planned"), "Unchecked")
        self.assertEqual(normalize_camera_status("normal"), "Unchecked")

        # Verified / Checked
        self.assertEqual(normalize_camera_status("checked"), "Checked")
        self.assertEqual(normalize_camera_status("Checked"), "Checked")
        self.assertEqual(normalize_camera_status("verified"), "Checked")
        self.assertEqual(normalize_camera_status("ok"), "Checked")

        # Flagged / Problematic
        self.assertEqual(normalize_camera_status("flagged"), "Flagged")
        self.assertEqual(normalize_camera_status("Flagged"), "Flagged")
        self.assertEqual(normalize_camera_status("critical"), "Flagged")
        self.assertEqual(normalize_camera_status("strong"), "Flagged")
        self.assertEqual(normalize_camera_status("needs verification"), "Flagged")

    def test_03_camera_fleet_status_initialization(self):
        """All cameras returned by loader must strictly possess one of the 3 statuses and default to Unchecked."""
        cams = self.loader.get_camera_list()
        self.assertEqual(len(cams), 98)
        self.assertEqual(set(c["status"] for c in cams), {"Unchecked"})
        for c in cams:
            self.assertIn(c["status"], VALID_CAMERA_STATUSES)
            self.assertNotIn("Critical", c["status"])
            self.assertNotIn("Strong", c["status"])

    def test_04_status_toggle_and_persistence(self):
        """Test updating camera status repeatedly and verifying persistence to disk."""
        cam_id = "Shelf M Wow Spagheti, Energen, Sterile Milk (4mm)"
        statuses: Dict[str, str] = {}

        # 1. Set to Flagged
        statuses[cam_id] = "Flagged"
        save_camera_statuses(self.tmp_dir, statuses)
        loaded = load_camera_statuses(self.tmp_dir)
        self.assertEqual(loaded.get(cam_id), "Flagged")

        # 2. Toggle to Checked
        statuses[cam_id] = "Checked"
        save_camera_statuses(self.tmp_dir, statuses)
        loaded = load_camera_statuses(self.tmp_dir)
        self.assertEqual(loaded.get(cam_id), "Checked")

        # 3. Toggle back to Unchecked
        statuses[cam_id] = "Unchecked"
        save_camera_statuses(self.tmp_dir, statuses)
        loaded = load_camera_statuses(self.tmp_dir)
        self.assertEqual(loaded.get(cam_id), "Unchecked")

    def test_05_filtering_logic_by_status(self):
        """Verify segmentation by status filter."""
        custom_statuses = {
            "D1": "Checked",
            "D2": "Flagged",
            "D3": "Unchecked",
        }
        cams = self.loader.get_camera_list(custom_statuses)
        cams_by_id = {c["id"]: c for c in cams}

        self.assertEqual(cams_by_id["D1"]["status"], "Checked")
        self.assertEqual(cams_by_id["D2"]["status"], "Flagged")
        self.assertEqual(cams_by_id["D3"]["status"], "Unchecked")

        # Filter subsets
        flagged_list = [c for c in cams if c["status"] == "Flagged"]
        checked_list = [c for c in cams if c["status"] == "Checked"]
        unchecked_list = [c for c in cams if c["status"] == "Unchecked"]

        self.assertIn("D2", [c["id"] for c in flagged_list])
        self.assertIn("D1", [c["id"] for c in checked_list])
        self.assertIn("D3", [c["id"] for c in unchecked_list])

    def test_06_export_format_a_and_b_status_integrity(self):
        """Verify camera status is properly written to Format A and Format B exports."""
        cam_id = "Lorong Bawah 1"
        out_a = os.path.join(self.tmp_dir, "test_format_a.json")
        out_b = os.path.join(self.tmp_dir, "test_format_b.json")

        # Format A
        self.exporter.export_format_a_overrides(
            camera_id=cam_id,
            camera_label=cam_id,
            frame="Lorong_Bawah_1_1.jpg",
            mode="slider_transform",
            parameters={"dx": 0.0, "dy": 0.0},
            baseline_H=None,
            calibrated_H=None,
            T_matrix=None,
            metrics={},
            calibrated_pairs=[],
            output_path=out_a,
            camera_status="Flagged",
        )
        with open(out_a, "r", encoding="utf-8") as f:
            data_a = json.load(f)
        self.assertEqual(data_a["cameras"][cam_id]["camera_status"], "Flagged")

        # Format B
        self.exporter.export_format_b_master_compatible(
            calibrated_pairs_by_camera={cam_id: []},
            output_path=out_b,
            camera_statuses={cam_id: "Checked"},
        )
        with open(out_b, "r", encoding="utf-8") as f:
            data_b = json.load(f)
        cam_entry = next(c for c in data_b["cameras"] if c["id"] == cam_id)
        self.assertEqual(cam_entry["review_status"], "Checked")


if __name__ == "__main__":
    unittest.main()
