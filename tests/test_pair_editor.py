"""Test Pair Editor and Custom Pairs Persistence Flow."""

import copy
import json
import os
import sys
import unittest
import numpy as np

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from core.config_loader import DataLoader
from core.homography_engine import compute_homography_dlt, build_transform_matrix, apply_calibration_to_pairs
from core.exporter import CalibrationExporter


class TestPairEditorFlow(unittest.TestCase):
    def setUp(self):
        self.loader = DataLoader()
        self.exporter = CalibrationExporter(self.loader.export_data)
        self.session_state = {
            "session_calibrations": {},
            "session_custom_pairs": {},
        }

    def test_01_edit_single_pair_point_flow(self):
        """Simulate editing a single pair point and verifying persistence."""
        cam_id = "D1"
        cam_pairs = self.loader.homography_by_cam.get(cam_id, {}).get("pairs", [])
        self.assertGreaterEqual(len(cam_pairs), 4)

        # Baseline point
        p0 = cam_pairs[0]
        pid = p0["id"]
        orig_map_x, orig_map_y = p0["mapPt"]

        # Simulate initializing session_custom_pairs
        self.session_state["session_custom_pairs"][cam_id] = [
            {"id": p["id"], "framePt": list(p["framePt"]), "mapPt": list(p["mapPt"])}
            for p in cam_pairs
        ]

        # Edit point: Frame (208, 166), Map (2914, 1000)
        target = next(p for p in self.session_state["session_custom_pairs"][cam_id] if p["id"] == pid)
        target["framePt"] = [208.0, 166.0]
        target["mapPt"] = [2914.0, 1000.0]

        # Assert session state holds edited coordinates
        updated_target = next(p for p in self.session_state["session_custom_pairs"][cam_id] if p["id"] == pid)
        self.assertEqual(updated_target["framePt"], [208.0, 166.0])
        self.assertEqual(updated_target["mapPt"], [2914.0, 1000.0])

        # Verify baseline in loader remains untouched
        self.assertEqual(self.loader.homography_by_cam[cam_id]["pairs"][0]["mapPt"], [orig_map_x, orig_map_y])

    def test_02_homography_recalculated_from_edited_pairs(self):
        """Verify that homography recomputed from edited pairs alters the resulting projection."""
        cam_id = "D1"
        cam_pairs = self.loader.homography_by_cam.get(cam_id, {}).get("pairs", [])
        
        # Original homography
        src_orig = np.array([p["framePt"] for p in cam_pairs], dtype=float)
        dst_orig = np.array([p["mapPt"] for p in cam_pairs], dtype=float)
        H_orig, _ = compute_homography_dlt(src_orig, dst_orig)

        # Edited pairs
        edited_pairs = copy.deepcopy(cam_pairs)
        edited_pairs[0]["mapPt"] = [edited_pairs[0]["mapPt"][0] + 150.0, edited_pairs[0]["mapPt"][1] - 200.0]
        src_edit = np.array([p["framePt"] for p in edited_pairs], dtype=float)
        dst_edit = np.array([p["mapPt"] for p in edited_pairs], dtype=float)
        H_edit, _ = compute_homography_dlt(src_edit, dst_edit)

        self.assertIsNotNone(H_orig)
        self.assertIsNotNone(H_edit)
        self.assertFalse(np.allclose(H_orig, H_edit), "Homography matrix should change after editing pair coordinates.")

    def test_03_format_b_export_with_custom_pairs(self):
        """Verify Format B master export integrates customized pairs."""
        cam_id = "D1"
        cam_pairs = copy.deepcopy(self.loader.homography_by_cam.get(cam_id, {}).get("pairs", []))
        pid = cam_pairs[0]["id"]
        cam_pairs[0]["mapPt"] = [2914.0, 1000.0]

        all_calib_pairs = {
            cam_id: cam_pairs
        }

        tmp_out = os.path.join(WORKSPACE_DIR, "tests", "tmp", "test_export_custom_pairs.json")
        try:
            self.exporter.export_format_b_master_compatible(all_calib_pairs, tmp_out)

            with open(tmp_out, "r", encoding="utf-8") as f:
                exported_data = json.load(f)

            exported_h = next(h for h in exported_data["homography"] if h["cameraId"] == cam_id)
            exported_pair = next(p for p in exported_h["pairs"] if p["id"] == pid)
            self.assertEqual(exported_pair["mapPt"], [2914.0, 1000.0])
        finally:
            if os.path.exists(tmp_out):
                os.remove(tmp_out)

    def test_04_reset_custom_pairs(self):
        """Verify resetting custom pairs removes override from session."""
        cam_id = "D1"
        self.session_state["session_custom_pairs"][cam_id] = [{"id": 1, "framePt": [1, 2], "mapPt": [3, 4]}]
        self.assertIn(cam_id, self.session_state["session_custom_pairs"])

        # Reset action
        del self.session_state["session_custom_pairs"][cam_id]
        self.assertNotIn(cam_id, self.session_state["session_custom_pairs"])

    def test_05_dynamic_pairs_render(self):
        """Verify dynamic pairs side-by-side rendering succeeds."""
        from PIL import Image
        from core.floorplan_renderer import FloorplanRenderer

        renderer = FloorplanRenderer(self.loader.regions, self.loader.floorplan_path, self.loader.floorplan_size)
        frame_path = self.loader.get_frame_path("D1_1.jpg")
        self.assertIsNotNone(frame_path)
        frame_img = Image.open(frame_path).convert("RGB")

        cam_pairs = copy.deepcopy(self.loader.homography_by_cam["D1"]["pairs"])
        # Modify pair 6 to mapPt [2914, 7]
        for p in cam_pairs:
            if p["id"] == 6:
                p["mapPt"] = [2914.0, 7.0]

        dyn_img = renderer.render_dynamic_pairs_side_by_side(
            frame_img=frame_img,
            pairs=cam_pairs,
            camera_id="D1",
            frame_name="D1_1.jpg",
            highlight_pid=6,
        )
        self.assertIsNotNone(dyn_img)
        self.assertGreater(dyn_img.width, frame_img.width)
        self.assertGreater(dyn_img.height, frame_img.height)


if __name__ == "__main__":
    unittest.main()
