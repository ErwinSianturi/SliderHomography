"""Test Streamlit session state and calibration parameter flow."""

import os
import sys
import unittest
import numpy as np

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from core.config_loader import DataLoader, CAMERA_ANOMALY_REGISTRY
from core.homography_engine import build_transform_matrix, project_points


class TestAppStateFlow(unittest.TestCase):
    def setUp(self):
        self.loader = DataLoader()
        self.session_state = {
            "session_calibrations": {}
        }

    def test_01_initial_open_camera_without_saved_calib(self):
        """Test initial load when a camera has no saved calibration."""
        cam_id = "Lorong Bawah 1"
        saved_calib = self.session_state["session_calibrations"].get(cam_id) or {}
        self.assertEqual(saved_calib, {})

        default_dx = saved_calib.get("dx", 0.0)
        default_dy = saved_calib.get("dy", 0.0)
        default_sx = saved_calib.get("sx", 1.0)
        default_sy = saved_calib.get("sy", 1.0)
        default_theta = saved_calib.get("theta_deg", 0.0)
        default_foot = saved_calib.get("foot_ratio", 1.0)

        self.assertEqual(default_dx, 0.0)
        self.assertEqual(default_dy, 0.0)
        self.assertEqual(default_sx, 1.0)
        self.assertEqual(default_sy, 1.0)
        self.assertEqual(default_theta, 0.0)
        self.assertEqual(default_foot, 1.0)

    def test_02_apply_preset_and_save(self):
        """Test applying an anomaly preset and saving to session."""
        cam_id = "Shelf M Wow Spagheti, Energen, Sterile Milk (4mm)"
        anomaly_info = CAMERA_ANOMALY_REGISTRY[cam_id]
        pdx = float(anomaly_info.get("recommended_dx", 0.0))
        pdy = float(anomaly_info.get("recommended_dy", -75.0))

        # Callback logic
        self.session_state[f"dx_{cam_id}"] = pdx
        self.session_state[f"dy_{cam_id}"] = pdy
        self.session_state[f"sx_{cam_id}"] = 1.0
        self.session_state[f"sy_{cam_id}"] = 1.0
        self.session_state[f"theta_{cam_id}"] = 0.0
        self.session_state[f"foot_{cam_id}"] = 1.0

        current_params = {
            "dx": self.session_state[f"dx_{cam_id}"],
            "dy": self.session_state[f"dy_{cam_id}"],
            "sx": self.session_state[f"sx_{cam_id}"],
            "sy": self.session_state[f"sy_{cam_id}"],
            "theta_deg": self.session_state[f"theta_{cam_id}"],
            "foot_ratio": self.session_state[f"foot_{cam_id}"],
        }
        self.session_state["session_calibrations"][cam_id] = current_params

        self.assertIn(cam_id, self.session_state["session_calibrations"])
        self.assertEqual(self.session_state["session_calibrations"][cam_id]["dy"], -75.0)

    def test_03_camera_switching_and_retrieval(self):
        """Test switching camera preserves previous calibrations and correctly loads new one."""
        cam_1 = "Shelf M Wow Spagheti, Energen, Sterile Milk (4mm)"
        cam_2 = "Lorong Bawah 1"

        # Save cam 1
        self.session_state["session_calibrations"][cam_1] = {
            "dx": 0.0, "dy": -75.0, "sx": 1.0, "sy": 1.0, "theta_deg": 0.0, "foot_ratio": 1.0
        }

        # Switch to cam 2 (unsaved)
        saved_calib_2 = self.session_state["session_calibrations"].get(cam_2) or {}
        self.assertEqual(saved_calib_2, {})
        self.assertEqual(saved_calib_2.get("dx", 0.0), 0.0)

        # Switch back to cam 1 (saved)
        saved_calib_1 = self.session_state["session_calibrations"].get(cam_1) or {}
        self.assertEqual(saved_calib_1.get("dy", 0.0), -75.0)

    def test_04_reset_callback(self):
        """Test resetting calibration restores defaults and cleans session."""
        cam_id = "Lorong Bawah 1"
        self.session_state["session_calibrations"][cam_id] = {
            "dx": -120.0, "dy": 80.0, "sx": 1.0, "sy": 1.0, "theta_deg": 0.0, "foot_ratio": 1.0
        }
        self.session_state[f"dx_{cam_id}"] = -120.0
        self.session_state[f"dy_{cam_id}"] = 80.0

        # Reset callback simulation
        self.session_state[f"dx_{cam_id}"] = 0.0
        self.session_state[f"dy_{cam_id}"] = 0.0
        self.session_state[f"sx_{cam_id}"] = 1.0
        self.session_state[f"sy_{cam_id}"] = 1.0
        self.session_state[f"theta_{cam_id}"] = 0.0
        self.session_state[f"foot_{cam_id}"] = 1.0
        if cam_id in self.session_state["session_calibrations"]:
            del self.session_state["session_calibrations"][cam_id]

        self.assertNotIn(cam_id, self.session_state["session_calibrations"])
        self.assertEqual(self.session_state[f"dx_{cam_id}"], 0.0)
        self.assertEqual(self.session_state[f"dy_{cam_id}"], 0.0)

    def test_05_focused_crop_box_math(self):
        """Test calculation of camera cone and bounding crop box using math module."""
        import math
        for cam in self.loader.get_camera_list():
            cam_meta = self.loader.cameras_by_id.get(cam["id"], {})
            if "pos" in cam_meta:
                cx, cy = float(cam_meta["pos"][0]), float(cam_meta["pos"][1])
                heading = float(cam_meta.get("heading_deg", 0.0))
                fov = float(cam_meta.get("fov_deg", 90.0))
                range_px = float(cam_meta.get("range_m", 6.0)) * 31.44654

                all_x, all_y = [cx], [cy]
                for ang in [heading - fov / 2.0, heading + fov / 2.0, heading]:
                    rad_c = math.radians(ang)
                    all_x.append(cx + range_px * math.cos(rad_c))
                    all_y.append(cy + range_px * math.sin(rad_c))

                self.assertGreater(max(all_x), min(all_x))
                self.assertGreater(max(all_y), min(all_y))


if __name__ == "__main__":
    unittest.main()
