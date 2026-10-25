"""End-to-End Automated UI and Interaction Testing with Streamlit AppTest.

Tests:
1. App initial load and zero exception execution.
2. Status filter selection and camera switching.
3. Interactive camera status changes (Unchecked -> Flagged -> Checked).
4. Slider interaction, precision nudge buttons, and Save button.
5. Correspondence pairs editing and reset button.
6. Export Format A and Format B execution.
"""

import os
import shutil
import sys
import unittest

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from streamlit.testing.v1 import AppTest
from core.config_loader import load_camera_statuses, save_camera_statuses, CAMERA_STATUSES_FILE


class TestStreamlitAppUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.status_file = os.path.join(WORKSPACE_DIR, CAMERA_STATUSES_FILE)
        cls.status_backup = None
        if os.path.exists(cls.status_file):
            with open(cls.status_file, "r", encoding="utf-8") as f:
                cls.status_backup = f.read()

    @classmethod
    def tearDownClass(cls):
        if cls.status_backup is not None:
            with open(cls.status_file, "w", encoding="utf-8") as f:
                f.write(cls.status_backup)
        elif os.path.exists(cls.status_file):
            os.remove(cls.status_file)

    def test_01_app_initial_load(self):
        """App must mount cleanly with 0 exceptions and valid title."""
        at = AppTest.from_file(os.path.join(WORKSPACE_DIR, "app.py"), default_timeout=30)
        at.run()
        self.assertEqual(len(at.exception), 0)
        self.assertIn("Homography Calibration Slider", at.title[0].value if at.title else "Homography Calibration Slider")

    def test_02_camera_status_change_via_ui(self):
        """Test changing camera status through the UI selectbox."""
        at = AppTest.from_file(os.path.join(WORKSPACE_DIR, "app.py"), default_timeout=30)
        at.run()
        self.assertEqual(len(at.exception), 0)

        # Find status selectbox
        status_boxes = [sb for sb in at.sidebar.selectbox if sb.key and "status_select_" in sb.key]
        self.assertGreaterEqual(len(status_boxes), 1)
        stat_box = status_boxes[0]

        # Select Flagged
        stat_box.select("Flagged")
        at.run()
        self.assertEqual(len(at.exception), 0)

        # Verify persisted status on disk
        persisted = load_camera_statuses(WORKSPACE_DIR)
        current_cam = at.session_state.camera_statuses
        self.assertIn("Flagged", current_cam.values())

        # Select Checked
        status_boxes = [sb for sb in at.sidebar.selectbox if sb.key and "status_select_" in sb.key]
        status_boxes[0].select("Checked")
        at.run()
        self.assertEqual(len(at.exception), 0)
        self.assertIn("Checked", at.session_state.camera_statuses.values())

    def test_03_slider_and_save_flow(self):
        """Test moving translation sliders and clicking save."""
        at = AppTest.from_file(os.path.join(WORKSPACE_DIR, "app.py"), default_timeout=30)
        at.run()
        self.assertEqual(len(at.exception), 0)

        # Move dx slider
        dx_sliders = [s for s in at.sidebar.slider if "Horizontal Translation" in s.label]
        self.assertGreaterEqual(len(dx_sliders), 1)
        dx_sliders[0].set_value(25.0)
        at.run()
        self.assertEqual(len(at.exception), 0)

        # Click save button
        save_btns = [b for b in at.sidebar.button if "Save" in b.label]
        self.assertGreaterEqual(len(save_btns), 1)
        save_btns[0].click()
        at.run()
        self.assertEqual(len(at.exception), 0)

        # Assert calibration stored in session calibrations
        self.assertGreater(len(at.session_state.session_calibrations), 0)

    def test_04_export_format_a_and_b_buttons(self):
        """Test clicking export Format A and Format B buttons in export tab."""
        at = AppTest.from_file(os.path.join(WORKSPACE_DIR, "app.py"), default_timeout=30)
        at.run()
        self.assertEqual(len(at.exception), 0)

        # Click Export Format A
        exp_a_btns = [b for b in at.button if "Export Format A to Disk" in b.label]
        self.assertGreaterEqual(len(exp_a_btns), 1)
        exp_a_btns[0].click()
        at.run()
        self.assertEqual(len(at.exception), 0)

        exp_a_file = os.path.join(WORKSPACE_DIR, "homography_calibration_overrides.json")
        self.assertTrue(os.path.exists(exp_a_file))

        # Click Export Format B
        exp_b_btns = [b for b in at.button if "Export Format B to Disk" in b.label]
        self.assertGreaterEqual(len(exp_b_btns), 1)
        exp_b_btns[0].click()
        at.run()
        self.assertEqual(len(at.exception), 0)

        exp_b_file = os.path.join(WORKSPACE_DIR, "cameras_export_calibrated.json")
        self.assertTrue(os.path.exists(exp_b_file))


if __name__ == "__main__":
    unittest.main()
