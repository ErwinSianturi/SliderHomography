"""Exporter Module.

Exports calibrated camera homography in two distinct, verifiable formats:
1. Format A: Calibration Overrides & Experiment Audit Log (JSON).
2. Format B: Master Production-Compatible Configuration (cameras_export_calibrated.json),
   preserving all other master data while updating correspondence pairs.
"""

from datetime import datetime, timezone
import json
import os
from typing import Any, Dict, List, Optional
import numpy as np


class CalibrationExporter:
    def __init__(self, master_export_data: Dict[str, Any]):
        self.master_data = master_export_data

    def export_format_a_overrides(
        self,
        camera_id: str,
        camera_label: str,
        frame: str,
        mode: str,
        parameters: Dict[str, Any],
        baseline_H: Optional[np.ndarray],
        calibrated_H: Optional[np.ndarray],
        T_matrix: Optional[np.ndarray],
        metrics: Dict[str, Any],
        calibrated_pairs: List[Dict[str, Any]],
        output_path: str,
        camera_status: Optional[str] = None,
    ) -> str:
        """Exports Format A: Independent calibration overrides and experiment report."""
        record = {
            "_about": "Homography Calibration Slider Experiment Override Record",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "camera_id": camera_id,
            "camera_label": camera_label,
            "frame": frame,
            "camera_status": camera_status or "Unchecked",
            "calibration_mode": mode,
            "parameters": {
                "delta_x_px": float(parameters.get("dx", 0.0)),
                "delta_y_px": float(parameters.get("dy", 0.0)),
                "delta_x_m": round(float(parameters.get("dx", 0.0)) / 31.44654, 3),
                "delta_y_m": round(float(parameters.get("dy", 0.0)) / 31.44654, 3),
                "scale_x": float(parameters.get("sx", 1.0)),
                "scale_y": float(parameters.get("sy", 1.0)),
                "rotation_deg": float(parameters.get("theta_deg", 0.0)),
                "foot_ratio": float(parameters.get("foot_ratio", 1.0)),
            },
            "baseline_homography_matrix": (
                [[round(float(v), 8) for v in row] for row in baseline_H]
                if baseline_H is not None else None
            ),
            "calibrated_homography_matrix": (
                [[round(float(v), 8) for v in row] for row in calibrated_H]
                if calibrated_H is not None else None
            ),
            "transform_matrix_T": (
                [[round(float(v), 8) for v in row] for row in T_matrix]
                if T_matrix is not None else None
            ),
            "spatial_evaluation_metrics": metrics,
            "calibrated_pairs": calibrated_pairs,
        }

        # If file exists, append/update camera entry
        overrides = {}
        if os.path.exists(output_path):
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                    if isinstance(existing, dict) and "cameras" in existing:
                        overrides = existing
            except Exception:
                overrides = {}

        if not overrides:
            overrides = {
                "_about": "Collection of Homography Calibration Overrides",
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "cameras": {},
            }

        overrides["last_updated"] = datetime.now(timezone.utc).isoformat()
        overrides["cameras"][camera_id] = record

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        tmp_path = output_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(overrides, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, output_path)

        return output_path

    def export_format_b_master_compatible(
        self,
        calibrated_pairs_by_camera: Dict[str, List[Dict[str, Any]]],
        output_path: str,
        camera_statuses: Optional[Dict[str, str]] = None,
    ) -> str:
        """Exports Format B: Full cameras_export.json clone with updated pairs for calibrated cameras.
        
        Guarantees 100% schema compatibility with the production pipeline (stage2_project_draw.py).
        """
        import copy
        new_master = copy.deepcopy(self.master_data)
        now_iso = datetime.now(timezone.utc).isoformat()

        # Update homography section
        updated_cams = set(calibrated_pairs_by_camera.keys())
        for h in new_master.get("homography", []):
            cid = h.get("cameraId")
            if cid in updated_cams:
                h["pairs"] = calibrated_pairs_by_camera[cid]
                h["updatedAt"] = now_iso

        # Update camera review statuses if provided
        if camera_statuses:
            for c in new_master.get("cameras", []):
                cid = c.get("id")
                if cid in camera_statuses:
                    c["review_status"] = camera_statuses[cid]

        new_master["exportedAt"] = now_iso
        new_master["_calibration_notice"] = (
            f"Calibrated via Homography Slider at {now_iso} for cameras: {sorted(list(updated_cams))}"
        )

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        tmp_path = output_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(new_master, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, output_path)

        return output_path
