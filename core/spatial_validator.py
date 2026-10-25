"""Spatial Validator Module.

Implements Point-in-Polygon validation using Shapely against store layout
regions (aisles and shelves). Quantifies calibration metrics (shift distance,
aisle containment change, and out-of-bounds rates).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from shapely.geometry import Point, Polygon


class SpatialValidator:
    def __init__(self, regions: List[Dict[str, Any]], scale_px_per_m: float = 31.4465408805):
        self.scale_px_per_m = scale_px_per_m
        self.aisles: List[Tuple[str, int, Polygon]] = []
        self.shelves: List[Tuple[str, int, Polygon]] = []
        self._build_polygons(regions)

    def _build_polygons(self, regions: List[Dict[str, Any]]):
        for r in regions:
            pts = r.get("points", [])
            if len(pts) >= 3:
                try:
                    poly = Polygon(pts)
                    if poly.is_valid and not poly.is_empty:
                        r_type = r.get("type", "")
                        label = r.get("label", "")
                        rid = r.get("id", 0)
                        if r_type == "aisle":
                            self.aisles.append((label, rid, poly))
                        elif "shelf" in r_type:
                            self.shelves.append((label, rid, poly))
                except Exception:
                    continue

    def find_aisle_for_point(self, x: float, y: float) -> Optional[str]:
        p = Point(x, y)
        for label, _, poly in self.aisles:
            if poly.contains(p):
                return label
        return None

    def find_region_for_point(self, x: float, y: float) -> Optional[str]:
        p = Point(x, y)
        # Check aisle first
        for label, _, poly in self.aisles:
            if poly.contains(p):
                return f"Aisle: {label}"
        # Check shelf
        for label, _, poly in self.shelves:
            if poly.contains(p):
                return f"Shelf: {label}"
        return "Outside Defined Regions"

    def evaluate_points(
        self,
        baseline_pts: np.ndarray,
        calibrated_pts: np.ndarray,
        intended_aisle: Optional[str] = None,
        store_bounds: Tuple[int, int] = (3750, 3368),
        boundary_tolerance_px: float = 1.0,
    ) -> Dict[str, Any]:
        """Evaluates calibration improvements between baseline and calibrated points.
        
        Strictly distinguishes aisle walkways from product shelves to ensure metrics
        are not artificially inflated by shelf penetration.
        """
        b_pts = np.asarray(baseline_pts, dtype=float)
        c_pts = np.asarray(calibrated_pts, dtype=float)
        count = len(b_pts)

        if count == 0:
            return {
                "count": 0,
                "mean_shift_px": 0.0,
                "mean_shift_m": 0.0,
                "median_shift_px": 0.0,
                "median_shift_m": 0.0,
                "out_of_bounds_baseline": 0,
                "out_of_bounds_calibrated": 0,
                "intended_aisle_baseline_pct": 0.0,
                "intended_aisle_calibrated_pct": 0.0,
                "intended_shelf_baseline_pct": 0.0,
                "intended_shelf_calibrated_pct": 0.0,
            }

        # Shift distances
        shifts_px = np.hypot(c_pts[:, 0] - b_pts[:, 0], c_pts[:, 1] - b_pts[:, 1])
        shifts_m = shifts_px / self.scale_px_per_m

        # Bounds checks
        W, H = store_bounds
        oob_b = np.sum((b_pts[:, 0] < 0) | (b_pts[:, 0] > W) | (b_pts[:, 1] < 0) | (b_pts[:, 1] > H))
        oob_c = np.sum((c_pts[:, 0] < 0) | (c_pts[:, 0] > W) | (c_pts[:, 1] < 0) | (c_pts[:, 1] > H))

        # Identify all relevant aisle and shelf polygons for intended target
        target_aisle_polys = []
        target_shelf_polys = []
        if intended_aisle:
            # Tokenize intended keywords, e.g. 'Instant & Import Noodles' -> ['instant', 'noodle']
            tokens = [w.lower().strip() for w in intended_aisle.replace("&", " ").replace("/", " ").split() if len(w) > 2]
            for label, _, poly in self.aisles:
                l_lower = label.lower()
                if any(t in l_lower for t in tokens):
                    target_aisle_polys.append(poly.buffer(boundary_tolerance_px))
            for label, _, poly in self.shelves:
                l_lower = label.lower()
                if any(t in l_lower for t in tokens):
                    target_shelf_polys.append(poly.buffer(boundary_tolerance_px))

        in_target_aisle_b = 0
        in_target_aisle_c = 0
        in_target_shelf_b = 0
        in_target_shelf_c = 0

        b_aisles = []
        c_aisles = []

        for (bx, by), (cx, cy) in zip(b_pts, c_pts):
            p_b = Point(bx, by)
            p_c = Point(cx, cy)

            b_aisle = self.find_aisle_for_point(bx, by)
            c_aisle = self.find_aisle_for_point(cx, cy)
            b_aisles.append(b_aisle)
            c_aisles.append(c_aisle)

            # Check target aisle walkways
            if any(p.contains(p_b) for p in target_aisle_polys):
                in_target_aisle_b += 1
            if any(p.contains(p_c) for p in target_aisle_polys):
                in_target_aisle_c += 1

            # Check target product shelves (distinct metric)
            if any(p.contains(p_b) for p in target_shelf_polys):
                in_target_shelf_b += 1
            if any(p.contains(p_c) for p in target_shelf_polys):
                in_target_shelf_c += 1

        return {
            "count": count,
            "mean_shift_px": float(np.mean(shifts_px)),
            "mean_shift_m": float(np.mean(shifts_m)),
            "median_shift_px": float(np.median(shifts_px)),
            "median_shift_m": float(np.median(shifts_m)),
            "max_shift_px": float(np.max(shifts_px)),
            "max_shift_m": float(np.max(shifts_m)),
            "out_of_bounds_baseline": int(oob_b),
            "out_of_bounds_calibrated": int(oob_c),
            "intended_aisle": intended_aisle,
            "intended_aisle_baseline_count": in_target_aisle_b,
            "intended_aisle_calibrated_count": in_target_aisle_c,
            "intended_aisle_baseline_pct": round(in_target_aisle_b / count * 100, 1),
            "intended_aisle_calibrated_pct": round(in_target_aisle_c / count * 100, 1),
            "intended_shelf_baseline_count": in_target_shelf_b,
            "intended_shelf_calibrated_count": in_target_shelf_c,
            "intended_shelf_baseline_pct": round(in_target_shelf_b / count * 100, 1),
            "intended_shelf_calibrated_pct": round(in_target_shelf_c / count * 100, 1),
            "baseline_aisles": b_aisles,
            "calibrated_aisles": c_aisles,
        }
