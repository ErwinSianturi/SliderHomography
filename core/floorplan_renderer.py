"""Floorplan Renderer Module.

Renders store layout either using an existing floorplan.png background image
or synthesizing high-precision vector maps directly from the 1,077 genuine
store regions (shelves, aisles, and Mayora products) stored in cameras_export.json.
Supports auto-cropping around camera FOV and rendering camera vision cones.
"""

import math
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class FloorplanRenderer:
    def __init__(
        self,
        regions: List[Dict[str, Any]],
        floorplan_image_path: Optional[str] = None,
        canvas_size: Tuple[int, int] = (3750, 3368),
    ):
        self.regions = regions
        self.canvas_size = canvas_size
        self.floorplan_image_path = floorplan_image_path
        self._cached_base_map: Optional[Image.Image] = None

    def get_base_map(self) -> Image.Image:
        """Returns background map: loads floorplan.png if exists, else renders vector map."""
        if self._cached_base_map is not None:
            return self._cached_base_map.copy()

        if self.floorplan_image_path and os.path.exists(self.floorplan_image_path):
            try:
                img = Image.open(self.floorplan_image_path).convert("RGB")
                if img.size == self.canvas_size:
                    self._cached_base_map = img
                    return img.copy()
            except Exception:
                pass

        # Synthesize vector map from genuine polygon regions
        W, H = self.canvas_size
        canvas = Image.new("RGB", (W, H), (248, 249, 250))
        draw = ImageDraw.Draw(canvas)

        # 1. Draw aisles first (background walkways)
        for r in self.regions:
            if r.get("type") == "aisle":
                pts = [tuple(p) for p in r.get("points", [])]
                if len(pts) >= 3:
                    draw.polygon(pts, fill=(238, 242, 246), outline=(203, 213, 225))

        # 2. Draw standard shelves
        for r in self.regions:
            r_type = r.get("type", "")
            if r_type == "shelf":
                pts = [tuple(p) for p in r.get("points", [])]
                if len(pts) >= 3:
                    draw.polygon(pts, fill=(226, 232, 240), outline=(148, 163, 184))

        # 3. Draw Mayora shelves (prominent color)
        for r in self.regions:
            if r.get("type") == "shelf_mayora":
                pts = [tuple(p) for p in r.get("points", [])]
                if len(pts) >= 3:
                    draw.polygon(pts, fill=(167, 243, 208), outline=(5, 150, 105))

        # 4. Draw checkouts and entrances
        for r in self.regions:
            r_type = r.get("type", "")
            if r_type in ("checkout", "entrance"):
                pts = [tuple(p) for p in r.get("points", [])]
                if len(pts) >= 3:
                    draw.polygon(pts, fill=(254, 215, 170), outline=(234, 88, 12))

        # 5. Draw text labels for aisles to make store layout clearly identifiable
        font_label = ImageFont.load_default()
        for r in self.regions:
            if r.get("type") == "aisle":
                pts = r.get("points", [])
                if len(pts) >= 3:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    cx_a = sum(xs) / len(xs)
                    cy_a = sum(ys) / len(ys)
                    draw.text((cx_a - 25, cy_a - 5), r.get("label", ""), fill=(100, 116, 139), font=font_label)

        self._cached_base_map = canvas
        return canvas.copy()

    def render_overlay(
        self,
        camera_meta: Optional[Dict[str, Any]],
        baseline_pts: np.ndarray,
        calibrated_pts: np.ndarray,
        pairs: Optional[List[Dict[str, Any]]] = None,
        crop_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> Image.Image:
        """Renders camera FOV cone, baseline points, calibrated points, and shift vectors."""
        base = self.get_base_map()
        draw = ImageDraw.Draw(base)

        # Draw camera position and viewing cone using correct Cartesian angles (0 = East/+X, 90 = South/+Y)
        if camera_meta and "pos" in camera_meta:
            cx, cy = float(camera_meta["pos"][0]), float(camera_meta["pos"][1])
            heading = float(camera_meta.get("heading_deg", 0.0))
            fov = float(camera_meta.get("fov_deg", 90.0))
            range_px = float(camera_meta.get("range_m", 6.0)) * 31.44654

            rad_center = math.radians(heading)
            rad_left = math.radians(heading - fov / 2.0)
            rad_right = math.radians(heading + fov / 2.0)

            # Cartesian projection: dx = cos(angle), dy = sin(angle)
            x1 = cx + range_px * math.cos(rad_left)
            y1 = cy + range_px * math.sin(rad_left)
            x2 = cx + range_px * math.cos(rad_right)
            y2 = cy + range_px * math.sin(rad_right)
            xc = cx + range_px * math.cos(rad_center)
            yc = cy + range_px * math.sin(rad_center)

            # Draw cone outline and directional line
            draw.polygon([(cx, cy), (x1, y1), (x2, y2)], outline=(220, 38, 38), fill=None)
            draw.line([(cx, cy), (xc, yc)], fill=(220, 38, 38), width=2)
            draw.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], fill=(220, 38, 38), outline=(255, 255, 255), width=2)
            # Camera label
            cam_lbl = camera_meta.get("label", camera_meta.get("id", "Camera"))
            draw.text((cx + 12, cy - 8), f"📷 {cam_lbl}", fill=(185, 28, 28), font=ImageFont.load_default())

        # Draw correspondence pairs (mapPt)
        if pairs:
            for p in pairs:
                if p.get("mapPt") is not None:
                    mx, my = p["mapPt"]
                    draw.rectangle([mx - 4, my - 4, mx + 4, my + 4], fill=(59, 130, 246), outline=(255, 255, 255))

        # Draw baseline points (Orange) & Calibrated points (Green)
        b_pts = np.asarray(baseline_pts, dtype=float)
        c_pts = np.asarray(calibrated_pts, dtype=float)
        R = 6

        for i, ((bx, by), (cx, cy)) in enumerate(zip(b_pts, c_pts), start=1):
            # Shift line
            draw.line([(bx, by), (cx, cy)], fill=(147, 51, 234), width=2)

            # Baseline dot (Orange)
            draw.ellipse([bx - R, by - R, bx + R, by + R], fill=(249, 115, 22), outline=(255, 255, 255), width=2)

            # Calibrated dot (Green)
            draw.ellipse([cx - R, cy - R, cx + R, cy + R], fill=(16, 185, 129), outline=(255, 255, 255), width=2)

        # Crop if requested
        if crop_box:
            x0, y0, x1, y1 = crop_box
            x0 = max(0, min(self.canvas_size[0] - 1, x0))
            y0 = max(0, min(self.canvas_size[1] - 1, y0))
            x1 = max(x0 + 10, min(self.canvas_size[0], x1))
            y1 = max(y0 + 10, min(self.canvas_size[1], y1))
            return base.crop((x0, y0, x1, y1))

        return base

    def render_calibrated_overlay(
        self,
        base_overlay_img: Image.Image,
        baseline_dets: List[Dict[str, Any]],
        calibrated_pts: np.ndarray,
        crop_bounds: Optional[Tuple[int, int, int, int]],
        frame_size: Tuple[int, int] = (640, 360),
        val_foot: float = 1.0,
        header_height: int = 64,
        gutter_width: int = 40,
        current_params: Optional[Dict[str, float]] = None,
        pairs: Optional[List[Dict[str, Any]]] = None,
    ) -> Image.Image:
        """Renders live calibration shift points directly on top of the pre-existing overlay image."""
        img = base_overlay_img.copy()
        if not crop_bounds or len(baseline_dets) == 0:
            return img

        draw = ImageDraw.Draw(img)
        font_label = ImageFont.load_default(size=14)
        font_badge = ImageFont.load_default(size=15)

        x0, y0, x1, y1 = crop_bounds
        fw, fh = frame_size
        crop_x_offset = fw + gutter_width
        crop_y_offset = header_height

        c_pts = np.asarray(calibrated_pts, dtype=float)
        R = 6
        any_shift = False

        # Draw correspondence pairs if provided
        if pairs:
            for p in pairs:
                pid = p.get("id")
                mpt = p.get("mapPt")
                if mpt is not None:
                    mx, my = float(mpt[0]), float(mpt[1])
                    if (x0 - 30 <= mx <= x1 + 30) and (y0 - 30 <= my <= y1 + 30):
                        mx_canv = crop_x_offset + (mx - x0)
                        my_canv = crop_y_offset + (my - y0)
                        draw.rectangle(
                            [mx_canv - 5, my_canv - 5, mx_canv + 5, my_canv + 5],
                            fill=(234, 179, 8),
                            outline=(255, 255, 255),
                            width=2,
                        )
                        draw.text(
                            (mx_canv + 7, my_canv - 7),
                            f"P#{pid}",
                            fill=(234, 179, 8),
                            font=font_label,
                            stroke_width=2,
                            stroke_fill=(0, 0, 0),
                        )

        for idx, d in enumerate(baseline_dets):
            if idx >= len(c_pts):
                break
            b_map = d.get("map_px")
            if not b_map:
                continue
            bx, by = float(b_map[0]), float(b_map[1])
            cx, cy = float(c_pts[idx, 0]), float(c_pts[idx, 1])

            dist = math.hypot(cx - bx, cy - by)
            if dist > 0.2:
                any_shift = True
                bx_canv = crop_x_offset + (bx - x0)
                by_canv = crop_y_offset + (by - y0)
                cx_canv = crop_x_offset + (cx - x0)
                cy_canv = crop_y_offset + (cy - y0)

                # Draw shift vector (Purple/Magenta)
                draw.line([(bx_canv, by_canv), (cx_canv, cy_canv)], fill=(168, 85, 247), width=3)

                # Draw new calibrated position (Vivid Cyan)
                draw.ellipse(
                    [cx_canv - R, cy_canv - R, cx_canv + R, cy_canv + R],
                    fill=(6, 182, 212),
                    outline=(255, 255, 255),
                    width=2,
                )
                draw.text(
                    (cx_canv + R + 3, cy_canv - R - 3),
                    str(d.get("n", idx + 1)),
                    fill=(6, 182, 212),
                    font=font_label,
                    stroke_width=2,
                    stroke_fill=(0, 0, 0),
                )

            # Foot ratio adjustment on frame side
            if abs(val_foot - 1.0) > 0.005:
                ltrb = d.get("bbox_native_ltrb")
                if ltrb:
                    l, t, r, b = ltrb
                    fy_new = t + (b - t) * val_foot
                    fx_new = (l + r) / 2.0
                    draw.ellipse(
                        [fx_new - 4, fy_new + header_height - 4, fx_new + 4, fy_new + header_height + 4],
                        fill=(6, 182, 212),
                        outline=(255, 255, 255),
                        width=2,
                    )

        # Header status badge
        if (any_shift or pairs) and current_params:
            dx = current_params.get("dx", 0.0)
            dy = current_params.get("dy", 0.0)
            sx = current_params.get("sx", 1.0)
            sy = current_params.get("sy", 1.0)
            th = current_params.get("theta_deg", 0.0)
            badge_text = f"CALIBRATED: ΔX={dx:+.0f}px | ΔY={dy:+.0f}px | Sx={sx:.2f} | Sy={sy:.2f} | θ={th:+.1f}°"
            bx_pos = max(10, img.width - 550)
            draw.rectangle([bx_pos - 6, 8, img.width - 10, 36], fill=(16, 185, 129), outline=(255, 255, 255), width=1)
            draw.text((bx_pos, 13), badge_text, fill=(255, 255, 255), font=font_badge)

        return img

    def render_dynamic_pairs_side_by_side(
        self,
        frame_img: Image.Image,
        pairs: List[Dict[str, Any]],
        camera_id: str = "",
        frame_name: str = "",
        highlight_pid: Optional[int] = None,
        margin: int = 80,
    ) -> Image.Image:
        """Dynamically renders side-by-side view of camera frame and floorplan crop with pairs.
        
        Reflects any manual pair modifications immediately in real-time.
        """
        base_floor = self.get_base_map()
        FW, FH = self.canvas_size

        drawn = [
            p for p in pairs
            if p.get("framePt") is not None and p.get("mapPt") is not None
        ]
        if not drawn:
            return frame_img

        W, H = frame_img.size
        mxs = [float(p["mapPt"][0]) for p in drawn]
        mys = [float(p["mapPt"][1]) for p in drawn]

        x0 = max(0, math.floor(min(mxs) - margin))
        y0 = max(0, math.floor(min(mys) - margin))
        x1 = min(FW, math.ceil(max(mxs) + margin))
        y1 = min(FH, math.ceil(max(mys) + margin))

        # Enforce minimum crop dimension for good context
        if (x1 - x0) < 300:
            cx = (x0 + x1) / 2.0
            x0 = max(0, math.floor(cx - 150))
            x1 = min(FW, math.ceil(cx + 150))
        if (y1 - y0) < 300:
            cy = (y0 + y1) / 2.0
            y0 = max(0, math.floor(cy - 150))
            y1 = min(FH, math.ceil(cy + 150))

        crop = base_floor.crop((x0, y0, x1, y1))
        cw, ch = crop.size

        gutter = 40
        header_h = 64
        canvas_w = W + gutter + cw
        canvas_h = header_h + max(H, ch)

        canvas = Image.new("RGB", (canvas_w, canvas_h), (248, 250, 252))
        canvas.paste(frame_img, (0, header_h))
        canvas.paste(crop, (W + gutter, header_h))

        draw = ImageDraw.Draw(canvas)
        font_header = ImageFont.load_default(size=16)
        font_label = ImageFont.load_default(size=14)

        # Header background and text
        draw.rectangle([0, 0, canvas_w, header_h], fill=(30, 41, 59))
        draw.text(
            (16, 10),
            f"📷 Kamera: {camera_id} | Frame: {frame_name} ({W}x{H} px) | {len(drawn)} Pasangan Titik (Pairs)",
            fill=(255, 255, 255),
            font=font_header,
        )
        draw.text(
            (16, 36),
            f"Denah Crop: X:[{x0}, {x1}], Y:[{y0}, {y1}] (Render Real-time Dinamis)",
            fill=(148, 163, 184),
            font=font_label,
        )

        palette = [
            (239, 68, 68), (34, 197, 94), (59, 130, 246), (249, 115, 22), (168, 85, 247),
            (20, 184, 166), (236, 72, 153), (234, 179, 8), (99, 102, 241), (14, 165, 233),
            (244, 63, 94), (132, 204, 22), (217, 70, 239), (245, 158, 11), (16, 185, 129),
        ]

        R = 7
        for i, p in enumerate(drawn):
            pid = p["id"]
            color = palette[i % len(palette)]
            is_hl = (pid == highlight_pid)

            fx, fy = float(p["framePt"][0]), float(p["framePt"][1])
            mx, my = float(p["mapPt"][0]), float(p["mapPt"][1])

            f_xy = (fx, fy + header_h)
            m_xy = (W + gutter + (mx - x0), header_h + (my - y0))

            # Connect line
            line_w = 4 if is_hl else 2
            draw.line([f_xy, m_xy], fill=color if not is_hl else (239, 68, 68), width=line_w)

            # Draw circles on frame and map
            for x, y in (f_xy, m_xy):
                rad = R + 4 if is_hl else R
                draw.ellipse(
                    [x - rad, y - rad, x + rad, y + rad],
                    fill=(239, 68, 68) if is_hl else color,
                    outline=(255, 255, 255),
                    width=3 if is_hl else 2,
                )
                lbl = f"P#{pid}" + (" [EDITED]" if is_hl else "")
                draw.text(
                    (x + rad + 4, y - rad - 2),
                    lbl,
                    fill=(239, 68, 68) if is_hl else (0, 0, 0),
                    font=font_label,
                    stroke_width=2,
                    stroke_fill=(255, 255, 255),
                )

        return canvas
