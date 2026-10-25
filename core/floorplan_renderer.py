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

    def render_annotated_frame(
        self,
        frame_img: Image.Image,
        baseline_dets: List[Dict[str, Any]],
        pairs: Optional[List[Dict[str, Any]]] = None,
        val_foot: float = 1.0,
        highlight_idx: Optional[int] = None,
    ) -> Image.Image:
        """Annotates camera frame with distinct, high-contrast bounding boxes, numbered badges,
        and foot-point markers matching the floorplan pins."""
        img = frame_img.convert("RGB").copy()
        draw = ImageDraw.Draw(img)
        font_large = ImageFont.load_default(size=18)
        font_small = ImageFont.load_default(size=13)

        # Draw pairs on frame if provided
        if pairs:
            for p in pairs:
                pid = p.get("id")
                fpt = p.get("framePt")
                if fpt is not None:
                    fx, fy = float(fpt[0]), float(fpt[1])
                    draw.rectangle([fx - 6, fy - 6, fx + 6, fy + 6], fill=(234, 179, 8), outline=(255, 255, 255), width=2)
                    draw.text((fx + 8, fy - 8), f"P#{pid}", fill=(234, 179, 8), font=font_small, stroke_width=2, stroke_fill=(0, 0, 0))

        palette = [
            (239, 68, 68),    # Red
            (34, 197, 94),    # Bright Green
            (14, 165, 233),   # Cyan / Sky
            (245, 158, 11),   # Amber
            (168, 85, 247),   # Purple
            (236, 72, 153),   # Pink
            (132, 204, 22),   # Lime
            (99, 102, 241),   # Indigo
            (20, 184, 166),   # Teal
            (249, 115, 22),   # Deep Orange
        ]

        from core.homography_engine import calculate_foot_point

        for idx, d in enumerate(baseline_dets, start=1):
            n = d.get("n", idx)
            color = palette[(n - 1) % len(palette)]
            is_hl = (highlight_idx == n)

            ltrb = d.get("bbox_native_ltrb")
            if ltrb and len(ltrb) == 4:
                x1, y1, x2, y2 = [float(v) for v in ltrb]
                bw = 5 if is_hl else 3
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=bw)

                # Draw top label badge
                conf = float(d.get("conf", 0.0))
                tag_text = f" #{n} ({conf*100:.0f}%) " if conf > 0 else f" #{n} "
                t_bbox = draw.textbbox((x1, max(0, y1 - 24)), tag_text, font=font_large)
                draw.rectangle([t_bbox[0], t_bbox[1], t_bbox[2], t_bbox[3]], fill=color)
                draw.text((x1, max(0, y1 - 24)), tag_text, fill=(255, 255, 255), font=font_large)

                # Draw foot point
                fx, fy = calculate_foot_point(x1, y1, x2, y2, foot_ratio=val_foot)
                R_foot = 7
                draw.ellipse([fx - R_foot, fy - R_foot, fx + R_foot, fy + R_foot], fill=color, outline=(255, 255, 255), width=2)
                draw.line([(fx - 12, fy), (fx + 12, fy)], fill=(255, 255, 255), width=2)
                draw.line([(fx, fy - 12), (fx, fy + 12)], fill=(255, 255, 255), width=2)
                draw.text((fx + 10, fy - 8), f"Kaki #{n}", fill=color, font=font_small, stroke_width=2, stroke_fill=(0, 0, 0))

        return img

    def render_smart_floorplan_crop(
        self,
        camera_meta: Dict[str, Any],
        baseline_dets: List[Dict[str, Any]],
        calibrated_pts: np.ndarray,
        pairs: Optional[List[Dict[str, Any]]] = None,
        zoom_margin: int = 350,
        show_fov: bool = True,
        show_labels: bool = True,
        show_minimap: bool = True,
        show_pairs: bool = True,
        show_vectors: bool = True,
        highlight_idx: Optional[int] = None,
    ) -> Image.Image:
        """Renders an expansive, ultra-clear high-resolution floorplan crop with synchronized person pins,
        visible shelf/aisle labels, camera FOV cone, and store locator minimap."""
        base_map = self.get_base_map()
        FW, FH = self.canvas_size

        # 1. Collect all points to determine bounding box
        xs: List[float] = []
        ys: List[float] = []

        pos = camera_meta.get("pos")
        if pos and len(pos) == 2:
            xs.append(float(pos[0]))
            ys.append(float(pos[1]))

        for d in baseline_dets:
            m = d.get("map_px")
            if m and len(m) == 2:
                xs.append(float(m[0]))
                ys.append(float(m[1]))

        c_arr = np.asarray(calibrated_pts, dtype=float)
        if len(c_arr) > 0 and c_arr.ndim == 2:
            for pt in c_arr:
                if not np.isnan(pt[0]) and not np.isnan(pt[1]):
                    xs.append(float(pt[0]))
                    ys.append(float(pt[1]))

        if show_pairs and pairs:
            for p in pairs:
                m = p.get("mapPt")
                if m and len(m) == 2:
                    xs.append(float(m[0]))
                    ys.append(float(m[1]))

        if not xs:
            xs = [FW / 2.0]
            ys = [FH / 2.0]

        # 2. Compute crop coordinates with minimum dimensions
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        margin = max(150, int(zoom_margin))
        x0 = max(0, math.floor(min_x - margin))
        y0 = max(0, math.floor(min_y - margin))
        x1 = min(FW, math.ceil(max_x + margin))
        y1 = min(FH, math.ceil(max_y + margin))

        # Enforce minimum dimension of 700x500 for generous context
        MIN_W, MIN_H = 700, 500
        if (x1 - x0) < MIN_W:
            cx_box = (x0 + x1) / 2.0
            x0 = max(0, math.floor(cx_box - MIN_W / 2.0))
            x1 = min(FW, math.ceil(cx_box + MIN_W / 2.0))
        if (y1 - y0) < MIN_H:
            cy_box = (y0 + y1) / 2.0
            y0 = max(0, math.floor(cy_box - MIN_H / 2.0))
            y1 = min(FH, math.ceil(cy_box + MIN_H / 2.0))

        # 3. Crop base floorplan
        crop = base_map.crop((x0, y0, x1, y1)).convert("RGBA")
        cw, ch = crop.size

        # Create overlay for transparent drawing
        overlay = Image.new("RGBA", (cw, ch), (255, 255, 255, 0))
        draw_ov = ImageDraw.Draw(overlay)
        draw = ImageDraw.Draw(crop)

        font_pin = ImageFont.load_default(size=15)
        font_shelf = ImageFont.load_default(size=12)
        font_header = ImageFont.load_default(size=14)
        font_small = ImageFont.load_default(size=11)

        # 4. Draw Camera FOV Cone & Position
        if show_fov and pos and len(pos) == 2:
            cam_x = float(pos[0]) - x0
            cam_y = float(pos[1]) - y0

            heading_deg = float(camera_meta.get("heading_deg", 0.0))
            fov_deg = float(camera_meta.get("fov_deg", 90.0))
            range_m = float(camera_meta.get("range_m", 6.5))
            range_px = range_m * 31.4465408805

            rad_center = math.radians(heading_deg)
            rad_left = math.radians(heading_deg - fov_deg / 2.0)
            rad_right = math.radians(heading_deg + fov_deg / 2.0)

            x_left = cam_x + range_px * math.cos(rad_left)
            y_left = cam_y + range_px * math.sin(rad_left)
            x_right = cam_x + range_px * math.cos(rad_right)
            y_right = cam_y + range_px * math.sin(rad_right)

            # Draw transparent FOV cone
            draw_ov.polygon([(cam_x, cam_y), (x_left, y_left), (x_right, y_right)], fill=(59, 130, 246, 45), outline=(37, 99, 235, 180))
            draw.line([(cam_x, cam_y), (cam_x + range_px * 0.9 * math.cos(rad_center), cam_y + range_px * 0.9 * math.sin(rad_center))], fill=(37, 99, 235), width=2)

            # Camera marker
            cam_id = camera_meta.get("id", "Camera")
            draw.ellipse([cam_x - 10, cam_y - 10, cam_x + 10, cam_y + 10], fill=(30, 41, 59), outline=(255, 255, 255), width=3)
            cam_tag = f"📷 {cam_id}"
            c_tag_box = draw.textbbox((cam_x + 14, cam_y - 10), cam_tag, font=font_header)
            draw.rounded_rectangle([c_tag_box[0] - 4, c_tag_box[1] - 2, c_tag_box[2] + 4, c_tag_box[3] + 2], radius=4, fill=(30, 41, 59))
            draw.text((cam_x + 14, cam_y - 10), cam_tag, fill=(255, 255, 255), font=font_header)

        # 5. Draw Nearby Shelf & Aisle Labels
        if show_labels and self.regions:
            seen_labels: Dict[str, Tuple[float, float]] = {}
            for r in self.regions:
                lbl = r.get("label", "").strip()
                if not lbl or len(lbl) < 2 or lbl.lower() in ("custom", "floor", "perimeter", "zone"):
                    continue
                pts = r.get("points", [])
                if not pts:
                    continue
                rx = [p[0] for p in pts]
                ry = [p[1] for p in pts]
                rcx = sum(rx) / len(rx)
                rcy = sum(ry) / len(ry)
                if (x0 + 40) <= rcx <= (x1 - 40) and (y0 + 30) <= rcy <= (y1 - 30):
                    dup = False
                    for ex_lbl, (ex_x, ex_y) in seen_labels.items():
                        if ex_lbl == lbl and math.hypot(rcx - ex_x, rcy - ex_y) < 140:
                            dup = True
                            break
                    if not dup and len(seen_labels) < 35:
                        seen_labels[lbl] = (rcx, rcy)

            for lbl, (rcx, rcy) in seen_labels.items():
                lx = int(rcx - x0)
                ly = int(rcy - y0)
                t_box = draw.textbbox((lx, ly), lbl, font=font_shelf)
                draw_ov.rounded_rectangle([t_box[0] - 4, t_box[1] - 2, t_box[2] + 4, t_box[3] + 2], radius=4, fill=(255, 255, 255, 215), outline=(148, 163, 184, 180))
                draw.text((lx, ly), lbl, fill=(15, 23, 42), font=font_shelf)

        # Combine overlay before drawing solid pins
        crop = Image.alpha_composite(crop, overlay).convert("RGB")
        draw = ImageDraw.Draw(crop)

        # 6. Draw Correspondence Pairs (mapPt)
        if show_pairs and pairs:
            for p in pairs:
                pid = p.get("id")
                m = p.get("mapPt")
                if m is not None:
                    pmx = float(m[0]) - x0
                    pmy = float(m[1]) - y0
                    if 0 <= pmx <= cw and 0 <= pmy <= ch:
                        draw.rectangle([pmx - 6, pmy - 6, pmx + 6, pmy + 6], fill=(234, 179, 8), outline=(255, 255, 255), width=2)
                        draw.text((pmx + 8, pmy - 8), f"P#{pid}", fill=(161, 98, 7), font=font_small, stroke_width=2, stroke_fill=(255, 255, 255))

        # 7. Draw Person Detection Points (#1, #2, #3...)
        palette = [
            (239, 68, 68),    # Red
            (34, 197, 94),    # Bright Green
            (14, 165, 233),   # Cyan / Sky
            (245, 158, 11),   # Amber
            (168, 85, 247),   # Purple
            (236, 72, 153),   # Pink
            (132, 204, 22),   # Lime
            (99, 102, 241),   # Indigo
            (20, 184, 166),   # Teal
            (249, 115, 22),   # Deep Orange
        ]

        for idx, d in enumerate(baseline_dets, start=1):
            n = d.get("n", idx)
            color = palette[(n - 1) % len(palette)]
            is_hl = (highlight_idx == n)

            b_map = d.get("map_px")
            if not b_map:
                continue

            bx_c = float(b_map[0]) - x0
            by_c = float(b_map[1]) - y0

            has_calib = False
            cx_c, cy_c = bx_c, by_c
            if idx - 1 < len(c_arr):
                pt = c_arr[idx - 1]
                if not np.isnan(pt[0]) and not np.isnan(pt[1]):
                    cx_c = float(pt[0]) - x0
                    cy_c = float(pt[1]) - y0
                    has_calib = True

            shift_dist = math.hypot(cx_c - bx_c, cy_c - by_c)

            if has_calib and shift_dist > 1.0 and show_vectors:
                # Draw shift arrow (Purple/Magenta)
                draw.line([(bx_c, by_c), (cx_c, cy_c)], fill=(168, 85, 247), width=4)
                ang = math.atan2(cy_c - by_c, cx_c - bx_c)
                arr_len = 10
                a1 = ang + math.pi * 0.85
                a2 = ang - math.pi * 0.85
                draw.polygon([(cx_c, cy_c), (cx_c + arr_len * math.cos(a1), cy_c + arr_len * math.sin(a1)), (cx_c + arr_len * math.cos(a2), cy_c + arr_len * math.sin(a2))], fill=(168, 85, 247))

                # Baseline point
                R_base = 7
                draw.ellipse([bx_c - R_base, by_c - R_base, bx_c + R_base, by_c + R_base], fill=(255, 255, 255), outline=color, width=2)
                draw.text((bx_c + 9, by_c - 9), f"#{n} (awal)", fill=(100, 116, 139), font=font_small, stroke_width=2, stroke_fill=(255, 255, 255))

                # Calibrated point
                R_calib = 13 if is_hl else 11
                draw.ellipse([cx_c - R_calib, cy_c - R_calib, cx_c + R_calib, cy_c + R_calib], fill=color, outline=(255, 255, 255), width=3)
                pin_lbl = f"#{n}"
                draw.text((cx_c - 6, cy_c - 7), pin_lbl, fill=(255, 255, 255), font=font_pin)
                draw.text((cx_c + R_calib + 4, cy_c - 12), f"#{n} Kalibrasi", fill=color, font=font_header, stroke_width=3, stroke_fill=(255, 255, 255))
            else:
                R_pin = 13 if is_hl else 11
                draw.ellipse([bx_c - R_pin, by_c - R_pin, bx_c + R_pin, by_c + R_pin], fill=color, outline=(255, 255, 255), width=3)
                draw.text((bx_c - 6, by_c - 7), f"#{n}", fill=(255, 255, 255), font=font_pin)
                draw.text((bx_c + R_pin + 4, by_c - 12), f"#{n}", fill=color, font=font_header, stroke_width=3, stroke_fill=(255, 255, 255))

        # 8. Draw Inset Minimap (Store Locator)
        if show_minimap:
            mw, mh = 170, 150
            try:
                thumb = base_map.resize((mw, mh), Image.Resampling.BILINEAR)
                sx = mw / float(FW)
                sy = mh / float(FH)
                rx0 = int(x0 * sx)
                ry0 = int(y0 * sy)
                rx1 = int(x1 * sx)
                ry1 = int(y1 * sy)

                d_thumb = ImageDraw.Draw(thumb)
                d_thumb.rectangle([0, 0, mw - 1, mh - 1], outline=(15, 23, 42), width=2)
                d_thumb.rectangle([rx0, ry0, rx1, ry1], outline=(239, 68, 68), width=3)
                d_thumb.rectangle([0, 0, mw - 1, 20], fill=(15, 23, 42))
                d_thumb.text((6, 3), "📍 Posisi di Toko", fill=(255, 255, 255), font=font_small)

                paste_x = cw - mw - 12
                paste_y = 12
                crop.paste(thumb, (paste_x, paste_y))
                draw.rectangle([paste_x, paste_y, paste_x + mw, paste_y + mh], outline=(30, 41, 59), width=2)
            except Exception:
                pass

        # 9. Top Info Bar
        bar_h = 32
        info_bar = Image.new("RGB", (cw, bar_h), (30, 41, 59))
        d_bar = ImageDraw.Draw(info_bar)
        d_bar.text(
            (12, 7),
            f"🗺️ Denah HD: X:[{x0}, {x1}] Y:[{y0}, {y1}] ({cw}x{ch} px) | Zoom Margin: {margin}px | {len(baseline_dets)} Deteksi",
            fill=(255, 255, 255),
            font=font_header,
        )
        final_img = Image.new("RGB", (cw, ch + bar_h), (30, 41, 59))
        final_img.paste(info_bar, (0, 0))
        final_img.paste(crop, (0, bar_h))

        return final_img
