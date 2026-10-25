"""Homography Engine Module.

Provides robust Direct Linear Transformation (DLT) with Hartley normalization,
perspective projection, geometric transformation composition (translation,
scale, rotation), foot-point calculation, and condition-number stability checks.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def normalize_2d_points(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hartley isotropic normalization for numerical stability in SVD."""
    pts = np.asarray(pts, dtype=float)
    centroid = np.mean(pts, axis=0)
    shifted = pts - centroid
    mean_dist = np.mean(np.hypot(shifted[:, 0], shifted[:, 1]))
    scale = np.sqrt(2.0) / mean_dist if mean_dist > 1e-9 else 1.0

    T = np.array([
        [scale, 0.0, -scale * centroid[0]],
        [0.0, scale, -scale * centroid[1]],
        [0.0, 0.0, 1.0],
    ], dtype=float)

    pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=float)])
    pts_norm = (T @ pts_h.T).T
    return pts_norm[:, :2], T


def compute_homography_dlt(src_pts: np.ndarray, dst_pts: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Calculates 3x3 homography matrix H mapping src_pts -> dst_pts using Normalized DLT."""
    src = np.asarray(src_pts, dtype=float)
    dst = np.asarray(dst_pts, dtype=float)

    if len(src) < 4 or len(dst) < 4 or len(src) != len(dst):
        return None, {"error": "Need at least 4 point pairs", "valid": False}

    src_norm, T_src = normalize_2d_points(src)
    dst_norm, T_dst = normalize_2d_points(dst)

    A = []
    for (x, y), (X, Y) in zip(src_norm, dst_norm):
        A.append([-x, -y, -1.0, 0.0, 0.0, 0.0, x * X, y * X, X])
        A.append([0.0, 0.0, 0.0, -x, -y, -1.0, x * Y, y * Y, Y])

    A = np.array(A, dtype=float)
    _, s, Vh = np.linalg.svd(A)

    # Condition number check
    cond = s[0] / s[-1] if s[-1] > 1e-12 else float("inf")
    H_norm = Vh[-1].reshape(3, 3)

    try:
        T_dst_inv = np.linalg.inv(T_dst)
    except np.linalg.LinAlgError:
        return None, {"error": "Target normalization matrix singular", "valid": False}

    H = T_dst_inv @ H_norm @ T_src
    norm_factor = H[2, 2] if abs(H[2, 2]) > 1e-9 else (np.max(np.abs(H)) or 1.0)
    H = H / norm_factor

    # Calculate residuals
    residuals = []
    for (x, y), (X, Y) in zip(src, dst):
        p = H @ np.array([x, y, 1.0], dtype=float)
        if abs(p[2]) < 1e-9:
            residuals.append(float("inf"))
        else:
            residuals.append(float(np.hypot(p[0] / p[2] - X, p[1] / p[2] - Y)))

    info = {
        "valid": bool(cond < 1e8 and not np.isnan(H).any()),
        "condition_number": float(cond),
        "singular_values": [float(x) for x in s],
        "median_residual_px": float(np.median(residuals)) if residuals else 0.0,
        "max_residual_px": float(np.max(residuals)) if residuals else 0.0,
        "residuals": residuals,
    }
    return H, info


def project_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Projects 2D points using homography matrix H."""
    pts = np.asarray(pts, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, 2)
    pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=float)])
    proj_h = (H @ pts_h.T).T
    w = proj_h[:, 2:3]
    # Guard against division by near-zero
    safe_w = np.where(np.abs(w) < 1e-9, np.sign(w) * 1e-9 + (w == 0) * 1e-9, w)
    return proj_h[:, :2] / safe_w


def calculate_foot_point(l: float, t: float, r: float, b: float, foot_ratio: float = 1.0) -> Tuple[float, float]:
    """Calculates representative foot point from bounding box (l, t, r, b).
    
    foot_ratio = 1.0 -> bottom edge (bottom_center).
    foot_ratio = 0.92 -> 92% down from top to bottom (ankle level).
    """
    fx = (l + r) / 2.0
    fy = t + foot_ratio * (b - t)
    return fx, fy


def build_transform_matrix(
    dx: float = 0.0,
    dy: float = 0.0,
    sx: float = 1.0,
    sy: float = 1.0,
    theta_deg: float = 0.0,
    center: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Constructs 2D affine transformation matrix around a pivot center (cx, cy)."""
    cx, cy = center
    theta = math.radians(theta_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    # Move pivot to origin
    T_to_orig = np.array([
        [1.0, 0.0, -cx],
        [0.0, 1.0, -cy],
        [0.0, 0.0, 1.0],
    ], dtype=float)

    # Scale and Rotate
    R_S = np.array([
        [sx * cos_t, -sy * sin_t, 0.0],
        [sx * sin_t,  sy * cos_t, 0.0],
        [0.0,        0.0,         1.0],
    ], dtype=float)

    # Move pivot back and apply translation (dx, dy)
    T_back = np.array([
        [1.0, 0.0, cx + dx],
        [0.0, 1.0, cy + dy],
        [0.0, 0.0, 1.0],
    ], dtype=float)

    return T_back @ R_S @ T_to_orig


def apply_calibration_to_pairs(
    pairs: List[Dict[str, Any]],
    T_corr: np.ndarray,
) -> List[Dict[str, Any]]:
    """Applies map transformation T_corr to correspondence pairs."""
    new_pairs = []
    for p in pairs:
        p_copy = dict(p)
        if p.get("mapPt") is not None:
            mx, my = p["mapPt"]
            m_h = T_corr @ np.array([mx, my, 1.0], dtype=float)
            if abs(m_h[2]) > 1e-9:
                new_pairs.append({
                    "id": p.get("id"),
                    "framePt": list(p.get("framePt")) if p.get("framePt") is not None else None,
                    "mapPt": [round(float(m_h[0] / m_h[2]), 1), round(float(m_h[1] / m_h[2]), 1)],
                })
            else:
                new_pairs.append(p_copy)
        else:
            new_pairs.append(p_copy)
    return new_pairs
