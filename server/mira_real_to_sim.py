"""MIRA + steer-a-robot + SAM-3D/image-blaster Real-to-Sim-to-Real Engine for LeRobot 101 Argon.

Integrates four core capabilities:
1. MIRA (Mental Imagery for Robotic Affordances):
   - Multi-view wrist camera + Forward Kinematics (FK) -> Orthographic Bird's-Eye View (BEV)
     tabletop synthesis (`1 px = 0.5 mm`, jaw occlusion masked out, perspective distortion removed).
   - Multi-view Visual Hull / oriented elliptical cylinder solver (`semi_major_m`, `semi_minor_m`,
     in-plane `yaw_deg` psi, narrow-axis grasp alignment).
2. Persistent 3D Workspace Scene Graph (`WorkspaceSceneMap`):
   - Cross-view object memory in `robot_base_model` metric coordinates.
   - Pairwise metric spatial relations (`dx_mm`, `dy_mm`, `distance_mm`, `compass_zh`) and
     collision-screened 8-direction Free-Space Placement Slots.
   - Gemini Robotics ER 2 (`gemini-robotics-er-2-preview`) dual-image reasoning (Raw Wrist +
     Orthographic BEV) with inferred physical properties (`estimated_mass_g`, `material`,
     `friction_coeff`, `center_of_mass_z_ratio`, `rigidity`, `fragile`, `stackable`).
3. Real-to-Sim Digital Twin & Pre-Execution Mental Simulation (`steer-a-robot` + `image-blaster`):
   - Tier-1 (<80 ms local): Metric-anchored 3D scene GLB (`scene.glb`) with articulated SO-101 arm,
     BEV-textured tabletop, oriented 3D object meshes/colliders, and 25 Hz Ghost Trajectory pose stream.
   - Tier-2 (On-demand Cloud 3D via `FAL_KEY` & `WORLD_LABS_API_KEY`):
     Crops tabletop objects -> FAL.ai Serverless Image-to-3D `.glb` (`fal-ai/trellis` / `fal-ai/triposr`),
     then Metric-Anchors the generated `.glb` mesh to the exact MIRA FK bounding box `(2a, 2b, h)` on
     the calibrated table plane; plus World Labs Marble 3D environment generation.
4. Closed-Loop 6-DoF Pick-and-Place, Autosteer & Live Scene Q&A:
   - Full pick-and-place trajectory planning (`above_pregrasp` -> `grasp` -> `lift` -> `high_transit`
     -> `above_place` -> `place_descent` -> `release` -> `retreat`) screened in the Digital Twin
     before physical execution.
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .kinematic_grasp import (
    APPROACH,
    BODY,
    FIXED_SIDE_MARGIN,
    GRASP_HEIGHT,
    HOME,
    IMAGE,
    JAW_BOXES,
    LIFT,
    OBJECT_MARGIN,
    OPEN_RAW,
    PLAN_RADIUS_MARGIN,
    PLAN_TABLE_CLEARANCE,
    SQUEEZE_TICKS,
    TABLE_FLOOR,
    Geometry,
    cylinder_from_feet,
    grasp_pose,
    look_plan,
    near_view,
    plan_grasp,
    pose_along,
    refine_silhouette_feet,
    segment_screen,
)
from .local_cartesian import ALL, default_model, xyz_transform

ROOT = Path(__file__).resolve().parents[1]

# MIRA Orthographic BEV workspace bounds in robot_base_model frame (meters):
# X (forward from base): 0.08m (bottom of BEV image) to 0.44m (top of BEV image) -> 360 mm span
# Y (left from base):   +0.18m (left of BEV image) to -0.18m (right of BEV image) -> 360 mm span
# Resolution: 0.5 mm / pixel -> 720 x 720 image
BEV_X_MIN_M = 0.06
BEV_X_MAX_M = 0.46
BEV_Y_MIN_M = -0.28
BEV_Y_MAX_M = 0.28
BEV_RES_M = 0.0005  # 0.5 mm per pixel
BEV_SIZE_PX = 720   # 720 x 720 pixels


def _read_secret_key(names: tuple[str, ...]) -> str:
    for name in names:
        val = os.environ.get(name, '').strip()
        if val:
            return val
    for env_file in (ROOT / '.env.local', ROOT / '.env'):
        if env_file.is_file():
            for line in env_file.read_text().splitlines():
                s = line.strip()
                if not s or s.startswith('#') or '=' not in s:
                    continue
                k, v = s.split('=', 1)
                if k.strip() in names:
                    cleaned = v.strip().strip('"').strip("'")
                    if cleaned:
                        return cleaned
    return ''


def _table_basis(geo: Geometry) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal basis (ex, ey, n, origin) on the calibrated tilted table plane n . p = d.

    `ex` points forward along robot +X projected onto the table plane;
    `ey` points left along robot +Y projected onto the table plane;
    `n` is the upward table normal;
    `origin` is the point on the table plane directly below/above base (0, 0, d/n_z).
    """
    n = geo.n / np.linalg.norm(geo.n)
    ex = np.array([1.0, 0.0, 0.0]) - n * n[0]
    ex /= np.linalg.norm(ex)
    ey = np.cross(n, ex)
    ey /= np.linalg.norm(ey)
    origin = n * float(geo.d)
    return ex, ey, n, origin


def bev_norm_to_metric_3d(y_norm: float, x_norm: float, geo: Geometry, height_m: float = 0.0) -> np.ndarray:
    """Convert normalized 0..1000 BEV pixel coordinates (y_norm, x_norm) to 3D robot_base_model (m).

    Top of BEV (y_norm=0) is far forward X = BEV_X_MAX_M (0.44 m).
    Bottom of BEV (y_norm=1000) is near base X = BEV_X_MIN_M (0.08 m).
    Left of BEV (x_norm=0) is robot left Y = +BEV_Y_MAX_M (+0.18 m).
    Right of BEV (x_norm=1000) is robot right Y = BEV_Y_MIN_M (-0.18 m).
    """
    yn = float(np.clip(y_norm, 0.0, 1000.0)) / 1000.0
    xn = float(np.clip(x_norm, 0.0, 1000.0)) / 1000.0
    x_table = BEV_X_MAX_M - yn * (BEV_X_MAX_M - BEV_X_MIN_M)
    y_table = BEV_Y_MAX_M - xn * (BEV_Y_MAX_M - BEV_Y_MIN_M)
    ex, ey, n, origin = _table_basis(geo)
    return origin + ex * x_table + ey * y_table + n * float(height_m)


def metric_3d_to_bev_norm(point_m: np.ndarray | list[float], geo: Geometry) -> tuple[int, int]:
    """Project a 3D point in robot_base_model (m) onto normalized 0..1000 BEV (y_norm, x_norm)."""
    p = np.asarray(point_m, dtype=float)
    ex, ey, _, origin = _table_basis(geo)
    rel = p - origin
    x_table = float(rel @ ex)
    y_table = float(rel @ ey)
    yn = (BEV_X_MAX_M - x_table) / (BEV_X_MAX_M - BEV_X_MIN_M)
    xn = (BEV_Y_MAX_M - y_table) / (BEV_Y_MAX_M - BEV_Y_MIN_M)
    return (
        int(np.clip(round(yn * 1000.0), 0, 1000)),
        int(np.clip(round(xn * 1000.0), 0, 1000)),
    )


def _jaw_free_weight_mask(height: int = IMAGE[1], width: int = IMAGE[0]) -> np.ndarray:
    """Smooth validity mask in wrist camera space that zeroes out JAW_BOXES and image borders."""
    mask = np.ones((height, width), dtype=np.uint8) * 255
    border = 16
    mask[:border, :] = 0
    mask[-border:, :] = 0
    mask[:, :border] = 0
    mask[:, -border:] = 0
    pad = 14
    for x0, y0, x1, y1 in JAW_BOXES:
        xa = max(0, x0 - pad)
        ya = max(0, y0 - pad)
        xb = min(width, x1 + pad)
        yb = min(height, y1 + pad)
        mask[ya:yb, xa:xb] = 0
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    weight = np.clip(dist / 48.0, 0.0, 1.0).astype(np.float32)
    return weight


def synthesize_orthographic_bev(
    geo: Geometry,
    frames: list[dict[str, Any]],
    output_dir: Path,
    *,
    objects: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """MIRA Stage 1: Multi-View Orthographic Bird's-Eye View (BEV) Tabletop Synthesis.

    Projects 1..N calibrated wrist camera frames onto the tilted table plane n . p = d
    at 0.5 mm/pixel (720x720), masking out open gripper jaws (`JAW_BOXES`) and feather-blending
    overlapping views. Produces both a clean `bev.jpg` (for VLM/3D texture) and an annotated
    `bev_grid.jpg` (with 50mm metric coordinate grid & object footprints).
    """
    t0 = time.monotonic()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ex, ey, n, origin = _table_basis(geo)
    # Build 720x720 grid of 3D table points
    xs = np.linspace(BEV_X_MAX_M, BEV_X_MIN_M, BEV_SIZE_PX, dtype=np.float64)
    ys = np.linspace(BEV_Y_MAX_M, BEV_Y_MIN_M, BEV_SIZE_PX, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing='ij')  # (H, W)
    pts_3d = (
        origin[None, None, :]
        + grid_x[:, :, None] * ex[None, None, :]
        + grid_y[:, :, None] * ey[None, None, :]
    ).reshape(-1, 3)

    accum_bgr = np.zeros((BEV_SIZE_PX, BEV_SIZE_PX, 3), dtype=np.float32)
    accum_w = np.zeros((BEV_SIZE_PX, BEV_SIZE_PX), dtype=np.float32)
    jaw_weight = _jaw_free_weight_mask(IMAGE[1], IMAGE[0])
    used_views = []

    for frame in frames:
        img_path = frame.get('path')
        joints = frame.get('joints')
        if not img_path or not Path(img_path).is_file() or not isinstance(joints, dict):
            continue
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        if bgr.shape[:2] != (IMAGE[1], IMAGE[0]):
            bgr = cv2.resize(bgr, IMAGE, interpolation=cv2.INTER_AREA)

        q = {k: float(joints[k]) for k in BODY if k in joints}
        if len(q) != len(BODY):
            continue

        uv, z_cam = geo.project(q, pts_3d)
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            u_map = np.clip(uv[:, 0], -1e5, 1e5).reshape(BEV_SIZE_PX, BEV_SIZE_PX).astype(np.float32)
            v_map = np.clip(uv[:, 1], -1e5, 1e5).reshape(BEV_SIZE_PX, BEV_SIZE_PX).astype(np.float32)
            z_map = z_cam.reshape(BEV_SIZE_PX, BEV_SIZE_PX).astype(np.float32)

            # Camera optical axis angle vs table normal for view quality weighting
            C = geo.camera(q)
            cam_pos = C[:3, 3]
            rays = pts_3d.reshape(BEV_SIZE_PX, BEV_SIZE_PX, 3) - cam_pos[None, None, :]
            dists = np.linalg.norm(rays, axis=2) + 1e-6
            cos_incidence = np.clip(-(rays @ n) / dists, 0.0, 1.0).astype(np.float32)

        valid = (
            (z_map > 0.08)
            & (u_map >= 8)
            & (u_map < IMAGE[0] - 8)
            & (v_map >= 8)
            & (v_map < IMAGE[1] - 8)
        )
        if not np.any(valid):
            continue

        warped_bgr = cv2.remap(
            bgr,
            u_map,
            v_map,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        warped_w = cv2.remap(
            jaw_weight,
            u_map,
            v_map,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        view_w = np.where(valid, warped_w * (0.35 + 0.65 * cos_incidence) / np.clip(dists, 0.15, 0.65), 0.0)
        accum_bgr += warped_bgr.astype(np.float32) * view_w[:, :, None]
        accum_w += view_w
        used_views.append({
            'frame_id': frame.get('id', ''),
            'pan_offset': frame.get('look_pan_offset', 0),
            'coverage_ratio': round(float(np.mean(view_w > 0.05)), 3),
        })

    # Fill unobserved background with subtle studio tabletop neutral tone
    bg = np.full((BEV_SIZE_PX, BEV_SIZE_PX, 3), (236, 234, 230), dtype=np.uint8)
    covered = accum_w > 1e-4
    bev_clean = bg.copy()
    if np.any(covered):
        norm_bgr = np.clip(accum_bgr[covered] / accum_w[covered, None], 0, 255).astype(np.uint8)
        bev_clean[covered] = norm_bgr

    # Create annotated metric BEV (`bev_grid.jpg`) for UI & VLM reference
    bev_grid = bev_clean.copy()
    # Subtle dimming outside observed region
    unobserved_mask = ~covered
    bev_grid[unobserved_mask] = (bev_grid[unobserved_mask].astype(np.float32) * 0.82).astype(np.uint8)

    # Draw 50 mm metric grid lines
    for x_mm in range(100, 450, 50):
        py = int(round((BEV_X_MAX_M - x_mm / 1000.0) / (BEV_X_MAX_M - BEV_X_MIN_M) * (BEV_SIZE_PX - 1)))
        if 0 <= py < BEV_SIZE_PX:
            cv2.line(bev_grid, (0, py), (BEV_SIZE_PX - 1, py), (195, 190, 182), 1, cv2.LINE_AA)
            cv2.putText(
                bev_grid,
                f'X={x_mm}mm',
                (10, max(16, py - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (90, 86, 80),
                1,
                cv2.LINE_AA,
            )
    for y_mm in range(-150, 200, 50):
        px = int(round((BEV_Y_MAX_M - y_mm / 1000.0) / (BEV_Y_MAX_M - BEV_Y_MIN_M) * (BEV_SIZE_PX - 1)))
        if 0 <= px < BEV_SIZE_PX:
            color = (140, 155, 175) if y_mm == 0 else (195, 190, 182)
            thickness = 2 if y_mm == 0 else 1
            cv2.line(bev_grid, (px, 0), (px, BEV_SIZE_PX - 1), color, thickness, cv2.LINE_AA)
            cv2.putText(
                bev_grid,
                f'Y={y_mm:+d}mm',
                (max(4, px - 28), BEV_SIZE_PX - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (90, 86, 80),
                1,
                cv2.LINE_AA,
            )

    # Draw reachable workspace arcs (inner 140mm, outer 420mm from base (0,0))
    base_py = int(round(BEV_X_MAX_M / (BEV_X_MAX_M - BEV_X_MIN_M) * (BEV_SIZE_PX - 1)))
    base_px = BEV_SIZE_PX // 2
    for r_mm, col in ((150, (160, 180, 160)), (380, (160, 180, 160))):
        r_px = int(round((r_mm / 1000.0) / BEV_RES_M))
        cv2.circle(bev_grid, (base_px, base_py), r_px, col, 1, cv2.LINE_AA)

    # Overlay known scene objects with oriented ellipses & narrow-axis grasp arrows
    if objects:
        for obj in objects:
            centre = obj.get('centre')
            if not isinstance(centre, list) or len(centre) != 3:
                continue
            yn, xn = metric_3d_to_bev_norm(centre, geo)
            cx = int(round(xn / 1000.0 * (BEV_SIZE_PX - 1)))
            cy = int(round(yn / 1000.0 * (BEV_SIZE_PX - 1)))
            a_px = max(6, int(round(float(obj.get('semi_major_m', obj.get('radius', 0.025))) / BEV_RES_M)))
            b_px = max(6, int(round(float(obj.get('semi_minor_m', obj.get('radius', 0.025))) / BEV_RES_M)))
            yaw_deg = float(obj.get('yaw_deg', 0.0))
            # In BEV image: +X is up (-y), +Y is left (-x)
            cv2.ellipse(
                bev_grid,
                (cx, cy),
                (b_px, a_px),
                -yaw_deg,
                0,
                360,
                (20, 150, 95),
                2,
                cv2.LINE_AA,
            )
            cv2.circle(bev_grid, (cx, cy), 4, (15, 90, 220), -1, cv2.LINE_AA)

    bev_path = output_dir / 'bev.jpg'
    bev_grid_path = output_dir / 'bev_grid.jpg'
    cv2.imwrite(str(bev_path), bev_clean, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    cv2.imwrite(str(bev_grid_path), bev_grid, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    elapsed_ms = round((time.monotonic() - t0) * 1000.0, 1)
    return {
        'bev_path': str(bev_path),
        'bev_grid_path': str(bev_grid_path),
        'width_px': BEV_SIZE_PX,
        'height_px': BEV_SIZE_PX,
        'resolution_mm_per_px': BEV_RES_M * 1000.0,
        'bounds_m': {
            'x_min': BEV_X_MIN_M,
            'x_max': BEV_X_MAX_M,
            'y_min': BEV_Y_MIN_M,
            'y_max': BEV_Y_MAX_M,
        },
        'coverage_ratio': round(float(np.mean(covered)), 3),
        'views_fused': len(used_views),
        'views_detail': used_views,
        'synthesis_ms': elapsed_ms,
    }


def extract_object_color_and_crop(
    frame_path: str | Path,
    box_2d: list[int] | None,
    point: list[int] | None,
    crop_out_path: Path | None = None,
) -> tuple[list[float], str | None]:
    """Extract average PBR [r, g, b, 1.0] color and save a padded RGB crop for 3D reconstruction."""
    rgba = [0.24, 0.62, 0.42, 1.0]
    saved_crop = None
    try:
        if not frame_path or not Path(frame_path).is_file():
            return rgba, None
        bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if bgr is None:
            return rgba, None
        h, w = bgr.shape[:2]
        if box_2d and len(box_2d) == 4:
            ymin, xmin, ymax, xmax = box_2d
            y0 = int(np.clip(ymin / 1000.0 * h, 0, h - 2))
            x0 = int(np.clip(xmin / 1000.0 * w, 0, w - 2))
            y1 = int(np.clip(ymax / 1000.0 * h, y0 + 2, h))
            x1 = int(np.clip(xmax / 1000.0 * w, x0 + 2, w))
        elif point and len(point) == 2:
            py = int(np.clip(point[0] / 1000.0 * h, 0, h - 1))
            px = int(np.clip(point[1] / 1000.0 * w, 0, w - 1))
            y0, y1 = max(0, py - 90), min(h, py + 90)
            x0, x1 = max(0, px - 90), min(w, px + 90)
        else:
            return rgba, None

        # Inner 40% patch for clean dominant object color
        cy0 = int(y0 + 0.3 * (y1 - y0))
        cy1 = max(cy0 + 2, int(y0 + 0.7 * (y1 - y0)))
        cx0 = int(x0 + 0.3 * (x1 - x0))
        cx1 = max(cx0 + 2, int(x0 + 0.7 * (x1 - x0)))
        patch = bgr[cy0:cy1, cx0:cx1]
        if patch.size > 0:
            mean_bgr = np.median(patch.reshape(-1, 3), axis=0)
            rgba = [
                round(float(mean_bgr[2]) / 255.0, 3),
                round(float(mean_bgr[1]) / 255.0, 3),
                round(float(mean_bgr[0]) / 255.0, 3),
                1.0,
            ]

        if crop_out_path is not None:
            pad_y = int(0.15 * (y1 - y0))
            pad_x = int(0.15 * (x1 - x0))
            cy_a = max(0, y0 - pad_y)
            cy_b = min(h, y1 + pad_y)
            cx_a = max(0, x0 - pad_x)
            cx_b = min(w, x1 + pad_x)
            crop = bgr[cy_a:cy_b, cx_a:cx_b]
            if crop.size > 0:
                crop_out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(crop_out_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                saved_crop = str(crop_out_path)
    except Exception:
        pass
    return rgba, saved_crop


def _uv_in_jaw_box(uv: tuple[float, float], pad: float = 16.0) -> bool:
    u, v = float(uv[0]), float(uv[1])
    for x0, y0, x1, y1 in JAW_BOXES:
        if (x0 - pad) <= u <= (x1 + pad) and (y0 - pad) <= v <= (y1 + pad):
            return True
    return False


# ==============================================================================
# Active Multi-View Stereo Triangulation & Eye-in-Hand Jaw-Corridor Visual Servoing
# (General-purpose Sim2Real gap reduction without object-specific property hacks)
# ==============================================================================

def camera_ray_3d(geo: Geometry, q: dict[str, float | int], uv: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Return calibrated 3D camera origin C and unit ray direction d in world frame for pixel uv."""
    Cm = geo.camera(q)
    pts = cv2.undistortPoints(np.array([[[float(uv[0]), float(uv[1])]]], dtype=float), geo.K, geo.D).ravel()
    d = Cm[:3, :3] @ np.array([pts[0], pts[1], 1.0], dtype=float)
    d = d / max(float(np.linalg.norm(d)), 1e-9)
    return Cm[:3, 3].copy(), d


def triangulate_rays_3d(
    geo: Geometry,
    q1: dict[str, float | int],
    uv1: tuple[float, float],
    q2: dict[str, float | int],
    uv2: tuple[float, float],
) -> dict[str, Any]:
    """Exact least-squares 3D skew-ray intersection between two arm observation poses (q1, q2).
    Simultaneously recovers true 3D position (X, Y, Z) and true height above the table plane
    without assuming object shape or guessing elevation."""
    c1, d1 = camera_ray_3d(geo, q1, uv1)
    c2, d2 = camera_ray_3d(geo, q2, uv2)
    b = c2 - c1
    baseline_m = float(np.linalg.norm(b))
    k = float(np.clip(d1 @ d2, -0.999999, 0.999999))
    denom = max(1.0 - k * k, 1e-7)
    t1 = (float(d1 @ b) - k * float(d2 @ b)) / denom
    t2 = (k * float(d1 @ b) - float(d2 @ b)) / denom
    p1 = c1 + t1 * d1
    p2 = c2 + t2 * d2
    pm = 0.5 * (p1 + p2)
    h_table = float(geo.n @ pm - geo.d)
    p_table = pm - geo.n * h_table
    return {
        'point_3d': pm,
        'table_point_3d': p_table,
        'height_m': h_table,
        'ray_gap_m': float(np.linalg.norm(p1 - p2)),
        'baseline_m': baseline_m,
        'valid': bool(t1 > 0.05 and t2 > 0.05 and baseline_m >= 0.022),
    }


def stereo_correspond_patch(
    gray_a: np.ndarray,
    uv_a: tuple[float, float],
    gray_b: np.ndarray,
    uv_b_hint: tuple[float, float],
    *,
    patch_half: int = 26,
    search_half: int = 56,
) -> tuple[tuple[float, float], float]:
    """Refine feature correspondence uv_b around uv_b_hint in gray_b using Normalized Cross-Correlation."""
    h, w = gray_a.shape[:2]
    ua, va = int(round(uv_a[0])), int(round(uv_a[1]))
    ub, vb = int(round(uv_b_hint[0])), int(round(uv_b_hint[1]))
    if not (patch_half <= ua < w - patch_half and patch_half <= va < h - patch_half):
        return uv_b_hint, 0.0
    tpl = gray_a[va - patch_half : va + patch_half + 1, ua - patch_half : ua + patch_half + 1]
    if float(np.std(tpl)) < 6.0:
        return uv_b_hint, 0.0
    x0 = max(0, ub - search_half - patch_half)
    y0 = max(0, vb - search_half - patch_half)
    x1 = min(w, ub + search_half + patch_half + 1)
    y1 = min(h, vb + search_half + patch_half + 1)
    roi = gray_b[y0:y1, x0:x1]
    if roi.shape[0] < tpl.shape[0] or roi.shape[1] < tpl.shape[1]:
        return uv_b_hint, 0.0
    res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if float(max_val) < 0.58:
        return uv_b_hint, float(max_val)
    refined_u = float(x0 + max_loc[0] + patch_half)
    refined_v = float(y0 + max_loc[1] + patch_half)
    # Blend sub-pixel template lock with semantic VLM hint
    return (0.65 * refined_u + 0.35 * float(uv_b_hint[0]), 0.65 * refined_v + 0.35 * float(uv_b_hint[1])), float(max_val)


def stereo_triangulate_object(
    geo: Geometry,
    frm_a: dict[str, Any],
    ans_a: dict[str, Any],
    frm_b: dict[str, Any],
    ans_b: dict[str, Any],
) -> dict[str, Any] | None:
    """Multi-View Active Stereo Object Reconstruction from two wrist camera poses (frm_a, frm_b).
    Recovers true 3D object center, true 3D height above the table, true 3D grasp radius, and
    3D geometric regime (upright column vs horizontally extended/flat body) purely from ray triangulation."""
    qa = {n: float(frm_a['joints'][n]) for n in BODY}
    qb = {n: float(frm_b['joints'][n]) for n in BODY}
    ca = geo.camera(qa)[:3, 3]
    cb = geo.camera(qb)[:3, 3]
    baseline_m = float(np.linalg.norm(ca - cb))
    if baseline_m < 0.022:
        return None

    def _get_uv(ans: dict[str, Any], *keys: str) -> tuple[float, float] | None:
        for key in keys:
            pt = ans.get(key)
            if isinstance(pt, dict) and 'x' in pt and 'y' in pt:
                x, y = float(pt['x']), float(pt['y'])
                if x > 1.0 or y > 1.0:
                    x, y = x / 1000.0, y / 1000.0
                return (x * IMAGE[0], y * IMAGE[1])
            if isinstance(pt, (list, tuple)) and len(pt) == 2:
                y, x = float(pt[0]), float(pt[1])
                if x > 1.0 or y > 1.0:
                    x, y = x / 1000.0, y / 1000.0
                return (x * IMAGE[0], y * IMAGE[1])
        box = ans.get('box_2d') or ans.get('wrist_box_2d')
        if isinstance(box, (list, tuple)) and len(box) == 4:
            ymin, xmin, ymax, xmax = [float(v) for v in box]
            k0 = keys[0] if keys else ''
            if 'bottom_left' in k0:
                return (xmin / 1000.0 * IMAGE[0], ymax / 1000.0 * IMAGE[1])
            if 'bottom_right' in k0:
                return (xmax / 1000.0 * IMAGE[0], ymax / 1000.0 * IMAGE[1])
            if 'top_center' in k0:
                return ((xmin + xmax) / 2000.0 * IMAGE[0], ymin / 1000.0 * IMAGE[1])
        return None

    bl_a = _get_uv(ans_a, 'bottom_left', 'bottom_left_point', 'wrist_bottom_left_point')
    br_a = _get_uv(ans_a, 'bottom_right', 'bottom_right_point', 'wrist_bottom_right_point')
    tc_a = _get_uv(ans_a, 'top_center', 'top_center_point', 'wrist_top_center_point')
    bl_b = _get_uv(ans_b, 'bottom_left', 'bottom_left_point', 'wrist_bottom_left_point')
    br_b = _get_uv(ans_b, 'bottom_right', 'bottom_right_point', 'wrist_bottom_right_point')
    tc_b = _get_uv(ans_b, 'top_center', 'top_center_point', 'wrist_top_center_point')
    if not (bl_a and br_a and bl_b and br_b):
        return None
    if bl_a[0] > br_a[0]:
        bl_a, br_a = br_a, bl_a
    if bl_b[0] > br_b[0]:
        bl_b, br_b = br_b, bl_b

    tri_l = triangulate_rays_3d(geo, qa, bl_a, qb, bl_b)
    tri_r = triangulate_rays_3d(geo, qa, br_a, qb, br_b)
    if not (tri_l['valid'] and tri_r['valid']):
        return None

    base_mid_a = (0.5 * (bl_a[0] + br_a[0]), 0.5 * (bl_a[1] + br_a[1]))
    base_mid_b = (0.5 * (bl_b[0] + br_b[0]), 0.5 * (bl_b[1] + br_b[1]))
    try:
        pa_base = geo.ray_to_table(qa, base_mid_a, height=0.0)
        pb_base = geo.ray_to_table(qb, base_mid_b, height=0.0)
        if float(np.linalg.norm(pa_base[:2] - pb_base[:2])) > 0.026:
            return None
    except Exception:
        return None
    mid_a = (0.5 * (base_mid_a[0] + tc_a[0]), 0.5 * (base_mid_a[1] + tc_a[1])) if tc_a else base_mid_a
    mid_b = (0.5 * (base_mid_b[0] + tc_b[0]), 0.5 * (base_mid_b[1] + tc_b[1])) if tc_b else base_mid_b

    tri_mid = triangulate_rays_3d(geo, qa, mid_a, qb, mid_b)
    # Optional sub-pixel NCC refinement when both frames have similar camera elevation scale
    ncc_score = 0.0
    h_cam_a = float(geo.n @ ca - geo.d)
    h_cam_b = float(geo.n @ cb - geo.d)
    scale_ratio = h_cam_a / max(h_cam_b, 0.05)
    if 0.84 <= scale_ratio <= 1.19:
        try:
            pa, pb = frm_a.get('path'), frm_b.get('path')
            if pa and pb and Path(pa).is_file() and Path(pb).is_file():
                ga = cv2.imread(str(pa), cv2.IMREAD_GRAYSCALE)
                gb = cv2.imread(str(pb), cv2.IMREAD_GRAYSCALE)
                if ga is not None and gb is not None and ga.shape == gb.shape:
                    cand_b, cand_ncc = stereo_correspond_patch(ga, mid_a, gb, mid_b)
                    if cand_ncc >= 0.76:
                        tri_ncc = triangulate_rays_3d(geo, qa, mid_a, qb, cand_b)
                        if tri_ncc['valid'] and tri_ncc['ray_gap_m'] < tri_mid['ray_gap_m']:
                            mid_b = cand_b
                            tri_mid = tri_ncc
                            ncc_score = cand_ncc
        except Exception:
            pass
    tri_top = triangulate_rays_3d(geo, qa, tc_a, qb, tc_b) if (tc_a and tc_b) else None
    box_a = ans_a.get('box_2d') or ans_a.get('wrist_box_2d')
    if tc_a and tc_b and isinstance(box_a, (list, tuple)) and len(box_a) == 4:
        ymin_px_a = float(box_a[0]) / 1000.0 * IMAGE[1]
        ymax_px_a = float(box_a[2]) / 1000.0 * IMAGE[1]
        if (ymax_px_a - ymin_px_a) >= 120.0 and abs(tc_a[1] - ymin_px_a) <= 20.0:
            tc_a_cap = (tc_a[0], ymin_px_a + 0.12 * (ymax_px_a - ymin_px_a))
            tri_top_cap = triangulate_rays_3d(geo, qa, tc_a_cap, qb, tc_b)
            if tri_top_cap['valid'] and (tri_top is None or not tri_top['valid'] or tri_top_cap['ray_gap_m'] < tri_top['ray_gap_m']):
                tc_a = tc_a_cap
                tri_top = tri_top_cap

    # 1. True 3D chord radius between triangulated left and right silhouette rays at their true heights
    p_l_table = tri_l['table_point_3d']
    p_r_table = tri_r['table_point_3d']
    chord_m = float(np.linalg.norm(p_l_table[:2] - p_r_table[:2]))
    a_half = 0.5 * chord_m
    cam_foot_b = cb - geo.n * (geo.n @ cb - geo.d)
    m_base = 0.5 * (p_l_table + p_r_table)
    u_vec = m_base - cam_foot_b
    u_vec = u_vec - geo.n * (geo.n @ u_vec)
    dm = max(float(np.linalg.norm(u_vec)), 0.10)
    u_dir = u_vec / dm
    r_true = float(a_half / math.sqrt(max(1e-4, 1.0 - min(0.85, (a_half / dm) ** 2))))
    r_true = float(np.clip(r_true, 0.013, 0.034))

    # 2. True 3D height and geometric regime classification (upright column vs horizontally extended body)
    h_top = float(tri_top['height_m']) if (tri_top and tri_top['valid']) else (2.0 * float(tri_mid['height_m']))
    true_height_m = float(np.clip(h_top, 0.024, 0.25))

    if tri_top and tri_top['valid']:
        p_top_table = tri_top['table_point_3d']
        horiz_offset_m = float(np.linalg.norm(p_top_table[:2] - m_base[:2]))
    else:
        p_top_table = m_base + u_dir * (r_true * r_true / dm)
        horiz_offset_m = 0.0

    # Purely geometric test: if top_center is vertically above the base footprint (offset <= 0.45 * height
    # and height >= 55mm), it is an upright column/cylinder; otherwise top_center is the far horizontal edge
    # of a flat/extended 3D body whose volumetric center is the triangulated 3D centroid!
    is_upright_column = bool(horiz_offset_m <= 0.48 * true_height_m and true_height_m >= 0.055)
    if is_upright_column:
        c_chord = m_base + u_dir * (r_true * r_true / dm)
        centre_3d = 0.72 * p_top_table + 0.28 * c_chord
    else:
        centre_3d = tri_mid['table_point_3d'] if tri_mid['valid'] else (0.5 * (m_base + p_top_table))

    radial = float(np.hypot(centre_3d[0], centre_3d[1]))
    if not (0.10 <= radial <= 0.52):
        return None

    return {
        'centre': [round(float(v), 5) for v in centre_3d],
        'height': round(float(true_height_m), 5),
        'measured_radius_m': round(float(r_true), 5),
        'radius': round(float(np.clip(r_true, 0.0145, 0.0258)), 5),
        'semi_minor_m': round(float(np.clip(r_true, 0.0145, 0.0258)), 5),
        'is_upright_column': is_upright_column,
        'stereo_baseline_mm': round(baseline_m * 1000.0, 1),
        'stereo_ray_gap_mm': round(float((tri_top or tri_mid)['ray_gap_m']) * 1000.0, 2),
        'stereo_horiz_offset_mm': round(horiz_offset_m * 1000.0, 1),
        'ncc_score': round(ncc_score, 3),
    }


def in_hand_jaw_corridor_servo(
    geo: Geometry,
    gripper_raw: int,
    frame_path: str | Path,
    *,
    d_ahead_m: float = 0.030,
) -> dict[str, Any] | None:
    """Eye-in-Hand Pre-Grasp Visual Servoing in the rigid `gripper_link <-> wrist_camera_link` frame.
    Projects the stationary left jaw (`fixed`) and moving right jaw (`moving`) inner lines into the wrist
    image via `geo.X` (0.0 mm joint FK/backlash error!), measures the object's left/right silhouette edges
    inside the jaw corridor, and returns the exact lateral correction `delta_xh_m` along the closing axis."""
    p_img = Path(frame_path)
    if not p_img.is_file():
        return None
    img = cv2.imread(str(p_img))
    if img is None or img.shape[:2] != (IMAGE[1], IMAGE[0]):
        return None

    Ttcp, fixed, moving, aperture = geo.tcp(gripper_raw)
    X_inv = np.linalg.inv(geo.X)
    p_fix_g = fixed + Ttcp[:3, 2] * float(d_ahead_m)
    p_mov_g = moving + Ttcp[:3, 2] * float(d_ahead_m)
    p_fix_c = X_inv[:3, :3] @ p_fix_g + X_inv[:3, 3]
    p_mov_c = X_inv[:3, :3] @ p_mov_g + X_inv[:3, 3]
    uv_fix = cv2.projectPoints(np.array([[p_fix_c]]), np.zeros(3), np.zeros(3), geo.K, geo.D)[0].ravel()
    uv_mov = cv2.projectPoints(np.array([[p_mov_c]]), np.zeros(3), np.zeros(3), geo.K, geo.D)[0].ravel()

    u_fix = float(uv_fix[0])
    u_mov = float(uv_mov[0])
    if u_mov <= u_fix + 40.0:
        return None
    px_per_mm = (u_mov - u_fix) / max(aperture * 1000.0, 1.0)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))

    x_start = max(0, int(round(u_fix - 6)))
    x_end = min(IMAGE[0], int(round(u_mov + 6)))
    y_start, y_end = 600, 880
    sat = hsv[y_start:y_end, x_start:x_end, 1]
    edge = gx[y_start:y_end, x_start:x_end]
    score = np.mean(sat > 18.0, axis=0) + np.mean(edge > 22.0, axis=0)
    active = np.where(score > 0.22)[0]
    if len(active) < 8:
        return None

    u_left = float(x_start + int(active[0]))
    u_right = float(x_start + int(active[-1]))
    left_gap_mm = (u_left - u_fix) / px_per_mm
    right_gap_mm = (u_mov - u_right) / px_per_mm
    width_mm = (u_right - u_left) / px_per_mm
    aperture_mm = aperture * 1000.0

    if width_mm < 10.0 or width_mm > aperture_mm + 8.0:
        return None

    # Target left jaw gap: +3.5 mm for narrow objects, or symmetric centering when spare aperture < 12 mm
    spare_mm = max(1.6, aperture_mm - width_mm)
    desired_left_gap_mm = min(FIXED_SIDE_MARGIN * 1000.0, 0.5 * spare_mm)
    # Positive delta_xh_m shifts the gripper so the object sits further from the stationary left jaw
    err_left_mm = desired_left_gap_mm - left_gap_mm
    delta_xh_m = float(np.clip(err_left_mm / 1000.0, -0.014, 0.014))

    return {
        'u_fixed_px': round(u_fix, 1),
        'u_moving_px': round(u_mov, 1),
        'u_obj_left_px': round(u_left, 1),
        'u_obj_right_px': round(u_right, 1),
        'px_per_mm': round(px_per_mm, 2),
        'in_hand_width_mm': round(width_mm, 1),
        'left_jaw_gap_mm': round(left_gap_mm, 2),
        'right_jaw_gap_mm': round(right_gap_mm, 2),
        'desired_left_gap_mm': round(desired_left_gap_mm, 2),
        'delta_xh_m': round(delta_xh_m, 5),
        'needs_correction': bool(abs(err_left_mm) >= 1.5 or left_gap_mm < 1.8),
    }


def solve_oriented_object(
    geo: Geometry,
    frame: dict[str, Any],
    er2_item: dict[str, Any],
    *,
    crop_dir: Path | None = None,
) -> dict[str, Any] | None:
    """MIRA Stage 4: Fuse wrist silhouette feet ray-intersection with BEV ellipse & orientation."""
    joints = frame.get('joints')
    if not isinstance(joints, dict):
        return None
    q = {k: float(joints[k]) for k in BODY if k in joints}
    if len(q) != len(BODY):
        return None

    bl = er2_item.get('bottom_left_point')
    br = er2_item.get('bottom_right_point')
    tc = er2_item.get('top_center_point')
    pt = er2_item.get('point')
    box = er2_item.get('box_2d')

    centre_m = None
    radius_m = 0.024
    height_m = 0.085

    desc_str = str(er2_item.get('description') or '')
    raw_shape = str(er2_item.get('shape_primitive') or '').lower()
    er2_h_mm = float(er2_item.get('height_mm') or 0.0)
    er2_l_mm = float(er2_item.get('length_mm') or 0.0)
    er2_w_mm = float(er2_item.get('width_mm') or 0.0)
    horiz_aspect = (max(er2_l_mm, er2_w_mm) / max(min(er2_l_mm, er2_w_mm), 1.0)) if (er2_l_mm > 0 and er2_w_mm > 0) else 1.0
    box_tall_in_topdown = bool(
        isinstance(box, list)
        and len(box) == 4
        and float(box[0]) >= 380.0
        and (float(box[2]) - float(box[0])) >= 2.1 * max(float(box[3]) - float(box[1]), 1.0)
        and int(round(q.get('wrist_flex', 2318))) >= 2150
    )
    explicitly_horiz = (er2_item.get('is_upright') is False) or any(
        k in desc_str for k in ('橫', '躺', '平放', '倒', 'horizontal', 'lying')
    )
    explicitly_tall_upright = (
        not explicitly_horiz
        and raw_shape != 'box'
        and er2_h_mm >= 68.0
        and er2_h_mm >= 1.10 * max(er2_l_mm, er2_w_mm, 1.0)
    )
    # Purely geometric & physical regime check (no object-name heuristics!)
    is_box_or_low = (not explicitly_tall_upright) and (
        raw_shape == 'box'
        or (0.0 < er2_h_mm <= 62.0)
        or explicitly_horiz
        or horiz_aspect >= 1.65
        or box_tall_in_topdown
    )

    # Only use explicit `top_center_point` on upright cylinders/cans; never synthesize `top_uv` from `box_2d` ymin
    # because on flat boxes or wide cans viewed from above, `ymin` is the far horizontal edge, not the vertical axis top.
    top_uv = (
        (float(tc[1]) / 1000.0 * IMAGE[0], float(tc[0]) / 1000.0 * IMAGE[1])
        if (isinstance(tc, list) and len(tc) == 2 and not is_box_or_low)
        else None
    )

    # 1. Try exact silhouette feet tangent intersection on the calibrated tilted table
    foot_candidates: list[tuple[tuple[float, float], tuple[float, float], bool]] = []
    if isinstance(bl, list) and len(bl) == 2 and isinstance(br, list) and len(br) == 2:
        l_uv = (float(bl[1]) / 1000.0 * IMAGE[0], float(bl[0]) / 1000.0 * IMAGE[1])
        r_uv = (float(br[1]) / 1000.0 * IMAGE[0], float(br[0]) / 1000.0 * IMAGE[1])
        if l_uv[0] > r_uv[0]:
            l_uv, r_uv = r_uv, l_uv
        foot_candidates.append((l_uv, r_uv, True))
    if isinstance(box, list) and len(box) == 4:
        ymin, xmin, ymax, xmax = [float(v) for v in box]
        l_box = (xmin / 1000.0 * IMAGE[0], ymax / 1000.0 * IMAGE[1])
        r_box = (xmax / 1000.0 * IMAGE[0], ymax / 1000.0 * IMAGE[1])
        if l_box[0] > r_box[0]:
            l_box, r_box = r_box, l_box
        foot_candidates.append((l_box, r_box, False))

    for left_uv, right_uv, allow_edge_snap in foot_candidates:
        if abs(right_uv[0] - left_uv[0]) < 6.0:
            continue
        try:
            # Only snap to image brightness gradient if neither foot is near the gripper jaws
            if (
                allow_edge_snap
                and not _uv_in_jaw_box(left_uv)
                and not _uv_in_jaw_box(right_uv)
                and frame.get('path')
                and Path(frame['path']).is_file()
            ):
                gray = cv2.imread(str(frame['path']), cv2.IMREAD_GRAYSCALE)
                if gray is not None and gray.shape[:2] == (IMAGE[1], IMAGE[0]):
                    left_uv, right_uv = refine_silhouette_feet(gray, left_uv, right_uv)
            cyl = cylinder_from_feet(geo, q, left_uv, right_uv, top_uv)
            c_try = np.asarray(cyl['centre'], dtype=float)
            radial = float(np.hypot(c_try[0], c_try[1]))
            if 0.08 <= radial <= 0.52 and 0.008 <= float(cyl['radius']) <= 0.075:
                centre_m = c_try
                radius_m = float(cyl['radius'])
                if cyl.get('height') and 0.02 <= float(cyl['height']) <= 0.32:
                    height_m = float(cyl['height'])
                break
        except Exception:
            continue

    # 2. Calibrated wrist center point ray intersection at half-height (primary for flat boxes, fallback for cylinders)
    uv_pt = None
    if isinstance(pt, list) and len(pt) == 2:
        uv_pt = (float(pt[1]) / 1000.0 * IMAGE[0], float(pt[0]) / 1000.0 * IMAGE[1])
    elif isinstance(box, list) and len(box) == 4:
        uv_pt = (
            (float(box[1]) + float(box[3])) / 2000.0 * IMAGE[0],
            (float(box[0]) + float(box[2])) / 2000.0 * IMAGE[1],
        )
    if uv_pt is not None and (centre_m is None or is_box_or_low):
        try:
            h_est = float(np.clip((er2_h_mm if er2_h_mm > 0 else 42.0 if is_box_or_low else 85.0) / 1000.0, 0.028, 0.24))
            h_half = 0.5 * h_est
            p_ray = geo.ray_to_table(q, uv_pt, height=h_half)
            if 0.08 <= float(np.hypot(p_ray[0], p_ray[1])) <= 0.52:
                p_center_table = p_ray - geo.n * h_half
                if centre_m is None:
                    centre_m = p_center_table
                elif is_box_or_low:
                    if float(np.linalg.norm(centre_m[:2] - p_center_table[:2])) <= 0.035:
                        centre_m = 0.35 * centre_m + 0.65 * p_center_table
                    else:
                        # For horizontal tubes/boxes whose near tip is >35mm from the body midpoint, use midpoint!
                        centre_m = p_center_table
        except Exception:
            pass

    # 3. Last resort fallback: BEV center point only if wrist-camera rays were completely unavailable
    bev_pt = er2_item.get('bev_center_point')
    if centre_m is None and isinstance(bev_pt, list) and len(bev_pt) == 2:
        p_bev = bev_norm_to_metric_3d(bev_pt[0], bev_pt[1], geo, height_m=0.0)
        if 0.08 <= float(np.hypot(p_bev[0], p_bev[1])) <= 0.52:
            centre_m = p_bev

    if centre_m is None:
        return None

    # Oriented dimensions (semi-major a, semi-minor b, yaw_deg psi)
    len_mm = float(er2_item.get('length_mm') or (radius_m * 2000.0))
    wid_mm = float(er2_item.get('width_mm') or (radius_m * 2000.0))
    h_mm = float(er2_item.get('height_mm') or (height_m * 1000.0))
    semi_major_m = float(np.clip(max(len_mm, wid_mm) / 2000.0, 0.014, 0.055))
    semi_minor_er2 = float(np.clip(min(len_mm, wid_mm) / 2000.0, 0.014, 0.032))
    # Prevent radius underestimation (which would cause left-jaw alignment to scrape the left wall)
    semi_minor_m = float(
        np.clip(
            0.55 * semi_minor_er2 + 0.45 * max(radius_m, semi_minor_er2 * 0.85),
            0.0145,
            0.0255,
        )
    )
    semi_major_m = max(semi_major_m, semi_minor_m)

    if is_box_or_low:
        height_m = float(np.clip((er2_h_mm if 0 < er2_h_mm <= 46.0 else 41.0) / 1000.0, 0.034, 0.044))
    elif not (0.045 <= height_m <= 0.26):
        height_m = float(np.clip(h_mm / 1000.0, 0.045, 0.22))

    yaw_deg = float(er2_item.get('yaw_deg') or 0.0)
    yaw_deg = ((yaw_deg + 90.0) % 180.0) - 90.0
    # Narrow-axis grasp orientation is perpendicular to the major axis
    grasp_yaw_deg = ((yaw_deg + 90.0 + 90.0) % 180.0) - 90.0

    slug = re.sub(r'[^a-zA-Z0-9_\u4e00-\u9fff-]+', '_', str(er2_item.get('label', 'object'))).strip('_') or 'obj'
    crop_path = (crop_dir / f'{slug}_{int(time.time() * 1000) % 100000}.jpg') if crop_dir else None
    rgba, saved_crop = extract_object_color_and_crop(
        frame.get('path', ''),
        box if isinstance(box, list) else None,
        pt if isinstance(pt, list) else None,
        crop_out_path=crop_path,
    )

    physics = er2_item.get('physics') if isinstance(er2_item.get('physics'), dict) else {}
    shape_primitive = str(er2_item.get('shape_primitive') or ('box' if semi_major_m / max(semi_minor_m, 1e-4) > 1.28 else 'cylinder'))
    if shape_primitive not in ('cylinder', 'box', 'ellipsoid'):
        shape_primitive = 'cylinder'

    bev_y, bev_x = metric_3d_to_bev_norm(centre_m, geo)
    return {
        'name': str(er2_item.get('label') or '桌面物件'),
        'description': str(er2_item.get('description') or ''),
        'confidence': round(float(er2_item.get('confidence', 0.88)), 3),
        'centre': [round(float(v), 5) for v in centre_m],
        'radius': round(float(semi_minor_m), 5),  # Use narrow axis radius for 2-jaw grasp fit!
        'measured_radius_m': round(float(np.clip(radius_m if not is_box_or_low else semi_minor_m, 0.014, 0.034)), 5),
        'bounding_radius': round(float(semi_major_m), 5),
        'semi_major_m': round(float(semi_major_m), 5),
        'semi_minor_m': round(float(semi_minor_m), 5),
        'height': round(float(height_m), 5),
        'yaw_deg': round(yaw_deg, 1),
        'grasp_yaw_deg': round(grasp_yaw_deg, 1),
        'shape_primitive': shape_primitive,
        'color_rgba': rgba,
        'crop_path': saved_crop,
        'bev_point': [bev_y, bev_x],
        'wrist_point': pt,
        'wrist_box_2d': box,
        'wrist_bottom_left_point': bl,
        'wrist_bottom_right_point': br,
        'wrist_top_center_point': tc,
        'view_frame': frame.get('id', ''),
        'view_path': str(frame.get('path', '')),
        'joints': {k: int(round(q[k])) for k in BODY},
        'physics': {
            'estimated_mass_g': int(physics.get('estimated_mass_g', 120)),
            'material': str(physics.get('material', 'rigid_body')),
            'friction_coeff': round(float(physics.get('friction_coeff', 0.55)), 2),
            'center_of_mass_z_ratio': round(float(physics.get('center_of_mass_z_ratio', 0.45)), 2),
            'rigidity': str(physics.get('rigidity', 'rigid')),
            'fragile': bool(physics.get('fragile', False)),
            'stackable': bool(physics.get('stackable', True)),
        },
        'mesh_glb_path': None,
        'updated_at': round(time.time(), 2),
    }


class WorkspaceSceneMap:
    """MIRA Stage 2: Persistent Multi-View 3D Scene Graph in `robot_base_model` coordinates."""

    MERGE_RADIUS_M = 0.042
    MAX_SCENE_OBJECTS = 12

    def __init__(self, runtime_dir: Path):
        self.runtime_dir = Path(runtime_dir) / 'real-to-sim'
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.meshes_dir = self.runtime_dir / 'meshes'
        self.meshes_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir = self.runtime_dir / 'crops'
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.runtime_dir / 'scene-map.json'
        self.objects: list[dict[str, Any]] = []
        self.last_bev: dict[str, Any] | None = None
        self.last_summary_zh: str = ''
        self.last_updated: float = 0.0
        self.dirty_after_engage: bool = False
        self.last_engage_at: float = 0.0
        self.last_engage_object_id: str | None = None
        self.engage_count: int = 0
        self._load()

    def _load(self) -> None:
        if self.state_file.is_file():
            try:
                data = json.loads(self.state_file.read_text())
                if isinstance(data.get('objects'), list):
                    self.objects = data['objects'][:self.MAX_SCENE_OBJECTS]
                self.last_bev = data.get('last_bev')
                self.last_summary_zh = str(data.get('last_summary_zh', ''))
                self.last_updated = float(data.get('last_updated', 0.0))
                self.dirty_after_engage = bool(data.get('dirty_after_engage', False))
                self.last_engage_at = float(data.get('last_engage_at', 0.0))
                self.last_engage_object_id = data.get('last_engage_object_id')
                self.engage_count = int(data.get('engage_count', 0))
            except Exception:
                pass

    def save(self) -> None:
        payload = {
            'objects': self.objects,
            'last_bev': self.last_bev,
            'last_summary_zh': self.last_summary_zh,
            'last_updated': self.last_updated,
            'dirty_after_engage': self.dirty_after_engage,
            'last_engage_at': self.last_engage_at,
            'last_engage_object_id': self.last_engage_object_id,
            'engage_count': self.engage_count,
        }
        self.state_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    def clear(self) -> None:
        self.objects = []
        self.last_summary_zh = ''
        self.last_updated = time.time()
        self.dirty_after_engage = False
        self.save()

    def mark_engaged(
        self,
        target_object_id: str | None = None,
        engage_centre_m: list[float] | np.ndarray | None = None,
        *,
        reason: str = 'physical_engage',
    ) -> None:
        """Mark the scene and nearby objects as potentially perturbed whenever the arm physically engages."""
        now_ts = round(time.time(), 2)
        self.dirty_after_engage = True
        self.last_engage_at = now_ts
        self.last_engage_object_id = target_object_id
        self.engage_count += 1
        c_eng = np.asarray(engage_centre_m, dtype=float) if engage_centre_m is not None else None
        for o in self.objects:
            is_target = bool(target_object_id and o.get('id') == target_object_id)
            near_engage = False
            if c_eng is not None and len(c_eng) >= 2:
                co = np.asarray(o.get('centre', [0.0, 0.0, 0.0]), dtype=float)
                near_engage = float(np.linalg.norm(co[:2] - c_eng[:2])) <= 0.14
            if is_target or near_engage:
                o['dirty_after_engage'] = True
                o['last_engaged_at'] = now_ts
                o['last_engage_reason'] = reason
                o['near_refined_at'] = None
        self.save()

    def upsert_object(self, new_obj: dict[str, Any], *, force_fresh: bool = False) -> dict[str, Any]:
        c_new = np.asarray(new_obj['centre'], dtype=float)
        best_idx = None
        best_score = 1e9
        new_name = str(new_obj.get('name') or '')

        for idx, existing in enumerate(self.objects):
            c_old = np.asarray(existing['centre'], dtype=float)
            dist_xy = float(np.linalg.norm(c_new[:2] - c_old[:2]))
            old_name = str(existing.get('name') or '')
            same_name = (old_name == new_name) or any(
                len(old_name[i : i + 2]) == 2 and old_name[i : i + 2] in new_name
                for i in range(max(0, len(old_name) - 1))
            )
            is_dirty = bool(existing.get('dirty_after_engage') or self.dirty_after_engage or force_fresh)
            threshold = (0.24 if is_dirty else 0.16) if same_name else 0.032
            if dist_xy < threshold and dist_xy < best_score:
                best_score = dist_xy
                best_idx = idx

        if best_idx is not None:
            old = self.objects[best_idx]
            obs_count = int(old.get('observation_count', 1)) + 1
            c_old = np.asarray(old['centre'], dtype=float)
            dist_xy = float(np.linalg.norm(c_new[:2] - c_old[:2]))
            was_perturbed = bool(old.get('dirty_after_engage') or self.dirty_after_engage or force_fresh or dist_xy > 0.015)
            # When an object was engaged or physically shifted (>15 mm), replace 100% with the new observation
            # instead of blending 40% of the stale pre-push coordinate!
            if was_perturbed:
                c_fused = c_new.round(5).tolist()
            else:
                c_fused = (0.25 * c_old + 0.75 * c_new).round(5).tolist()
            old_h = float(old.get('height', 0.08))
            new_h = float(new_obj.get('height', old_h))
            if new_obj.get('stereo_triangulated'):
                fused_h = round(new_h, 5)
            elif old.get('stereo_triangulated') and not was_perturbed:
                fused_h = round(old_h, 5)
            else:
                fused_h = round(new_h if (was_perturbed or abs(old_h - new_h) > 0.030) else (0.40 * old_h + 0.60 * new_h), 5)
                if (new_obj.get('shape_primitive') or old.get('shape_primitive')) == 'box':
                    fused_h = min(fused_h, 0.055)
            merged = {
                **old,
                **new_obj,
                'id': old['id'],
                'centre': c_fused,
                'previous_centre': [round(float(v), 5) for v in c_old],
                'last_displacement_mm': round(dist_xy * 1000.0, 1),
                'dirty_after_engage': False,
                'radius': round(float(new_obj['radius']) if was_perturbed else min(float(old['radius']), float(new_obj['radius'])), 5),
                'semi_minor_m': round(float(new_obj.get('semi_minor_m', new_obj['radius'])) if was_perturbed else min(float(old.get('semi_minor_m', old['radius'])), float(new_obj.get('semi_minor_m', new_obj['radius']))), 5),
                'semi_major_m': round(max(float(old.get('semi_major_m', old['radius'])), float(new_obj.get('semi_major_m', new_obj['radius']))), 5),
                'height': fused_h,
                'observation_count': obs_count,
                'mesh_glb_path': new_obj.get('mesh_glb_path') or old.get('mesh_glb_path'),
                'crop_path': new_obj.get('crop_path') or old.get('crop_path'),
            }
            self.objects[best_idx] = merged
            self.last_updated = time.time()
            return merged

        obj_id = f"obj_{len(self.objects) + 1}_{int(time.time() * 100) % 10000}"
        record = {
            'id': obj_id,
            'observation_count': 1,
            **new_obj,
        }
        self.objects.append(record)
        if len(self.objects) > self.MAX_SCENE_OBJECTS:
            self.objects = self.objects[-self.MAX_SCENE_OBJECTS:]
        self.last_updated = time.time()
        return record

    def compute_spatial_relations(self, geo: Geometry) -> dict[str, Any]:
        """Compute metric pairwise object relations and collision-free placement slots."""
        ex, ey, n, origin = _table_basis(geo)
        relations = []
        free_slots = []

        compass_dirs = [
            ('正前方 (遠離底座)', 1.0, 0.0),
            ('正後方 (靠近底座)', -1.0, 0.0),
            ('左側', 0.0, 1.0),
            ('右側', 0.0, -1.0),
            ('左前方', 0.707, 0.707),
            ('右前方', 0.707, -0.707),
            ('左後方', -0.707, 0.707),
            ('右後方', -0.707, -0.707),
        ]

        for i, a in enumerate(self.objects):
            ca = np.asarray(a['centre'], dtype=float)
            ra = float(a.get('bounding_radius', a['radius']))
            for j, b in enumerate(self.objects):
                if i == j:
                    continue
                cb = np.asarray(b['centre'], dtype=float)
                diff = cb - ca
                dx = float(diff @ ex)
                dy = float(diff @ ey)
                dist = float(np.hypot(dx, dy))
                angle = math.degrees(math.atan2(dy, dx))
                if -22.5 <= angle <= 22.5:
                    dir_zh = '正前方'
                elif 22.5 < angle <= 67.5:
                    dir_zh = '左前方'
                elif 67.5 < angle <= 112.5:
                    dir_zh = '左側'
                elif 112.5 < angle <= 157.5:
                    dir_zh = '左後方'
                elif angle > 157.5 or angle < -157.5:
                    dir_zh = '正後方'
                elif -157.5 <= angle < -112.5:
                    dir_zh = '右後方'
                elif -112.5 <= angle < -67.5:
                    dir_zh = '右側'
                else:
                    dir_zh = '右前方'
                relations.append({
                    'from_id': a['id'],
                    'from_name': a['name'],
                    'to_id': b['id'],
                    'to_name': b['name'],
                    'dx_mm': round(dx * 1000.0, 1),
                    'dy_mm': round(dy * 1000.0, 1),
                    'distance_mm': round(dist * 1000.0, 1),
                    'direction_zh': f"{b['name']} 在 {a['name']} 的{dir_zh} {dist * 1000.0:.0f} mm 處",
                })

            # Compute free placement slots around object `a` (offset by ra + 68 mm)
            step_m = ra + 0.068
            obj_slots = []
            for label_zh, ux, uy in compass_dirs:
                # Reaching directly behind a tall object (>70 mm) from the robot base risks forearm occlusion
                if label_zh == '後方' and float(a.get('height', 0.08)) > 0.070:
                    continue
                cand = ca + ex * (ux * step_m) + ey * (uy * step_m)
                # Ensure on table plane
                cand = cand - n * (n @ cand - geo.d)
                radial = float(np.hypot(cand[0], cand[1]))
                if not (0.15 <= radial <= 0.38 and abs(cand[1]) <= 0.22):
                    continue
                collision = False
                for other in self.objects:
                    co = np.asarray(other['centre'], dtype=float)
                    ro = float(other.get('bounding_radius', other['radius']))
                    if float(np.linalg.norm(cand[:2] - co[:2])) < (ro + 0.046):
                        collision = True
                        break
                if not collision:
                    bev_y, bev_x = metric_3d_to_bev_norm(cand, geo)
                    obj_slots.append({
                        'slot_label': f"{a['name']}{label_zh}",
                        'reference_object_id': a['id'],
                        'reference_object_name': a['name'],
                        'direction': label_zh,
                        'position_m': [round(float(v), 4) for v in cand],
                        'bev_point': [bev_y, bev_x],
                    })
            free_slots.extend(obj_slots)

        # Also add global workspace canonical zones (桌面中央, 桌面左側空位, 桌面右側空位)
        for zone_name, zx, zy in (
            ('桌面正中央空位', 0.25, 0.0),
            ('桌面左側空位', 0.24, 0.11),
            ('桌面右側空位', 0.24, -0.11),
            ('桌面近端中央', 0.19, 0.0),
        ):
            cand = origin + ex * zx + ey * zy
            if all(
                float(np.linalg.norm(cand[:2] - np.asarray(o['centre'])[:2]))
                >= float(o.get('bounding_radius', o['radius'])) + 0.040
                for o in self.objects
            ):
                bev_y, bev_x = metric_3d_to_bev_norm(cand, geo)
                free_slots.append({
                    'slot_label': zone_name,
                    'reference_object_id': None,
                    'reference_object_name': '桌面',
                    'direction': zone_name,
                    'position_m': [round(float(v), 4) for v in cand],
                    'bev_point': [bev_y, bev_x],
                })

        return {
            'relations': relations,
            'free_space_slots': free_slots[:24],
        }

    def snapshot_dict(self, geo: Geometry | None = None) -> dict[str, Any]:
        rel_info = self.compute_spatial_relations(geo) if geo is not None else {'relations': [], 'free_space_slots': []}
        return {
            'object_count': len(self.objects),
            'objects': self.objects,
            'relations': rel_info['relations'],
            'free_space_slots': rel_info['free_space_slots'],
            'last_bev': self.last_bev,
            'last_summary_zh': self.last_summary_zh,
            'last_updated': self.last_updated,
            'dirty_after_engage': self.dirty_after_engage,
            'last_engage_at': self.last_engage_at,
            'last_engage_object_id': self.last_engage_object_id,
            'engage_count': self.engage_count,
        }


def analyze_dual_view_with_er2(
    er2_brain,
    wrist_frame_path: str | Path | list[str | Path],
    bev_image_path: str | Path,
    query: str = '',
    *,
    thinking_level: str = 'low',
) -> dict[str, Any]:
    """Call Gemini Robotics ER 2 (`gemini-robotics-er-2-preview`) with:
    - 1 or more Wrist Camera Views (1920x1080)
    - MIRA Orthographic Top-Down BEV Map (720x720, 1px = 0.5mm)

    Returns multi-view object detections (`view_index`), oriented bounding ellipses (`yaw_deg`, `length_mm`,
    `width_mm`, `height_mm`), and inferred physical properties (`estimated_mass_g`, `material`,
    `friction_coeff`, `fragile`, `stackable`).
    """
    from google.genai import types

    t0 = time.time()
    frame_paths = [wrist_frame_path] if isinstance(wrist_frame_path, (str, Path)) else list(wrist_frame_path)
    wrist_rgbs = []
    for fp in frame_paths:
        with Image.open(fp) as im_w:
            wrist_rgbs.append(im_w.convert('RGB'))
    with Image.open(bev_image_path) as im2:
        bev_rgb = im2.convert('RGB')

    num_views = len(wrist_rgbs)
    focus = query.strip() or '桌面上所有可抓取或互動的實體物件與空曠放置區'
    view_desc = (
        '- IMAGE 1 (view_index=0): Perspective view from the wrist camera.\n- IMAGE 2: MIRA Orthographic Bird\'s-Eye View (BEV) top-down map'
        if num_views == 1
        else f'- IMAGES 1..{num_views} (view_index=0..{num_views - 1}): {num_views} perspective wrist camera views swept across the tabletop.\n- IMAGE {num_views + 1} (last image): MIRA Orthographic Bird\'s-Eye View (BEV) top-down map'
    )
    prompt = f"""You are Gemini Robotics ER 2 (`gemini-robotics-er-2-preview`), the spatial & physical intelligence brain for LeRobot SO-101.
You are given synchronized images of the workspace:
{view_desc} synthesized via Forward Kinematics (`1 px = 0.5 mm`, top is far X=440mm, bottom is near robot base X=80mm, left is Y=+180mm, right is Y=-180mm).

User task / focus: {focus!r}

Analyze all views together, deduplicate objects that appear in multiple wrist views, and return a JSON object with this EXACT structure:
{{
  "summary": "台灣繁體中文摘要：說明桌面上有哪些物件、它們的相對方位與建議操作策略",
  "items": [
    {{
      "label": "繁體中文物件名稱 (例如：綠色品客罐、藍色紙盒、紅色馬克杯)",
      "view_index": 0,
      "point": [y_wrist, x_wrist],
      "box_2d": [ymin_w, xmin_w, ymax_w, xmax_w],
      "bottom_left_point": [y_bl, x_bl],
      "bottom_right_point": [y_br, x_br],
      "top_center_point": [y_tc, x_tc],
      "bev_center_point": [y_bev, x_bev],
      "yaw_deg": 0.0,
      "length_mm": 65.0,
      "width_mm": 65.0,
      "height_mm": 115.0,
      "shape_primitive": "cylinder",
      "confidence": 0.94,
      "description": "繁體中文外觀與方位描述",
      "matches_query": true,
      "physics": {{
        "estimated_mass_g": 135,
        "material": "cardboard_metal",
        "friction_coeff": 0.58,
        "center_of_mass_z_ratio": 0.46,
        "rigidity": "rigid",
        "fragile": false,
        "stackable": true
      }}
    }}
  ],
  "recommended_place_bev_point": [y_place_bev, x_place_bev],
  "recommended_place_reason": "繁體中文說明建議放置目標位置的原因（若任務包含放置或移動）"
}}

Coordinate rules:
- All points `[y, x]` and `box_2d` `[ymin, xmin, ymax, xmax]` use integers normalized to `0..1000`.
- For each unique tabletop object, choose `view_index` (0..{num_views - 1}) where the object is most clearly and completely visible near the center of that wrist image.
- `point`, `box_2d`, `bottom_left_point`, `bottom_right_point`, `top_center_point` refer to that chosen wrist image (`view_index`).
- `bev_center_point` and `recommended_place_bev_point` refer to the last image (MIRA Orthographic BEV top-down map).
- `yaw_deg` (-90 to +90) is the in-plane angle of the object's longer horizontal axis in the top-down view (0 = aligned with forward X axis).
- `shape_primitive` must be `"cylinder"`, `"box"`, or `"ellipsoid"`.
- Report EVERY distinct physical object visible across ANY of the wrist views (ignore the robot's own gripper jaws)."""

    client = er2_brain._client()
    budget_map = {'none': 0, 'low': 256, 'medium': 1024, 'high': -1}
    thinking_cfg = types.ThinkingConfig(thinking_budget=budget_map.get(thinking_level, 256))
    config = types.GenerateContentConfig(
        temperature=0.2,
        response_mime_type='application/json',
        thinking_config=thinking_cfg,
    )
    response = client.models.generate_content(
        model=er2_brain.model,
        contents=[*wrist_rgbs, bev_rgb, prompt],
        config=config,
    )
    raw_text = response.text or '{}'
    cleaned = raw_text.strip()
    fence = re.search(r'```(?:json)?\s*([\s\S]*?)```', cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    parsed = json.loads(cleaned)
    if isinstance(parsed, list):
        parsed = {'summary': f'偵測到 {len(parsed)} 個物件', 'items': parsed}
    parsed['latency_ms'] = round((time.time() - t0) * 1000.0, 1)
    er2_brain._verified = True
    er2_brain._last_latency_ms = parsed['latency_ms']
    return parsed


# ==============================================================================
# Tier-2 Cloud 3D Mesh & World Reconstruction (`FAL_KEY` & `WORLD_LABS_API_KEY`)
# ==============================================================================
def reconstruct_object_3d_fal(
    obj: dict[str, Any],
    output_glb_path: Path,
    *,
    timeout_s: float = 95.0,
) -> dict[str, Any]:
    """Generate a high-fidelity 3D `.glb` mesh for a tabletop object via FAL.ai Serverless,
    then Metric-Anchor its bounding box to the exact MIRA FK dimensions `(2a, 2b, height)`!
    """
    import trimesh

    fal_key = _read_secret_key(('FAL_KEY',))
    if not fal_key:
        raise RuntimeError('尚未設定 FAL_KEY，無法呼叫 FAL.ai 3D 重建服務')

    crop_path = obj.get('crop_path')
    if not crop_path or not Path(crop_path).is_file():
        raise RuntimeError('找不到此物件的裁切影像，請先執行 MIRA 桌面掃描')

    t0 = time.time()
    mime = 'image/jpeg'
    b64 = base64.b64encode(Path(crop_path).read_bytes()).decode('ascii')
    data_uri = f'data:{mime};base64,{b64}'

    # Try FAL's fast image-to-3D endpoints (trellis -> triposr fallback)
    endpoints = [
        ('fal-ai/trellis', {'image_url': data_uri, 'ss_sampling_steps': 12, 'slat_sampling_steps': 12, 'mesh_simplify': 0.95, 'texture_size': 1024}),
        ('fal-ai/triposr', {'image_url': data_uri, 'output_format': 'glb', 'do_remove_background': True}),
    ]
    glb_url = None
    used_endpoint = ''
    last_err = ''

    for ep, payload in endpoints:
        url = f'https://fal.run/{ep}'
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={
                'Authorization': f'Key {fal_key}',
                'Content-Type': 'application/json',
                'Accept': 'application/json',
            },
            method='POST',
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                result = json.loads(resp.read().decode('utf-8'))
            # Extract GLB URL from standard FAL 3D response schemas
            cand = (
                (result.get('model_mesh') or {}).get('url')
                or (result.get('mesh') or {}).get('url')
                or (result.get('model_glb') or {}).get('url')
                or result.get('glb_url')
            )
            if cand:
                glb_url = str(cand)
                used_endpoint = ep
                break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode('utf-8', errors='ignore')[:240]
            last_err = f'{ep} HTTP {exc.code}: {body}'
        except Exception as exc:
            last_err = f'{ep}: {exc}'

    if not glb_url:
        raise RuntimeError(f'FAL.ai 3D 模型生成失敗：{last_err}')

    # Download raw GLB bytes
    with urllib.request.urlopen(glb_url, timeout=45.0) as resp:
        raw_glb = resp.read()

    # Metric-Anchor the generated GLB to MIRA's measured physical bounding volume (2a, 2b, height)
    loaded = trimesh.load(io.BytesIO(raw_glb), file_type='glb', force='mesh')
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.dump(concatenate=True)
    else:
        mesh = loaded

    bounds = mesh.bounds  # [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    extents = np.maximum(bounds[1] - bounds[0], 1e-5)
    center = 0.5 * (bounds[0] + bounds[1])
    mesh.apply_translation(-center)

    # FAL meshes typically have Y-up; rotate +90 deg around X so Z is up in robot_base_model
    rot_x90 = np.eye(4)
    rot_x90[:3, :3] = Rotation.from_euler('x', 90, degrees=True).as_matrix()
    mesh.apply_transform(rot_x90)

    b2 = mesh.bounds
    ext2 = np.maximum(b2[1] - b2[0], 1e-5)
    target_x = float(obj.get('semi_major_m', obj['radius'])) * 2.0
    target_y = float(obj.get('semi_minor_m', obj['radius'])) * 2.0
    target_z = float(obj.get('height', 0.085))
    scale_factors = np.array([
        target_x / ext2[0],
        target_y / ext2[1],
        target_z / ext2[2],
    ], dtype=float)
    S = np.eye(4)
    S[0, 0], S[1, 1], S[2, 2] = scale_factors
    mesh.apply_transform(S)

    # Shift so the bottom of the mesh sits at local z = -target_z/2 (consistent with centered primitive convention)
    b3 = mesh.bounds
    mesh.apply_translation(-0.5 * (b3[0] + b3[1]))

    output_glb_path = Path(output_glb_path)
    output_glb_path.parent.mkdir(parents=True, exist_ok=True)
    output_glb_path.write_bytes(mesh.export(file_type='glb'))

    return {
        'object_id': obj['id'],
        'object_name': obj['name'],
        'provider': used_endpoint,
        'source_glb_url': glb_url,
        'local_glb_path': str(output_glb_path),
        'metric_extents_mm': [
            round(target_x * 1000.0, 1),
            round(target_y * 1000.0, 1),
            round(target_z * 1000.0, 1),
        ],
        'elapsed_s': round(time.time() - t0, 2),
    }


def generate_world_worldlabs(
    frame_path: str | Path,
    output_dir: Path,
    prompt_text: str = 'Clean robotic manipulation tabletop studio environment',
) -> dict[str, Any]:
    """Trigger World Labs Marble 3D environment generation (`WORLD_LABS_API_KEY`) as in `image-blaster`."""
    wl_key = _read_secret_key(('WORLD_LABS_API_KEY',))
    if not wl_key:
        raise RuntimeError('尚未設定 WORLD_LABS_API_KEY，無法呼叫 World Labs 服務')

    output_dir = Path(output_dir) / 'worldlabs'
    output_dir.mkdir(parents=True, exist_ok=True)
    img_bytes = Path(frame_path).read_bytes()
    b64 = base64.b64encode(img_bytes).decode('ascii')

    payload = {
        'display_name': f'lerobot-argon-{int(time.time())}',
        'model': 'marble-1.0-draft',
        'world_prompt': {
            'type': 'image',
            'image_prompt': {
                'source': 'data_base64',
                'data_base64': b64,
            },
            'text_prompt': prompt_text,
        },
    }
    req = urllib.request.Request(
        'https://api.worldlabs.ai/marble/v1/worlds:generate',
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'WLT-Api-Key': wl_key,
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        },
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            result = json.loads(resp.read().decode('utf-8'))
        meta_path = output_dir / 'latest_operation.json'
        meta_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return {
            'status': 'submitted',
            'model': 'marble-1.0-draft',
            'operation_id': result.get('name') or result.get('operation_id') or 'worldlabs-op',
            'metadata_path': str(meta_path),
            'response': result,
        }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode('utf-8', errors='ignore')[:300]
        raise RuntimeError(f'World Labs API 回應錯誤 ({exc.code})：{body}') from exc


# ==============================================================================
# Real-to-Sim Digital Twin Exporter (`scene.glb` + 25Hz Pose Stream + Collision Check)
# ==============================================================================
def _jaw_display_world(model, raw_joints: dict[str, int]) -> tuple[str, np.ndarray]:
    joint = model.tree.find("joint[@name='gripper']")
    cal = model.calibration['gripper']
    lo, hi = cal['range_min'], cal['range_max']
    limit = joint.find('limit')
    lower, upper = float(limit.get('lower')), float(limit.get('upper'))
    frac = float(np.clip((raw_joints.get('gripper', OPEN_RAW) - lo) / max(hi - lo, 1), 0.0, 1.0))
    angle = lower + frac * (upper - lower)
    rot = np.eye(4)
    axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
    rot[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
    local = xyz_transform(joint.find('origin')) @ rot
    world = model.fk.transforms(raw_joints)[joint.find('parent').get('link')] @ local
    return joint.find('child').get('link'), world


def export_real_to_sim_twin(
    geo: Geometry,
    scene_map: WorkspaceSceneMap,
    current_joints: dict[str, int],
    output_dir: Path,
    *,
    active_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the complete Real-to-Sim Digital Twin GLB (`scene.glb`) and `twin_state.json`.

    Includes:
    1. Articulated SO-101 robot arm visual links at `current_joints`.
    2. Calibrated tilted tabletop plane (`n . p = d`).
    3. Every object in `WorkspaceSceneMap` rendered either as its FAL.ai reconstructed `.glb`
       mesh (Tier-2) or its MIRA Visual Hull oriented primitive (Tier-1) with physical metadata.
    4. If `active_plan` is present, renders the Ghost Trajectory tube, target grasp markers,
       and placement target ring!
    """
    import trimesh

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = default_model(ROOT)

    raw = {k: int(np.clip(round(current_joints.get(k, 2047)), 0, 4095)) for k in ALL}
    transforms = dict(model.fk.transforms(raw))
    jaw_link, jaw_world = _jaw_display_world(model, raw)
    transforms[jaw_link] = jaw_world

    scene = trimesh.Scene()
    materials = {
        '3d_printed': [0.93, 0.93, 0.94, 1.0],
        'sts3215': [0.09, 0.10, 0.11, 1.0],
    }

    def add_mesh(mesh_obj, name: str, T: np.ndarray, rgba: list[float], *, role: str = 'visual'):
        mesh_obj.metadata['role'] = role
        has_custom_texture = (
            role == 'reconstructed_glb'
            and getattr(mesh_obj, 'visual', None) is not None
            and getattr(getattr(mesh_obj.visual, 'material', None), 'baseColorTexture', None) is not None
        )
        if has_custom_texture:
            mesh_obj.visual.material.name = name
        else:
            mesh_obj.visual = trimesh.visual.TextureVisuals(
                material=trimesh.visual.material.PBRMaterial(
                    name=name,
                    baseColorFactor=np.rint(np.asarray(rgba) * 255).astype(np.uint8),
                    roughnessFactor=0.65,
                    metallicFactor=0.15 if role == 'robot' else 0.05,
                    alphaMode='BLEND' if rgba[3] < 1.0 else 'OPAQUE',
                )
            )
        scene.add_geometry(mesh_obj, node_name=name, geom_name=name, transform=T)

    # 1. Add SO-101 robot links
    for link in model.tree.findall('link'):
        lname = link.get('name')
        if lname not in transforms:
            continue
        for idx, visual in enumerate(link.findall('visual')):
            geom = visual.find('geometry')
            mesh_el = geom.find('mesh')
            if mesh_el is not None:
                mpath = (model.mesh_root / mesh_el.get('filename')).resolve()
                m = trimesh.load_mesh(mpath, process=False)
                m.apply_scale(np.fromstring(mesh_el.get('scale', '1 1 1'), sep=' '))
            elif geom.find('box') is not None:
                m = trimesh.creation.box(np.fromstring(geom.find('box').get('size'), sep=' '))
            elif geom.find('cylinder') is not None:
                cyl = geom.find('cylinder')
                m = trimesh.creation.cylinder(float(cyl.get('radius')), float(cyl.get('length')))
            else:
                continue
            mat_el = visual.find('material')
            mat_name = mat_el.get('name') if mat_el is not None else ''
            rgba = materials.get(mat_name, [0.58, 0.62, 0.66, 1.0])
            T_vis = transforms[lname] @ xyz_transform(visual.find('origin'))
            add_mesh(m, f'robot_{lname}_{idx}', T_vis, rgba, role='robot')

    # 2. Add Calibrated Tilted Tabletop Plane (matching BEV bounds X in [0.08, 0.44], Y in [-0.18, 0.18])
    ex, ey, n, origin = _table_basis(geo)
    R_table = np.column_stack([ex, ey, n])
    T_table = np.eye(4)
    T_table[:3, :3] = R_table
    table_center_x = 0.5 * (BEV_X_MIN_M + BEV_X_MAX_M)
    T_table[:3, 3] = origin + ex * table_center_x - n * 0.006
    table_box = trimesh.creation.box([BEV_X_MAX_M - BEV_X_MIN_M, BEV_Y_MAX_M - BEV_Y_MIN_M, 0.012])
    add_mesh(table_box, 'calibrated_tabletop', T_table, [0.85, 0.87, 0.90, 1.0], role='table')

    # 3. Add Reconstructed Scene Objects (Tier-2 FAL GLB if available, else Tier-1 MIRA Oriented Mesh)
    for idx, obj in enumerate(scene_map.objects):
        c = np.asarray(obj['centre'], dtype=float)
        h = float(obj.get('height', 0.085))
        a = float(obj.get('semi_major_m', obj['radius']))
        b = float(obj.get('semi_minor_m', obj['radius']))
        yaw_rad = math.radians(float(obj.get('yaw_deg', 0.0)))
        rgba = list(obj.get('color_rgba') or [0.22, 0.68, 0.42, 1.0])

        R_yaw = Rotation.from_AxisAngle(n, yaw_rad) if hasattr(Rotation, 'from_AxisAngle') else Rotation.from_rotvec(n * yaw_rad)
        T_obj = np.eye(4)
        T_obj[:3, :3] = R_yaw.as_matrix() @ R_table
        # Place center at height h/2 above the tilted table plane
        c_on_table = c - n * (n @ c - geo.d)
        T_obj[:3, 3] = c_on_table + n * (0.5 * h)

        glb_path = obj.get('mesh_glb_path')
        loaded_custom = False
        if glb_path and Path(glb_path).is_file():
            try:
                custom = trimesh.load(str(glb_path), file_type='glb', force='mesh')
                if isinstance(custom, trimesh.Scene):
                    custom = custom.dump(concatenate=True)
                add_mesh(custom, f"object_{obj['id']}", T_obj, rgba, role='reconstructed_glb')
                loaded_custom = True
            except Exception:
                loaded_custom = False

        if not loaded_custom:
            if obj.get('shape_primitive') == 'box':
                prim = trimesh.creation.box([2.0 * a, 2.0 * b, h])
            else:
                prim = trimesh.creation.cylinder(radius=b, height=h, sections=32)
                if abs(a - b) > 0.002:
                    S = np.eye(4)
                    S[0, 0] = a / max(b, 1e-4)
                    prim.apply_transform(S)
            add_mesh(prim, f"object_{obj['id']}", T_obj, rgba, role='mira_visual_hull')

    # 4. Add Active Plan Ghost Trajectory & Target Placement Markers
    if active_plan and isinstance(active_plan.get('pose_stream'), list):
        stream = active_plan['pose_stream']
        for step_idx, frame_pose in enumerate(stream[::2]):
            tcp = frame_pose.get('tcp_xyz')
            if isinstance(tcp, list) and len(tcp) == 3:
                T_pt = np.eye(4)
                T_pt[:3, 3] = tcp
                dot = trimesh.creation.icosphere(subdivisions=1, radius=0.0035)
                add_mesh(dot, f'ghost_wp_{step_idx}', T_pt, [0.18, 0.52, 0.98, 0.78], role='ghost_trajectory')

        place_target = active_plan.get('place_target_m')
        if isinstance(place_target, list) and len(place_target) == 3:
            T_pl = np.eye(4)
            T_pl[:3, :3] = R_table
            T_pl[:3, 3] = np.asarray(place_target, dtype=float) + n * 0.003
            ring = trimesh.creation.cylinder(radius=0.032, height=0.004, sections=32)
            add_mesh(ring, 'target_placement_zone', T_pl, [0.98, 0.62, 0.12, 0.65], role='place_target')

    glb_out = output_dir / 'scene.glb'
    glb_out.write_bytes(scene.export(file_type='glb'))

    # Export link poses & camera optical pose for Three.js dual-camera (`Agentview` + `Eye-in-Hand`)
    cam_T = transforms['wrist_camera_optical']
    links_pose = {
        name: {
            'position_m': [round(float(v), 5) for v in T[:3, 3]],
            'quaternion_xyzw': [round(float(v), 6) for v in Rotation.from_matrix(T[:3, :3]).as_quat()],
        }
        for name, T in transforms.items()
    }
    twin_state = {
        'exported_at': round(time.time(), 3),
        'joints': raw,
        'wrist_camera_optical': links_pose.get('wrist_camera_optical'),
        'links': links_pose,
        'table_plane': {
            'normal': [round(float(v), 5) for v in n],
            'd_m': round(float(geo.d), 5),
        },
        'scene_map': scene_map.snapshot_dict(geo),
        'active_plan': active_plan,
        'cloud_providers': {
            'fal_ai_configured': bool(_read_secret_key(('FAL_KEY',))),
            'world_labs_configured': bool(_read_secret_key(('WORLD_LABS_API_KEY',))),
            'er2_configured': bool(_read_secret_key(('GEMINI_API_KEY', 'GOOGLE_API_KEY'))),
        },
    }
    (output_dir / 'twin_state.json').write_text(json.dumps(twin_state, ensure_ascii=False, indent=2))
    return twin_state


# ==============================================================================
# Mental Simulation & 6-DoF Pick-and-Place Trajectory Planner
# ==============================================================================
def build_pose_stream_and_simulate(
    geo: Geometry,
    waypoints: list[dict[str, Any]],
    q_start: dict[str, int],
    target_obj: dict[str, Any],
    scene_objects: list[dict[str, Any]],
    gripper_raw: int,
    *,
    fps: int = 25,
) -> dict[str, Any]:
    """Interpolate a planned waypoint sequence at 25 Hz (`steer-a-robot` style pose stream)
    and run pre-execution collision & physical stability checks in the Digital Twin!
    """
    model = default_model(ROOT)
    Ttcp, _, _, aperture_m = geo.tcp(gripper_raw)
    neighbors = [
        o for o in scene_objects
        if o.get('id') != target_obj.get('id')
        and float(np.linalg.norm(np.asarray(o['centre'])[:2] - np.asarray(target_obj['centre'])[:2])) > 0.032
    ]

    stream = []
    min_table_mm = 999.0
    min_target_mm = 999.0
    min_neighbor_mm = 999.0
    worst_neighbor_name = ''
    sim_time = 0.0
    prev = {n: float(q_start[n]) for n in BODY}
    curr_gripper = int(gripper_raw)

    for wp in waypoints:
        label = wp['label']
        goal = {n: float(wp['target'][n]) for n in BODY}
        wp_gripper = int(wp.get('gripper_raw', curr_gripper))
        max_ticks = max(abs(goal[n] - prev[n]) for n in BODY)
        duration_s = max(0.32, min(1.35, max_ticks / 420.0))
        num_steps = max(4, int(round(duration_s * fps)))

        check_target = label in ('above_pregrasp', 'approach_70mm', 'approach_50mm', 'approach_30mm', 'approach_15mm', 'grasp')

        t_mm = 50.0
        t_link = 'gripper_link'
        obj_mm = None
        for i in range(1, num_steps + 1):
            alpha = i / float(num_steps)
            # Smooth cosine easing (`lerp`)
            s = 0.5 * (1.0 - math.cos(math.pi * alpha))
            q_interp = {n: int(round(prev[n] + (goal[n] - prev[n]) * s)) for n in BODY}
            g_interp = int(round(curr_gripper + (wp_gripper - curr_gripper) * s))

            if i == 1 or i == num_steps or (i % 2 == 0):
                t_clear, t_link = geo.table_clearance(q_interp)
                t_mm = round(float(t_clear) * 1000.0, 1)
                min_table_mm = min(min_table_mm, t_mm)

                obj_mm = None
                if check_target:
                    g_clear, _ = geo.object_clearance(q_interp, target_obj, gripper_raw=g_interp)
                    obj_mm = round(float(g_clear) * 1000.0, 1)
                    min_target_mm = min(min_target_mm, obj_mm)

                if label not in ('pre_high_lift', 'pre_high_transit', 'high_lift', 'high_transit', 'high_retreat', 'return_home'):
                    for nb in neighbors:
                        nb_clear, _ = geo.object_clearance(q_interp, nb, gripper_raw=g_interp)
                        nb_mm = round(float(nb_clear) * 1000.0, 1)
                        if nb_mm < min_neighbor_mm:
                            min_neighbor_mm = nb_mm
                            worst_neighbor_name = nb.get('name', '鄰近物件')

            raw_full = {**q_interp, 'gripper': g_interp}
            transforms = model.fk.transforms(raw_full)
            G = transforms['gripper_link']
            tcp_world = (G[:3, :3] @ Ttcp[:3, 3] + G[:3, 3]).round(4).tolist()
            sim_time += 1.0 / fps

            stream.append({
                'step': len(stream),
                'time_s': round(sim_time, 3),
                'phase': label,
                'joints': raw_full,
                'tcp_xyz': tcp_world,
                'table_clearance_mm': t_mm,
                'table_link': t_link,
                'target_clearance_mm': obj_mm,
            })

        prev = goal
        curr_gripper = wp_gripper

    # Physical grasp stability evaluation using ER 2's inferred physics
    physics = target_obj.get('physics') or {}
    mass_g = float(physics.get('estimated_mass_g', 120))
    friction = float(physics.get('friction_coeff', 0.55))
    diam_mm = float(target_obj.get('semi_minor_m', target_obj['radius'])) * 2000.0
    aperture_mm = aperture_m * 1000.0
    fit_margin_mm = round(aperture_mm - diam_mm, 1)
    # Normal squeeze force of STS3215 jaw at low PWM ~ 3.5 N; safe payload = 2 * mu * F_n
    max_safe_mass_g = round(2.0 * friction * 3.5 / 9.81 * 1000.0, 0)
    stability_score = round(
        float(np.clip(
            0.95
            - (0.20 if fit_margin_mm < 6.0 else 0.0)
            - (0.15 if mass_g > max_safe_mass_g else 0.0)
            - (0.25 if min_table_mm < 5.0 else 0.0),
            0.1,
            0.99,
        )),
        2,
    )

    sim_problems = []
    if min_table_mm < TABLE_FLOOR * 1000.0:
        sim_problems.append(f'數位孿生預演發現桌面淨空不足 ({min_table_mm:.1f} mm < 5.0 mm)')
    if min_neighbor_mm < -3.5 and worst_neighbor_name:
        sim_problems.append(f'軌跡太靠近鄰近物件「{worst_neighbor_name}」({min_neighbor_mm:.1f} mm)')

    return {
        'passed': len(sim_problems) == 0,
        'problems': sim_problems,
        'duration_s': round(sim_time, 2),
        'num_frames': len(stream),
        'fps': fps,
        'min_table_clearance_mm': min_table_mm,
        'min_target_clearance_mm': None if min_target_mm > 500 else min_target_mm,
        'min_neighbor_clearance_mm': None if min_neighbor_mm > 500 else min_neighbor_mm,
        'worst_neighbor_name': worst_neighbor_name,
        'physics_verification': {
            'object_mass_g': mass_g,
            'max_friction_payload_g': max_safe_mass_g,
            'narrow_axis_diameter_mm': round(diam_mm, 1),
            'jaw_aperture_mm': round(aperture_mm, 1),
            'fit_margin_mm': fit_margin_mm,
            'stability_score': stability_score,
        },
        'pose_stream': stream,
    }


def plan_mira_grasp_or_place(
    geo: Geometry,
    scene_map: WorkspaceSceneMap,
    target_obj: dict[str, Any],
    q_now: dict[str, int],
    gripper_raw: int,
    *,
    place_target_m: list[float] | None = None,
    place_label_zh: str = '',
) -> dict[str, Any]:
    """Plan a 6-DoF Grasp or Pick-and-Place trajectory using MIRA narrow-axis geometry
    and verify it in the Real-to-Sim Digital Twin!
    """
    # Use narrow-axis radius (`semi_minor_m`) so wide boxes or cans fit cleanly
    aperture = float(geo.tcp(gripper_raw)[3])
    max_fit_radius = max(0.012, 0.5 * (aperture - FIXED_SIDE_MARGIN - 0.012))
    raw_narrow = min(
        float(target_obj.get('semi_minor_m', target_obj['radius'])),
        float(target_obj['radius']),
    )
    narrow_r = min(raw_narrow, max_fit_radius)
    # Preserve true un-inflated narrow-axis radius so `grasp_pose` aligns the stationary left jaw
    # at 3.5 mm (`FIXED_SIDE_MARGIN`) from the object's true left edge without double inflation.
    uninflated_r = min(narrow_r, float(target_obj.get('uninflated_radius_m') or raw_narrow))
    c_true = np.asarray(target_obj['centre'], dtype=float)
    grasp_c = c_true.copy()
    # Neighbor-Avoidance V-Groove Offset: when a neighbor is touching or within 20 mm surface gap,
    # shift the IK grasp center up to 13.5 mm away from the neighbor into the moving-jaw spare aperture
    # so the stationary left jaw outer housing slides cleanly into the V-groove without clipping the neighbor.
    for nb in scene_map.objects:
        if nb.get('id') == target_obj.get('id'):
            continue
        c_nb = np.asarray(nb['centre'], dtype=float)
        diff_xy = c_true[:2] - c_nb[:2]
        dist_xy = float(np.linalg.norm(diff_xy))
        nb_r = float(nb.get('semi_minor_m', nb['radius']))
        surf_gap = dist_xy - (narrow_r + nb_r)
        if 1e-4 < dist_xy and surf_gap < 0.015:
            shift_m = float(np.clip(0.015 - surf_gap, 0.0, 0.0045))
            u_away = diff_xy / dist_xy
            grasp_c[0] = float(c_true[0] + u_away[0] * shift_m)
            grasp_c[1] = float(c_true[1] + u_away[1] * shift_m)
            break
    obj_for_ik = dict(
        target_obj,
        centre=[round(float(v), 5) for v in grasp_c],
        true_centre=[round(float(v), 5) for v in c_true],
        radius=narrow_r,
        uninflated_radius_m=uninflated_r,
    )

    base_plan = plan_grasp(
        geo,
        obj_for_ik,
        gripper_raw,
        q_now,
        pitches=(25, 18, 12, 32, 8, 38, 0, 44),
        heights=(0.020, 0.022, 0.024, 0.028, 0.032, 0.036, 0.045, 0.055, 0.068, 0.082, 0.096, 0.110),
    )

    waypoints = list(base_plan.get('waypoints', []))
    problems = list(base_plan.get('problems', []))
    closed_raw = max(1580, int(gripper_raw) - 360)

    # If panning across the workspace (>160 ticks) to reach `above_pregrasp`, insert a high-clearance
    # overhead bridge so the arm never sweeps horizontally through tall objects sitting near the center.
    if waypoints and not problems and abs(int(q_now['shoulder_pan']) - int(waypoints[0]['target']['shoulder_pan'])) > 160:
        q_ap = waypoints[0]['target']
        q_pre_lift = {
            'shoulder_pan': int(round(q_now['shoulder_pan'])),
            'shoulder_lift': min(int(round(q_now['shoulder_lift'])), 1260),
            'elbow_flex': min(int(round(q_now['elbow_flex'])), 1420),
            'wrist_flex': 1900,
            'wrist_roll': int(round(q_now['wrist_roll'])),
        }
        q_pre_transit = {
            'shoulder_pan': int(round(q_ap['shoulder_pan'])),
            'shoulder_lift': min(int(round(q_ap['shoulder_lift'])), 1260),
            'elbow_flex': min(int(round(q_ap['elbow_flex'])), 1420),
            'wrist_flex': 1900,
            'wrist_roll': int(round(q_ap['wrist_roll'])),
        }
        pre_wps = []
        prev_pre = q_now
        for lbl_pre, q_pre in (('pre_high_lift', q_pre_lift), ('pre_high_transit', q_pre_transit)):
            scr_pre = segment_screen(geo, prev_pre, q_pre, None, int(gripper_raw), object_check=False)
            pre_wps.append({
                'label': lbl_pre,
                'target': q_pre,
                'gripper_raw': int(gripper_raw),
                'ik_error_mm': 0.5,
                'ik_rotation_deg': 0.5,
                **scr_pre,
                'max_joint_step_ticks': int(max(abs(q_pre[n] - prev_pre[n]) for n in BODY)),
            })
            prev_pre = q_pre
        waypoints = pre_wps + waypoints

    # Tag gripper state on grasp waypoints
    for wp in waypoints:
        wp['gripper_raw'] = closed_raw if wp['label'] == 'lift' else int(gripper_raw)

    # If a placement destination `place_target_m` is given and grasp is valid, append Pick-and-Place arc!
    if place_target_m is not None and not problems and waypoints:
        p_place = np.asarray(place_target_m, dtype=float)
        # Ensure place point is on the table plane
        p_place = p_place - geo.n * (geo.n @ p_place - geo.d)
        place_obj_dummy = {
            'centre': p_place.tolist(),
            'radius': narrow_r,
            'uninflated_radius_m': uninflated_r,
            'height': float(target_obj.get('height', 0.08)),
        }
        q_lift = waypoints[-1]['target']
        grasp_h = float(base_plan['grasp'].get('height_m', GRASP_HEIGHT))
        pitch_deg = float(base_plan['grasp'].get('pitch_target_deg', 22.0))
        neighbors = [o for o in scene_map.objects if o['id'] != target_obj.get('id')]

        try:
            # Keep placement pitch aligned with grasp pitch so upright cylinders stay vertical on release,
            # and place at grasp_h + 2mm so the base gently rests on the table before jaws open.
            best_place_cand = None
            best_place_score = -1e9
            h0 = max(0.022, grasp_h + 0.002)
            h1 = max(0.025, grasp_h + 0.005)
            cand_pairs = (
                (h0, pitch_deg),
                (h0, min(28.0, pitch_deg + 5.0)),
                (h1, pitch_deg),
                (h1, max(0.0, pitch_deg - 5.0)),
            )
            for h_cand, p_cand in cand_pairs:
                try:
                    q_cand, info_cand = grasp_pose(
                        geo,
                        place_obj_dummy,
                        gripper_raw,
                        q_lift,
                        height=h_cand,
                        pitch_deg=p_cand,
                    )
                except Exception:
                    continue
                pos_err = float(info_cand.get('position_error_mm', 99.0))
                if pos_err > 10.0:
                    continue
                t_c, _ = geo.table_clearance(q_cand)
                t_mm = float(t_c) * 1000.0
                if t_mm < PLAN_TABLE_CLEARANCE * 1000.0:
                    continue
                nb_min_mm = 999.0
                for nb in neighbors:
                    nc, _ = geo.object_clearance(q_cand, nb, gripper_raw=gripper_raw)
                    nb_min_mm = min(nb_min_mm, float(nc) * 1000.0)
                score = (
                    min(nb_min_mm, 45.0) * 2.0
                    + min(t_mm, 30.0)
                    - pos_err * 2.5
                    - abs(p_cand - pitch_deg) * 3.0
                    - abs(h_cand - h0) * 400.0
                )
                if score > best_place_score:
                    best_place_score = score
                    best_place_cand = (q_cand, info_cand)
                if nb_min_mm >= 14.0 and t_mm >= 10.0 and pos_err <= 4.0 and abs(p_cand - pitch_deg) <= 1.0:
                    break

            if best_place_cand is not None:
                q_place_low, place_info = best_place_cand
            else:
                q_place_low, place_info = grasp_pose(
                    geo,
                    place_obj_dummy,
                    gripper_raw,
                    q_lift,
                    height=h0,
                    pitch_deg=pitch_deg,
                )

            q_above_place, _, _ = pose_along(geo, q_place_low, geo.n * 0.075, q_place_low)
            # High-clearance overhead bridge so panning across the table clears all tall cans (>160 mm)
            q_high_pick = {
                'shoulder_pan': int(round(q_lift['shoulder_pan'])),
                'shoulder_lift': min(int(round(q_lift['shoulder_lift'])), 1340),
                'elbow_flex': min(int(round(q_lift['elbow_flex'])), 1480),
                'wrist_flex': 1920,
                'wrist_roll': int(round(q_lift['wrist_roll'])),
            }
            q_high_transit = {
                'shoulder_pan': int(round(q_above_place['shoulder_pan'])),
                'shoulder_lift': min(int(round(q_above_place['shoulder_lift'])), 1340),
                'elbow_flex': min(int(round(q_above_place['elbow_flex'])), 1480),
                'wrist_flex': 1920,
                'wrist_roll': int(round(q_above_place['wrist_roll'])),
            }

            Ttcp, *_ = geo.tcp(gripper_raw)
            zb_place = geo.transforms(q_place_low)['gripper_link'][:3, :3] @ Ttcp[:3, 2]
            # Slide straight back along -zb_place first so open fingers clear a tall standing cylinder cleanly
            q_place_retreat, _, _ = pose_along(geo, q_place_low, -zb_place * 0.065 + geo.n * 0.025, q_place_low)
            q_high_retreat = {
                'shoulder_pan': int(round(q_place_retreat['shoulder_pan'])),
                'shoulder_lift': min(int(round(q_place_retreat['shoulder_lift'])), 1340),
                'elbow_flex': min(int(round(q_place_retreat['elbow_flex'])), 1480),
                'wrist_flex': 1920,
                'wrist_roll': int(round(q_place_retreat['wrist_roll'])),
            }

            extra_wps = [
                ('high_lift', q_high_pick, closed_raw),
                ('high_transit', q_high_transit, closed_raw),
                ('above_place', q_above_place, closed_raw),
                ('place_descent', q_place_low, closed_raw),
                ('release_open', q_place_low, int(gripper_raw)),
                ('place_retreat', q_place_retreat, int(gripper_raw)),
                ('high_retreat', q_high_retreat, int(gripper_raw)),
                ('return_home', dict(HOME), int(gripper_raw)),
            ]
            prev_q = q_lift
            for lbl, q_raw, g_val in extra_wps:
                qi = {n: int(round(q_raw[n])) for n in BODY}
                scr = segment_screen(geo, prev_q, qi, None, g_val, object_check=False)
                if scr['table_mm'] < PLAN_TABLE_CLEARANCE * 1000.0:
                    problems.append(f"{lbl}: 放置路徑離桌面太近 ({scr['table_mm']} mm)")
                waypoints.append({
                    'label': lbl,
                    'target': qi,
                    'gripper_raw': g_val,
                    'ik_error_mm': round(float(place_info.get('position_error_mm', 0.8)), 2),
                    'ik_rotation_deg': 0.5,
                    **scr,
                    'max_joint_step_ticks': int(max(abs(qi[n] - prev_q[n]) for n in BODY)),
                })
                prev_q = qi
        except Exception as exc:
            problems.append(f'無法規劃放置點 IK 軌跡：{exc}')

    sim_report = build_pose_stream_and_simulate(
        geo,
        waypoints,
        q_now,
        obj_for_ik,
        scene_map.objects,
        gripper_raw,
    )
    if not sim_report['passed']:
        problems.extend(sim_report['problems'])

    return {
        'mode': 'pick_and_place' if place_target_m is not None else 'grasp',
        'object': obj_for_ik,
        'place_target_m': [round(float(v), 4) for v in place_target_m] if place_target_m is not None else None,
        'place_label_zh': place_label_zh,
        'grasp': base_plan.get('grasp', {}),
        'waypoints': waypoints,
        'problems': problems,
        'executable': len(problems) == 0 and len(waypoints) > 0,
        'simulation': sim_report,
        'pose_stream': sim_report['pose_stream'],
    }


class _FallbackJawSample:
    def __init__(self, raw: int, aperture_m: float):
        self.raw = raw
        self.aperture_m = aperture_m
        T = np.eye(4)
        T[2, 3] = -0.095
        self.T_wrist_tcp = type('_Arr', (), {'as_array': lambda self_: T})()


class _FallbackJawProfile:
    def __init__(self):
        self.samples = [_FallbackJawSample(1500, 0.008), _FallbackJawSample(2100, 0.068)]

    def at(self, raw: int):
        frac = float(np.clip((raw - 1500) / 600.0, 0.0, 1.0))
        return _FallbackJawSample(raw, 0.008 + frac * 0.060)


def load_calibrated_geometry(root_dir: Path = ROOT) -> Geometry:
    root_dir = Path(root_dir)
    config = json.loads((root_dir / 'config/robot.json').read_text())
    try:
        from .spatial_runtime import _profile_snapshot
        jaw = _profile_snapshot(default_model(root_dir), config, include_jaws=True)[2]
        if jaw is not None:
            return Geometry(root_dir, config, jaw)
    except Exception:
        pass
    return Geometry(root_dir, config, _FallbackJawProfile())


class RealToSimOrchestrator:
    """High-level coordinator uniting MIRA BEV, WorkspaceSceneMap, ER 2, FAL/WorldLabs 3D, and Pick & Place."""

    def __init__(self, robot, camera, er2_brain, brain, grasp_prog, kgrasp_prog, runtime_dir: Path, root_dir: Path = ROOT):
        self.robot = robot
        self.camera = camera
        self.er2_brain = er2_brain
        self.brain = brain
        self.grasp_prog = grasp_prog
        self.kgrasp_prog = kgrasp_prog
        self.runtime_dir = Path(runtime_dir)
        self.root_dir = Path(root_dir)
        self.scene_map = WorkspaceSceneMap(self.runtime_dir)
        self.current_dir = self.scene_map.runtime_dir / 'current'
        self.current_dir.mkdir(parents=True, exist_ok=True)
        self.active_plan: dict[str, Any] | None = None
        self.worldlabs_last: dict[str, Any] | None = None
        self.qa_history: list[dict[str, Any]] = []

    def geometry(self) -> Geometry:
        try:
            return self.kgrasp_prog.geometry()
        except Exception:
            return load_calibrated_geometry(self.root_dir)

    def _current_joints_safe(self, *, auto_connect: bool = False) -> dict[str, int]:
        try:
            if auto_connect and not getattr(self.robot, 'connected', False) and not getattr(self.robot, 'moving', False):
                if hasattr(self.robot, 'connect'):
                    self.robot.connect()
            if getattr(self.robot, 'connected', False):
                if getattr(self.robot, 'moving', False):
                    goals = getattr(self.robot, 'goals', None)
                    if isinstance(goals, dict) and all(k in goals for k in BODY):
                        return {k: int(goals.get(k, HOME.get(k, OPEN_RAW))) for k in (*BODY, 'gripper')}
                return self.robot.read()
        except Exception:
            pass
        goals = getattr(self.robot, 'goals', None)
        if isinstance(goals, dict) and all(k in goals for k in BODY):
            return {k: int(goals.get(k, HOME.get(k, OPEN_RAW))) for k in (*BODY, 'gripper')}
        return {**HOME, 'gripper': OPEN_RAW}

    def _capture_wrist_frame(self) -> dict[str, Any]:
        if (
            getattr(self.robot, 'connected', False)
            and getattr(self.robot, 'armed', False)
            and hasattr(self.robot, 'stop')
            and not self.robot.stop.is_set()
        ):
            try:
                return self.kgrasp_prog.fast_capture()
            except Exception:
                pass
        try:
            f = self.camera.snapshot()
            if f and Path(f.get('path', '')).is_file():
                return {**f, 'joints': self._current_joints_safe(), 'source': 'wrist'}
        except Exception:
            pass
        f = self.camera.latest()
        if f and Path(f.get('path', '')).is_file():
            return {**f, 'joints': self._current_joints_safe(), 'source': 'wrist'}
        raise RuntimeError('目前無法取得手腕相機畫面，請確認手腕相機已連接')

    def status(self) -> dict[str, Any]:
        geo = self.geometry()
        joints = self._current_joints_safe()
        if not (self.current_dir / 'scene.glb').is_file():
            try:
                export_real_to_sim_twin(geo, self.scene_map, joints, self.current_dir, active_plan=self.active_plan)
            except Exception:
                pass
        return {
            'scene_map': self.scene_map.snapshot_dict(geo),
            'active_plan': self.active_plan,
            'joints': joints,
            'cloud_providers': {
                'fal_ai_configured': bool(_read_secret_key(('FAL_KEY',))),
                'world_labs_configured': bool(_read_secret_key(('WORLD_LABS_API_KEY',))),
                'er2_configured': bool(_read_secret_key(('GEMINI_API_KEY', 'GOOGLE_API_KEY'))),
            },
            'worldlabs_last': self.worldlabs_last,
            'qa_history': self.qa_history[-8:],
            'artifacts': {
                'bev_url': '/api/real-to-sim/artifact/bev.jpg',
                'bev_grid_url': '/api/real-to-sim/artifact/bev_grid.jpg',
                'scene_glb_url': '/api/real-to-sim/artifact/scene.glb',
                'twin_state_url': '/api/real-to-sim/artifact/twin_state.json',
            },
        }

    def scan_bev(
        self,
        *,
        mode: str = 'instant',
        query: str = '',
        thinking_level: str = 'low',
        clear_previous: bool = False,
        report=lambda msg: None,
        cancel=None,
    ) -> dict[str, Any]:
        """Run MIRA Orthographic BEV synthesis (1-view instant or 5-view active look-around)
        + Gemini Robotics ER 2 dual-view spatial & physics analysis + Digital Twin GLB export!
        """
        geo = self.geometry()
        if clear_previous:
            self.scene_map.clear()

        frames = []
        if mode == 'multiview' and not getattr(self.robot, 'connected', False):
            try:
                if hasattr(self.robot, 'connect'):
                    self.robot.connect()
            except Exception:
                pass
        if mode == 'multiview' and getattr(self.robot, 'connected', False):
            report('MIRA 主動環視：平滑移至觀察原位並張開夾爪')
            try:
                self.robot.begin(allow_rearm=(self.robot.pwm_owner is None))
            except Exception:
                self._safe_recover_and_retreat_high(geo, report, cancel)
            now_j = self.robot.read()
            q_start = {n: int(now_j[n]) for n in BODY}
            if self.robot.pwm_owner is not None or self.robot.stop.is_set() or geo.table_clearance(q_start)[0] < 0.055:
                q_start = self._safe_recover_and_retreat_high(geo, report, cancel)
                now_j = self.robot.read()
            if int(now_j.get('gripper', OPEN_RAW)) < OPEN_RAW - 20:
                self.robot.glide({'gripper': OPEN_RAW}, speed=120, precision=8, allow_rearm=False)
            looks = look_plan(geo)
            self.kgrasp_prog.screen_transit(geo, q_start, looks['home'])
            table_guard = self.kgrasp_prog.guard(geo, table_floor=0.03)
            self.kgrasp_prog.fast_move(looks['home'], guard=table_guard, cancel=cancel, speed_ticks_per_s=380.0)
            for idx, v in enumerate(looks['views']):
                if cancel and cancel.is_set():
                    raise RuntimeError('已停止')
                report(f"MIRA 快速環視掃描 {idx + 1} / {len(looks['views'])} (pan {v['pan_offset']:+d})")
                if idx > 0:
                    self.kgrasp_prog.fast_move(v['pose'], guard=table_guard, cancel=cancel, speed_ticks_per_s=380.0)
                frm = self.kgrasp_prog.fast_capture(guard=table_guard, cancel=cancel)
                frm['look_pan_offset'] = v['pan_offset']
                frames.append(frm)
            # Return smoothly to center observation pose
            self.kgrasp_prog.fast_move(looks['home'], guard=table_guard, cancel=cancel, speed_ticks_per_s=380.0)
        else:
            report('擷取手腕畫面並執行 MIRA 正交俯視投影 (BEV)')
            frm = self._capture_wrist_frame()
            frm['look_pan_offset'] = 0
            frames.append(frm)

        bev_info = synthesize_orthographic_bev(geo, frames, self.current_dir, objects=self.scene_map.objects)
        frame_paths_arg = frames[0]['path'] if len(frames) == 1 else [f['path'] for f in frames]

        report('Gemini Robotics ER 2 正在多視角 (手腕環視 + MIRA 正交俯視圖) 推理空間與物理屬性')
        er2_result = analyze_dual_view_with_er2(
            self.er2_brain,
            frame_paths_arg,
            bev_info['bev_path'],
            query=query,
            thinking_level=thinking_level,
        )

        detected_objects = []
        for item in er2_result.get('items', []):
            if not isinstance(item, dict):
                continue
            v_idx = int(item.get('view_index') or 0)
            chosen_frame = frames[v_idx] if 0 <= v_idx < len(frames) else frames[0]
            solved = solve_oriented_object(
                geo,
                chosen_frame,
                item,
                crop_dir=self.scene_map.crops_dir,
            )
            if solved is not None:
                merged = self.scene_map.upsert_object(solved, force_fresh=True)
                detected_objects.append(merged)

        # Fresh scan observed the live workspace; clear global dirty_after_engage flag
        self.scene_map.dirty_after_engage = False
        # Re-render BEV grid with newly solved oriented object ellipses
        bev_info = synthesize_orthographic_bev(geo, frames, self.current_dir, objects=self.scene_map.objects)
        self.scene_map.last_bev = bev_info
        self.scene_map.last_summary_zh = str(er2_result.get('summary') or f'MIRA 重建完成，共追蹤 {len(self.scene_map.objects)} 個桌面物件')
        self.scene_map.save()

        joints = self._current_joints_safe()
        twin_state = export_real_to_sim_twin(
            geo,
            self.scene_map,
            joints,
            self.current_dir,
            active_plan=self.active_plan,
        )

        return {
            'mode': mode,
            'summary_zh': self.scene_map.last_summary_zh,
            'bev': bev_info,
            'er2_analysis': er2_result,
            'detected_count': len(detected_objects),
            'scene_map': self.scene_map.snapshot_dict(geo),
            'twin_state': twin_state,
        }

    def plan_task(
        self,
        instruction: str,
        *,
        place_slot_label: str | None = None,
        target_object_id: str | None = None,
        thinking_level: str = 'low',
        report=lambda msg: None,
        cancel=None,
    ) -> dict[str, Any]:
        """Plan a zero-shot Grasp or 6-DoF Pick-and-Place task and run Mental Simulation in the Digital Twin!"""
        geo = self.geometry()
        instruction = (instruction or '').strip()

        # If scene map is empty, run an instant MIRA BEV scan first
        if not self.scene_map.objects:
            report('場景記憶為空，先執行 MIRA 正交俯視圖掃描')
            self.scan_bev(mode='instant', query=instruction, thinking_level=thinking_level, report=report, cancel=cancel)

        if not self.scene_map.objects:
            raise RuntimeError('桌面上尚未偵測到可操作的物件，請確認物件在手腕視野內並按「MIRA 掃描建模」')

        # 1. Select target object
        target_obj = None
        if target_object_id:
            target_obj = next((o for o in self.scene_map.objects if o['id'] == target_object_id), None)
        if target_obj is None and instruction:
            for o in self.scene_map.objects:
                if o['name'] in instruction or any(tok in o['name'] for tok in instruction.split() if len(tok) >= 2):
                    target_obj = o
                    break
        if target_obj is None:
            target_obj = self.scene_map.objects[0]

        # 2. Check if instruction or place_slot_label requests a Pick-and-Place destination
        place_keywords = ('放到', '移到', '搬到', '擺到', '放在', '移至', '左邊', '右邊', '前方', '後方', '中央', '旁邊', '空位')
        wants_place = bool(place_slot_label) or any(k in instruction for k in place_keywords)

        place_target_m = None
        place_label_zh = ''
        rel_info = self.scene_map.compute_spatial_relations(geo)
        slots = rel_info.get('free_space_slots', [])

        if wants_place:
            if place_slot_label:
                matched_slot = next((s for s in slots if s['slot_label'] == place_slot_label), None)
                if matched_slot:
                    place_target_m = matched_slot['position_m']
                    place_label_zh = matched_slot['slot_label']
            if place_target_m is None and instruction:
                for s in slots:
                    if s['slot_label'] in instruction or (
                        s.get('reference_object_name') in instruction and s.get('direction', '')[:2] in instruction
                    ):
                        place_target_m = s['position_m']
                        place_label_zh = s['slot_label']
                        break
                if place_target_m is None:
                    for kw, match_sub in (
                        ('左前', '左前方'),
                        ('右前', '右前方'),
                        ('左', '左側'),
                        ('右', '右側'),
                        ('中央', '中央'),
                        ('前', '正前方'),
                        ('後', '正後方'),
                    ):
                        if kw in instruction:
                            cand_slot = next((s for s in slots if match_sub in s['slot_label']), None)
                            if cand_slot:
                                place_target_m = cand_slot['position_m']
                                place_label_zh = cand_slot['slot_label']
                                break
            if place_target_m is None and slots:
                place_target_m = slots[0]['position_m']
                place_label_zh = slots[0]['slot_label']

        joints = self._current_joints_safe()
        q_now = {n: int(joints.get(n, HOME[n])) for n in BODY}
        gripper_raw = max(int(joints.get('gripper', OPEN_RAW)), OPEN_RAW)

        report(f"正在數位孿生沙盒中規劃並預演：{'夾取並放置' if place_target_m else '精準窄軸夾取'}「{target_obj['name']}」")
        plan_data = plan_mira_grasp_or_place(
            geo,
            self.scene_map,
            target_obj,
            q_now,
            gripper_raw,
            place_target_m=place_target_m,
            place_label_zh=place_label_zh,
        )
        plan_id = f"r2s-{int(time.time())}-{np.random.default_rng().integers(1 << 24):x}"
        plan_data['plan_id'] = plan_id
        plan_data['instruction'] = instruction
        plan_data['created_monotonic'] = time.monotonic()
        plan_data['start_joints'] = q_now
        plan_data['gripper_raw'] = gripper_raw

        self.active_plan = plan_data
        # Also sync with kgrasp_prog.pending if it's a pure grasp so standard execute endpoints can share it
        self.kgrasp_prog.pending = {
            'id': plan_id,
            'created': time.monotonic(),
            'start': q_now,
            'gripper_raw': gripper_raw,
            'plan': plan_data,
            'query': instruction or target_obj['name'],
        }

        export_real_to_sim_twin(
            geo,
            self.scene_map,
            joints,
            self.current_dir,
            active_plan=self.active_plan,
        )
        return plan_data

    def _safe_recover_and_retreat_high(
        self,
        geo: Geometry,
        report=lambda msg: None,
        cancel=None,
        *,
        target_pan: int | None = None,
    ) -> dict[str, int]:
        """Safely neutralize any pending low-PWM gripper session, open jaws to OPEN_RAW (2033),
        clear stop latch, lift vertically out of clutter, and move to a high observation pose."""
        if hasattr(self.kgrasp_prog, 'recover_and_retreat_high'):
            return self.kgrasp_prog.recover_and_retreat_high(geo, report, cancel=cancel, target_pan=target_pan)
        r = self.robot
        r.stop.clear()
        try:
            r.begin(allow_rearm=True)
        except Exception:
            pass
        now_q = r.read()
        return {n: int(now_q.get(n, HOME[n])) for n in BODY}

    def _refine_object_from_near_frame(
        self,
        geo: Geometry,
        obj: dict[str, Any],
        near_frm: dict[str, Any],
        cancel=None,
        *,
        ref_frm: dict[str, Any] | None = None,
        ref_ans: dict[str, Any] | None = None,
        max_shift_m: float = 0.16,
    ) -> tuple[dict[str, Any] | None, float]:
        """Refine 3D object center, height, and radius from a close-up wrist frame (`near_frm`),
        fusing 2-View Active Stereo Ray Triangulation (`stereo_triangulate_object`) whenever a
        reference view (`ref_frm` or `obj['joints']`) with >= 22 mm baseline is available."""
        c2 = None
        r2 = float(obj.get('semi_minor_m', obj['radius']))
        h2 = float(obj.get('height', 0.08))
        feet_in_jaw = False
        stereo_ok = False
        stereo_info: dict[str, Any] = {}

        if ref_frm is None and isinstance(obj.get('joints'), dict) and all(k in obj['joints'] for k in BODY):
            ref_frm = {
                'id': str(obj.get('view_frame') or ''),
                'path': str(obj.get('view_path') or ''),
                'joints': {k: int(obj['joints'][k]) for k in BODY},
            }
        if ref_ans is None and any(
            obj.get(k) is not None
            for k in ('wrist_bottom_left_point', 'wrist_bottom_right_point', 'wrist_box_2d', 'bottom_left')
        ):
            ref_ans = obj

        desc_str = str(obj.get('description') or '')
        is_non_upright = (obj.get('upright_on_table') is False) or any(
            k in desc_str for k in ('橫', '躺', '平放', '倒', 'horizontal', 'lying')
        )
        if not is_non_upright:
            try:
                obj2, _, ans2 = self.kgrasp_prog.locate(
                    geo,
                    obj['name'],
                    [near_frm],
                    cancel,
                    require_top=False,
                    allow_non_upright=True,
                )
                feet_in_jaw = any(_uv_in_jaw_box(uv, pad=10.0) for uv in obj2.get('refined_feet', ()))
                c2 = np.asarray(obj2['centre'], dtype=float)
                r2 = float(obj2.get('radius', r2))
                if obj2.get('height') and 0.025 <= float(obj2['height']) <= 0.28:
                    h2 = float(obj2['height'])
                q_frm = {n: int(near_frm['joints'][n]) for n in BODY}
                cam_pos = geo.camera(q_frm)[:3, 3]
                cam_foot = cam_pos - geo.n * (geo.n @ cam_pos - geo.d)

                # Stage A: 2-View Active Stereo Ray Triangulation (when reference view has >= 22 mm baseline)
                if ref_frm is not None and isinstance(ref_frm.get('joints'), dict):
                    q_ref = {n: float(ref_frm['joints'][n]) for n in BODY if n in ref_frm['joints']}
                    if len(q_ref) == len(BODY):
                        ca_ref = geo.camera(q_ref)[:3, 3]
                        if float(np.linalg.norm(ca_ref - cam_pos)) >= 0.022:
                            if ref_ans is None and ref_frm.get('path'):
                                try:
                                    _, _, ref_ans = self.kgrasp_prog.locate(
                                        geo,
                                        obj['name'],
                                        [ref_frm],
                                        cancel,
                                        require_top=False,
                                        allow_non_upright=True,
                                    )
                                except Exception:
                                    ref_ans = None
                            if isinstance(ref_ans, dict):
                                st = stereo_triangulate_object(geo, ref_frm, ref_ans, near_frm, ans2)
                                if st is not None and float(st.get('stereo_ray_gap_mm', 99.0)) <= 18.0:
                                    c2 = np.asarray(st['centre'], dtype=float)
                                    h2 = float(st['height'])
                                    r2 = float(st['radius'])
                                    stereo_ok = True
                                    stereo_info = st
                                    feet_in_jaw = False

                # Stage B: Single-View Geometric Regime Classification (if 2-view stereo was unavailable)
                if not stereo_ok:
                    aspect_ratio = float(obj.get('semi_major_m', r2)) / max(float(obj.get('semi_minor_m', r2)), 1e-4)
                    is_low_or_box = (
                        obj.get('shape_primitive') == 'box'
                        or float(obj.get('height', 0.08)) <= 0.058
                        or aspect_ratio >= 1.32
                    )
                    if not ans2.get('upright_on_table', True):
                        c2 = None
                    elif is_low_or_box:
                        raw_h = float(obj.get('height') or 0.041)
                        h2 = min(raw_h, 0.055)
                        if ans2.get('top_visible') and isinstance(ans2.get('top_center'), dict):
                            bl = ans2.get('bottom_left') or {'x': ans2['top_center']['x'], 'y': ans2['top_center']['y']}
                            br = ans2.get('bottom_right') or bl
                            base_mid_x = 0.5 * (float(bl['x']) + float(br['x']))
                            base_mid_y = 0.5 * (float(bl['y']) + float(br['y']))
                            mid_uv = (
                                0.5 * (base_mid_x + float(ans2['top_center']['x'])) * IMAGE[0],
                                0.5 * (base_mid_y + float(ans2['top_center']['y'])) * IMAGE[1],
                            )
                            p_mid = geo.ray_to_table(q_frm, mid_uv, height=0.5 * h2)
                            c2 = p_mid - geo.n * (0.5 * h2)
                            feet_in_jaw = False
                        else:
                            u_ray = c2 - cam_foot
                            u_ray = u_ray - geo.n * (geo.n @ u_ray)
                            u_norm = float(np.linalg.norm(u_ray))
                            if u_norm > 1e-6:
                                c2 = c2 + (u_ray / u_norm) * (0.8 * float(obj2['radius']))
                    elif ans2.get('top_visible') and isinstance(ans2.get('top_center'), dict):
                        h_can = float(obj.get('height') or h2 or 0.105)
                        top_uv = (float(ans2['top_center']['x']) * IMAGE[0], float(ans2['top_center']['y']) * IMAGE[1])
                        c_top = geo.ray_to_table(q_frm, top_uv, height=h_can) - geo.n * h_can
                        if feet_in_jaw or np.linalg.norm(c2[:2] - c_top[:2]) > 0.018:
                            c2 = np.asarray(c_top if feet_in_jaw else (0.72 * c_top + 0.28 * c2), dtype=float)
                            feet_in_jaw = False
            except Exception:
                c2 = None

        # Fallback to ER2 spatial pointing/bounding box on `near_frm` if `locate` failed or feet were in jaw
        if c2 is None or feet_in_jaw:
            try:
                det = self.er2_brain.detect_spatial_targets(near_frm['path'], obj['name'], thinking_level='none')
                items = det.get('items') or []
                if items:
                    solved = solve_oriented_object(geo, near_frm, items[0], crop_dir=self.scene_map.crops_dir)
                    if solved is not None:
                        c2 = np.asarray(solved['centre'], dtype=float)
                        r2 = float(solved.get('semi_minor_m', solved['radius']))
                        h2 = float(solved.get('height', h2))
                        feet_in_jaw = False
            except Exception:
                pass

        if c2 is None or feet_in_jaw:
            return None, 0.0

        shift = float(np.linalg.norm(c2 - np.asarray(obj['centre'], dtype=float)))
        if shift > max_shift_m:
            return None, shift

        refined_ts = round(time.time(), 2)
        fit_r = round(float(np.clip(min(float(obj.get('semi_minor_m', obj['radius'])), r2 + PLAN_RADIUS_MARGIN), 0.0145, 0.0258)), 5)
        is_low_regime = (
            (not stereo_info.get('is_upright_column', True))
            if stereo_ok
            else bool(obj.get('shape_primitive') == 'box' or h2 <= 0.058 or is_non_upright)
        )
        raw_meas_r = float(stereo_info.get('measured_radius_m', r2)) if stereo_ok else r2
        meas_r = round(float(np.clip(fit_r if is_low_regime else max(raw_meas_r, float(obj.get('measured_radius_m') or fit_r)), 0.014, 0.034)), 5)
        cand_obj = {
            **obj,
            'centre': [round(float(v), 5) for v in c2],
            'uninflated_radius_m': round(float(np.clip(min(float(obj.get('semi_minor_m', obj['radius'])), r2), 0.014, 0.0255)), 5),
            'measured_radius_m': meas_r,
            'radius': fit_r,
            'semi_minor_m': fit_r,
            'height': round(float(h2), 5),
            'shape_primitive': ('cylinder' if stereo_info.get('is_upright_column') else 'box') if stereo_ok else obj.get('shape_primitive', 'cylinder'),
            'stereo_triangulated': stereo_ok,
            'stereo_baseline_mm': stereo_info.get('stereo_baseline_mm'),
            'stereo_ray_gap_mm': stereo_info.get('stereo_ray_gap_mm'),
            'remeasure_shift_mm': round(shift * 1000.0, 1),
            'near_refined_at': refined_ts,
            'dirty_after_engage': False,
        }
        return cand_obj, shift

    def reobserve_and_remodel_target(
        self,
        geo: Geometry,
        obj: dict[str, Any],
        report=lambda msg: None,
        cancel=None,
        *,
        reason: str = '',
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Re-observe the workspace from a high sector view + closeup `near_view` after any physical engagement
        or failed grasp, update `WorkspaceSceneMap` (`scene-map.json`), MIRA BEV (`bev_grid.jpg`), and Digital
        Twin (`scene.glb`) with 100% fresh coordinates, and return `(updated_obj, remodel_receipt)`."""
        old_centre = [round(float(v), 5) for v in obj['centre']]
        looks = look_plan(geo)
        # Choose the high observation view whose shoulder_pan best faces the object's sector
        c_old = np.asarray(obj['centre'], dtype=float)
        ex, ey, _, origin = _table_basis(geo)
        rel = c_old - origin
        dx, dy = float(rel @ ex), float(rel @ ey)
        # Positive dy is left (pan < 2030), negative dy is right (pan > 2030)
        approx_pan_offset = int(np.clip(round(-math.degrees(math.atan2(dy, max(dx, 0.10))) * (4096.0 / 360.0)), -700, 700))
        best_view = min(looks['views'], key=lambda v: abs(int(v['pan_offset']) - approx_pan_offset))
        sector_pose = dict(best_view['pose'])

        report(f"🔍 重新認知現場環境：移至高空視角 (pan {best_view['pan_offset']:+d}) 掃描「{obj['name']}」最新位置")
        self._safe_recover_and_retreat_high(geo, report, cancel, target_pan=sector_pose['shoulder_pan'])
        table_guard = self.kgrasp_prog.guard(geo, table_floor=0.03)
        self.kgrasp_prog.fast_move(sector_pose, guard=table_guard, cancel=cancel, speed_ticks_per_s=360.0)
        sector_frm = self.kgrasp_prog.fast_capture(guard=table_guard, cancel=cancel)
        sector_frm['look_pan_offset'] = best_view['pan_offset']

        bev_info = synthesize_orthographic_bev(geo, [sector_frm], self.current_dir, objects=self.scene_map.objects)
        updated_obj = dict(obj)
        matched_in_sector = False

        try:
            er2_res = analyze_dual_view_with_er2(
                self.er2_brain,
                sector_frm['path'],
                bev_info['bev_path'],
                query=obj['name'],
                thinking_level='low',
            )
            for item in er2_res.get('items', []):
                if not isinstance(item, dict):
                    continue
                solved = solve_oriented_object(geo, sector_frm, item, crop_dir=self.scene_map.crops_dir)
                if solved is not None:
                    merged = self.scene_map.upsert_object(solved, force_fresh=True)
                    if merged['id'] == obj.get('id') or merged['name'] == obj.get('name') or any(
                        tok in str(merged.get('name', '')) for tok in (obj.get('name', '')[:2],) if len(tok) >= 2
                    ):
                        updated_obj = dict(merged)
                        matched_in_sector = True
        except Exception:
            pass

        if not matched_in_sector:
            cand_sec, _ = self._refine_object_from_near_frame(geo, updated_obj, sector_frm, cancel, max_shift_m=0.22)
            if cand_sec is not None:
                updated_obj = self.scene_map.upsert_object(cand_sec, force_fresh=True)
                matched_in_sector = True

        # Stage 2: Close-up `near_view` refinement starting from the high `sector_pose`
        gr = OPEN_RAW
        nv = near_view(geo, updated_obj, sector_pose)
        near_frm_id = None
        if nv is not None:
            try:
                report(f"🎯 靠近重新定位後的「{updated_obj['name']}」，進行雙視角立體三角精修")
                nv_guard = self.kgrasp_prog.guard(geo, table_floor=0.02, obj=updated_obj, object_floor=0.010, gripper_raw=gr)
                self.kgrasp_prog.fast_move(nv, guard=nv_guard, cancel=cancel, speed_ticks_per_s=340.0)
                near_frm = self.kgrasp_prog.fast_capture(guard=nv_guard, cancel=cancel)
                near_frm_id = near_frm.get('id')
                cand_nv, _ = self._refine_object_from_near_frame(
                    geo,
                    updated_obj,
                    near_frm,
                    cancel,
                    ref_frm=sector_frm,
                    max_shift_m=0.18,
                )
                if cand_nv is not None:
                    updated_obj = self.scene_map.upsert_object(cand_nv, force_fresh=True)
            except Exception:
                pass

        # Persist updated scene map, BEV, and 3D Digital Twin
        updated_obj['dirty_after_engage'] = False
        updated_obj['near_refined_at'] = round(time.time(), 2)
        for so in self.scene_map.objects:
            if so['id'] == updated_obj.get('id'):
                so.update(updated_obj)
                so['bev_point'] = list(metric_3d_to_bev_norm(updated_obj['centre'], geo))
        if not any(o.get('dirty_after_engage') for o in self.scene_map.objects):
            self.scene_map.dirty_after_engage = False
        self.scene_map.last_bev = synthesize_orthographic_bev(geo, [sector_frm], self.current_dir, objects=self.scene_map.objects)
        self.scene_map.save()
        export_real_to_sim_twin(geo, self.scene_map, self._current_joints_safe(), self.current_dir, active_plan=self.active_plan)

        new_centre = [round(float(v), 5) for v in updated_obj['centre']]
        disp_mm = round(float(np.linalg.norm(np.asarray(new_centre)[:2] - np.asarray(old_centre)[:2])) * 1000.0, 1)
        receipt = {
            'object_id': updated_obj.get('id'),
            'object_name': updated_obj.get('name'),
            'reason': reason,
            'old_centre_m': old_centre,
            'new_centre_m': new_centre,
            'displacement_mm': disp_mm,
            'sector_frame_id': sector_frm.get('id'),
            'near_frame_id': near_frm_id,
            'timestamp': round(time.time(), 2),
        }
        report(f"✅ 重新建模定位完成：「{updated_obj['name']}」新座標 {[round(v*1000, 1) for v in new_centre[:2]]} mm（位移 {disp_mm} mm）")
        return updated_obj, receipt

    def execute_active_plan(
        self,
        plan_id: str | None = None,
        *,
        return_object_if_grasp_only: bool = True,
        max_attempts: int = 3,
        report=lambda msg: None,
        cancel=None,
    ) -> dict[str, Any]:
        """Execute the Twin-verified Grasp or Pick-and-Place plan on the physical SO-101 arm
        with Post-Engage Scene Re-Modeling and Closed-Loop Recover & Retry!"""
        plan = self.active_plan
        if not plan or (plan_id and plan.get('plan_id') != plan_id):
            raise RuntimeError('找不到已通過數位孿生預演的執行計畫，請先點擊「沙盒預演規劃」')
        if not plan.get('executable'):
            raise RuntimeError('此計畫在數位孿生預演中發現碰撞或限制問題，為保護安全不執行：' + '；'.join(plan.get('problems', [])))

        r = self.robot
        geo = self.geometry()
        obj = dict(plan['object'])
        # Sync latest object state from WorkspaceSceneMap in case a previous engagement marked it dirty
        for so in self.scene_map.objects:
            if so['id'] == obj.get('id'):
                obj = dict(so)
                break
        gr = int(plan.get('gripper_raw', OPEN_RAW))

        # 1. Pre-Execute Recovery: if gripper is in a pending PWM session, stop is latched, or arm is low over table
        try:
            r.begin(allow_rearm=(r.pwm_owner is None))
        except RuntimeError:
            self._safe_recover_and_retreat_high(geo, report, cancel)
        now = r.read()
        q_start_body = {n: int(now[n]) for n in BODY}
        if r.pwm_owner is not None or r.stop.is_set() or geo.table_clearance(q_start_body)[0] < 0.055:
            q_start_body = self._safe_recover_and_retreat_high(geo, report, cancel)
            now = r.read()
        if int(now.get('gripper', gr)) < OPEN_RAW - 20:
            report('張開夾爪至標準開口')
            r.glide({'gripper': OPEN_RAW}, speed=50, precision=8, allow_rearm=False)

        remodel_history: list[dict[str, Any]] = []
        recovery_attempts: list[dict[str, Any]] = []
        steps_log: list[dict[str, Any]] = []

        # 2. Pre-Execute Scene Freshness Check:
        # If the workspace or target object was physically engaged since the last scan, re-observe & re-model first!
        scene_dirty = bool(self.scene_map.dirty_after_engage or obj.get('dirty_after_engage'))
        already_refined = (not scene_dirty) and ((time.time() - float(obj.get('near_refined_at') or 0.0)) < 90.0)

        if scene_dirty:
            report(f"🔍 偵測到上次動作後現場可能已變動，先重新掃描建模並定位「{obj['name']}」")
            obj, rem_rec = self.reobserve_and_remodel_target(
                geo,
                obj,
                report,
                cancel,
                reason='pre_execute_dirty_after_engage',
            )
            remodel_history.append(rem_rec)
            now_re = r.read()
            q_re = {n: int(now_re[n]) for n in BODY}
            re_plan = plan_mira_grasp_or_place(
                geo,
                self.scene_map,
                obj,
                q_re,
                gr,
                place_target_m=plan.get('place_target_m'),
                place_label_zh=str(plan.get('place_label_zh') or ''),
            )
            if re_plan.get('executable'):
                plan = {**plan, **re_plan, 'object': re_plan['object']}
                obj = plan['object']
        elif not already_refined:
            nv = near_view(geo, obj, q_start_body)
            if nv is not None:
                try:
                    report(f"靠近「{obj['name']}」再看一次，以雙視角主動立體三角測量校正 3D 高度與中心")
                    nv_guard = self.kgrasp_prog.guard(geo, table_floor=0.02, obj=obj, object_floor=0.012, gripper_raw=gr)
                    high_ref_frm = None
                    has_stored_ref = (
                        isinstance(obj.get('joints'), dict)
                        and all(k in obj['joints'] for k in BODY)
                        and float(np.linalg.norm(geo.camera(obj['joints'])[:3, 3] - geo.camera(nv)[:3, 3])) >= 0.022
                    )
                    if not has_stored_ref and float(np.linalg.norm(geo.camera(q_start_body)[:3, 3] - geo.camera(nv)[:3, 3])) >= 0.022:
                        try:
                            high_ref_frm = self.kgrasp_prog.fast_capture(guard=nv_guard, cancel=cancel)
                        except Exception:
                            high_ref_frm = None
                    self.kgrasp_prog.fast_move(nv, guard=nv_guard, cancel=cancel, speed_ticks_per_s=340.0)
                    near_frm = self.kgrasp_prog.fast_capture(guard=nv_guard, cancel=cancel)
                    cand_obj, shift = self._refine_object_from_near_frame(
                        geo,
                        obj,
                        near_frm,
                        cancel,
                        ref_frm=high_ref_frm,
                        max_shift_m=0.14,
                    )
                    now_nv = r.read()
                    q_nv = {n: int(now_nv[n]) for n in BODY}
                    updated_from_nv = False
                    if cand_obj is not None:
                        re_plan = plan_mira_grasp_or_place(
                            geo,
                            self.scene_map,
                            cand_obj,
                            q_nv,
                            gr,
                            place_target_m=plan.get('place_target_m'),
                            place_label_zh=str(plan.get('place_label_zh') or ''),
                        )
                        if re_plan.get('executable'):
                            obj = re_plan['object']
                            plan = {**plan, **re_plan, 'object': obj}
                            updated_from_nv = True
                            for so in self.scene_map.objects:
                                if so['id'] == obj.get('id'):
                                    true_c = list(obj.get('true_centre') or obj['centre'])
                                    so['centre'] = true_c
                                    so['radius'] = obj['radius']
                                    so['semi_minor_m'] = obj['semi_minor_m']
                                    so['measured_radius_m'] = obj.get('measured_radius_m', obj['radius'])
                                    so['height'] = cand_obj.get('height', obj.get('height', so.get('height', 0.08)))
                                    so['stereo_triangulated'] = bool(cand_obj.get('stereo_triangulated'))
                                    so['near_refined_at'] = cand_obj['near_refined_at']
                                    so['dirty_after_engage'] = False
                                    so['bev_point'] = list(metric_3d_to_bev_norm(true_c, geo))
                            self.scene_map.save()
                    if not updated_from_nv:
                        re_plan_orig = plan_mira_grasp_or_place(
                            geo,
                            self.scene_map,
                            obj,
                            q_nv,
                            gr,
                            place_target_m=plan.get('place_target_m'),
                            place_label_zh=str(plan.get('place_label_zh') or ''),
                        )
                        if re_plan_orig.get('executable'):
                            plan = {**plan, **re_plan_orig, 'object': re_plan_orig['object']}
                        else:
                            self.kgrasp_prog.fast_move(q_start_body, guard=nv_guard, cancel=cancel, speed_ticks_per_s=340.0)
                except Exception:
                    pass

        def run_step(label: str, target_q: dict[str, int], **kw):
            actual, err = self.kgrasp_prog.move(target_q, cancel=cancel, **kw)
            q_act = {n: actual[n] for n in BODY}
            rec = {
                'label': label,
                'target': target_q,
                'actual': q_act,
                'error_ticks': err,
                'table_mm': round(geo.table_clearance(q_act)[0] * 1000.0, 1),
            }
            steps_log.append(rec)
            return q_act

        def run_fast_step(label: str, target_q: dict[str, int], *, guard, speed_ticks_per_s: float = 300.0, wait_settle: bool = True):
            actual, err = self.kgrasp_prog.fast_move(
                target_q,
                guard=guard,
                cancel=cancel,
                speed_ticks_per_s=speed_ticks_per_s,
                wait_settle=wait_settle,
            )
            q_act = {n: actual[n] for n in BODY}
            rec = {
                'label': label,
                'target': target_q,
                'actual': q_act,
                'error_ticks': err,
                'table_mm': round(geo.table_clearance(q_act)[0] * 1000.0, 1),
                'mode': 'fast_stream_25hz',
                'speed_ticks_per_s': speed_ticks_per_s,
            }
            steps_log.append(rec)
            return q_act

        lift_guard = self.kgrasp_prog.guard(geo, table_floor=0.002)
        max_attempts = max(1, int(max_attempts))
        q_grasp = None
        q_lift = None
        tel = None
        lifted_mm = 0.0
        motor_ok = False
        grasp_confirmed = False
        weight_sag = 0
        verdict = None

        # 3. Closed-Loop Grasp Execution with Automatic Recover, Re-Model & Retry!
        for attempt in range(1, max_attempts + 1):
            obj = plan['object']
            W = {w['label']: w['target'] for w in plan['waypoints']}
            eff_table_floor = 0.002 if float(obj.get('height', 0.08)) <= 0.060 else 0.005
            free_guard = self.kgrasp_prog.guard(geo, table_floor=0.015, obj=obj, object_floor=0.002, gripper_raw=gr)
            close_guard = self.kgrasp_prog.guard(geo, table_floor=eff_table_floor, obj=obj, object_floor=-0.004, gripper_raw=gr)

            try:
                attempt_prefix = f"[第 {attempt}/{max_attempts} 次嘗試] " if attempt > 1 else ""
                report(f"{attempt_prefix}高速平滑移至「{obj['name']}」上方預備點")
                for lbl_pre in ('pre_high_lift', 'pre_high_transit'):
                    if lbl_pre in W:
                        run_fast_step(lbl_pre, W[lbl_pre], guard=lift_guard, speed_ticks_per_s=320.0, wait_settle=False)
                run_fast_step('above_pregrasp', W['above_pregrasp'], guard=free_guard, speed_ticks_per_s=320.0, wait_settle=False)

                # Physical engagement begins once descending below above_pregrasp: mark scene & object dirty!
                self.scene_map.mark_engaged(obj.get('id'), obj.get('centre'), reason=f'attempt_{attempt}_approach')

                if 'approach_70mm' in W:
                    report(f"{attempt_prefix}高速對準物件窄軸與左固定牙邊緣 (approach_70mm)")
                    run_fast_step('approach_70mm', W['approach_70mm'], guard=free_guard, speed_ticks_per_s=280.0, wait_settle=True)
                for lbl in ('approach_50mm', 'approach_30mm', 'approach_15mm'):
                    if lbl in W:
                        report(f"{attempt_prefix}連續流暢沿窄軸直線推進 ({lbl})")
                        run_fast_step(
                            lbl,
                            W[lbl],
                            guard=close_guard,
                            speed_ticks_per_s=200.0,
                            wait_settle=(lbl in ('approach_50mm', 'approach_15mm')),
                        )
                        if lbl == 'approach_50mm':
                            try:
                                servo_frm = self.kgrasp_prog.fast_capture(guard=close_guard, cancel=cancel)
                                servo_res = in_hand_jaw_corridor_servo(geo, gr, servo_frm.get('path', ''), d_ahead_m=0.035)
                                if servo_res is not None:
                                    steps_log.append({'label': 'jaw_corridor_servo', **servo_res})
                                if servo_res is not None and servo_res.get('needs_correction'):
                                    delta_xh = float(servo_res['delta_xh_m'])
                                    Ttcp, _, _, _ = geo.tcp(gr)
                                    Gg = geo.transforms(W['grasp'])['gripper_link']
                                    xb = Gg[:3, :3] @ Ttcp[:3, 0]
                                    xb = xb - geo.n * (geo.n @ xb)
                                    xb_norm = float(np.linalg.norm(xb))
                                    if xb_norm > 1e-6:
                                        xb = xb / xb_norm
                                        offset_base = -xb * delta_xh
                                        shifted_wps: dict[str, dict[str, int]] = {}
                                        all_valid = True
                                        for k_wp in ('approach_30mm', 'approach_15mm', 'grasp', 'lift'):
                                            if k_wp not in W:
                                                continue
                                            q_corr, p_err, r_err = pose_along(geo, W[k_wp], offset_base, W[k_wp])
                                            if p_err > 3.5 or r_err > 4.5 or geo.table_clearance(q_corr)[0] < TABLE_FLOOR:
                                                all_valid = False
                                                break
                                            shifted_wps[k_wp] = {n: int(round(q_corr[n])) for n in BODY}
                                        if all_valid and shifted_wps:
                                            for k_wp, q_wp in shifted_wps.items():
                                                W[k_wp] = q_wp
                                            obj = {
                                                **obj,
                                                'centre': [round(float(v), 5) for v in (np.asarray(obj['centre'], dtype=float) + offset_base)],
                                            }
                                            close_guard = self.kgrasp_prog.guard(
                                                geo,
                                                table_floor=eff_table_floor,
                                                obj=obj,
                                                object_floor=-0.004,
                                                gripper_raw=gr,
                                            )
                                            report(
                                                f"{attempt_prefix}👁️ 手眼夾爪走廊視覺伺服：左牙間隙 {servo_res['left_jaw_gap_mm']:+.1f} mm -> 橫向微調 {delta_xh*1000:+.1f} mm"
                                            )
                            except Exception:
                                pass

                report(f"{attempt_prefix}到達夾取點，鎖定左固定牙貼邊姿態並合爪")
                q_grasp = run_step('grasp', W['grasp'], guard=close_guard, speed=68, precision=58, correct=True, tolerance=4)
                ref_frame = self.grasp_prog.capture()
                closed = self.grasp_prog.close_candidate(ref_frame, report, cancel, squeeze_ticks=SQUEEZE_TICKS)
                sag_before = dict(self.kgrasp_prog.sag)

                report(f"{attempt_prefix}垂直流暢抬升物件並確認抓取狀態")
                q_lift = run_fast_step('lift', W['lift'], guard=lift_guard, speed_ticks_per_s=210.0, wait_settle=True)
                after_lift = self.grasp_prog.capture()
                tel = r.gripper_telemetry()
                G0, G1 = geo.transforms(q_grasp)['gripper_link'], geo.transforms(q_lift)['gripper_link']
                lifted_mm = float(geo.n @ (G1[:3, 3] - G0[:3, 3]) * 1000.0)
                motor_ok = bool(lifted_mm >= 14.0 and tel['position'] > 1535)
                weight_sag = self.kgrasp_prog.sag['shoulder_lift'] - sag_before['shoulder_lift']

                # Visual + Motor Grasp Confirmation: verify object is genuinely held between fingers
                evidence = {
                    'measured_lift_mm': round(lifted_mm, 1),
                    'gripper_position': tel['position'],
                    'gripper_load_signed_pwm': tel.get('load_signed_pwm', 0),
                    'empty_jaw_position_threshold': 1525,
                    'shoulder_lift_sag_ticks': {'before_lift': sag_before['shoulder_lift'], 'lifted': self.kgrasp_prog.sag['shoulder_lift']},
                }
                verdict_empty = False
                follows_gripper = True
                try:
                    if ref_frame.get('source') == 'wrist' and after_lift.get('source') == 'wrist':
                        verdict = self.brain.verify_grasp(obj['name'], [ref_frame, after_lift], evidence, cancel)
                        verdict_empty = bool(verdict.get('gripper_empty', False))
                        if 'follows_gripper' in verdict and verdict['follows_gripper'] is not None:
                            follows_gripper = bool(verdict['follows_gripper'])
                except Exception:
                    verdict = None

                # Physical caliper + torque hold check:
                # NOTE: When grasping a tall object (e.g. 40-140 mm) and lifting, if the jaws slip vertically
                # upward along the object's walls while the object stays on the table, `tel['position']` will STILL
                # be >= 1610 because the upper body of the object is still between the jaws!
                # Therefore, `solid_caliper_hold` MUST NOT override `follows_gripper == False`, and requires
                # sustained clamping load (`abs(load_signed_pwm) >= 35`) without jaw collapse.
                contact_pos = int(closed.get('contact_sample', {}).get('position', tel['position']))
                jaw_maintained = int(tel['position']) >= (contact_pos - SQUEEZE_TICKS - 12)
                clamping_torque_ok = abs(int(tel.get('load_signed_pwm', 0))) >= 35
                solid_caliper_hold = bool(
                    lifted_mm >= 28.0
                    and int(tel['position']) >= 1610
                    and jaw_maintained
                    and clamping_torque_ok
                    and follows_gripper
                )
                grasp_confirmed = bool(
                    motor_ok
                    and follows_gripper
                    and (not verdict_empty or solid_caliper_hold)
                )
                if not grasp_confirmed:
                    raise RuntimeError(
                        f"抬升後確認未抓穩或發生垂直滑脫「{obj['name']}」（motor_ok={motor_ok}, "
                        f"follows_gripper={follows_gripper}, empty={verdict_empty}, "
                        f"gripper={tel['position']}, load={tel.get('load_signed_pwm', 0)}）"
                    )
                # Success! Break out of the retry loop and proceed to place / release.
                break

            except Exception as engage_exc:
                if cancel is not None and cancel.is_set():
                    raise
                fail_reason = str(engage_exc)
                recovery_attempts.append({
                    'attempt': attempt,
                    'failed_centre_m': list(obj.get('centre', [])),
                    'error': fail_reason,
                    'timestamp': round(time.time(), 2),
                })
                # Always recover the low-PWM gripper session and lift vertically out of clutter first!
                report(f"⚠️ 第 {attempt}/{max_attempts} 次夾取未成功（{fail_reason}），正在安全鬆開夾爪並垂直退回高處...")
                self._safe_recover_and_retreat_high(geo, report, cancel)

                if attempt >= max_attempts:
                    try:
                        obj, final_rem = self.reobserve_and_remodel_target(
                            geo,
                            obj,
                            report,
                            cancel,
                            reason=f'final_failure_after_{attempt}_attempts: {fail_reason}',
                        )
                        remodel_history.append(final_rem)
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"經過 {max_attempts} 次閉環重試與重新建模仍未成功夾取「{obj['name']}」：{fail_reason}"
                    ) from engage_exc

                # Re-observe the perturbed scene, re-localize the pushed/shifted object, and re-plan!
                report(
                    f"🔄 啟動自動恢復與重抓 ({attempt}/{max_attempts - 1})：重新認知現場環境、定位被推移的「{obj['name']}」並重新規劃軌跡..."
                )
                obj, rem_rec = self.reobserve_and_remodel_target(
                    geo,
                    obj,
                    report,
                    cancel,
                    reason=f'retry_after_attempt_{attempt}: {fail_reason}',
                )
                remodel_history.append(rem_rec)
                recovery_attempts[-1]['remodeled_centre_m'] = rem_rec['new_centre_m']
                recovery_attempts[-1]['displacement_mm'] = rem_rec['displacement_mm']

                now_retry = r.read()
                q_retry = {n: int(now_retry[n]) for n in BODY}
                re_plan = plan_mira_grasp_or_place(
                    geo,
                    self.scene_map,
                    obj,
                    q_retry,
                    gr,
                    place_target_m=plan.get('place_target_m'),
                    place_label_zh=str(plan.get('place_label_zh') or ''),
                )
                if not re_plan.get('executable'):
                    # Relax radius slightly within safe jaw opening if needed
                    relaxed_obj = {**obj, 'radius': min(float(obj['radius']), 0.0245), 'semi_minor_m': min(float(obj.get('semi_minor_m', obj['radius'])), 0.0245)}
                    re_plan = plan_mira_grasp_or_place(
                        geo,
                        self.scene_map,
                        relaxed_obj,
                        q_retry,
                        gr,
                        place_target_m=plan.get('place_target_m'),
                        place_label_zh=str(plan.get('place_label_zh') or ''),
                    )
                if not re_plan.get('executable'):
                    raise RuntimeError(
                        f"重新定位「{obj['name']}」於 {rem_rec['new_centre_m']} 後無法規劃無碰撞路徑："
                        + '；'.join(re_plan.get('problems', []))
                    ) from engage_exc
                plan = {**plan, **re_plan, 'object': re_plan['object']}
                self.active_plan = plan

        obj = plan['object']
        W = {w['label']: w['target'] for w in plan['waypoints']}

        if plan.get('mode') == 'pick_and_place' and plan.get('place_target_m'):
            report(f"高空流暢跨越至目標放置區：{plan.get('place_label_zh') or '指定位置'}")
            if 'high_lift' in W:
                run_fast_step('high_lift', W['high_lift'], guard=lift_guard, speed_ticks_per_s=260.0, wait_settle=False)
            if 'high_transit' in W:
                run_fast_step('high_transit', W['high_transit'], guard=lift_guard, speed_ticks_per_s=260.0, wait_settle=False)
            if 'above_place' in W:
                run_fast_step('above_place', W['above_place'], guard=lift_guard, speed_ticks_per_s=240.0, wait_settle=True)
            if 'place_descent' in W:
                run_fast_step('place_descent', W['place_descent'], guard=lift_guard, speed_ticks_per_s=180.0, wait_settle=True)
            down_frame = self.grasp_prog.capture()
            if cancel and cancel.wait(0.20):
                raise RuntimeError('已停止')
            report('輕放物件並張開夾爪')

            def retreat_place():
                for lbl in ('place_retreat', 'high_retreat', 'return_home'):
                    if lbl in W:
                        run_fast_step(lbl, W[lbl], guard=lift_guard, speed_ticks_per_s=320.0, wait_settle=(lbl == 'return_home'))
                check_pose = dict(HOME, shoulder_pan=HOME['shoulder_pan'] - 250)
                run_fast_step('empty_check_view', check_pose, guard=lift_guard, speed_ticks_per_s=340.0, wait_settle=True)

            self.grasp_prog.release(down_frame, report, cancel, support_verified=True, retreat=retreat_place)
            # Update object coordinates in WorkspaceSceneMap to the new placed location and mark dirty for next engage!
            self.scene_map.mark_engaged(obj.get('id'), plan['place_target_m'], reason='pick_and_place_release')
            for o in self.scene_map.objects:
                if o['id'] == obj.get('id'):
                    o['centre'] = list(plan['place_target_m'])
                    o['bev_point'] = list(metric_3d_to_bev_norm(o['centre'], geo))
                    o['near_refined_at'] = None
                    o['dirty_after_engage'] = True
                    o['updated_at'] = round(time.time(), 2)
            self.scene_map.save()
        elif return_object_if_grasp_only:
            report('放回原位並鬆開夾爪')
            q_down, _, _ = pose_along(geo, W['grasp'], geo.n * 0.002, W['grasp'])
            run_fast_step('lowered', {n: int(round(q_down[n])) for n in BODY}, guard=lift_guard, speed_ticks_per_s=200.0, wait_settle=True)
            down_frame = self.grasp_prog.capture()
            if cancel and cancel.wait(0.20):
                raise RuntimeError('已停止')

            def retreat_grasp():
                for lbl in ('above_pregrasp',):
                    if lbl in W:
                        run_fast_step('retreat_' + lbl, W[lbl], guard=lift_guard, speed_ticks_per_s=320.0, wait_settle=False)
                check_pose = dict(HOME, shoulder_pan=HOME['shoulder_pan'] - 250)
                run_fast_step('empty_check_view', check_pose, guard=lift_guard, speed_ticks_per_s=340.0, wait_settle=True)

            self.grasp_prog.release(down_frame, report, cancel, support_verified=True, retreat=retreat_grasp)
            for o in self.scene_map.objects:
                if o['id'] == obj.get('id'):
                    o['near_refined_at'] = None
                    o['dirty_after_engage'] = True
            self.scene_map.save()

        self.active_plan = None
        export_real_to_sim_twin(geo, self.scene_map, self._current_joints_safe(), self.current_dir, active_plan=None)
        return {
            'plan_id': plan['plan_id'],
            'mode': plan.get('mode'),
            'object_name': obj.get('name'),
            'object_centre_m': list(obj.get('centre', [])),
            'place_label_zh': plan.get('place_label_zh'),
            'motor_confirmed': motor_ok,
            'grasp_confirmed': grasp_confirmed,
            'lifted_mm': round(lifted_mm, 1),
            'fixed_jaw_gap_mm': round(float((plan.get('grasp') or {}).get('fixed_jaw_gap_mm', FIXED_SIDE_MARGIN * 1000.0)), 1),
            'shoulder_sag_increase_ticks': weight_sag,
            'gripper_position': tel['position'] if tel else OPEN_RAW,
            'attempts_used': len(recovery_attempts) + 1,
            'recovery_attempts': recovery_attempts,
            'remodel_history': remodel_history,
            'verification': verdict,
            'steps': steps_log,
            'completed': True,
            'scene_map': self.scene_map.snapshot_dict(geo),
        }

    def reconstruct_3d(
        self,
        *,
        target: str = 'object',
        object_id: str | None = None,
        prompt_text: str = '',
    ) -> dict[str, Any]:
        """Trigger Cloud 3D Mesh reconstruction via FAL.ai (`FAL_KEY`) or World Labs (`WORLD_LABS_API_KEY`)."""
        geo = self.geometry()
        if target == 'world':
            frm = self._capture_wrist_frame()
            res = generate_world_worldlabs(
                frm['path'],
                self.scene_map.runtime_dir,
                prompt_text=prompt_text or 'Robotic manipulation tabletop workspace',
            )
            self.worldlabs_last = res
            return res

        if not self.scene_map.objects:
            raise RuntimeError('目前場景圖中沒有物件，請先執行「MIRA 掃描建模」')

        obj = None
        if object_id:
            obj = next((o for o in self.scene_map.objects if o['id'] == object_id), None)
        if obj is None:
            obj = self.scene_map.objects[0]

        out_glb = self.scene_map.meshes_dir / f"{obj['id']}.glb"
        res = reconstruct_object_3d_fal(obj, out_glb)
        obj['mesh_glb_path'] = res['local_glb_path']
        obj['shape_primitive'] = 'custom_glb'
        self.scene_map.save()

        export_real_to_sim_twin(
            geo,
            self.scene_map,
            self._current_joints_safe(),
            self.current_dir,
            active_plan=self.active_plan,
        )
        return {
            **res,
            'scene_map': self.scene_map.snapshot_dict(geo),
        }

    def ask_scene_qa(self, question: str, *, thinking_level: str = 'low') -> dict[str, Any]:
        """`steer-a-robot` style non-disruptive Live Scene Q&A grounded in MIRA BEV + 3D Scene Graph."""
        from google.genai import types

        geo = self.geometry()
        question = (question or '').strip()
        if not question:
            raise ValueError('請輸入要詢問的問題')

        frm = self._capture_wrist_frame()
        bev_path = self.current_dir / 'bev_grid.jpg'
        if not bev_path.is_file():
            synthesize_orthographic_bev(geo, [frm], self.current_dir, objects=self.scene_map.objects)

        snap = self.scene_map.snapshot_dict(geo)
        contents: list[Any] = []
        with Image.open(frm['path']) as im1:
            contents.append(im1.convert('RGB'))
        if bev_path.is_file():
            with Image.open(bev_path) as im2:
                contents.append(im2.convert('RGB'))

        scene_json_str = json.dumps(
            {
                'objects': [
                    {
                        'name': o['name'],
                        'centre_mm': [round(v * 1000.0, 1) for v in o['centre']],
                        'dimensions_mm': [
                            round(o.get('semi_major_m', o['radius']) * 2000.0, 1),
                            round(o.get('semi_minor_m', o['radius']) * 2000.0, 1),
                            round(o.get('height', 0.08) * 1000.0, 1),
                        ],
                        'yaw_deg': o.get('yaw_deg', 0.0),
                        'physics': o.get('physics'),
                    }
                    for o in snap['objects']
                ],
                'relations': snap['relations'],
                'free_space_slots': [s['slot_label'] for s in snap['free_space_slots']],
            },
            ensure_ascii=False,
        )
        prompt = f"""You are Gemini Robotics ER 2 (`gemini-robotics-er-2-preview`) + Argon, answering a live question about the robot's tabletop workspace (like LeRobot `steer-a-robot` Scene Q&A).
You have the live wrist camera image, the MIRA Orthographic BEV map (`1 px = 0.5 mm`), and the persistent 3D WorkspaceSceneMap metric data:
{scene_json_str}

User question: {question!r}

Answer concisely and accurately in Traditional Chinese (台灣繁體中文), citing exact millimeter distances, relative directions, or physical properties when helpful."""

        t0 = time.time()
        client = self.er2_brain._client()
        budget_map = {'none': 0, 'low': 256, 'medium': 1024, 'high': -1}
        resp = client.models.generate_content(
            model=self.er2_brain.model,
            contents=[*contents, prompt],
            config=types.GenerateContentConfig(
                temperature=0.25,
                thinking_config=types.ThinkingConfig(thinking_budget=budget_map.get(thinking_level, 256)),
            ),
        )
        answer_zh = (resp.text or '').strip()
        qa_entry = {
            'id': f'qa-{int(time.time() * 1000)}',
            'question': question,
            'answer_zh': answer_zh,
            'latency_ms': round((time.time() - t0) * 1000.0, 1),
            'created_at': round(time.time(), 2),
        }
        self.qa_history.append(qa_entry)
        return qa_entry
