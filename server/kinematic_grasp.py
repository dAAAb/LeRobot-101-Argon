"""Lean grasp of an upright object with the fitted kinematic model (scripts/kinematic_cal).

Locate once, plan locally, execute with measured feedback:
  1. home -> look around (model-planned views, high above the table) -> one VLM call picks the view and
     marks where the object's left/right silhouette meets the table;
  2. those two pixels are intersected with the calibrated (tilted) table plane: centre and radius are
     measured, never taken from a known object size; a closer view repeats the measurement;
  3. local IK plans above-pregrasp -> pregrasp -> straight approach -> grasp -> lift, every segment
     screened against the table plane and the measured cylinder;
  4. execution uses Robot.glide with the servo steady-state error fed forward through Robot.bias and a
     measured-pose guard; the jaw closes through WristGrasp.close_candidate (low PWM); the VLM is called
     again only to verify the lift.

The planning half (Geometry, plans) is pure and has no hardware access.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from .local_cartesian import default_model, stl_vertices, xyz_transform

ROOT = Path(__file__).resolve().parents[1]
BODY = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')
TICK = 2 * math.pi / 4096
IMAGE = (1920, 1080)
JAW_BOXES = ((280, 620, 860, 1080), (1330, 650, 1680, 1080))   # wrist-image regions hidden by the open jaws
LIMIT_MARGIN = 60          # ticks inside the onboard range for every planned pose
TRANSIT_HEIGHT = 0.15      # m: gripper/wrist/forearm stay this high while looking or travelling (object unknown)
TABLE_FLOOR = 0.005        # m: abort if any measured link (moving jaw: full-360 envelope) gets closer to the table
PLAN_TABLE_CLEARANCE = 0.010
OBJECT_MARGIN = 0.0032     # m: robot surfaces (except the moving jaw pad, see below) vs measured cylinder surface
FIXED_SIDE_MARGIN = 0.0035 # m: asymmetric SO-101 trick: stationary left jaw (fixed_pad) sits 3.5 mm from true left edge
PLAN_RADIUS_MARGIN = 0.0020  # m: added to the measured base radius for collision screening
GRASP_HEIGHT = 0.040       # m: pad cross-section above the table (default when no height preference applies)
SQUEEZE_TICKS = 55         # goal advance past the detected contact, under the low PWM cap
APPROACH = 0.070           # m: straight approach length along the approach axis
LIFT = 0.058               # m: lift high enough (58 mm) to clearly clear tabletop and verify suspension
MAX_PITCH_DEG = 40.0
OPEN_RAW = 2033            # empty jaw opening used for grasping (~63 mm, inside the measured jaw profile)


class Geometry:
    """Kinematic model + table plane + robot meshes. Pure; no hardware."""

    def __init__(self, root=ROOT, config=None, jaw=None):
        root = Path(root)
        config = config if config is not None else json.loads((root / 'config/robot.json').read_text())
        if not config.get('kinematic_model'):
            raise ValueError('config/robot.json has no kinematic_model')
        m = json.loads((root / config['kinematic_model']).read_text())
        self.model_path = config['kinematic_model']
        self.K = np.array(m['camera_matrix'], float)
        self.D = np.array(m['distortion'], float)
        self.X = np.array(m['handeye_gripper_link_T_optical'], float)
        tilt = np.array(m['table_tilt_T'], float)
        self.n = tilt[:3, 2] / np.linalg.norm(tilt[:3, 2])
        self.d = float(m['table_z_m'])            # plane: n . p = d (tilt is a pure rotation)
        from .kinematic_offsets import model_offsets
        self.cad = default_model(root, offsets=model_offsets(root / config['kinematic_model']))
        cal = json.loads((root / 'config/follower-calibration.json').read_text())
        self.limits = {n: (cal[n]['range_min'] + LIMIT_MARGIN, cal[n]['range_max'] - LIMIT_MARGIN) for n in BODY}
        self.jaw = jaw
        # Dense CAD surface samples (vertices + triangle centroids) for object proximity; hulls would
        # fill the gap between the jaws. The moving jaw is excluded (its CAD angle is not calibrated).
        self.dense = {}
        for link in self.cad.tree.findall('link'):
            name = link.get('name')
            if name in ('base_link', 'moving_jaw_so101_v1_link'):
                continue
            parts = []
            for col in link.findall('collision'):
                mesh = col.find('geometry/mesh')
                if mesh is None:
                    continue
                v = np.ascontiguousarray(
                    stl_vertices(self.cad.mesh_root / mesh.get('filename')) * np.fromstring(mesh.get('scale', '1 1 1'), sep=' '),
                    dtype=np.float64,
                )
                tri = v.reshape(-1, 3, 3)
                pts = np.ascontiguousarray(np.r_[v, tri.mean(1)], dtype=np.float64)
                T = np.ascontiguousarray(xyz_transform(col.find('origin')), dtype=np.float64)
                with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                    pts = pts @ T[:3, :3].T + T[:3, 3]
                _, keep = np.unique(np.round(pts / 0.002), axis=0, return_index=True)   # 2 mm voxels
                parts.append(pts[keep])
            if parts:
                self.dense[name] = np.concatenate(parts)
        if 'wrist_camera_link' in self.cad.vertices:
            self.dense['wrist_camera_link'] = self.cad.vertices['wrist_camera_link']

    # ------------------------------------------------------------------ kinematics
    def full(self, q, gripper=2047):
        out = {n: float(q[n]) for n in BODY}
        out['gripper'] = float(q.get('gripper', gripper))
        return out

    def transforms(self, q):
        return self.cad.fk.transforms(self.full(q))

    def camera(self, q):
        return self.transforms(q)['gripper_link'] @ self.X

    def tcp(self, gripper_raw):
        """gripper_link -> grasp TCP from the jaw profile (x: fixed->moving pad, z: approach), pads, aperture."""
        g = self.jaw.at(int(gripper_raw))
        T = g.T_wrist_tcp.as_array()
        half = g.aperture_m / 2
        fixed = T[:3, 3] - T[:3, 0] * half
        return T, fixed, T[:3, 3] + T[:3, 0] * half, g.aperture_m

    # ------------------------------------------------------------------ table and object clearance
    def table_clearance(self, q):
        """Lowest robot point above the tilted table (moving jaw as its full-360 envelope). base/shoulder exempt."""
        with np.errstate(all='ignore'):   # numpy 2.2 + Accelerate raises spurious matmul FP warnings
            best, where = self._table_clearance(q)
        if not math.isfinite(best):
            raise RuntimeError('桌面淨空計算出非有限值，停止')
        return best, where

    def _table_clearance(self, q):
        T = self.transforms(q)
        best, where = np.inf, ''
        for link, v in self.cad.vertices.items():
            if link in ('base_link', 'shoulder_link'):
                continue
            h = float(np.min(v @ (T[link][:3, :3].T @ self.n)) + self.n @ T[link][:3, 3] - self.d)
            if h < best:
                best, where = h, link
        H = T['gripper_link'] @ self.cad.jaw_origin
        mm = H[:3, :3].T @ self.n
        v = self.cad.jaw
        h = float(np.min(mm[2] * v[:, 2] - np.hypot(mm[0], mm[1]) * np.hypot(v[:, 0], v[:, 1])) + self.n @ H[:3, 3] - self.d)
        if h < best:
            best, where = h, 'moving_jaw_full360'
        return best, where

    def object_clearance(self, q, obj, *, gripper_raw):
        with np.errstate(all='ignore'):
            best, where = self._object_clearance(q, obj, gripper_raw=gripper_raw)
        if not math.isfinite(best):
            raise RuntimeError('物件間隙計算出非有限值，停止')
        return best, where

    def _object_clearance(self, q, obj, *, gripper_raw):
        """Smallest gap between the robot and the measured upright cylinder (negative = overlap).

        Rigid links use dense CAD samples; the moving jaw is modelled by its measured pad line (from 10 mm
        behind the pad to the fingertip, 27 mm ahead) inflated by 6 mm, since its CAD angle is unknown."""
        T = self.transforms(q)
        c, r, top = np.asarray(obj['centre']), obj['radius'], obj['height']
        best, where = np.inf, ''

        def gap(points, label, inflate=0.0):
            nonlocal best, where
            rel = points - c
            up = rel @ self.n
            keep = (up > -0.002) & (up < top + 0.01)
            if not keep.any():
                return
            horiz = rel[keep] - np.outer(up[keep], self.n)
            g = float(np.min(np.linalg.norm(horiz, axis=1)) - r - inflate)
            if g < best:
                best, where = g, label

        for link, v in self.dense.items():
            gap(v @ T[link][:3, :3].T + T[link][:3, 3], link)
        Ttcp, fixed, moving, _ = self.tcp(gripper_raw)
        G = T['gripper_link']
        z = G[:3, :3] @ Ttcp[:3, 2]
        line = np.linspace(-0.010, 0.027, 16)[:, None]
        gap((G[:3, :3] @ moving + G[:3, 3]) + line * z, 'moving_jaw_pad_line', inflate=0.0025)
        return best, where

    # ------------------------------------------------------------------ camera rays
    def ray_to_table(self, q, uv, height=0.0):
        C = self.camera(q)
        p = cv2.undistortPoints(np.array([[uv]], float), self.K, self.D).ravel()
        dirn = C[:3, :3] @ np.array([p[0], p[1], 1.0])
        o = C[:3, 3]
        denom = self.n @ dirn
        if denom > -1e-6:
            raise ValueError('pixel ray does not hit the table')
        t = (self.d + height - self.n @ o) / denom
        return o + t * dirn

    def project(self, q, points):
        C = self.camera(q)
        pts_arr = np.ascontiguousarray(points, dtype=np.float64)
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            P = (np.c_[pts_arr, np.ones(len(pts_arr))] @ np.linalg.inv(C).T)[:, :3]
        uv, _ = cv2.projectPoints(np.ascontiguousarray(P, dtype=np.float64), np.zeros(3), np.zeros(3), self.K, self.D)
        return uv.reshape(-1, 2), P[:, 2]

    # ------------------------------------------------------------------ IK
    def solve(self, residual, q0, extra_starts=()):
        lo = np.array([self.limits[n][0] for n in BODY], float)
        hi = np.array([self.limits[n][1] for n in BODY], float)
        best = None
        for start in (q0, *extra_starts):
            x0 = np.clip(np.array([float(start[n]) for n in BODY]), lo + 1, hi - 1)
            sol = least_squares(lambda x: residual(dict(zip(BODY, x))), x0, bounds=(lo, hi), x_scale=100.0)
            if best is None or sol.cost < best.cost:
                best = sol
        return {n: float(v) for n, v in zip(BODY, best.x)}

    def tilt_deg(self, axis):
        return float(np.degrees(np.arcsin(np.clip(-(axis @ self.n), -1, 1))))   # >0: pointing down


# ---------------------------------------------------------------------- cylinder from two silhouette-foot pixels
def cylinder_from_feet(geo, q, left_uv, right_uv, top_uv=None):
    """Upright cylinder from where its left/right silhouette meets the table.

    The tangent planes of an upright cylinder through the camera are vertical, so on the table the two foot
    points are the tangent points of the circle seen from the camera's foot F. With half-chord a and
    d = |F - O|: a = r sqrt(1 - r^2/d^2) and the chord midpoint lies r^2/d from O towards F."""
    A, B = geo.ray_to_table(q, left_uv), geo.ray_to_table(q, right_uv)
    C = geo.camera(q)[:3, 3]
    F = C - geo.n * (geo.n @ C - geo.d)
    M = (A + B) / 2
    a = float(np.linalg.norm(A - B) / 2)
    u = M - F
    u -= geo.n * (geo.n @ u)
    dm = float(np.linalg.norm(u))
    u /= dm
    r = a
    for _ in range(50):
        d = dm + r * r / max(dm, 1e-6)
        r = a / math.sqrt(max(1e-9, 1 - (r / d) ** 2))
    d = dm + r * r / dm
    O = M + u * (r * r / d)
    height = None
    if top_uv is not None:
        # closest point of the top-centre ray to the cylinder axis
        Cm = geo.camera(q)
        p = cv2.undistortPoints(np.array([[top_uv]], float), geo.K, geo.D).ravel()
        v = Cm[:3, :3] @ np.array([p[0], p[1], 1.0])
        w0 = Cm[:3, 3] - O
        a_, b_, c_ = v @ v, v @ geo.n, geo.n @ geo.n
        d_, e_ = v @ w0, geo.n @ w0
        den = a_ * c_ - b_ * b_
        if abs(den) > 1e-9:
            t_axis = (a_ * e_ - b_ * d_) / den
            height = float(t_axis)
    return {'centre': O.tolist(), 'radius': r, 'height': height, 'foot_left': A.tolist(), 'foot_right': B.tolist(),
            'camera_distance_m': float(np.linalg.norm(O - C))}


def refine_silhouette_feet(gray, left, right, search=15, rows=7):
    """Snap the VLM foot points to the strongest horizontal brightness edge within +-search px (median of rows).

    Left foot: background -> object going right; right foot: object -> background. Points without a clear
    edge are kept as given."""
    g = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 1.5)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    h, w = gray.shape[:2]
    out = []
    for (u, v), sign in ((left, 1.0), (right, -1.0)):
        u0, v0 = int(round(u)), int(round(v))
        if not (search + 2 <= u0 < w - search - 2 and rows <= v0 < h - rows):
            out.append((float(u), float(v)))
            continue
        hits = []
        for dv in range(-(rows // 2), rows // 2 + 1):
            row = gx[v0 + dv, u0 - search:u0 + search + 1] * sign
            i = int(np.argmax(row))
            hits.append((float(row[i]), u0 - search + i))
        strength = float(np.median([s for s, _ in hits]))
        out.append((float(np.median([x for _, x in hits])), float(v)) if strength > 20 else (float(u), float(v)))
    return out


# ---------------------------------------------------------------------- look poses
def look_pose(geo, target, q_ref, rng, *, dist=(0.15, 0.34), centre_box=(600, 1320, 250, 700), half=0.05,
              min_height=TRANSIT_HEIGHT, n=300_000, batches=6, prefer='motion', pan_window=900):
    """Model-planned wrist view of a table point: target near the image centre, a +-half square fully visible
    and clear of the jaw regions, gripper/jaws/wrist/camera at least min_height above the table."""
    fk = _batch_fk(geo)
    e1 = np.cross(geo.n, [0.0, 1.0, 0.0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(geo.n, e1)
    pts = np.array([target] + [target + sx * half * e1 + sy * half * e2 for sx in (-1, 1) for sy in (-1, 1)])
    # vectorised height proxy: fingertip ends of both pads, gripper/wrist origins (exact mesh check follows)
    if geo.jaw is not None:
        Ttcp, fixed, moving, _ = geo.tcp(geo.jaw.samples[-1].raw)
        tips = np.array([fixed + Ttcp[:3, 2] * 0.03, moving + Ttcp[:3, 2] * 0.03, [0, 0, 0]])
    else:
        tips = np.array([[-0.011, 0, -0.105], [0.05, 0, -0.1], [0, 0, 0]])
    found = []
    for _ in range(batches):
        S = {m: np.round(rng.uniform(*geo.limits[m], n)) for m in BODY}
        S['shoulder_pan'] = np.round(rng.uniform(max(geo.limits['shoulder_pan'][0], q_ref['shoulder_pan'] - pan_window),
                                                 min(geo.limits['shoulder_pan'][1], q_ref['shoulder_pan'] + pan_window), n))
        S['wrist_roll'] = np.round(rng.uniform(max(geo.limits['wrist_roll'][0], 998 - 110),
                                               min(geo.limits['wrist_roll'][1], 998 + 110), n))
        G = fk(S)
        C = G @ geo.X
        ok = (C[:, :3, 3] @ geo.n - geo.d) > min_height + 0.02
        low = np.min(np.einsum('nij,kj->nki', G[:, :3, :3], tips) @ geo.n + (G[:, :3, 3] @ geo.n)[:, None], axis=1) - geo.d
        ok &= low > min_height + 0.005
        Pc = np.einsum('nij,kj->nki', np.linalg.inv(C), np.c_[pts, np.ones(len(pts))])[..., :3]
        z = Pc[..., 2]
        ok &= (z > 0.1).all(1)
        x, y = Pc[..., 0] / np.where(z > 0, z, 1), Pc[..., 1] / np.where(z > 0, z, 1)
        u, v = geo.K[0, 0] * x + geo.K[0, 2], geo.K[1, 1] * y + geo.K[1, 2]
        ok &= (u[:, 0] > centre_box[0]) & (u[:, 0] < centre_box[1]) & (v[:, 0] > centre_box[2]) & (v[:, 0] < centre_box[3])
        ok &= ((u > 80) & (u < IMAGE[0] - 80) & (v > 60) & (v < IMAGE[1] - 60)).all(1)
        for x0, y0, x1, y1 in JAW_BOXES:
            ok &= ~((u > x0 - 30) & (u < x1 + 30) & (v > y0 - 30) & (v < y1 + 30)).any(1)
        dd = np.linalg.norm(C[:, :3, 3] - target, axis=1)
        ok &= (dd > dist[0]) & (dd < dist[1])
        for i in np.flatnonzero(ok):
            q = {m: int(S[m][i]) for m in BODY}
            score = -float(C[i, :3, 3] @ geo.n) if prefer == 'high' else float(max(abs(q[m] - q_ref[m]) for m in BODY))
            found.append((score, q))
        if len(found) >= 40:
            break
    for _, q in sorted(found, key=lambda f: f[0])[:60]:
        if transit_clearance(geo, q) >= min_height:
            return q
    return None


def transit_clearance(geo, q):
    """Height of the gripper, jaw envelope, camera, wrist and forearm above the table."""
    T = geo.transforms(q)
    low = np.inf
    for link in ('gripper_link', 'wrist_link', 'lower_arm_link', 'wrist_camera_link'):
        v = geo.cad.vertices.get(link)
        if v is not None:
            low = min(low, float(np.min(v @ (T[link][:3, :3].T @ geo.n)) + geo.n @ T[link][:3, 3] - geo.d))
    h, where = geo.table_clearance(q)
    if where == 'moving_jaw_full360':
        low = min(low, h)
    return low


def _batch_fk(geo):
    bychild = {j[3]: j for j in geo.cad.fk.joints}
    chain, link = [], 'gripper_link'
    while link in bychild:
        chain.append(bychild[link])
        link = bychild[link][2]
    chain = list(reversed(chain))
    offs = geo.cad.fk.offsets

    def fk(raw):
        n = len(np.atleast_1d(raw['shoulder_pan']))
        T = np.tile(np.eye(4), (n, 1, 1))
        for name, kind, parent, child, To, axis in chain:
            T = T @ To
            if kind == 'fixed':
                continue
            ang = (np.atleast_1d(raw[name]).astype(float) + offs.get(name, 0.0) - 2047) * TICK
            a = axis / np.linalg.norm(axis)
            Kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
            s, c = np.sin(ang)[:, None, None], np.cos(ang)[:, None, None]
            M = np.tile(np.eye(4), (n, 1, 1))
            M[:, :3, :3] = np.eye(3) + s * Kx + (1 - c) * (Kx @ Kx)
            T = T @ M
        return T
    return fk


# Home: Astra's previously screened wide-observation shape, panned to face straight ahead. The camera is
# ~22 cm above the table looking 27-65 cm out; the fingertips stay ~13 cm high. Looking around only turns the
# shoulder, so the tips keep their height and distance from the base and never sweep a new region.
HOME = {'shoulder_pan': 2030, 'shoulder_lift': 1103, 'elbow_flex': 2915, 'wrist_flex': 2318, 'wrist_roll': 998}
LOOK_PAN_OFFSETS = (0, -350, -700, 350, 700)
EMPTY_CHECK_PAN_OFFSET = -250   # clean background for the empty-jaw image (no table marks between the jaws)


def look_plan(geo, home=None, pan_offsets=LOOK_PAN_OFFSETS):
    home = dict(home or HOME)
    views = []
    for d in pan_offsets:
        q = {**home, 'shoulder_pan': home['shoulder_pan'] + d}
        if not all(geo.limits[n][0] <= q[n] <= geo.limits[n][1] for n in BODY):
            continue
        views.append({'pan_offset': d, 'pose': q, 'transit_m': round(transit_clearance(geo, q), 4)})
    return {'home': home, 'views': views}


def near_view(geo, obj, q_ref, *, seed=7):
    """Closer wrist view of a located cylinder foot, clear of the cylinder and the table; None if unreachable."""
    rng = np.random.default_rng(seed)
    centre = np.asarray(obj['centre'])
    g_raw = geo.jaw.samples[-1].raw
    # If q_ref is currently low near the table or close to the object (e.g. after a previous attempt),
    # screen from a lifted transit pose so near_view never fails just because q_ref started low.
    q_high_ref = q_ref
    if geo.table_clearance(q_ref)[0] < 0.045 or geo.object_clearance(q_ref, obj, gripper_raw=g_raw)[0] < 0.030:
        q_high_ref = {**HOME, 'shoulder_pan': int(np.clip(q_ref.get('shoulder_pan', HOME['shoulder_pan']), geo.limits['shoulder_pan'][0], geo.limits['shoulder_pan'][1]))}
    for prefer in ('high', 'motion'):
        for min_h, dist in ((0.10, (0.17, 0.30)), (0.07, (0.15, 0.32)), (0.05, (0.13, 0.34))):
            q = look_pose(geo, centre, q_high_ref, rng, dist=dist, half=0.045, min_height=min_h,
                          n=120_000, batches=3, prefer=prefer, pan_window=900)
            if q is None:
                continue
            gap, _ = geo.object_clearance(q, obj, gripper_raw=g_raw)
            t_clear, _ = geo.table_clearance(q)
            if gap >= 0.030 and t_clear >= 0.035:
                s = segment_screen(geo, q_high_ref, q, obj, g_raw, object_check=True)
                if s['table_mm'] >= 30.0 and (s['object_mm'] is None or s['object_mm'] >= 22.0):
                    return q
                s_home = segment_screen(geo, HOME, q, obj, g_raw, object_check=True)
                if s_home['table_mm'] >= 30.0 and (s_home['object_mm'] is None or s_home['object_mm'] >= 22.0):
                    return q
    return None


# ---------------------------------------------------------------------- grasp plan
def grasp_pose(geo, obj, gripper_raw, q_ref, *, height=GRASP_HEIGHT, pitch_deg=None):
    """Pose that puts the measured cylinder axis between the open pads at `height`, closing axis level.

    SO-101 Asymmetric Jaw Trick:
    Because the left jaw (`fixed`) is stationary and only the right jaw (`moving`) closes toward it,
    we align the stationary left jaw at `FIXED_SIDE_MARGIN` (3.5 mm) from the object's true un-inflated
    narrow-axis surface (`r_contact`). That prevents the moving right jaw from pushing or tipping the
    object sideways during jaw closure while keeping safe approach clearance."""
    Ttcp, fixed, moving, aperture = geo.tcp(gripper_raw)
    r = float(obj['radius'])
    r_contact = float(obj.get('uninflated_radius_m') or obj.get('measured_radius') or obj.get('semi_minor_m') or r)
    r_contact = min(r_contact, r)
    if aperture - 2 * r_contact < FIXED_SIDE_MARGIN + 0.007:
        raise ValueError(f'measured diameter {2000 * r_contact:.0f} mm does not fit the {1000 * aperture:.0f} mm opening')
    xh, zh = Ttcp[:3, 0], Ttcp[:3, 2]
    # Ensure `local` sits at least `r + OBJECT_MARGIN + 0.5mm` from `fixed` so `gripper_link` clearance passes,
    # and at least `r + 0.0025 + OBJECT_MARGIN + 0.5mm` from `moving` so `moving_jaw_pad_line` (inflate=2.5mm) passes.
    r_true = float(obj.get('measured_radius_m') or r_contact)
    r_true = float(np.clip(r_true, r_contact, 0.5 * aperture - 0.0015))
    min_eff = max(r + OBJECT_MARGIN + 0.0005, min(r_true + 0.0022, 0.5 * aperture))
    max_eff = max(min_eff, aperture - (r + 0.0025 + OBJECT_MARGIN + 0.0005))
    eff_offset = float(np.clip(r_true + FIXED_SIDE_MARGIN, min_eff, max_eff))
    fixed_jaw_gap_m = max(0.0, eff_offset - r_true)
    moving_jaw_gap_m = max(0.0, aperture - eff_offset - r_true)
    local = fixed + xh * eff_offset
    target = np.asarray(obj['centre']) + geo.n * height
    radial = target - geo.n * (geo.n @ target)
    radial /= np.linalg.norm(radial)

    def res(q):
        G = geo.transforms(q)['gripper_link']
        p = G[:3, :3] @ local + G[:3, 3]
        x, z = G[:3, :3] @ xh, G[:3, :3] @ zh
        tilt = geo.tilt_deg(z)
        out = [(p - target) * 1000, [20 * (x @ geo.n)], [20 * (np.cross(z - geo.n * (z @ geo.n), radial) @ geo.n)]]
        out.append([0.15 * (tilt - (pitch_deg or 0.0)) if pitch_deg is not None else 0.05 * tilt])
        out.append([2.0 * max(0.0, tilt - MAX_PITCH_DEG), 2.0 * max(0.0, -tilt - 5.0)])
        out.append([0.5 * max(0.0, abs(float(q['wrist_roll']) - float(HOME['wrist_roll'])) - 160.0)])
        return np.concatenate([np.ravel(o) for o in out])

    starts = [
        HOME,
        {**HOME, 'shoulder_pan': q_ref['shoulder_pan']},
        *({**q_ref, 'shoulder_pan': q_ref['shoulder_pan'] + d} for d in (-200, 200)),
    ]
    q = geo.solve(res, q_ref, starts)
    G = geo.transforms(q)['gripper_link']
    err = float(np.linalg.norm(G[:3, :3] @ local + G[:3, 3] - target) * 1000)
    return q, {'position_error_mm': err, 'pitch_deg': geo.tilt_deg(G[:3, :3] @ zh),
               'closing_axis_vs_table_deg': float(np.degrees(np.arcsin(abs((G[:3, :3] @ xh) @ geo.n)))),
               'local_point_mm': (local * 1000).round(2).tolist(), 'aperture_mm': aperture * 1000,
               'fixed_jaw_gap_mm': round(fixed_jaw_gap_m * 1000, 2),
               'moving_jaw_clearance_mm': round(moving_jaw_gap_m * 1000, 2),
               'asymmetric_left_jaw_aligned': True}


def pose_along(geo, q_goal, offset, q_ref):
    """Same gripper orientation as q_goal, translated by `offset` (base frame, m)."""
    Gg = geo.transforms(q_goal)['gripper_link']

    def res(q):
        G = geo.transforms(q)['gripper_link']
        dR = Gg[:3, :3].T @ G[:3, :3]
        return np.r_[(G[:3, 3] - Gg[:3, 3] - offset) * 1000, 30 * (dR - np.eye(3))[[2, 0, 1], [1, 2, 0]]]
    q = geo.solve(res, q_ref)
    G = geo.transforms(q)['gripper_link']
    rot = float(np.degrees(np.arccos(np.clip((np.trace(Gg[:3, :3].T @ G[:3, :3]) - 1) / 2, -1, 1))))
    return q, float(np.linalg.norm(G[:3, 3] - Gg[:3, 3] - offset) * 1000), rot


def segment_screen(geo, a, b, obj, gripper_raw, *, steps=24, object_check=True):
    """Worst table clearance and object gap along a joint-space segment (the executed interpolation)."""
    table, obj_gap = (np.inf, ''), (np.inf, '')
    for s in np.linspace(0, 1, steps + 1):
        q = {n: a[n] + (b[n] - a[n]) * s for n in BODY}
        t = geo.table_clearance(q)
        if t[0] < table[0]:
            table = t
        if object_check and obj is not None:
            g = geo.object_clearance(q, obj, gripper_raw=gripper_raw)
            if g[0] < obj_gap[0]:
                obj_gap = g
    return {'table_mm': round(table[0] * 1000, 1), 'table_link': table[1],
            'object_mm': None if obj_gap[0] == np.inf else round(obj_gap[0] * 1000, 1), 'object_link': obj_gap[1]}


def plan_grasp(geo, obj, gripper_raw, q_now, pitches=(25, 15, 35, 5, 45), heights=(0.040, 0.060, 0.080, 0.100, 0.115)):
    """above-pregrasp -> pregrasp -> straight approach -> grasp -> lift, all screened.

    Tries grasp heights (low first) x approach pitches (deg below horizontal). A candidate grasp pose is
    screened alone first (wrist camera/palm vs the measured cylinder, table); only survivors get a full path.
    Returns the first plan without problems, else the one with the fewest (`problems` says why it must not run)."""
    if not obj.get('height'):
        raise ValueError('object height was not measured')
    H = float(obj['height'])
    # For tall upright objects (H > 65 mm), prefer grasping ~25 mm below the top rim so wrist_camera_link
    # stays >= 25 mm above the object top even under servo gravity droop at extended reach.
    prefer = min(max(0.45 * H, H - 0.025), 0.105)
    heights = sorted(set(heights) | {round(prefer, 3)}, key=lambda h: (abs(h - prefer), h))
    R_xy = float(np.linalg.norm(np.asarray(obj['centre'], dtype=float)[:2]))
    sag_ticks = int(np.clip((R_xy - 0.28) * 220.0, 8, 38))
    best, tried = None, []
    for h in heights:
        if h > max(H - 0.018, 0.022):
            continue
        for pitch in pitches:
            try:
                q, info = grasp_pose(geo, obj, gripper_raw, q_now, height=h, pitch_deg=pitch)
            except ValueError as exc:
                tried.append({'height_mm': h * 1000, 'pitch': pitch, 'rejected': str(exc)})
                continue
            gap = geo.object_clearance(q, obj, gripper_raw=gripper_raw)
            table = geo.table_clearance(q)
            q_sag = {**q, 'shoulder_lift': q['shoulder_lift'] + 4, 'elbow_flex': q['elbow_flex'] + sag_ticks}
            gap_sag = geo.object_clearance(q_sag, obj, gripper_raw=gripper_raw) if H >= 0.065 else gap
            ok = (info['position_error_mm'] <= 1.5 and info['closing_axis_vs_table_deg'] <= 3
                  and abs(info['pitch_deg'] - pitch) <= 8 and gap[0] >= OBJECT_MARGIN
                  and gap_sag[0] >= 0.001 and table[0] >= PLAN_TABLE_CLEARANCE)
            tried.append({'height_mm': h * 1000, 'pitch': pitch, 'ik_mm': round(info['position_error_mm'], 2),
                          'object_mm': round(gap[0] * 1000, 1), 'object_link': gap[1], 'table_mm': round(table[0] * 1000, 1)})
            if not ok:
                continue
            try:
                plan = _plan_grasp_at(geo, obj, gripper_raw, q_now, pitch, h)
            except ValueError as exc:
                plan = {'problems': [str(exc)], 'grasp': {'pitch_target_deg': pitch, 'height_m': h}, 'waypoints': []}
            plan['candidates'] = tried
            if not plan['problems']:
                return plan
            if best is None or len(plan['problems']) < len(best['problems']):
                best = plan
    return best or {'problems': ['no grasp height/pitch keeps the wrist camera and palm clear of the object'],
                    'candidates': tried, 'waypoints': [], 'grasp': {}}


def _plan_grasp_at(geo, obj, gripper_raw, q_now, pitch, height=GRASP_HEIGHT):
    q_grasp, info = grasp_pose(geo, obj, gripper_raw, q_now, height=height, pitch_deg=pitch)
    info.update(pitch_target_deg=pitch, height_m=height)
    if info['position_error_mm'] > 1.5 or info['closing_axis_vs_table_deg'] > 3 or abs(info['pitch_deg'] - pitch) > 8:
        raise ValueError(f'no reachable level grasp pose at {pitch} deg: '
                         f"err {info['position_error_mm']:.1f} mm, pitch {info['pitch_deg']:.1f}, closing {info['closing_axis_vs_table_deg']:.1f}")
    Ttcp, *_ = geo.tcp(gripper_raw)
    G_grasp = geo.transforms(q_grasp)['gripper_link']
    xb = G_grasp[:3, :3] @ Ttcp[:3, 0]
    zb = G_grasp[:3, :3] @ Ttcp[:3, 2]
    steps = [('grasp', 0.0)]
    for back in (0.015, 0.030, 0.050, APPROACH):
        steps.append((f'approach_{int(back * 1000)}mm', back))
    poses, prev = {}, q_grasp
    for label, back in steps:
        if back == 0:
            poses[label] = (q_grasp, 0.0, 0.0)
            continue
        q, e, rot = pose_along(geo, q_grasp, -zb * back, prev)
        poses[label] = (q, e, rot)
        prev = q
    pre = poses[f'approach_{int(APPROACH * 1000)}mm'][0]
    above, e_a, r_a = pose_along(geo, q_grasp, -zb * APPROACH + geo.n * 0.08, pre)
    lift, e_l, r_l = pose_along(geo, q_grasp, geo.n * LIFT, q_grasp)
    order = [('above_pregrasp', above, e_a, r_a)]
    for label, back in reversed(steps):
        q, e, rot = poses[label]
        order.append((label, q, e, rot))
    order.append(('lift', lift, e_l, r_l))
    waypoints, prev = [], q_now
    for label, q, e, rot in order:
        qi = {n: int(round(q[n])) for n in BODY}
        check = segment_screen(geo, prev, qi, obj, gripper_raw, object_check=label != 'lift')
        waypoints.append({'label': label, 'target': qi, 'ik_error_mm': round(e, 2), 'ik_rotation_deg': round(rot, 2),
                          **check, 'max_joint_step_ticks': int(max(abs(qi[n] - prev[n]) for n in BODY))})
        prev = qi
    problems = []
    for w in waypoints:
        max_pos_err = 3.8 if w['label'] in ('above_pregrasp', 'lift') else 2.0
        max_rot_err = 9.5 if w['label'] in ('above_pregrasp', 'lift') else 4.2
        if w['ik_error_mm'] > max_pos_err or w['ik_rotation_deg'] > max_rot_err:
            problems.append(f"{w['label']}: IK {w['ik_error_mm']} mm / {w['ik_rotation_deg']} deg")
        floor = 30.0 if w['label'] == 'above_pregrasp' else PLAN_TABLE_CLEARANCE * 1000
        if w['table_mm'] < floor:
            problems.append(f"{w['label']}: table {w['table_mm']} mm ({w['table_link']})")
        if w['object_mm'] is not None and w['object_mm'] < OBJECT_MARGIN * 1000:
            problems.append(f"{w['label']}: object gap {w['object_mm']} mm ({w['object_link']})")
    return {'grasp': info, 'waypoints': waypoints, 'problems': problems, 'object': obj, 'gripper_raw': int(gripper_raw)}


# ====================================================================== runtime (hardware via Robot/WristGrasp)
class KinematicGraspProgram:
    """Two calls: plan (moves only to observe: home, look around, one closer view) then execute (approved plan)."""

    PLAN_TTL_S = 900
    MAX_BIAS = 68

    def __init__(self, robot, brain, grasp, runtime, root=ROOT):
        self.robot, self.brain, self.grasp = robot, brain, grasp
        self.runtime, self.root = Path(runtime), Path(root)
        self.pending = None
        self.sag = {n: 0 for n in BODY}

    # ------------------------------------------------------------------ helpers
    def geometry(self):
        from .spatial_runtime import _profile_snapshot
        config = json.loads((self.root / 'config/robot.json').read_text())
        # Pads are measured in gripper_link and do not depend on joint offsets; the jaw profile stays bound
        # to Astra's active (global) geometry revision.
        jaw = _profile_snapshot(default_model(self.root), config, include_jaws=True)[2]
        if jaw is None:
            raise RuntimeError('沒有夾爪校準，無法規劃抓取')
        return Geometry(self.root, config, jaw)

    def log(self, event):
        with (self.runtime / 'kinematic-grasp.jsonl').open('a') as f:
            f.write(json.dumps({'time': time.time(), **event}, ensure_ascii=False, default=float) + '\n')

    def guard(self, geo, *, table_floor=TABLE_FLOOR, obj=None, object_floor=0.0, gripper_raw=None):
        def check(actual):
            q = {n: actual[n] for n in BODY}
            h, link = geo.table_clearance(q)
            if h < table_floor:
                raise RuntimeError(f'實測姿勢離桌面只剩 {h * 1000:.1f} mm（{link}），停止並保持')
            if obj is not None:
                g, glink = geo.object_clearance(q, obj, gripper_raw=gripper_raw)
                if g < object_floor:
                    raise RuntimeError(f'實測姿勢會碰到物件（{glink} {g * 1000:.1f} mm），停止並保持')
        return check

    def move(self, target, *, guard, cancel, speed=120, precision=45, correct=False, tolerance=4):
        """glide with the measured servo steady-state error fed forward through Robot.bias."""
        r = self.robot
        target = {n: int(target[n]) for n in BODY}
        eff_speed = min(float(speed), float(r.config.get('max_ticks_per_second', 90)))
        for attempt in range(3 if correct else 1):
            if (cancel is not None and cancel.is_set()) or r.stop.is_set():
                raise RuntimeError('已停止')
            r.bias.update({n: int(np.clip(self.sag[n], -self.MAX_BIAS, self.MAX_BIAS)) for n in BODY})
            r.glide(target, speed=eff_speed, precision=precision, measured_guard=guard, allow_rearm=False, settle_timeout_s=1.8)
            if r.stop.wait(.10 if not correct else .18):
                raise RuntimeError('已停止')
            actual = r.read()
            err = {n: target[n] - actual[n] for n in BODY}
            self.sag = {n: int(np.clip(r.bias[n] + err[n], -self.MAX_BIAS, self.MAX_BIAS)) for n in BODY}
            if max(abs(v) for v in err.values()) <= tolerance:
                break
        return actual, err

    def fast_move(self, target, *, guard=None, cancel=None, speed_ticks_per_s=380.0, wait_settle=True):
        """Fast cosine-eased 25 Hz trajectory streaming for agile look-around and transit motions."""
        r = self.robot
        target_body = {n: int(target[n]) for n in BODY}
        r.bias.update({n: int(np.clip(self.sag[n], -self.MAX_BIAS, self.MAX_BIAS)) for n in BODY})
        actual = r.fast_move_to(
            target_body,
            speed_ticks_per_s=speed_ticks_per_s,
            guard=guard,
            cancel=cancel,
            settle_tol_ticks=2,
            settle_window_s=0.075,
            wait_settle=wait_settle,
        )
        err = {n: target_body[n] - actual[n] for n in BODY}
        return actual, err

    def fast_capture(self, *, guard=None, cancel=None):
        """Snap a sharp wrist image immediately once mechanical vibration settles (<=2 ticks over ~75ms)."""
        frame = self.robot.fast_settled_snapshot(self.grasp.snapshot, guard=guard, cancel=cancel)
        if getattr(self.grasp, 'state_memory', None) is not None:
            self.grasp.state_memory.remember(frame)
        return frame

    def screen_transit(self, geo, a, b, *, obj=None, gripper_raw=None, table_floor=0.05):
        start_h = geo.table_clearance(a)[0]
        need = min(table_floor, start_h - 0.002)
        s = segment_screen(geo, a, b, obj, gripper_raw or geo.jaw.samples[-1].raw, object_check=obj is not None)
        if s['table_mm'] < need * 1000:
            raise RuntimeError(f"路徑離桌面只剩 {s['table_mm']} mm（{s['table_link']}），不移動")
        if obj is not None and s['object_mm'] is not None and s['object_mm'] < 25:
            raise RuntimeError(f"路徑離物件只剩 {s['object_mm']} mm（{s['object_link']}），不移動")
        return s

    def locate(self, geo, query, frames, cancel, *, require_top=True, allow_non_upright=False):
        answer = self.brain.locate_upright_feet(query, frames, cancel)
        self.log({'type': 'locate', 'answer': answer, 'frames': [{'id': f['id'], 'joints': f['joints']} for f in frames]})
        if not answer['found'] or answer['confidence'] < .6:
            raise RuntimeError('Argon ER 2 沒有在畫面中找到要抓的物件：' + answer['reason'])
        if not ((answer['upright_on_table'] or allow_non_upright) and (answer['bottom_fully_visible'] or allow_non_upright) and (answer['top_visible'] or not require_top)):
            raise RuntimeError('物件不是直立在桌面上，或底部／頂部看不完整：' + answer['reason'])
        frame = frames[min(answer['view_index'], len(frames) - 1)]
        gray = cv2.imread(frame['path'], cv2.IMREAD_GRAYSCALE)
        if gray.shape[:2] != (IMAGE[1], IMAGE[0]):
            raise RuntimeError('腕部影像解析度和相機模型不同')
        px = {k: (answer[k]['x'] * IMAGE[0], answer[k]['y'] * IMAGE[1]) for k in ('bottom_left', 'bottom_right', 'top_center')}
        refined = refine_silhouette_feet(gray, px['bottom_left'], px['bottom_right'])
        q = {n: frame['joints'][n] for n in BODY}
        obj = cylinder_from_feet(geo, q, refined[0], refined[1], px['top_center'] if answer['top_visible'] else None)
        obj.update(view_frame=frame['id'], joints=q, pixels=px, refined_feet=refined,
                   crowded=answer['other_objects_within_10cm'], name=answer['object_name'],
                   upright_on_table=bool(answer.get('upright_on_table', True)))
        max_r = 0.068 if allow_non_upright else 0.048
        if not 0.008 <= obj['radius'] <= max_r:
            raise RuntimeError(f"量到的直徑 {obj['radius'] * 2000:.0f} mm 不合理，不抓")
        if require_top and (not obj['height'] or not 0.03 <= obj['height'] <= 0.30):
            raise RuntimeError(f"量到的尺寸不合理（直徑 {obj['radius'] * 2000:.0f} mm、高 {0 if not obj['height'] else obj['height'] * 1000:.0f} mm），不抓")
        radial = float(np.hypot(*np.asarray(obj['centre'])[:2]))
        if not 0.12 <= radial <= 0.48:
            raise RuntimeError(f'物件離底座 {radial * 1000:.0f} mm，超出可抓範圍')
        return obj, frame, answer

    # ------------------------------------------------------------------ plan (observation motions only)
    def plan(self, query, report, cancel):
        r = self.robot
        if r.pwm_owner is not None:
            raise RuntimeError('夾爪正在保持物件，請先釋放')
        deadline = time.monotonic() + 8
        while not r.camera.fresh() and time.monotonic() < deadline:
            if cancel.wait(.2):
                raise RuntimeError('已停止')
        r.begin(allow_rearm=False)
        geo = self.geometry()
        start = self.grasp.capture()
        gripper_raw = int(start['joints']['gripper'])
        if gripper_raw < OPEN_RAW - 15 and (r.gripper_power._record() or {}).get('phase') in (None, 'completed'):
            report('張開空夾爪')
            r.glide({'gripper': OPEN_RAW}, speed=50, precision=8, allow_rearm=False)
            start = self.grasp.capture()
            gripper_raw = int(start['joints']['gripper'])
        if not geo.jaw.samples[0].raw + 20 <= gripper_raw <= geo.jaw.samples[-1].raw:
            raise RuntimeError('夾爪目前沒有張開到已校準的開口，請先張開夾爪')
        if geo.tcp(gripper_raw)[3] < 0.050:
            raise RuntimeError('夾爪開口小於 50 mm，請先張開夾爪')
        looks = look_plan(geo)
        q = {n: start['joints'][n] for n in BODY}
        report('回到觀察原位')
        self.screen_transit(geo, q, looks['home'])
        self.fast_move(looks['home'], guard=self.guard(geo, table_floor=0.03), cancel=cancel, speed_ticks_per_s=380.0)
        frames = []
        for i, v in enumerate(looks['views']):
            report(f"張望 {i + 1} / {len(looks['views'])}")
            if i:
                self.fast_move(v['pose'], guard=self.guard(geo, table_floor=0.03), cancel=cancel, speed_ticks_per_s=380.0)
            frame = self.fast_capture(guard=self.guard(geo, table_floor=0.03), cancel=cancel)
            frame['look_pan_offset'] = v['pan_offset']
            frames.append(frame)
        report('Argon ER 2 正在從各個角度找物件')
        try:
            obj, view_frame, answer = self.locate(geo, query, frames, cancel)
        except Exception:
            if not cancel.is_set() and not r.stop.is_set():
                try:
                    self.fast_move(looks['home'], guard=self.guard(geo, table_floor=0.03), cancel=cancel, speed_ticks_per_s=380.0)
                except Exception:
                    pass
            raise
        report(f"在桌面上量到物件：直徑 {obj['radius'] * 2000:.0f} mm、高 {obj['height'] * 1000:.0f} mm")
        here = r.read()
        q_here = {n: here[n] for n in BODY}
        q_view = {n: view_frame['joints'][n] for n in BODY}
        self.screen_transit(geo, q_here, q_view, obj=obj, gripper_raw=gripper_raw)
        self.fast_move(
            q_view,
            guard=self.guard(geo, table_floor=0.03, obj=obj, object_floor=0.015, gripper_raw=gripper_raw),
            cancel=cancel,
            speed_ticks_per_s=340.0,
        )
        first = obj
        nv = near_view(geo, obj, q_view)
        if nv is not None:
            report('靠近一點再看一次')
            self.screen_transit(geo, q_view, nv, obj=obj, gripper_raw=gripper_raw, table_floor=0.04)
            nv_guard = self.guard(geo, table_floor=0.02, obj=obj, object_floor=0.015, gripper_raw=gripper_raw)
            self.fast_move(nv, guard=nv_guard, cancel=cancel, speed_ticks_per_s=320.0)
            near = self.fast_capture(guard=nv_guard, cancel=cancel)
            try:
                obj2, _, _ = self.locate(geo, query, [near], cancel, require_top=False)
                shift = float(np.linalg.norm(np.asarray(obj2['centre']) - np.asarray(obj['centre'])))
                self.log({'type': 'remeasure', 'shift_mm': shift * 1000, 'first': first, 'near': obj2})
                if shift <= 0.048:
                    obj = dict(obj2, height=first['height'], first_centre=first['centre'], remeasure_shift_mm=round(shift * 1000, 1))
            except Exception as near_exc:
                self.log({'type': 'remeasure_fallback', 'reason': str(near_exc)})
        # Silhouette feet give the base width; the body can be wider higher up (caps, bevels). Plan with margin.
        obj['measured_radius'] = obj['radius']
        obj['radius'] = max(obj['radius'], first['radius']) + PLAN_RADIUS_MARGIN
        here = r.read()
        q_here = {n: here[n] for n in BODY}
        report('本機規劃接近、夾取與抬起')
        plan = plan_grasp(geo, obj, gripper_raw, q_here)
        if plan['problems']:
            # Wider cylindrical cans (e.g. Pringles ~62-66mm) can slightly exceed the strict jaw margin after
            # oblique silhouette inflation + PLAN_RADIUS_MARGIN. Retry with a compliant effective radius that
            # fits the calibrated jaw opening while still enforcing table & wrist camera clearance.
            aperture = float(geo.tcp(gripper_raw)[3])
            max_fit_radius = max(0.012, 0.5 * (aperture - FIXED_SIDE_MARGIN - 0.016))
            relaxed_r = min(float(obj['measured_radius']), float(first['radius']), max_fit_radius)
            obj_relaxed = dict(obj, radius=relaxed_r, relaxed_from_radius_m=round(float(obj['radius']), 4))
            relaxed_plan = plan_grasp(
                geo,
                obj_relaxed,
                gripper_raw,
                q_here,
                pitches=(25, 15, 20, 10, 5, 30, 35, 0, 45),
                heights=(0.035, 0.045, 0.055, 0.065, 0.080, 0.095, 0.110),
            )
            if not relaxed_plan['problems']:
                obj = obj_relaxed
                plan = relaxed_plan
            elif not cancel.is_set() and not r.stop.is_set():
                try:
                    self.move(looks['home'], guard=self.guard(geo, table_floor=0.03), cancel=cancel, precision=45)
                except Exception:
                    pass
        plan_id = f'{int(time.time())}-{np.random.default_rng().integers(1 << 30):x}'
        self.pending = {'id': plan_id, 'created': time.monotonic(), 'start': q_here, 'gripper_raw': gripper_raw,
                        'plan': plan, 'query': query}
        summary = {'plan_id': plan_id, 'object': obj, 'first_measurement': first, 'grasp': plan['grasp'],
                   'waypoints': plan['waypoints'], 'problems': plan['problems'], 'gripper_raw': gripper_raw,
                   'look_views': [{'frame': f['id'], 'pan_offset': f['look_pan_offset']} for f in frames],
                   'chosen_view': answer['view_index'], 'executable': not plan['problems'],
                   'kinematic_model': geo.model_path}
        self.log({'type': 'plan', **summary})
        return summary

    def recover_and_retreat_high(self, geo, report, cancel=None, *, target_pan=None):
        """Safely neutralize any pending low-PWM gripper session, open jaws to OPEN_RAW, clear stop latch,
        and lift the arm vertically out of clutter back to a high observation pose."""
        import uuid as _uuid
        r = self.robot
        r.stop.clear()
        with r.io:
            r.connect()
            gp = r.gripper_power
            rec = gp.journal.get()
            try:
                gp._guard()
                gp._write(48, gp.LOW)
                pos = gp._position()
                target_goal = int(np.clip(max(pos, OPEN_RAW), gp.bounds[0], gp.bounds[1]))
                gp._write(42, target_goal)
                t_wait = time.monotonic() + 1.15
                while time.monotonic() < t_wait:
                    time.sleep(0.08)
                    if abs(gp._position() - target_goal) <= 8:
                        break
                settled_pos = gp._position()
                gp._write(42, settled_pos)
                time.sleep(0.10)
                gp._write(48, gp.ORIGINAL)
                if settled_pos < target_goal - 8:
                    gp._write(42, target_goal)
                    time.sleep(0.28)
                    settled_pos = gp._position()
                    gp._write(42, settled_pos)
                gp.journal.put({
                    'version': 2,
                    'session_id': (rec or {}).get('session_id') or _uuid.uuid4().hex,
                    'phase': 'completed',
                    'contact_possible': False,
                    'servo_id': gp.SERVO_ID,
                    'model': gp.MODEL,
                    'device_key': gp.device_key,
                    'position_bounds': list(gp.bounds),
                    'original_cap': gp.ORIGINAL,
                    'low_cap': gp.LOW,
                    'restored_cap': gp.ORIGINAL,
                    'open_position': gp.open_position,
                    'opening_direction': gp.opening_direction,
                    'neutral_goal': settled_pos,
                    'neutral_position': settled_pos,
                    'neutral_samples': 3,
                })
                gp.state = 'completed'
                gp.open_retreat_token = None
                r.pwm_owner = None
                self.grasp.state = 'idle'
            except Exception:
                pass
            state = r.read()
            for n, v in state.items():
                r._write(n, 42, int(v))
                r._write(n, 40, 1, 1)
            r.goals = {n: int(v) for n, v in state.items()}
            r.position = {n: int(v) for n, v in state.items()}
            r.bias = {n: 0 for n in state}
            r.armed = True
            self.sag = {n: 0 for n in BODY}
        r.stop.clear()
        q_cur = {n: int(state[n]) for n in BODY}
        table_guard = self.guard(geo, table_floor=TABLE_FLOOR)
        h_cur, _ = geo.table_clearance(q_cur)
        escape_guard = self.guard(geo, table_floor=min(-0.008, h_cur - 0.008))
        if h_cur < 0.085:
            report('垂直抬離桌面與物件區，避免橫向碰撞')
            q_lift_up = {
                'shoulder_pan': q_cur['shoulder_pan'],
                'shoulder_lift': 1180,
                'elbow_flex': 1680,
                'wrist_flex': 1980,
                'wrist_roll': 998,
            }
            self.fast_move(q_lift_up, guard=escape_guard, cancel=cancel, speed_ticks_per_s=280.0, wait_settle=True)
        looks = look_plan(geo)
        high_pose = dict(looks['home'])
        if target_pan is not None:
            high_pose['shoulder_pan'] = int(np.clip(target_pan, geo.limits['shoulder_pan'][0], geo.limits['shoulder_pan'][1]))
        report('退回高處觀察姿態，準備重新檢視桌面環境')
        self.fast_move(high_pose, guard=escape_guard, cancel=cancel, speed_ticks_per_s=340.0, wait_settle=True)
        return high_pose

    # ------------------------------------------------------------------ execute an approved plan
    def execute(self, plan_id, report, cancel, *, return_object=True, max_attempts=2):
        r = self.robot
        p = self.pending
        if p is None or p['id'] != plan_id or time.monotonic() - p['created'] > self.PLAN_TTL_S:
            raise RuntimeError('沒有這份有效的抓取計畫，請重新規劃')
        if p['plan']['problems']:
            raise RuntimeError('這份計畫有未解決的問題，不能執行')
        r.begin(allow_rearm=False)
        geo = self.geometry()
        now = r.read()
        if max(abs(now[n] - p['start'][n]) for n in BODY) > 8 or abs(now['gripper'] - p['gripper_raw']) > 6:
            raise RuntimeError('手臂或夾爪在規劃後移動過，請重新規劃')
        self.pending = None
        obj, gr = p['plan']['object'], p['gripper_raw']
        W = {w['label']: w['target'] for w in p['plan']['waypoints']}
        result = {
            'plan_id': plan_id,
            'object_name': obj.get('name') or p.get('query', ''),
            'fixed_jaw_gap_mm': p['plan']['grasp'].get('fixed_jaw_gap_mm', round(FIXED_SIDE_MARGIN * 1000, 2)),
            'steps': [],
            'grasp_confirmed': False,
            'completed': False,
        }

        def step(label, target, **kw):
            actual, err = self.move(target, cancel=cancel, **kw)
            q = {n: actual[n] for n in BODY}
            rec = {'label': label, 'target': target, 'actual': q, 'error_ticks': err,
                   'table_mm': round(geo.table_clearance(q)[0] * 1000, 1)}
            if label not in ('lift', 'lifted'):
                rec['object_mm'] = round(geo.object_clearance(q, obj, gripper_raw=gr)[0] * 1000, 1)
            result['steps'].append(rec)
            self.log({'type': 'step', **rec})
            return q

        def fast_step(label, target, *, guard, speed_ticks_per_s=300.0, wait_settle=True):
            actual, err = self.fast_move(
                target,
                guard=guard,
                cancel=cancel,
                speed_ticks_per_s=speed_ticks_per_s,
                wait_settle=wait_settle,
            )
            q = {n: actual[n] for n in BODY}
            rec = {'label': label, 'target': target, 'actual': q, 'error_ticks': err,
                   'table_mm': round(geo.table_clearance(q)[0] * 1000, 1)}
            if label not in ('lift', 'lifted'):
                rec['object_mm'] = round(geo.object_clearance(q, obj, gripper_raw=gr)[0] * 1000, 1)
            result['steps'].append(rec)
            self.log({'type': 'step', **rec})
            return q

        free = self.guard(geo, table_floor=0.02, obj=obj, object_floor=0.004, gripper_raw=gr)
        close = self.guard(geo, table_floor=TABLE_FLOOR, obj=obj, object_floor=-0.003, gripper_raw=gr)
        report('快速移到物件後上方')
        fast_step('above_pregrasp', W['above_pregrasp'], guard=free, speed_ticks_per_s=320.0, wait_settle=False)
        report('快速對準物件窄軸與左牙邊緣')
        fast_step('approach_70mm', W['approach_70mm'], guard=free, speed_ticks_per_s=280.0, wait_settle=True)
        for label in ('approach_50mm', 'approach_30mm', 'approach_15mm'):
            report('流暢沿直線接近物件')
            fast_step(label, W[label], guard=close, speed_ticks_per_s=200.0, wait_settle=(label == 'approach_15mm'))
        report('到達夾取位置，鎖定左固定牙貼邊姿態並修正靜差')
        q = step('grasp', W['grasp'], guard=close, speed=65, precision=48, correct=True, tolerance=3)
        Ttcp, fixed, moving, _ = geo.tcp(gr)
        G = geo.transforms(q)['gripper_link']
        local = np.array(p['plan']['grasp']['local_point_mm']) / 1000
        want = np.asarray(obj['centre']) + geo.n * p['plan']['grasp']['height_m']
        miss = float(np.linalg.norm(G[:3, :3] @ local + G[:3, 3] - want) * 1000)
        result['grasp_point_error_mm'] = round(miss, 2)
        if miss > 4:
            raise RuntimeError(f'夾取點偏差 {miss:.1f} mm，超過 4 mm，不合爪')
        reference = self.grasp.capture()
        report('低出力合爪')
        closed = self.grasp.close_candidate(reference, report, cancel, squeeze_ticks=SQUEEZE_TICKS)
        result['close'] = {'contact_position': closed['contact_sample']['position'],
                           'position': closed['sample']['position'], 'goal': closed['sample']['goal'],
                           'load': closed['sample']['load_signed_pwm'], 'squeeze_ticks': SQUEEZE_TICKS}
        sag_before = dict(self.sag)
        report('抬起 30 mm')
        lift_guard = self.guard(geo, table_floor=TABLE_FLOOR)
        # The held object's weight adds sag (shoulder 30-50 ticks measured); correct from measured joints.
        q_lift = step('lift', W['lift'], guard=lift_guard, speed=65, precision=60, correct=True, tolerance=6)
        after = self.grasp.capture()
        tel = r.gripper_telemetry()
        G0, G1 = geo.transforms(q)['gripper_link'], geo.transforms(q_lift)['gripper_link']
        lifted_mm = float(geo.n @ (G1[:3, 3] - G0[:3, 3]) * 1000)
        evidence = {'measured_lift_mm': round(lifted_mm, 1), 'gripper_position': tel['position'],
                    'gripper_load_signed_pwm': tel['load_signed_pwm'], 'empty_jaw_position_threshold': 1525}
        result['lift'] = evidence
        result['lifted_mm'] = round(lifted_mm, 1)
        report('Argon ER 2 核對是否真的拿起來')
        evidence['shoulder_lift_sag_ticks'] = {'before_lift': sag_before['shoulder_lift'], 'lifted': self.sag['shoulder_lift']}
        verdict = self.brain.verify_grasp(p['query'], [reference, after], evidence, cancel)
        result['verification'] = verdict
        self.log({'type': 'verify', 'evidence': evidence, 'verdict': verdict})
        # Motor evidence is primary: measured lift, jaw still blocked short of the empty position, and the
        # object's weight showing up as extra shoulder sag. The image may not show the table gap (the wrist
        # camera is beside the object), so it must only not contradict the hold.
        motor = (lifted_mm >= 15 and tel['position'] > 1540
                 and tel['position'] >= closed['contact_sample']['position'] - SQUEEZE_TICKS - 8)
        weight = self.sag['shoulder_lift'] - sag_before['shoulder_lift']
        result['motor_confirmed'] = bool(motor)
        result['shoulder_sag_increase_ticks'] = weight
        confirmed = motor and not verdict['gripper_empty']
        result['grasp_confirmed'] = bool(confirmed)
        report('已拿起物件' if confirmed else '抬起後無法確認拿到物件')
        if return_object:
            report('放回原位')
            # Back to 2 mm above the grasp pose: the object bottom just clears the table, never pressed into it.
            q_down, _, _ = pose_along(geo, W['grasp'], geo.n * 0.002, W['grasp'])
            step('lowered', {n: int(round(q_down[n])) for n in BODY}, guard=lift_guard, speed=60, precision=60,
                 correct=True, tolerance=4)
            down = self.grasp.capture()
            if cancel.wait(.25):
                raise RuntimeError('已停止')
            report('鬆開夾爪')
            back = self.guard(geo, table_floor=TABLE_FLOOR)

            def retreat():
                for label in ('approach_30mm', 'approach_70mm', 'above_pregrasp'):
                    fast_step('retreat_' + label, W[label], guard=back, speed_ticks_per_s=280.0, wait_settle=False)
                # The empty-jaw image must not show the placed object or table marks between the fingers.
                check_pose = dict(HOME, shoulder_pan=HOME['shoulder_pan'] + EMPTY_CHECK_PAN_OFFSET)
                self.screen_transit(geo, {n: W['above_pregrasp'][n] for n in BODY}, check_pose, obj=obj, gripper_raw=gr)
                fast_step('empty_check_view', check_pose, guard=free, speed_ticks_per_s=340.0, wait_settle=True)
            result['release'] = self.grasp.release(down, report, cancel, support_verified=True, retreat=retreat)
        result['completed'] = True
        self.log({'type': 'result', **result})
        return result
