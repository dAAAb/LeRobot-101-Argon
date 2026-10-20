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
OBJECT_MARGIN = 0.004      # m: robot surfaces (except the moving jaw pad, see below) vs measured cylinder surface
FIXED_SIDE_MARGIN = 0.004  # m: planned gap between the fixed pad and the cylinder surface (closer to centred contact)
PLAN_RADIUS_MARGIN = 0.0025  # m: added to the measured base radius for planning
GRASP_HEIGHT = 0.040       # m: pad cross-section above the table (default when no height preference applies)
SQUEEZE_TICKS = 55         # goal advance past the detected contact, under the low PWM cap
APPROACH = 0.070           # m: straight approach length along the approach axis
LIFT = 0.030               # m
MAX_PITCH_DEG = 48.0
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
                v = stl_vertices(self.cad.mesh_root / mesh.get('filename')) * np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
                tri = v.reshape(-1, 3, 3)
                pts = np.r_[v, tri.mean(1)]
                T = xyz_transform(col.find('origin'))
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
        gap((G[:3, :3] @ moving + G[:3, 3]) + line * z, 'moving_jaw_pad_line', inflate=0.0015)
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
        P = (np.c_[points, np.ones(len(points))] @ np.linalg.inv(C).T)[:, :3]
        uv, _ = cv2.projectPoints(np.ascontiguousarray(P, dtype=np.float64), np.zeros(3), np.zeros(3), self.K, self.D)
        return uv.reshape(-1, 2), P[:, 2]

    # ------------------------------------------------------------------ IK
    def solve(self, residual, q0, extra_starts=()):
        lo = np.array([self.limits[n][0] for n in BODY], float)
        hi = np.array([self.limits[n][1] for n in BODY], float)
        best = None
        for start in (q0, *extra_starts):
            x0 = np.clip(np.array([float(start[n]) for n in BODY]), lo + 1, hi - 1)
            sol = least_squares(
                lambda x: residual(dict(zip(BODY, x))),
                x0,
                bounds=(lo, hi),
                x_scale=100.0,
                ftol=1e-3,
                xtol=1e-3,
                max_nfev=45,
            )
            if best is None or sol.cost < best.cost:
                best = sol
            if best.cost <= 1.25:
                break
        return {n: float(v) for n, v in zip(BODY, best.x)}

    def tilt_deg(self, axis):
        return float(np.degrees(np.arcsin(np.clip(-(axis @ self.n), -1, 1))))   # >0: pointing down


# ---------------------------------------------------------------------- cylinder from two silhouette-foot pixels
def cylinder_from_feet(geo, q, left_uv, right_uv, top_uv=None, *, lying_flat=False):
    """Upright cylinder (or lying-flat horizontal prism/cylinder) from where its left/right silhouette meets the table.

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
    u /= max(dm, 1e-6)
    if lying_flat:
        r_flat = float(np.clip(a * 0.86, 0.014, 0.024))
        horiz_len = 2.0 * r_flat
        if top_uv is not None:
            T_table = geo.ray_to_table(q, top_uv)
            axis_vec = T_table - M
            axis_vec -= geo.n * (geo.n @ axis_vec)
            horiz_len = float(np.linalg.norm(axis_vec))
            if horiz_len > 0.010:
                shift_m = float(np.clip(0.38 * horiz_len, 0.018, 0.034))
                O = M + (axis_vec / horiz_len) * shift_m
            else:
                O = M + u * r_flat
        else:
            O = M + u * r_flat
        height = round(float(np.clip(2.0 * r_flat + 0.004, 0.038, 0.052)), 4)
        return {
            'centre': O.tolist(),
            'radius': r_flat,
            'height': height,
            'foot_left': A.tolist(),
            'foot_right': B.tolist(),
            'camera_distance_m': float(np.linalg.norm(O - C)),
            'lying_flat': True,
            'posture': 'lying_flat',
            'horizontal_length_m': round(horiz_len, 4),
        }
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


# ---------------------------------------------------------------------- directional & compound query helpers
import re as _re


def parse_direction_hint(query: str) -> str | None:
    """Extract 'left', 'right', or 'center' directional hint from a natural-language query."""
    q = (query or '').strip().lower()
    if not q:
        return None
    left_tokens = ('往左', '向左', '左轉', '左邊', '左側', '左下', '左上', '左方', '左面', 'left')
    right_tokens = ('往右', '向右', '右轉', '右邊', '右側', '右下', '右上', '右方', '右面', 'right')
    center_tokens = ('正前方', '正中間', '中間', '前方', 'center', 'front', 'middle')
    l_pos = min((q.find(t) for t in left_tokens if t in q), default=10**9)
    r_pos = min((q.find(t) for t in right_tokens if t in q), default=10**9)
    if l_pos < r_pos and l_pos < 10**9:
        return 'left'
    if r_pos < l_pos and r_pos < 10**9:
        return 'right'
    if any(t in q for t in center_tokens):
        return 'center'
    return None


def extract_grasp_query_info(query: str) -> dict:
    """Parse a natural-language grasp instruction into direction_hint, target_query, and return_object."""
    raw = (query or '').strip()
    direction_hint = parse_direction_hint(raw)
    no_return_tokens = ('不要放回', '別放回', '不要放下', '拿著就好', '保持拿著', '不用放回', 'hold it')
    return_object = not any(tok in raw.lower() for tok in no_return_tokens)

    # Extract clean object noun phrase from compound instructions like「往左找到牙籤罐，把它拿起來再放回去」
    clauses = [c.strip() for c in _re.split(r'[，,；;。！!？?]+|然後|接著|再把|再將', raw) if c.strip()]
    candidate = ''
    pronouns = {'它', '他', '她', '這個', '那個', '東西', '物件', '目標', '原位', '原地', '桌面', '桌上', 'it', 'them'}
    for clause in clauses:
        cleaned = _re.sub(
            r'^(?:請|幫我|麻煩|先|試著|嘗試)*(?:往左邊?|向左邊?|往右邊?|向右邊?|左轉|右轉|到左邊|到右邊|去左邊|去右邊|在左邊|在右邊|在左下角?|在右下角?|往正前方|往中間|左邊的?|右邊的?|左側的?|右側的?|左下角?的?|右下角?的?)*'
            r'(?:看一看|看看|看|轉向|轉過去|尋找|搜尋|找到|找找看|找找|找一下|找|去拿|去抓|去夾|把|將|拿取|抓取|夾取|拿起|抓起|夾起|拿|抓|夾)*',
            '',
            clause,
        ).strip()
        cleaned = _re.sub(
            r'^(?:please\s+)?(?:pick\s+up|grasp|grab|find|get|take)\s+(?:the\s+)?(?:left\s+|right\s+)?',
            '',
            cleaned,
            flags=_re.IGNORECASE,
        ).strip()
        cleaned = _re.sub(
            r'\s+(?:on\s+the\s+left|on\s+the\s+right|and\s+put\s+it\s+back).*$',
            '',
            cleaned,
            flags=_re.IGNORECASE,
        ).strip()
        cleaned = _re.sub(
            r'(?:把它|將它)?(?:拿起來|抓起來|夾起來|撿起來|拿取|抓取|夾取|拿起|抓起|夾起|放回去|放回原位|放回桌面|放回|放下|再放回去|再放下)+.*$',
            '',
            cleaned,
        ).strip()
        cleaned = _re.sub(r'^(?:的|之)+|(?:的|之)+$', '', cleaned).strip()
        if cleaned and cleaned.lower() not in pronouns:
            candidate = cleaned
            break
    target_query = candidate or raw
    return {
        'raw_query': raw,
        'target_query': target_query,
        'direction_hint': direction_hint,
        'return_object': return_object,
    }


def is_kinematic_grasp_command(query: str) -> bool:
    """Return True if a search/command query asks to find and pick up (and optionally put back) an object."""
    q = (query or '').strip()
    if not q:
        return False
    lower = q.lower()
    if any(neg in lower for neg in ('不要抓', '不要拿', '不要夾', '別抓', '別拿', '別夾')):
        return False
    # Exclude multi-object placement / stacking onto a different destination object
    if any(tok in lower for tok in ('堆', '疊', '一疊', 'stack')):
        return False
    if '放到' in lower and not any(ret in lower for ret in ('放到原位', '放到原地', '放回')):
        return False
    grasp_phrases = (
        '拿起來', '抓起來', '夾起來', '撿起來', '拿取', '抓取', '夾取', '拿起', '抓起', '夾起',
        '把它拿', '把它抓', '把它夾', '幫我抓', '幫我拿', '幫我夾', '去抓', '去拿', '去夾',
        '試著抓', '嘗試輕夾', 'pick up', 'grasp', 'grab',
    )
    if any(p in lower for p in grasp_phrases):
        return True
    if _re.match(r'^(?:請|幫我|先)?(?:往左|向左|往右|向右)?(?:抓|夾|拿)\S+', q):
        return True
    return False


# ---------------------------------------------------------------------- look poses
def look_pose(geo, target, q_ref, rng, *, dist=(0.15, 0.34), centre_box=(600, 1320, 250, 700), half=0.05,
              min_height=TRANSIT_HEIGHT, n=300_000, batches=6, prefer='motion', pan_window=900, roll_window=120):
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
    center_u = 0.5 * (centre_box[0] + centre_box[1])
    center_v = 0.5 * (centre_box[2] + centre_box[3])
    found = []
    for _ in range(batches):
        S = {m: np.round(rng.uniform(*geo.limits[m], n)) for m in BODY}
        S['shoulder_pan'] = np.round(rng.uniform(max(geo.limits['shoulder_pan'][0], q_ref['shoulder_pan'] - pan_window),
                                                 min(geo.limits['shoulder_pan'][1], q_ref['shoulder_pan'] + pan_window), n))
        if roll_window is not None:
            roll_ref = q_ref.get('wrist_roll', HOME['wrist_roll'])
            S['wrist_roll'] = np.round(rng.uniform(max(geo.limits['wrist_roll'][0], roll_ref - roll_window),
                                                   min(geo.limits['wrist_roll'][1], roll_ref + roll_window), n))
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
            img_dist = float(np.hypot(u[i, 0] - center_u, v[i, 0] - center_v))
            score = (-float(C[i, :3, 3] @ geo.n) if prefer == 'high'
                     else float(max(abs(q[m] - q_ref[m]) for m in BODY)) + 0.35 * img_dist)
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
LOOK_PAN_OFFSETS = (0, -240, -450, 240, 450)
DIRECTIONAL_PAN_OFFSETS = {
    'left': (-220, -420, -650, 0, 320),
    'right': (220, 420, 650, 0, -320),
    'center': (0, -240, 240, -450, 450),
}
EMPTY_CHECK_PAN_OFFSET = -250   # clean background for the empty-jaw image (no table marks between the jaws)


def look_plan(geo, home=None, pan_offsets=None, *, direction_hint=None):
    home = dict(home or HOME)
    if pan_offsets is None:
        pan_offsets = DIRECTIONAL_PAN_OFFSETS.get(direction_hint, LOOK_PAN_OFFSETS)
    views = []
    for d in pan_offsets:
        q = {**home, 'shoulder_pan': home['shoulder_pan'] + d}
        if not all(geo.limits[n][0] <= q[n] <= geo.limits[n][1] for n in BODY):
            continue
        views.append({'pan_offset': d, 'pose': q, 'transit_m': round(transit_clearance(geo, q), 4)})
    return {'home': home, 'views': views, 'direction_hint': direction_hint}


def _extract_locate_uv(answer: dict) -> dict:
    bl = answer.get('bottom_left') or {'x': 0.0, 'y': 0.0}
    br = answer.get('bottom_right') or {'x': 0.0, 'y': 0.0}
    tc = answer.get('top_center') or {'x': 0.0, 'y': 0.0}
    bl_x, bl_y = float(bl.get('x', 0.0)), float(bl.get('y', 0.0))
    br_x, br_y = float(br.get('x', 0.0)), float(br.get('y', 0.0))
    tc_x, tc_y = float(tc.get('x', 0.0)), float(tc.get('y', 0.0))
    has_feet = (bl_x > 0.01 or bl_y > 0.01) and (br_x > 0.01 or br_y > 0.01) and (br_x > bl_x + 0.008)
    cx = 0.5 * (bl_x + br_x) if has_feet else tc_x
    by = max(bl_y, br_y) if has_feet else min(0.95, tc_y + 0.16)
    return {
        'has_feet': has_feet,
        'bl_x': bl_x, 'bl_y': bl_y,
        'br_x': br_x, 'br_y': br_y,
        'tc_x': tc_x, 'tc_y': tc_y,
        'cx': cx, 'by': by,
    }


def _point_in_jaw_box(u_px: float, v_px: float, margin_px: float = 35.0) -> bool:
    for x0, y0, x1, y1 in JAW_BOXES:
        if (x0 - margin_px) <= u_px <= (x1 + margin_px) and (y0 - margin_px) <= v_px <= (y1 + margin_px):
            return True
    return False


def _needs_foveation(answer: dict, *, require_top: bool = True) -> bool:
    """Return True if the located object is off-center, near frame borders/gripper jaws, or partially occluded."""
    if not answer.get('found'):
        return False
    uv = _extract_locate_uv(answer)
    if not answer.get('upright_on_table') or not answer.get('bottom_fully_visible'):
        return True
    if require_top and not answer.get('top_visible'):
        return True
    if float(answer.get('confidence', 0.0)) < 0.72:
        return True
    # Object is off-center horizontally (arm has not turned to face the object squarely)
    if not (0.36 <= uv['cx'] <= 0.62) or uv['bl_x'] < 0.12 or uv['br_x'] > 0.88:
        return True
    # Object bottom is close to bottom border or in the white gripper jaw occlusion zone
    if uv['by'] > 0.74 or (require_top and uv['tc_y'] < 0.06):
        return True
    if (_point_in_jaw_box(uv['bl_x'] * IMAGE[0], uv['bl_y'] * IMAGE[1])
            or _point_in_jaw_box(uv['br_x'] * IMAGE[0], uv['br_y'] * IMAGE[1])):
        return True
    return False


def foveate_poses(geo, view_frame, answer, *, seed=11):
    """Compute 1-2 centered wrist-camera poses that turn the arm (`shoulder_pan`) and pitch the wrist (`wrist_flex`)
    to face a detected edge/partially-occluded object squarely in the unobstructed upper-middle window."""
    q_rough = {n: int(view_frame['joints'][n]) for n in BODY}
    uv = _extract_locate_uv(answer)
    cx, by = uv['cx'], uv['by']

    # Direct visual-servo alignment:
    # - Turn shoulder_pan so cx moves to 0.49 (dead center between left jaw x<=0.448 and right jaw x>=0.693)
    # - Pitch wrist_flex so by moves above the jaws while keeping tall object tops (tc_y) inside the upper frame
    span_y = max(0.08, by - uv['tc_y'])
    target_by = float(np.clip(max(0.48, span_y + 0.14), 0.48, 0.76))
    pan_target = int(np.clip(round(q_rough['shoulder_pan'] + (cx - 0.49) * 960), *geo.limits['shoulder_pan']))
    wf_target = int(np.clip(round(q_rough['wrist_flex'] + (by - target_by) * 520),
                            max(1920, geo.limits['wrist_flex'][0]),
                            min(2620, geo.limits['wrist_flex'][1])))

    q_servo = {
        **HOME,
        'shoulder_pan': pan_target,
        'wrist_flex': wf_target,
        'wrist_roll': HOME['wrist_roll'],
    }
    q_retracted = {
        **HOME,
        'shoulder_pan': pan_target,
        'shoulder_lift': int(np.clip(HOME['shoulder_lift'] - 70, *geo.limits['shoulder_lift'])),
        'elbow_flex': int(np.clip(HOME['elbow_flex'] - 55, *geo.limits['elbow_flex'])),
        'wrist_flex': int(np.clip(wf_target + 35, max(1920, geo.limits['wrist_flex'][0]), min(2620, geo.limits['wrist_flex'][1]))),
        'wrist_roll': HOME['wrist_roll'],
    }

    candidates = [q_servo]
    u_px = float(np.clip(cx * IMAGE[0], 40.0, IMAGE[0] - 40.0))
    v_px = float(np.clip(by * IMAGE[1], 140.0, IMAGE[1] - 15.0))
    try:
        target_3d = geo.ray_to_table(q_rough, (u_px, v_px))
        rng = np.random.default_rng(seed)
        for min_h, dist in ((0.09, (0.15, 0.34)), (0.06, (0.13, 0.36))):
            q_ik = look_pose(
                geo,
                target_3d,
                q_servo,
                rng,
                dist=dist,
                centre_box=(740, 1180, 260, 560),
                half=0.035,
                min_height=min_h,
                pan_window=260,
                roll_window=90,
            )
            if q_ik is not None:
                candidates.append(q_ik)
                break
    except Exception:
        pass
    candidates.append(q_retracted)

    valid = []
    for q_cand in candidates:
        if not all(geo.limits[n][0] <= q_cand[n] <= geo.limits[n][1] for n in BODY):
            continue
        if transit_clearance(geo, q_cand) < 0.045:
            continue
        if any(max(abs(q_cand[n] - existing[n]) for n in BODY) < 30 for existing in valid):
            continue
        valid.append(q_cand)
        if len(valid) >= 2:
            break
    return valid


def near_view(geo, obj, q_ref, *, seed=7):
    """Closer wrist view of a located cylinder foot, clear of the cylinder and the table; None if unreachable."""
    rng = np.random.default_rng(seed)
    centre = np.asarray(obj['centre'])
    for min_h, dist in ((0.10, (0.17, 0.30)), (0.07, (0.15, 0.32)), (0.05, (0.13, 0.34))):
        q = look_pose(geo, centre, q_ref, rng, dist=dist, half=0.04, min_height=min_h, pan_window=800, roll_window=110)
        if q is None:
            continue
        gap, _ = geo.object_clearance(q, obj, gripper_raw=geo.jaw.samples[-1].raw)
        if gap >= 0.03:
            return q
    return None


# ---------------------------------------------------------------------- grasp plan
def grasp_pose(geo, obj, gripper_raw, q_ref, *, height=GRASP_HEIGHT, pitch_deg=None, depth_offset=0.0):
    """Pose that puts the measured cylinder axis between the open pads at `height`, closing axis level.

    The cylinder surface is FIXED_SIDE_MARGIN from the fixed pad (the moving pad closes the rest).
    `depth_offset` (m) shifts the contact point toward the palm so the gripper advances deeper around the object."""
    Ttcp, fixed, moving, aperture = geo.tcp(gripper_raw)
    r = obj['radius']
    if aperture - 2 * r < FIXED_SIDE_MARGIN + 0.005:
        raise ValueError(f'measured diameter {2000 * r:.0f} mm does not fit the {1000 * aperture:.0f} mm opening')
    xh, zh = Ttcp[:3, 0], Ttcp[:3, 2]
    local = fixed + xh * (r + FIXED_SIDE_MARGIN) - zh * float(depth_offset)
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
        return np.concatenate([np.ravel(o) for o in out])

    starts = [
        {**HOME, 'shoulder_pan': q_ref['shoulder_pan']},
        {**q_ref, 'shoulder_pan': q_ref['shoulder_pan'] - 180},
        {**q_ref, 'shoulder_pan': q_ref['shoulder_pan'] + 180},
    ]
    q = geo.solve(res, q_ref, starts)
    G = geo.transforms(q)['gripper_link']
    err = float(np.linalg.norm(G[:3, :3] @ local + G[:3, 3] - target) * 1000)
    return q, {'position_error_mm': err, 'pitch_deg': geo.tilt_deg(G[:3, :3] @ zh),
               'closing_axis_vs_table_deg': float(np.degrees(np.arcsin(abs((G[:3, :3] @ xh) @ geo.n)))),
               'local_point_mm': (local * 1000).round(2).tolist(), 'aperture_mm': aperture * 1000,
               'depth_offset_mm': round(float(depth_offset) * 1000.0, 1)}


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


def plan_grasp(geo, obj, gripper_raw, q_now, pitches=(25, 20, 15, 35, 5, 45), heights=(0.032, 0.040, 0.050, 0.055, 0.060, 0.080, 0.100, 0.115), *, tuning=None):
    """above-pregrasp -> pregrasp -> straight approach -> grasp -> lift, all screened.

    Tries grasp heights (low first) x approach pitches (deg below horizontal).
    A candidate grasp pose is screened alone first (wrist camera/palm vs the measured cylinder, table);
    only survivors get a full path. Returns the first plan without problems, else the one with the fewest."""
    if not obj.get('height'):
        raise ValueError('object height was not measured')
    tuning = tuning or {}
    h_tune = float(tuning.get('height_offset_mm', 0.0)) / 1000.0
    d_tune = float(tuning.get('depth_offset_mm', 0.0)) / 1000.0
    lat_tune = float(tuning.get('lateral_offset_mm', 0.0)) / 1000.0
    if abs(lat_tune) > 1e-5:
        obj = dict(obj)
        c = np.asarray(obj['centre'], dtype=float).copy()
        radial_v = c - geo.n * (geo.n @ c)
        rn = float(np.linalg.norm(radial_v))
        if rn > 1e-6:
            lat_dir = np.cross(geo.n, radial_v / rn)
            lat_dir /= max(float(np.linalg.norm(lat_dir)), 1e-6)
            c = c + lat_dir * lat_tune
            obj['centre'] = c.tolist()
    # For lying-flat/tipped-over containers (height ~38..56 mm), grasp low at 25..34 mm above the table.
    # For shorter upright boxes/containers (<=105 mm), prefer grasping lower on the body (34..58 mm);
    # for tall cylinders (>105 mm), keep depth_offset=0.0 so the upper wrist camera mount stays clear.
    if obj.get('lying_flat') or obj['height'] <= 0.056:
        prefer = float(np.clip(0.58 * obj['height'] + h_tune, 0.024, 0.035))
        depth_offsets = tuple(dict.fromkeys(round(float(np.clip(base + d_tune, -0.015, 0.025)), 4) for base in (0.006, 0.003, 0.0)))
        max_grasp_h = max(0.035, obj['height'] - 0.008)
        heights = tuple(heights) + (0.025, 0.027, 0.030, 0.033)
    elif obj['height'] <= 0.105:
        if obj['radius'] <= 0.020:
            prefer = float(np.clip(0.38 * obj['height'] + h_tune, 0.030, 0.048))
            base_depths = (0.006, 0.0)
        else:
            prefer = float(np.clip(0.62 * obj['height'] + h_tune, 0.034, 0.058))
            base_depths = (0.0, 0.006)
        depth_offsets = tuple(dict.fromkeys(round(float(np.clip(base + d_tune, -0.015, 0.025)), 4) for base in base_depths))
        max_grasp_h = max(0.042, obj['height'] - 0.015)
    else:
        prefer = float(np.clip(0.45 * obj['height'] + h_tune, 0.032, 0.080))
        depth_offsets = (round(float(np.clip(d_tune, -0.015, 0.020)), 4),) if abs(d_tune) > 1e-5 else (0.0,)
        max_grasp_h = max(0.042, obj['height'] - 0.015)
    heights = sorted(set(heights) | {round(prefer, 3)}, key=lambda h: (abs(h - prefer), h))
    best, tried = None, []
    for h in heights:
        if h > max_grasp_h:
            continue
        for pitch in pitches:
            for depth_offset in depth_offsets:
                try:
                    q, info = grasp_pose(geo, obj, gripper_raw, q_now, height=h, pitch_deg=pitch, depth_offset=depth_offset)
                except ValueError as exc:
                    tried.append({'height_mm': h * 1000, 'pitch': pitch, 'depth_mm': depth_offset * 1000, 'rejected': str(exc)})
                    break
                gap = geo.object_clearance(q, obj, gripper_raw=gripper_raw)
                table = geo.table_clearance(q)
                ok = (info['position_error_mm'] <= 1.5 and info['closing_axis_vs_table_deg'] <= 3
                      and abs(info['pitch_deg'] - pitch) <= 8 and gap[0] >= OBJECT_MARGIN and table[0] >= PLAN_TABLE_CLEARANCE)
                tried.append({'height_mm': h * 1000, 'pitch': pitch, 'depth_mm': depth_offset * 1000,
                              'ik_mm': round(info['position_error_mm'], 2),
                              'object_mm': round(gap[0] * 1000, 1), 'object_link': gap[1], 'table_mm': round(table[0] * 1000, 1)})
                if not ok:
                    continue
                try:
                    plan = _plan_grasp_at(
                        geo,
                        obj,
                        gripper_raw,
                        q_now,
                        pitch,
                        h,
                        depth_offset=depth_offset,
                        grasp_solution=(q, info),
                        fast_reject=True,
                    )
                except ValueError as exc:
                    plan = {'problems': [str(exc)], 'grasp': {'pitch_target_deg': pitch, 'height_m': h, 'depth_offset_mm': depth_offset * 1000}, 'waypoints': []}
                plan['candidates'] = tried
                if not plan['problems']:
                    return plan
                if best is None or len(plan['problems']) < len(best['problems']):
                    best = plan
    return best or {'problems': ['no grasp height/pitch keeps the wrist camera and palm clear of the object'],
                    'candidates': tried, 'waypoints': [], 'grasp': {}}


def _plan_grasp_at(geo, obj, gripper_raw, q_now, pitch, height=GRASP_HEIGHT, *, depth_offset=0.0, grasp_solution=None, fast_reject=False):
    if grasp_solution is not None:
        q_grasp, info = grasp_solution
        info = dict(info)
    else:
        q_grasp, info = grasp_pose(geo, obj, gripper_raw, q_now, height=height, pitch_deg=pitch, depth_offset=depth_offset)
    info.update(pitch_target_deg=pitch, height_m=height)
    if info['position_error_mm'] > 1.5 or info['closing_axis_vs_table_deg'] > 3 or abs(info['pitch_deg'] - pitch) > 8:
        raise ValueError(f'no reachable level grasp pose at {pitch} deg: '
                         f"err {info['position_error_mm']:.1f} mm, pitch {info['pitch_deg']:.1f}, closing {info['closing_axis_vs_table_deg']:.1f}")
    Ttcp, *_ = geo.tcp(gripper_raw)
    zb = geo.transforms(q_grasp)['gripper_link'][:3, :3] @ Ttcp[:3, 2]
    radial_dist = float(np.hypot(obj['centre'][0], obj['centre'][1]))
    scale = min(1.0, max(0.75, (radial_dist - 0.145) / 0.125))
    pre_label = f'approach_{int(APPROACH * 1000)}mm'
    steps = [
        ('grasp', 0.0),
        ('approach_15mm', 0.015 * scale),
        ('approach_30mm', 0.030 * scale),
        ('approach_50mm', 0.050 * scale),
        (pre_label, APPROACH * scale),
    ]
    poses = {'grasp': (q_grasp, 0.0, 0.0)}
    q_pre, e_pre, rot_pre = pose_along(geo, q_grasp, -zb * (APPROACH * scale), q_grasp)
    if fast_reject:
        max_ik, max_rot = (2.5, 12.0) if radial_dist < 0.30 else (1.5, 3.0)
        if round(e_pre, 2) > max_ik or round(rot_pre, 2) > max_rot:
            raise ValueError(f'{pre_label}: IK {round(e_pre, 2)} mm / {round(rot_pre, 2)} deg')
    poses[pre_label] = (q_pre, e_pre, rot_pre)
    prev = q_grasp
    for label, back in steps:
        if back == 0 or label == pre_label:
            continue
        q, e, rot = pose_along(geo, q_grasp, -zb * back, prev)
        poses[label] = (q, e, rot)
        prev = q
        if fast_reject:
            max_ik, max_rot = (2.0, 6.5) if radial_dist < 0.30 else (1.5, 3.0)
            if round(e, 2) > max_ik or round(rot, 2) > max_rot:
                raise ValueError(f'{label}: IK {round(e, 2)} mm / {round(rot, 2)} deg')
    pre = poses[pre_label][0]
    above, e_a, r_a = None, np.inf, np.inf
    for above_h in (0.08, 0.06, 0.045):
        cand_a, cand_e, cand_r = pose_along(geo, q_grasp, -zb * (APPROACH * scale) + geo.n * above_h, pre)
        if cand_e <= 1.5 and cand_r <= 3.0 and geo.table_clearance(cand_a)[0] >= 0.032:
            above, e_a, r_a = cand_a, cand_e, cand_r
            break
        if above is None or (cand_e + 0.5 * cand_r) < (e_a + 0.5 * r_a):
            above, e_a, r_a = cand_a, cand_e, cand_r
    lift, e_l, r_l = pose_along(geo, q_grasp, geo.n * LIFT, q_grasp)
    order = [('above_pregrasp', above, e_a, r_a)]
    for label, back in reversed(steps):
        q, e, rot = poses[label]
        order.append((label, q, e, rot))
    order.append(('lift', lift, e_l, r_l))

    ik_problems = []
    for label, q, e, rot in order:
        if label == 'above_pregrasp':
            max_ik, max_rot = (3.5, 25.0) if radial_dist < 0.30 else (3.0, 8.0)
        elif label == pre_label and radial_dist < 0.30:
            max_ik, max_rot = (2.5, 12.0)
        elif radial_dist < 0.30 and label != 'grasp':
            max_ik, max_rot = (2.0, 6.5)
        else:
            max_ik, max_rot = (1.5, 3.0)
        if round(e, 2) > max_ik or round(rot, 2) > max_rot:
            ik_problems.append(f'{label}: IK {round(e, 2)} mm / {round(rot, 2)} deg')
    if fast_reject and ik_problems:
        raise ValueError(ik_problems[0])

    screen_steps = 4 if ik_problems else 12
    waypoints, prev = [], q_now
    for label, q, e, rot in order:
        qi = {n: int(round(q[n])) for n in BODY}
        check = segment_screen(geo, prev, qi, obj, gripper_raw, steps=screen_steps, object_check=label != 'lift')
        if fast_reject:
            floor = 30.0 if label == 'above_pregrasp' else PLAN_TABLE_CLEARANCE * 1000
            if check['table_mm'] < floor:
                raise ValueError(f"{label}: table {check['table_mm']} mm ({check['table_link']})")
            if check['object_mm'] is not None and check['object_mm'] < OBJECT_MARGIN * 1000:
                raise ValueError(f"{label}: object gap {check['object_mm']} mm ({check['object_link']})")
        waypoints.append({'label': label, 'target': qi, 'ik_error_mm': round(e, 2), 'ik_rotation_deg': round(rot, 2),
                          **check, 'max_joint_step_ticks': int(max(abs(qi[n] - prev[n]) for n in BODY))})
        prev = qi
    problems = []
    for w in waypoints:
        if w['label'] == 'above_pregrasp':
            max_ik, max_rot = (3.5, 25.0) if radial_dist < 0.30 else (3.0, 8.0)
        elif w['label'] == pre_label and radial_dist < 0.30:
            max_ik, max_rot = (2.5, 12.0)
        elif radial_dist < 0.30 and w['label'] != 'grasp':
            max_ik, max_rot = (2.0, 6.5)
        else:
            max_ik, max_rot = (1.5, 3.0)
        if w['ik_error_mm'] > max_ik or w['ik_rotation_deg'] > max_rot:
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
    MAX_BIAS = 55

    def __init__(self, robot, brain, grasp, runtime, root=ROOT):
        self.robot, self.brain, self.grasp = robot, brain, grasp
        self.runtime, self.root = Path(runtime), Path(root)
        self.pending = None
        self.sag = {n: 0 for n in BODY}
        self.action_logger = None
        self.last_measured_objects = {}

    @staticmethod
    def _target_key(query: str) -> str:
        q = (query or '').strip().lower()
        if any(k in q for k in ('toothpick', '牙籤', '牙签', '方盒')):
            return 'toothpick_container'
        if any(k in q for k in ('發泡', 'effervescent', '圓筒', '橘蓋', 'vitus')):
            return 'effervescent_tube'
        return q or 'default'

    def _tuning_path(self) -> Path:
        return self.runtime / 'loop-engineer-tuning.json'

    def get_tuning(self, query: str) -> dict:
        path = self._tuning_path()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text())
            return dict(data.get(self._target_key(query)) or {})
        except Exception:
            return {}

    def update_tuning(self, query: str, patch: dict) -> dict:
        path = self._tuning_path()
        data = {}
        if path.exists():
            try:
                data = json.loads(path.read_text())
            except Exception:
                data = {}
        key = self._target_key(query)
        cur = dict(data.get(key) or {})
        for k, v in (patch or {}).items():
            if v is None:
                cur.pop(k, None)
            elif k in ('height_offset_mm', 'depth_offset_mm', 'lateral_offset_mm'):
                cur[k] = round(float(np.clip(float(v), -28.0, 28.0)), 1)
            elif k == 'extra_squeeze_ticks':
                cur[k] = int(np.clip(int(v), -20, 45))
            else:
                cur[k] = v
        cur['updated_at'] = time.time()
        data[key] = cur
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        return cur

    def reset_tuning(self, query: str | None = None) -> dict:
        path = self._tuning_path()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text())
        except Exception:
            data = {}
        if query is None:
            data = {}
        else:
            data.pop(self._target_key(query), None)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        return data

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

    def log(self, event, *, wrist_frame=None, gripper_telemetry=None):
        with (self.runtime / 'kinematic-grasp.jsonl').open('a') as f:
            f.write(json.dumps({'time': time.time(), **event}, ensure_ascii=False, default=float) + '\n')
        if self.action_logger is not None:
            step_name = str(event.get('label') or event.get('type') or 'event')
            self.action_logger.record(
                step_name,
                event,
                wrist_frame=wrist_frame,
                gripper_telemetry=gripper_telemetry,
            )

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

    def move(self, target, *, guard, cancel, speed=60, precision=45, correct=False, tolerance=4, track_sag=False):
        """glide with the measured servo steady-state error fed forward through Robot.bias."""
        r = self.robot
        target = {n: int(target[n]) for n in BODY}
        glide_precision = max(int(precision), 55)
        for attempt in range(4 if correct else 1):
            if cancel.is_set() or r.stop.is_set():
                raise RuntimeError('已停止')
            r.bias.update({n: int(np.clip(self.sag[n], -self.MAX_BIAS, self.MAX_BIAS)) for n in BODY})
            r.glide(target, speed=speed, precision=glide_precision, measured_guard=guard, allow_rearm=False, settle_timeout_s=2.0)
            if r.stop.wait(.35):
                raise RuntimeError('已停止')
            actual = r.read()
            err = {n: target[n] - actual[n] for n in BODY}
            if correct or track_sag:
                self.sag = {n: int(np.clip(r.bias[n] + err[n], -self.MAX_BIAS, self.MAX_BIAS)) for n in BODY}
            if max(abs(v) for v in err.values()) <= tolerance:
                break
        return actual, err

    def screen_transit(self, geo, a, b, *, obj=None, gripper_raw=None, table_floor=0.05, object_floor=0.010):
        start_h = geo.table_clearance(a)[0]
        need = min(table_floor, start_h - 0.002)
        s = segment_screen(geo, a, b, obj, gripper_raw or geo.jaw.samples[-1].raw, object_check=obj is not None)
        if s['table_mm'] < need * 1000:
            raise RuntimeError(f"路徑離桌面只剩 {s['table_mm']} mm（{s['table_link']}），不移動")
        if obj is not None and s['object_mm'] is not None and s['object_mm'] < object_floor * 1000:
            raise RuntimeError(f"路徑離物件只剩 {s['object_mm']} mm（{s['object_link']}），不移動")
        return s

    def locate(self, geo, query, frames, cancel, *, require_top=True, allow_foveated=False, report=None, tuning=None):
        answer = self.brain.locate_upright_feet(query, frames, cancel)
        self.log({'type': 'locate', 'answer': answer, 'frames': [{'id': f['id'], 'joints': f['joints']} for f in frames]})
        if not answer['found'] or answer['confidence'] < .48:
            raise RuntimeError('Argon ER 2 沒有在畫面中找到要抓的物件：' + answer['reason'])
        rough_answer = answer
        rough_frame = frames[min(answer['view_index'], len(frames) - 1)]
        foveated_done = False
        if allow_foveated and _needs_foveation(answer, require_top=require_top):
            fov_poses = foveate_poses(geo, rough_frame, answer)
            if fov_poses:
                fov_frames = []
                obj_label = answer.get('object_name') or query
                # Move to the primary centered pose first; only try a 2nd pose if the 1st doesn't locate cleanly
                for fi, q_fov in enumerate(fov_poses[:1]):
                    if report is not None:
                        report(f'轉動手臂面向「{obj_label}」，對準畫面中央再看清楚')
                    here = self.robot.read()
                    q_here = {n: here[n] for n in BODY}
                    if max(abs(q_here[n] - q_fov[n]) for n in BODY) > 15:
                        self.screen_transit(geo, q_here, q_fov, table_floor=0.03)
                        self.move(q_fov, guard=self.guard(geo, table_floor=0.025), cancel=cancel, precision=40)
                    ff = self.grasp.capture()
                    ff['look_pan_offset'] = int(q_fov['shoulder_pan'] - HOME['shoulder_pan'])
                    fov_frames.append(ff)
                fov_answer = self.brain.locate_upright_feet(query, fov_frames, cancel)
                self.log({'type': 'locate_foveated', 'answer': fov_answer, 'frames': [{'id': f['id'], 'joints': f['joints']} for f in fov_frames]})
                foveated_done = True
                if fov_answer['found'] and fov_answer['confidence'] >= .48:
                    if ((fov_answer['bottom_fully_visible'])
                            or not (answer['bottom_fully_visible'])
                            or fov_answer['confidence'] >= answer['confidence'] - 0.08):
                        answer = fov_answer
                        frames = fov_frames
        has_valid_feet = (
            abs(float(answer['bottom_right']['x']) - float(answer['bottom_left']['x'])) >= 0.015
            and 0.02 <= float(answer['bottom_left']['y']) <= 0.99
            and 0.02 <= float(answer['bottom_right']['y']) <= 0.99
        )
        strict_ok = (
            has_valid_feet
            and answer['bottom_fully_visible']
            and (answer['top_visible'] or not require_top or not answer.get('upright_on_table', True))
        )
        foveated_ok = (
            (foveated_done or not allow_foveated)
            and has_valid_feet
            and answer['confidence'] >= 0.48
        )
        if not (strict_ok or foveated_ok):
            raise RuntimeError('物件不是直立在桌面上，或底部／頂部看不完整：' + answer['reason'])
        frame = frames[min(answer['view_index'], len(frames) - 1)]
        gray = cv2.imread(frame['path'], cv2.IMREAD_GRAYSCALE)
        if gray.shape[:2] != (IMAGE[1], IMAGE[0]):
            raise RuntimeError('腕部影像解析度和相機模型不同')
        px = {k: (answer[k]['x'] * IMAGE[0], answer[k]['y'] * IMAGE[1]) for k in ('bottom_left', 'bottom_right', 'top_center')}
        refined = refine_silhouette_feet(gray, px['bottom_left'], px['bottom_right'])
        q = {n: frame['joints'][n] for n in BODY}
        has_top_pt = answer['top_visible'] or (
            0.0 <= float(answer['top_center']['y']) < min(float(answer['bottom_left']['y']), float(answer['bottom_right']['y'])) - 0.03
        )
        est_upright = cylinder_from_feet(geo, q, refined[0], refined[1], px['top_center'] if has_top_pt else None, lying_flat=False)
        reason_lc = str(answer.get('reason', '')).lower()
        name_lc = (str(query) + ' ' + str(answer.get('object_name', ''))).lower()
        is_toothpick_target = any(k in name_lc for k in ('toothpick', '牙籤', '牙签', '方盒'))
        # An upright toothpick box is ~84-90 mm tall with radius ~20.7 mm; when tipped over on its side, its horizontal body
        # spans far back along the table and produces a fake vertical ray height >= 94 mm with oblique silhouette radius >= 24.0 mm.
        geom_tipped_toothpick = bool(
            is_toothpick_target
            and (est_upright.get('height') or 0.0) >= 0.094
            and float(est_upright.get('radius') or 0.0) >= 0.0240
        )
        reason_lying = any(
            k in reason_lc
            for k in ('橫躺', '平放', '倒下', '側躺', '傾倒', '倒了', 'lying', 'tipped', 'on its side', 'horizontal')
        )
        force_posture = (tuning or {}).get('force_posture')
        is_lying_flat = bool(
            force_posture == 'lying_flat'
            or (
                force_posture != 'upright'
                and (not answer.get('upright_on_table', True) or reason_lying or geom_tipped_toothpick)
            )
        )
        if is_lying_flat:
            obj = cylinder_from_feet(geo, q, refined[0], refined[1], px['top_center'] if has_top_pt else None, lying_flat=True)
            if report is not None:
                report(f"偵測到「{answer.get('object_name') or query}」目前處於倒下／平躺狀態，自動改用低位平躺夾取模型（高度 {obj['height'] * 1000:.0f} mm）")
        else:
            obj = est_upright
            # If the top of a tall cylinder was clipped at the top border during foveation, preserve the pre-foveated height
            # and add a small margin to the top-of-frame ray height so a tall tube is never mistaken for a short 55 mm box.
            if foveated_done and not answer.get('top_visible'):
                if obj.get('height'):
                    obj['height'] = round(float(obj['height'] + 0.008), 4)
                try:
                    if rough_answer.get('top_visible') or 0.0 <= float(rough_answer['top_center']['y']) < 0.85:
                        q_r = {n: rough_frame['joints'][n] for n in BODY}
                        px_r = {k: (rough_answer[k]['x'] * IMAGE[0], rough_answer[k]['y'] * IMAGE[1]) for k in ('bottom_left', 'bottom_right', 'top_center')}
                        rough_obj = cylinder_from_feet(geo, q_r, px_r['bottom_left'], px_r['bottom_right'], px_r['top_center'])
                        if rough_obj.get('height') and 0.03 <= rough_obj['height'] <= 0.30:
                            obj['height'] = round(float(max(obj.get('height') or 0.0, rough_obj['height'])), 4)
                except Exception:
                    pass
            if require_top and not obj['height']:
                # Conservative fallback height for short upright boxes/jars whose top rim was slightly ambiguous
                obj['height'] = round(float(np.clip(obj['radius'] * 2.4, 0.055, 0.11)), 4)
        obj.update(view_frame=frame['id'], joints=q, pixels=px, refined_feet=refined,
                   crowded=answer['other_objects_within_10cm'], name=answer['object_name'],
                   foveated=foveated_done, posture='lying_flat' if is_lying_flat else 'upright',
                   upright_on_table=bool(not is_lying_flat))
        if not 0.008 <= obj['radius'] <= 0.052:
            raise RuntimeError(f"量到的直徑 {obj['radius'] * 2000:.0f} mm 不合理，不抓")
        if require_top and (not obj['height'] or not 0.03 <= obj['height'] <= 0.30):
            raise RuntimeError(f"量到的尺寸不合理（直徑 {obj['radius'] * 2000:.0f} mm、高 {0 if not obj['height'] else obj['height'] * 1000:.0f} mm），不抓")
        radial = float(np.hypot(*np.asarray(obj['centre'])[:2]))
        if not 0.13 <= radial <= 0.46:
            raise RuntimeError(f'物件離底座 {radial * 1000:.0f} mm，超出可抓範圍')
        return obj, frame, answer

    def directional_observe(self, direction_hint, report, cancel):
        """Actively turn shoulder_pan left/right to search for objects in that direction."""
        r = self.robot
        if (r.gripper_power._record() or {}).get('phase') not in (None, 'completed'):
            self.grasp.complete_pending_open_release()
        if r.pwm_owner is not None:
            raise RuntimeError('夾爪正在保持物件，請先釋放')
        deadline = time.monotonic() + 8
        while not r.camera.fresh() and time.monotonic() < deadline:
            if cancel.wait(.2):
                raise RuntimeError('已停止')
        self.sag = {n: 0 for n in BODY}
        r.bias.update({n: 0 for n in BODY})
        r.begin(allow_rearm=False)
        geo = self.geometry()
        start = self.grasp.capture()
        q = {n: start['joints'][n] for n in BODY}
        looks = look_plan(geo, direction_hint=direction_hint)
        if q['shoulder_lift'] > 1350:
            report('先沿原方向退回並抬高夾爪，避開前方物件…')
            retract_pose = {**looks['home'], 'shoulder_pan': q['shoulder_pan']}
            self.move(retract_pose, guard=self.guard(geo, table_floor=0.015), cancel=cancel, precision=45)
            start = self.grasp.capture()
            q = {n: start['joints'][n] for n in BODY}
        dir_views = looks['views'][:3] if direction_hint in ('left', 'right') else looks['views'][:4]
        dir_label = '往左轉動手臂尋找' if direction_hint == 'left' else ('往右轉動手臂尋找' if direction_hint == 'right' else '左右張望尋找')
        frames = []
        cur = q
        for i, v in enumerate(dir_views):
            report(f"{dir_label}（視角 {i + 1}/{len(dir_views)}）")
            if max(abs(cur[n] - v['pose'][n]) for n in BODY) > 15:
                self.screen_transit(geo, cur, v['pose'], table_floor=0.03)
                self.move(v['pose'], guard=self.guard(geo, table_floor=0.03), cancel=cancel, precision=45)
            frame = self.grasp.capture()
            frame['look_pan_offset'] = v['pan_offset']
            frames.append(frame)
            cur = {n: frame['joints'][n] for n in BODY}
        # Stay at the primary directional pose (view 0) so the user & subsequent grasp face that direction
        primary_pose = dir_views[0]['pose']
        if max(abs(cur[n] - primary_pose[n]) for n in BODY) > 20:
            self.screen_transit(geo, cur, primary_pose, table_floor=0.03)
            self.move(primary_pose, guard=self.guard(geo, table_floor=0.03), cancel=cancel, precision=45)
        return {
            'frames': frames,
            'measured_roll_range_deg': 0.0,
            'return_error_deg': 0.0,
            'camera_baseline_model_mm': 65.0,
            'not_external_metrology': True,
            'direction_hint': direction_hint,
        }

    def face_bbox_in_frame(self, frame, bbox, report, cancel):
        """Turn and tilt the arm so a bounding box in `frame` moves to the unobstructed image center."""
        r = self.robot
        if r.pwm_owner is not None or r.stop.is_set():
            return None
        geo = self.geometry()
        bx = float(bbox.get('x', 0.4))
        by = float(bbox.get('y', 0.4))
        bw = float(bbox.get('w', 0.2))
        bh = float(bbox.get('h', 0.2))
        fake_answer = {
            'bottom_left': {'x': max(0.02, bx), 'y': min(0.96, by + bh)},
            'bottom_right': {'x': min(0.98, bx + bw), 'y': min(0.96, by + bh)},
            'top_center': {'x': bx + 0.5 * bw, 'y': max(0.02, by)},
        }
        poses = foveate_poses(geo, frame, fake_answer)
        if not poses:
            return None
        target = poses[0]
        r.connect()
        r.begin(allow_rearm=False)
        here = r.read()
        q_here = {n: here[n] for n in BODY}
        if max(abs(q_here[n] - target[n]) for n in BODY) <= 18:
            return None
        if report is not None:
            report('轉動手臂正對選取的物件')
        self.screen_transit(geo, q_here, target, table_floor=0.03)
        self.move(target, guard=self.guard(geo, table_floor=0.025), cancel=cancel, precision=40)
        return self.grasp.capture()

    # ------------------------------------------------------------------ plan (observation motions only)
    def plan(self, query, report, cancel, *, direction_hint=None, tuning=None):
        from .d435_recorder import ActionRunLogger
        info = extract_grasp_query_info(query)
        direction_hint = direction_hint or info['direction_hint']
        target_query = info['target_query'] or query
        if tuning is None:
            tuning = self.get_tuning(target_query)
        r = self.robot
        if (r.gripper_power._record() or {}).get('phase') not in (None, 'completed'):
            report('正在自動復原上次未完成的夾爪狀態…')
            self.grasp.complete_pending_open_release()
        if r.pwm_owner is not None:
            raise RuntimeError('夾爪正在保持物件，請先釋放')
        deadline = time.monotonic() + 8
        while not r.camera.fresh() and time.monotonic() < deadline:
            if cancel.wait(.2):
                raise RuntimeError('已停止')
        plan_id = f'{int(time.time())}-{np.random.default_rng().integers(1 << 30):x}'
        logger = ActionRunLogger(self.runtime, r, run_id=plan_id, kind='plan')
        self.action_logger = logger
        try:
            self.sag = {n: 0 for n in BODY}
            r.bias.update({n: 0 for n in BODY})
            r.begin(allow_rearm=False)
            geo = self.geometry()
            start = self.grasp.capture()
            logger.record('plan_start', {'query': query, 'target_query': target_query, 'direction_hint': direction_hint, 'tuning': tuning}, wrist_frame=start)
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
            looks = look_plan(geo, direction_hint=direction_hint)
            q = {n: start['joints'][n] for n in BODY}
            if q['shoulder_lift'] > 1350 or q['elbow_flex'] < 2750 or q['wrist_flex'] > 2620:
                report('先退回觀察高度，重新確認桌面環境與物件姿態…')
                retract_pose = {**looks['home'], 'shoulder_pan': q['shoulder_pan']}
                self.move(retract_pose, guard=self.guard(geo, table_floor=0.015), cancel=cancel, precision=45)
                start = self.grasp.capture()
                q = {n: start['joints'][n] for n in BODY}
            frames = []
            obj = view_frame = answer = None

            # 1. If the arm was already aimed at an object (or no directional sweep was requested and it's in view),
            # check the current frame first before resetting to HOME and sweeping all 5 views!
            pre_aimed = (
                q['shoulder_lift'] <= 1350
                and (
                    abs(q['shoulder_pan'] - HOME['shoulder_pan']) > 60
                    or abs(q['wrist_flex'] - HOME['wrist_flex']) > 50
                )
            )
            if pre_aimed and (
                direction_hint is None
                or (direction_hint == 'left' and q['shoulder_pan'] < HOME['shoulder_pan'] - 60)
                or (direction_hint == 'right' and q['shoulder_pan'] > HOME['shoulder_pan'] + 60)
            ):
                report(f'先從目前視角尋找「{target_query}」並確認環境與物件姿態')
                start['look_pan_offset'] = int(q['shoulder_pan'] - HOME['shoulder_pan'])
                try:
                    obj, view_frame, answer = self.locate(
                        geo, target_query, [start], cancel, allow_foveated=True, report=report, tuning=tuning
                    )
                    frames = [view_frame]
                except Exception as first_exc:
                    self.log({'type': 'pre_aimed_miss', 'reason': str(first_exc)})

            # 2. If not found in the current pre-aimed view, perform the multi-angle look sweep
            if obj is None:
                if direction_hint in ('left', 'right'):
                    dir_label = '往左轉動手臂尋找' if direction_hint == 'left' else '往右轉動手臂尋找'
                    primary_views = looks['views'][:3]
                    fallback_views = looks['views'][3:]
                    cur = q
                    for i, v in enumerate(primary_views):
                        report(f"{dir_label}「{target_query}」（{i + 1} / {len(primary_views)}）")
                        if max(abs(cur[n] - v['pose'][n]) for n in BODY) > 15:
                            self.screen_transit(geo, cur, v['pose'], table_floor=0.03)
                            self.move(v['pose'], guard=self.guard(geo, table_floor=0.03), cancel=cancel, precision=45)
                        frame = self.grasp.capture()
                        frame['look_pan_offset'] = v['pan_offset']
                        frames.append(frame)
                        logger.record(f'look_view_{i + 1}', {'pan_offset': v['pan_offset']}, wrist_frame=frame)
                        cur = {n: frame['joints'][n] for n in BODY}
                    report(f'正在從{"左側" if direction_hint == "left" else "右側"}視角定位「{target_query}」')
                    try:
                        obj, view_frame, answer = self.locate(
                            geo, target_query, frames, cancel, allow_foveated=True, report=report, tuning=tuning
                        )
                    except Exception as dir_exc:
                        self.log({'type': 'directional_sweep_miss', 'direction': direction_hint, 'reason': str(dir_exc)})
                        for i, v in enumerate(fallback_views):
                            report(f"擴大範圍張望（{len(primary_views) + i + 1} / {len(looks['views'])}）")
                            if max(abs(cur[n] - v['pose'][n]) for n in BODY) > 15:
                                self.screen_transit(geo, cur, v['pose'], table_floor=0.03)
                                self.move(v['pose'], guard=self.guard(geo, table_floor=0.03), cancel=cancel, precision=45)
                            frame = self.grasp.capture()
                            frame['look_pan_offset'] = v['pan_offset']
                            frames.append(frame)
                            logger.record(f'look_fallback_{i + 1}', {'pan_offset': v['pan_offset']}, wrist_frame=frame)
                            cur = {n: frame['joints'][n] for n in BODY}
                        self.move(looks['home'], guard=self.guard(geo, table_floor=0.03), cancel=cancel, precision=45)
                        report('正在從各個角度找物件')
                        obj, view_frame, answer = self.locate(
                            geo, target_query, frames, cancel, allow_foveated=True, report=report, tuning=tuning
                        )
                else:
                    report('回到觀察原位')
                    self.screen_transit(geo, q, looks['home'])
                    self.move(looks['home'], guard=self.guard(geo, table_floor=0.03), cancel=cancel, precision=45)
                    for i, v in enumerate(looks['views']):
                        report(f"張望 {i + 1} / {len(looks['views'])}")
                        if i:
                            self.move(v['pose'], guard=self.guard(geo, table_floor=0.03), cancel=cancel, precision=45)
                        frame = self.grasp.capture()
                        frame['look_pan_offset'] = v['pan_offset']
                        frames.append(frame)
                        logger.record(f'look_view_{i + 1}', {'pan_offset': v['pan_offset']}, wrist_frame=frame)
                    # Return to low-torque centered home pose before waiting on cloud VLM inference
                    self.move(looks['home'], guard=self.guard(geo, table_floor=0.03), cancel=cancel, precision=45)
                    report('正在從各個角度找物件')
                    obj, view_frame, answer = self.locate(
                        geo, target_query, frames, cancel, allow_foveated=True, report=report, tuning=tuning
                    )

            tkey = self._target_key(target_query)
            prev_obj = self.last_measured_objects.get(tkey)
            posture_changed = bool(prev_obj and prev_obj.get('posture') != obj.get('posture'))
            height_delta_mm = round((float(obj.get('height') or 0.0) - float(prev_obj.get('height') or 0.0)) * 1000, 1) if prev_obj else 0.0
            pos_shift_mm = round(float(np.linalg.norm(np.asarray(obj['centre']) - np.asarray(prev_obj['centre']))) * 1000, 1) if prev_obj else 0.0
            env_changed = bool(prev_obj and (posture_changed or abs(height_delta_mm) >= 12.0 or pos_shift_mm >= 18.0))
            env_change = {
                'changed': env_changed,
                'posture': obj.get('posture', 'upright'),
                'previous_posture': prev_obj.get('posture') if prev_obj else None,
                'posture_changed': posture_changed,
                'height_delta_mm': height_delta_mm,
                'position_shift_mm': pos_shift_mm,
            }
            obj['environment_change'] = env_change
            self.last_measured_objects[tkey] = {
                'centre': list(obj['centre']),
                'radius': float(obj['radius']),
                'height': float(obj['height']),
                'posture': obj.get('posture', 'upright'),
                'time': time.time(),
            }
            if env_changed:
                posture_label = '橫躺／倒下' if obj.get('posture') == 'lying_flat' else '直立'
                report(
                    f"偵測到桌面環境改變：「{obj.get('name') or target_query}」目前為{posture_label}（高度 {obj['height'] * 1000:.0f} mm，高度變化 {height_delta_mm:+.0f} mm，位移 {pos_shift_mm:.0f} mm）"
                )
                logger.record('environment_change', env_change)
            else:
                posture_suffix = '（橫躺平放）' if obj.get('posture') == 'lying_flat' else ''
                report(f"在桌面上量到物件{posture_suffix}：寬 {obj['radius'] * 2000:.0f} mm、高 {obj['height'] * 1000:.0f} mm")
            here = r.read()
            q_here = {n: here[n] for n in BODY}
            q_view = {n: view_frame['joints'][n] for n in BODY}
            if max(abs(q_here[n] - q_view[n]) for n in BODY) > 35:
                try:
                    self.screen_transit(geo, q_here, q_view, obj=obj, gripper_raw=gripper_raw, object_floor=0.008)
                    self.move(q_view, guard=self.guard(geo, table_floor=0.03, obj=obj, object_floor=0.008, gripper_raw=gripper_raw),
                              cancel=cancel, precision=45)
                except Exception as view_exc:
                    self.log({'type': 'return_to_view_skipped', 'reason': str(view_exc)})
            first = obj
            # Skip redundant near_view if we already foveated/centered the object, if it is lying flat, or within 23 cm
            nv = None if (obj.get('foveated') or obj.get('lying_flat') or obj.get('camera_distance_m', 1.0) <= 0.23) else near_view(geo, obj, q_view)
            if nv is not None:
                try:
                    report('靠近一點再看一次')
                    self.screen_transit(geo, q_view, nv, obj=obj, gripper_raw=gripper_raw, table_floor=0.04, object_floor=0.010)
                    self.move(nv, guard=self.guard(geo, table_floor=0.02, obj=obj, object_floor=0.010, gripper_raw=gripper_raw),
                              cancel=cancel, precision=45)
                    near = self.grasp.capture()
                    obj2, _, _ = self.locate(geo, target_query, [near], cancel, require_top=False, allow_foveated=False, tuning=tuning)
                    shift = float(np.linalg.norm(np.asarray(obj2['centre']) - np.asarray(obj['centre'])))
                    self.log({'type': 'remeasure', 'shift_mm': shift * 1000, 'first': first, 'near': obj2}, wrist_frame=near)
                    if shift <= 0.048:
                        obj = dict(obj2, height=first['height'], first_centre=first['centre'], remeasure_shift_mm=round(shift * 1000, 1))
                except Exception as near_exc:
                    self.log({'type': 'remeasure_fallback', 'reason': str(near_exc)})
            # Silhouette feet give the base width; the body can be wider higher up (caps, bevels). Plan with margin,
            # but cap the effective planning radius for wider square/cylindrical containers so it fits the 63mm jaw opening.
            aperture = float(geo.tcp(gripper_raw)[3])
            max_fit_radius = max(0.012, 0.5 * (aperture - FIXED_SIDE_MARGIN - 0.0155))
            obj['measured_radius'] = obj['radius']
            raw_r = float(obj['radius']) if obj.get('foveated') else max(float(obj['radius']), float(first['radius']))
            obj['radius'] = min(raw_r + PLAN_RADIUS_MARGIN, max(raw_r, max_fit_radius)) if raw_r <= max_fit_radius + 0.002 else raw_r + PLAN_RADIUS_MARGIN
            here = r.read()
            q_here = {n: here[n] for n in BODY}
            report('本機規劃接近、夾取與抬起')
            plan = plan_grasp(geo, obj, gripper_raw, q_here, tuning=tuning)
            if plan['problems']:
                # Wider cylindrical cans or square/rectangular boxes can slightly exceed the strict jaw margin after
                # oblique silhouette inflation + PLAN_RADIUS_MARGIN. Retry with a compliant effective radius that
                # fits the calibrated jaw opening while still enforcing table & wrist camera clearance.
                relaxed_r = min(float(obj['measured_radius']), float(first['radius']), max_fit_radius)
                obj_relaxed = dict(obj, radius=relaxed_r, relaxed_from_radius_m=round(float(obj['radius']), 4))
                relaxed_plan = plan_grasp(
                    geo,
                    obj_relaxed,
                    gripper_raw,
                    q_here,
                    pitches=(25, 15, 20, 10, 5, 30, 35, 0, 40, 45),
                    heights=(0.024, 0.028, 0.032, 0.038, 0.045, 0.055, 0.065, 0.080, 0.095, 0.110),
                    tuning=tuning,
                )
                if not relaxed_plan['problems']:
                    obj = obj_relaxed
                    plan = relaxed_plan
                elif not cancel.is_set() and not r.stop.is_set():
                    try:
                        self.move(looks['home'], guard=self.guard(geo, table_floor=0.03), cancel=cancel, precision=45)
                    except Exception:
                        pass
            self.pending = {'id': plan_id, 'created': time.monotonic(), 'start': q_here, 'gripper_raw': gripper_raw,
                            'plan': plan, 'query': query, 'target_query': target_query, 'direction_hint': direction_hint,
                            'tuning': tuning}
            summary = {'plan_id': plan_id, 'object': obj, 'first_measurement': first, 'grasp': plan['grasp'],
                       'waypoints': plan['waypoints'], 'problems': plan['problems'], 'gripper_raw': gripper_raw,
                       'look_views': [{'frame': f['id'], 'pan_offset': f.get('look_pan_offset', 0)} for f in frames],
                       'chosen_view': answer['view_index'], 'executable': not plan['problems'],
                       'kinematic_model': geo.model_path, 'direction_hint': direction_hint, 'target_query': target_query,
                       'tuning': tuning, 'environment_change': env_change,
                       'action_log_dir': str(logger.run_dir)}
            self.log({'type': 'plan', **summary}, wrist_frame=view_frame)
            return summary
        finally:
            logger.close()
            self.action_logger = None

    # ------------------------------------------------------------------ execute an approved plan
    def execute(self, plan_id, report, cancel, *, return_object=True):
        from .d435_recorder import ActionRunLogger
        r = self.robot
        p = self.pending
        if p is None or p['id'] != plan_id or time.monotonic() - p['created'] > self.PLAN_TTL_S:
            raise RuntimeError('沒有這份有效的抓取計畫，請重新規劃')
        if p['plan']['problems']:
            raise RuntimeError('這份計畫有未解決的問題，不能執行')
        self.sag = {n: 0 for n in BODY}
        r.bias.update({n: 0 for n in BODY})
        r.begin(allow_rearm=False)
        geo = self.geometry()
        now = r.read()
        if max(abs(now[n] - p['start'][n]) for n in BODY) > 8 or abs(now['gripper'] - p['gripper_raw']) > 6:
            raise RuntimeError('手臂或夾爪在規劃後移動過，請重新規劃')
        self.pending = None
        obj, gr = p['plan']['object'], p['gripper_raw']
        tuning = p.get('tuning') or {}
        squeeze_ticks = int(np.clip(SQUEEZE_TICKS + int(tuning.get('extra_squeeze_ticks', 0)), 28, 90))
        W = {w['label']: w['target'] for w in p['plan']['waypoints']}
        logger = ActionRunLogger(self.runtime, r, run_id=plan_id, kind='execute')
        self.action_logger = logger
        result = {'plan_id': plan_id, 'steps': [], 'grasp_confirmed': False, 'completed': False,
                  'query': p.get('query'), 'target_query': p.get('target_query'),
                  'object': obj, 'grasp': p['plan']['grasp'], 'tuning': tuning,
                  'action_log_dir': str(logger.run_dir)}

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

        try:
            free = self.guard(geo, table_floor=0.018 if not obj.get('lying_flat') else 0.014, obj=obj, object_floor=0.005, gripper_raw=gr)
            close = self.guard(geo, table_floor=TABLE_FLOOR if not obj.get('lying_flat') else 0.014, obj=obj, object_floor=-0.004, gripper_raw=gr)
            report('移到物件後上方')
            step('above_pregrasp', W['above_pregrasp'], guard=free, precision=45)
            report('對準物件')
            step('approach_70mm', W['approach_70mm'], guard=free, precision=45, correct=True)
            for label in ('approach_50mm', 'approach_30mm', 'approach_15mm'):
                report('沿直線接近物件')
                step(label, W[label], guard=close, speed=40, precision=45, track_sag=True)
            report('到達夾取位置，修正舵機靜差')
            q = step('grasp', W['grasp'], guard=close, speed=30, precision=45, correct=True, tolerance=5)
            Ttcp, fixed, moving, _ = geo.tcp(gr)
            G = geo.transforms(q)['gripper_link']
            local = np.array(p['plan']['grasp']['local_point_mm']) / 1000
            want = np.asarray(obj['centre']) + geo.n * p['plan']['grasp']['height_m']
            miss = float(np.linalg.norm(G[:3, :3] @ local + G[:3, 3] - want) * 1000)
            result['grasp_point_error_mm'] = round(miss, 2)
            if miss > 6.5:
                raise RuntimeError(f'夾取點偏差 {miss:.1f} mm，超過 6.5 mm，不合爪')
            reference = self.grasp.capture()
            logger.record('before_close', {'grasp_point_error_mm': result['grasp_point_error_mm']}, wrist_frame=reference)
            report('低出力合爪')
            try:
                closed = self.grasp.close_candidate(reference, report, cancel, squeeze_ticks=squeeze_ticks)
            except Exception as close_exc:
                if getattr(close_exc, 'retryable', False) and not cancel.is_set():
                    report('合爪時偵測到瞬間通訊干擾，正在自動恢復並重試…')
                    r.resume()
                    r.begin(allow_rearm=True)
                    closed = self.grasp.close_candidate(reference, report, cancel, squeeze_ticks=squeeze_ticks)
                elif '夾爪已接近空夾位置' in str(close_exc):
                    after_miss = self.grasp.capture()
                    logger.record('missed_empty_close', {'error': str(close_exc)}, wrist_frame=after_miss)
                    report('偵測到抓空（夾爪已接近空夾位置），正在自動張開夾爪並安全退回原位…')
                    self.grasp.complete_pending_open_release()
                    r.resume()
                    r.begin(allow_rearm=True)
                    back = self.guard(geo, table_floor=0.014)
                    self.sag = {n: 0 for n in BODY}
                    r.bias.update({n: 0 for n in BODY})
                    for label in ('approach_30mm', 'approach_50mm', 'approach_70mm', 'above_pregrasp'):
                        if label in W:
                            step('retreat_' + label, W[label], guard=back, speed=45, precision=45)
                    step('home', dict(HOME), guard=back, speed=60, precision=45)
                    result.update(
                        grasp_confirmed=False,
                        missed_empty_close=True,
                        completed=False,
                        error='夾爪已接近空夾位置（抓空），已自動張開夾爪並退回原位',
                    )
                    self.log({'type': 'result', **result})
                    return result
                else:
                    raise
            result['close'] = {'contact_position': closed['contact_sample']['position'],
                               'position': closed['sample']['position'], 'goal': closed['sample']['goal'],
                               'load': closed['sample']['load_signed_pwm'], 'squeeze_ticks': squeeze_ticks}
            logger.record('after_close', result['close'], gripper_telemetry=closed['sample'])
            sag_before = dict(self.sag)
            report('抬起 30 mm')
            lift_guard = self.guard(geo, table_floor=TABLE_FLOOR)
            # The held object's weight adds sag (shoulder 30-50 ticks measured); correct from measured joints.
            q_lift = step('lift', W['lift'], guard=lift_guard, speed=30, precision=60, correct=True, tolerance=6)
            after = self.grasp.capture()
            tel = r.gripper_telemetry()
            G0, G1 = geo.transforms(q)['gripper_link'], geo.transforms(q_lift)['gripper_link']
            lifted_mm = float(geo.n @ (G1[:3, 3] - G0[:3, 3]) * 1000)
            load_signed = int(tel['gripper_load_signed_pwm'] if 'gripper_load_signed_pwm' in tel else tel['load_signed_pwm'])
            load_pwm = abs(load_signed)
            jaw_error = int(tel.get('position', 0)) - int(tel.get('goal', closed['sample']['goal']))
            evidence = {'measured_lift_mm': round(lifted_mm, 1), 'gripper_position': tel['position'],
                        'gripper_goal': tel.get('goal', closed['sample']['goal']),
                        'gripper_jaw_error_ticks': jaw_error,
                        'gripper_load_signed_pwm': load_signed,
                        'empty_jaw_position_threshold': 1525}
            result['lift'] = evidence
            report('正在核對是否真的拿起來')
            evidence['shoulder_lift_sag_ticks'] = {'before_lift': sag_before['shoulder_lift'], 'lifted': self.sag['shoulder_lift']}
            verdict = self.brain.verify_grasp(p['query'], [reference, after], evidence, cancel)
            result['verification'] = verdict
            self.log({'type': 'verify', 'evidence': evidence, 'verdict': verdict}, wrist_frame=after, gripper_telemetry=tel)
            # Motor + vision evidence: require actual clamping load (load_pwm >= 45) and jaw stall error (jaw_error >= 8)
            # so a slipped container that squared up and left the jaw at 0 error / 24 PWM is never falsely confirmed.
            # Compare against min(closed['sample']['position'], contact - SQUEEZE_TICKS) - 25 so square/rectangular
            # containers that touched corner-first and rotated flat during closing are not rejected by 2 ticks.
            min_held_pos = min(
                int(closed['sample']['position']),
                int(closed['contact_sample']['position']) - SQUEEZE_TICKS,
            ) - 25
            motor = (lifted_mm >= 15 and tel['position'] > 1540
                     and tel['position'] >= min_held_pos
                     and jaw_error >= 8 and load_pwm >= 45)
            weight = self.sag['shoulder_lift'] - sag_before['shoulder_lift']
            result['motor_confirmed'] = bool(motor)
            result['shoulder_sag_increase_ticks'] = weight
            vision_conf = float(verdict.get('confidence', 0))
            visibly_empty = bool(
                verdict.get('gripper_empty', False)
                and not verdict.get('between_fingers', False)
                and (load_pwm < 85 or jaw_error < 14)
            )
            vision_held = (vision_conf >= 0.72 and verdict.get('same_object', False)
                           and not visibly_empty
                           and (verdict.get('follows_gripper', False) or verdict.get('clear_of_table', False)))
            vision_slipped = (vision_conf >= 0.72
                              and not verdict.get('follows_gripper', False)
                              and not verdict.get('clear_of_table', False)
                              and (load_pwm < 85 or jaw_error < 14))
            confirmed = bool(
                motor
                and not visibly_empty
                and (vision_held or (not vision_slipped and load_pwm >= 55) or (load_pwm >= 85 and jaw_error >= 14))
            )
            result['grasp_confirmed'] = bool(confirmed)
            back = self.guard(geo, table_floor=TABLE_FLOOR)
            if not confirmed:
                report('偵測到未夾起物件（物件仍留在桌面），正在鬆開夾爪並安全退回原位…')

                def retreat_unconfirmed():
                    self.sag = {n: 0 for n in BODY}
                    r.bias.update({n: 0 for n in BODY})
                    for label in ('approach_30mm', 'approach_50mm', 'approach_70mm', 'above_pregrasp'):
                        step('retreat_' + label, W[label], guard=back, speed=45, precision=45)
                    check_pose = dict(HOME, wrist_flex=1880)
                    step('empty_check_view', check_pose, guard=back, speed=60, precision=45)

                result['release'] = self.grasp.release(after, report, cancel, support_verified=True, retreat=retreat_unconfirmed)
                report('歸回正前方原位')
                step('home', dict(HOME), guard=back, speed=60, precision=45)
            elif return_object:
                report('已拿起物件，準備放回原位')
                # Back to 2 mm above the grasp pose: the object bottom just clears the table, never pressed into it.
                q_down, _, _ = pose_along(geo, W['grasp'], geo.n * 0.002, W['grasp'])
                step('lowered', {n: int(round(q_down[n])) for n in BODY}, guard=lift_guard, speed=25, precision=60,
                     correct=True, tolerance=4)
                down = self.grasp.capture()
                logger.record('lowered', {}, wrist_frame=down)
                if cancel.wait(.6):
                    raise RuntimeError('已停止')
                report('鬆開夾爪')

                def retreat():
                    self.sag = {n: 0 for n in BODY}
                    r.bias.update({n: 0 for n in BODY})
                    for label in ('approach_15mm', 'approach_30mm', 'approach_50mm', 'approach_70mm', 'above_pregrasp'):
                        step('retreat_' + label, W[label], guard=back, speed=40, precision=45)
                    # Pitch wrist up at HOME so open fingers are silhouetted against upper background, not tabletop objects.
                    check_pose = dict(HOME, wrist_flex=1880)
                    step('empty_check_view', check_pose, guard=back, speed=60, precision=45)
                result['release'] = self.grasp.release(down, report, cancel, support_verified=True, retreat=retreat)
                report('歸回正前方原位')
                step('home', dict(HOME), guard=back, speed=60, precision=45)
            else:
                report('已拿起物件')
            result['completed'] = True
            self.log({'type': 'result', **result})
            return result
        except Exception as exc:
            self.log({'type': 'error', 'plan_id': plan_id, 'error': str(exc)})
            try:
                if (r.gripper_power._record() or {}).get('phase') not in (None, 'completed'):
                    self.grasp.complete_pending_open_release()
            except Exception:
                pass
            raise
        finally:
            self.sag = {n: 0 for n in BODY}
            r.bias.update({n: 0 for n in BODY})
            logger.close()
            self.action_logger = None
