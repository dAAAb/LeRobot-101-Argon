"""Shared helpers for SO-101 kinematic calibration (wrist camera + fixed checkerboard).

Hardware rules for every tool in this folder:
  * never write torque enable, EEPROM, gains, speed/acceleration registers;
  * never command the gripper;
  * only kcal_run.py writes Goal_Position (42) of the five body joints, in small
    interpolated steps, after the plan has been shown to and approved by the user.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

# setUseOpenCL(False) is per-thread; findChessboardCornersSB's worker threads still built
# OpenCL kernels and segfaulted in Apple's OpenCL driver on Neo. Disable it process-wide.
os.environ.setdefault('OPENCV_OPENCL_RUNTIME', 'disabled')
import cv2  # noqa: E402
import numpy as np

cv2.ocl.setUseOpenCL(False)

ASTRA = Path(__file__).resolve().parents[2]          # repository root (scripts/kinematic_cal/..)
ROOT = ASTRA / '.runtime' / 'kinematic-cal'           # new captures, plans and images (git-ignored)
ROOT.mkdir(parents=True, exist_ok=True)
V2_REPORT = ASTRA / 'config/calibration/98986db474f5473593fc6e535d9261ea-9b3d2b4643c2413ca10bcbdf604f78a2-report.json'
sys.path.insert(0, str(ASTRA))
from server.geometry import Kinematics  # noqa: E402
from server.local_cartesian import default_model  # noqa: E402

PORT = '/dev/tty.usbmodem5B415309611'
CAMERA = 'icspring camera'
BODY = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
IDS = {'shoulder_pan': 1, 'shoulder_lift': 2, 'elbow_flex': 3, 'wrist_flex': 4, 'wrist_roll': 5, 'gripper': 6}
TICK = 2 * math.pi / 4096
SQUARE = 0.020
PATTERN = (9, 6)                       # inner corners (cols, rows) of the printed 10x7 board
OBJ = np.zeros((PATTERN[0] * PATTERN[1], 3))
OBJ[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE
# Paper region around the inner-corner grid that must stay unoccluded (squares + quiet zone).
BOARD_OUTLINE = np.array([[-0.032, -0.032, 0], [0.192, -0.032, 0], [0.192, 0.132, 0], [-0.032, 0.132, 0]])
IMAGE_SIZE = (1920, 1080)


# --------------------------------------------------------------------------- camera model
def load_v2_camera():
    """Astra spatial-v2 camera profile (used only as the initial guess / baseline)."""
    prof = json.loads(V2_REPORT.read_text())['calibration_profile']
    return (np.array(prof['camera_matrix'], float), np.array(prof['distortion'], float),
            np.array(prof['handeye'], float))


def project(K, D, pts_cam):
    """OpenCV Brown model (k1,k2,p1,p2,k3), vectorised over leading dims."""
    z = pts_cam[..., 2]
    x, y = pts_cam[..., 0] / z, pts_cam[..., 1] / z
    k1, k2, p1, p2, k3 = (list(D) + [0] * 5)[:5]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([K[0, 0] * xd + K[0, 1] * yd + K[0, 2], K[1, 1] * yd + K[1, 2]], -1), z


def inv(T):
    out = np.zeros_like(T)
    R = np.swapaxes(T[..., :3, :3], -1, -2)
    out[..., :3, :3] = R
    out[..., :3, 3] = -np.einsum('...ij,...j->...i', R, T[..., :3, 3])
    out[..., 3, 3] = 1
    return out


# --------------------------------------------------------------------------- kinematics
class BatchFK:
    """Vectorised base_link -> gripper_link FK using Astra's URDF convention
    angle = (raw + offset - 2047) * 2pi/4096. offsets are in ticks."""

    def __init__(self, kin=None):
        kin = kin or Kinematics(ASTRA / 'assets/so101_follower_cam.urdf')
        bychild = {j[3]: j for j in kin.joints}
        chain, link = [], 'gripper_link'
        while link in bychild:
            chain.append(bychild[link])
            link = bychild[link][2]
        self.chain = list(reversed(chain))

    def __call__(self, raw, offsets=None):
        offsets = offsets or {}
        n = len(np.atleast_1d(raw['shoulder_pan']))
        T = np.tile(np.eye(4), (n, 1, 1))
        for name, kind, parent, child, To, axis in self.chain:
            T = T @ To
            if kind == 'fixed':
                continue
            ang = (np.atleast_1d(raw[name]).astype(float) + offsets.get(name, 0.0) - 2047) * TICK
            a = axis / np.linalg.norm(axis)
            Kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
            s, c = np.sin(ang)[:, None, None], np.cos(ang)[:, None, None]
            M = np.tile(np.eye(4), (n, 1, 1))
            M[:, :3, :3] = np.eye(3) + s * Kx + (1 - c) * (Kx @ Kx)
            T = T @ M
        return T


# --------------------------------------------------------------------------- board detection
def detect_board(gray):
    """Return 54x2 corners in canonical printed order, or None."""
    small = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2), interpolation=cv2.INTER_AREA)
    ok, c = cv2.findChessboardCornersSB(small, PATTERN, flags=cv2.CALIB_CB_EXHAUSTIVE)
    if not ok:
        ok, c = cv2.findChessboardCorners(small, PATTERN, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ok:
            return None
    c = (c.reshape(-1, 1, 2) * 2.0 + 0.5).astype(np.float32)
    cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.005))
    return c.reshape(-1, 2).astype(np.float64)


def _patch_mean(gray, p, r):
    x, y = int(round(p[0])), int(round(p[1]))
    if not (r <= x < gray.shape[1] - r and r <= y < gray.shape[0] - r):
        return None
    return float(gray[y - r:y + r + 1, x - r:x + r + 1].mean())


def canonical_corners(gray, corners, K, D):
    """Resolve the checkerboard ordering ambiguity.

    Printed board: top-left square black. Canonical corner 0 is the inner corner
    whose outward diagonal square is black, and the board z axis points away from
    the camera (into the table)."""
    g = corners.reshape(PATTERN[1], PATTERN[0], 2)
    options = {'id': g, 'flipx': g[:, ::-1], 'flipy': g[::-1, :], 'rot180': g[::-1, ::-1]}
    chosen = []
    for name, cand in options.items():
        c00, cx, cy = cand[0, 0], cand[0, 1], cand[1, 0]
        step = 0.5 * np.linalg.norm(cx - c00)
        r = max(2, int(step * 0.35))
        outer = _patch_mean(gray, c00 - 0.5 * (cx - c00) - 0.5 * (cy - c00), r)
        side = _patch_mean(gray, c00 + 0.5 * (cx - c00) - 0.5 * (cy - c00), r)
        if outer is None or side is None or not outer < side - 15:
            continue
        pts = cand.reshape(-1, 2)
        ok, rvec, tvec = cv2.solvePnP(OBJ, pts, K, D, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            continue
        rvec, tvec = cv2.solvePnPRefineLM(OBJ, pts, K, D, rvec, tvec)
        R, _ = cv2.Rodrigues(rvec)
        if R[:, 2] @ tvec.ravel() > 0:
            chosen.append((name, pts, rvec, tvec))
    if len(chosen) != 1:
        return None
    return chosen[0]


def pattern_ambiguous(gray):
    """A genuine 10x7-square board cannot also show 10x6 or 9x7 inner corners; if it does,
    spurious junctions (text, paper edge) can shift the 9x6 window by one square."""
    small = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2), interpolation=cv2.INTER_AREA)
    return any(cv2.findChessboardCornersSB(small, pat, flags=cv2.CALIB_CB_EXHAUSTIVE)[0] for pat in ((10, 6), (9, 7)))


def board_pose_cam(gray, K, D):
    corners = detect_board(gray)
    if corners is None:
        return None
    res = canonical_corners(gray, corners, K, D)
    if res is None:
        return None
    name, pts, rvec, tvec = res
    proj, _ = cv2.projectPoints(OBJ, rvec, tvec, K, D)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - pts) ** 2, 1))))
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return {'corners': pts, 'T_cam_board': T, 'pnp_rms_px': rms, 'ordering': name, 'ambiguous': pattern_ambiguous(gray)}


# --------------------------------------------------------------------------- table clearance
class Clearance:
    """Minimum height of every robot mesh above a (possibly tilted) plane.

    Uses Astra's convex hulls (incl. camera boxes) and its conservative full-360
    moving-jaw envelope. base_link is exempt (it rests on the table)."""

    def __init__(self):
        self.model = default_model(ASTRA)

    def min_height(self, raw, plane_up, plane_d):
        full = dict(raw)
        full.setdefault('gripper', 2047)
        T = self.model.fk.transforms(full)
        n = np.asarray(plane_up, float)
        out = {}
        for link, v in self.model.vertices.items():
            if link == 'base_link':
                continue
            pts = v @ T[link][:3, :3].T + T[link][:3, 3]
            out[link] = float(np.min(pts @ n) - plane_d)
        H = T['gripper_link'] @ self.model.jaw_origin
        m = H[:3, :3].T @ n
        v = self.model.jaw
        h = m[2] * v[:, 2] - np.hypot(m[0], m[1]) * np.hypot(v[:, 0], v[:, 1]) + n @ H[:3, 3] - plane_d
        out['moving_jaw_full360'] = float(h.min())
        return out


# --------------------------------------------------------------------------- hardware (read paths)
class Arm:
    def __init__(self, port=PORT):
        from scservo_sdk import PortHandler, PacketHandler, GroupSyncRead
        self.port = PortHandler(port)
        if not self.port.openPort():
            raise RuntimeError('cannot open follower port')
        self.port.setBaudRate(1_000_000)
        self.ph = PacketHandler(0)
        self._sync = {}
        for addr in (56, 42, 60):
            g = GroupSyncRead(self.port, self.ph, addr, 2)
            for i in IDS.values():
                g.addParam(i)
            self._sync[addr] = g

    def close(self):
        self.port.closePort()

    def sync_read(self, addr, names=tuple(IDS)):
        from scservo_sdk import COMM_SUCCESS
        g = self._sync[addr]
        for _ in range(6):
            if g.txRxPacket() == COMM_SUCCESS and all(g.isAvailable(IDS[n], addr, 2) for n in names):
                return {n: g.getData(IDS[n], addr, 2) for n in names}
            time.sleep(0.004)
        raise RuntimeError(f'sync read {addr} failed')

    def positions(self):
        return self.sync_read(56)

    def goals(self):
        return self.sync_read(42)

    def loads(self):
        raw = self.sync_read(60)
        return {n: (-(v & 0x3FF) if v & 0x400 else v & 0x3FF) for n, v in raw.items()}

    def read1(self, name, addr):
        from scservo_sdk import COMM_SUCCESS
        for _ in range(4):
            v, res, err = self.ph.read1ByteTxRx(self.port, IDS[name], addr)
            if res == COMM_SUCCESS:
                return v
            time.sleep(0.004)
        raise RuntimeError(f'read {name}@{addr} failed')

    def health(self):
        return {n: {'status': self.read1(n, 65), 'torque': self.read1(n, 40),
                    'temp': self.read1(n, 63), 'volt': self.read1(n, 62) / 10} for n in IDS}


# --------------------------------------------------------------------------- camera
def snapshot(path, warmup_s=1.2):
    """One fresh frame from the wrist camera (by device name, not index)."""
    path = Path(path)
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'avfoundation', '-pixel_format', 'nv12', '-framerate', '30',
                    '-video_size', '1920x1080', '-i', f'{CAMERA}:none', '-ss', str(warmup_s), '-frames:v', '1',
                    '-q:v', '2', '-y', str(path)], check=True, timeout=30)
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError('camera frame unreadable')
    return img
