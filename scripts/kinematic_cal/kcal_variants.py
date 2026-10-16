"""Offline comparison of kinematic error models on the captured placements. No hardware access."""
from __future__ import annotations

import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares

import kcal_model as km
from kcal_common import ASTRA, BODY, OBJ, ROOT, BatchFK, Kinematics, inv, project

URDF = ASTRA / 'assets/so101_follower_cam.urdf'
kin = Kinematics(URDF)
TREE = ET.parse(URDF).getroot()
MASS = {}
for link in TREE.findall('link'):
    i = link.find('inertial')
    if i is not None and float(i.find('mass').get('value')) > 1e-6:
        o = i.find('origin')
        MASS[link.get('name')] = (float(i.find('mass').get('value')),
                                  np.fromstring(o.get('xyz'), sep=' ') if o is not None else np.zeros(3))
MASS.pop('moving_jaw_so101_v1_link', None)            # jaw branch not in FK; lump it into the gripper
MASS['gripper_link'] = (MASS['gripper_link'][0] + 0.012, MASS['gripper_link'][1])
DOWNSTREAM = {'shoulder_lift': ['upper_arm_link', 'lower_arm_link', 'wrist_link', 'gripper_link', 'wrist_camera_link'],
              'elbow_flex': ['lower_arm_link', 'wrist_link', 'gripper_link', 'wrist_camera_link'],
              'wrist_flex': ['wrist_link', 'gripper_link', 'wrist_camera_link']}
CHILD = {'shoulder_lift': 'upper_arm_link', 'elbow_flex': 'lower_arm_link', 'wrist_flex': 'wrist_link'}


def gravity_torques(joints):
    """Gravity torque (N m) about each pitch joint axis at the measured pose."""
    T = kin.transforms({**joints, 'gripper': 2047})
    g = np.array([0, 0, -9.81])
    out = {}
    for j, links in DOWNSTREAM.items():
        p, a = T[CHILD[j]][:3, 3], T[CHILD[j]][:3, 2]
        tau = 0.0
        for l in links:
            m, com = MASS[l]
            c = T[l][:3, :3] @ com + T[l][:3, 3]
            tau += a @ np.cross(c - p, m * g)
        out[j] = tau
    return out


def main():
    sessions = [Path(s).resolve() if Path(s).exists() else km.COMMITTED_SESSIONS / s for s in sys.argv[1:]] or \
        sorted(p for p in km.COMMITTED_SESSIONS.glob('session-2026*') if not p.name.endswith('-test'))
    views = [v for v in km.load_views(sessions) if not (v['name'] == 'view_02' and v['placement'].endswith('173458'))]
    train = [v for v in views if v['split'] == 'train']
    val = [v for v in views if v['split'] != 'train']
    K, D, _ = km.intrinsics(train, km.load_v2_camera()[0])
    base = json.loads(km.MODEL_PATH.read_text())
    X0 = np.array(base['handeye_gripper_link_T_optical'])
    for v in views:
        v['tau'] = gravity_torques(v['joints'])
    placements = sorted({v['placement'] for v in views})
    b0 = km.initial_boards(train, K, D, X0, base['offsets_deg'])
    bfk = BatchFK()
    L = {'elbow_flex': None, 'wrist_flex': None}   # link-length scaling acts on these joint origins

    def chain(vs, offs, comp, X, boards, scale=(1.0, 1.0)):
        raw = {n: np.array([v['joints'][n] for v in vs], float) for n in BODY}
        tick = {n: np.zeros(len(vs)) for n in BODY}
        for n in km.OFFSET_JOINTS:
            tick[n] = offs.get(n, 0.0) * km.TPD + np.degrees(np.array([comp.get(n, 0.0) * v['tau'][n] for v in vs])) * km.TPD
        orig = [(i, j[4].copy()) for i, j in enumerate(bfk.chain) if j[0] in ('elbow_flex', 'wrist_flex')]
        try:
            for (i, To), s in zip(orig, scale):
                name, kind, parent, child, _, axis = bfk.chain[i]
                Ts = To.copy()
                Ts[:3, 3] *= s
                bfk.chain[i] = (name, kind, parent, child, Ts, axis)
            C = bfk({n: raw[n] + tick[n] for n in BODY}) @ X
        finally:
            for i, To in orig:
                name, kind, parent, child, _, axis = bfk.chain[i]
                bfk.chain[i] = (name, kind, parent, child, To, axis)
        out = []
        for v, Cv in zip(vs, C):
            Pc = (np.c_[OBJ, np.ones(54)] @ boards[v['placement']].T @ inv(Cv).T)[:, :3]
            out.append(project(K, D, Pc)[0])
        return np.array(out)

    def run(name, level=True, use_comp=False, use_scale=False, tilted=False):

        def unpack(p):
            k = 0
            offs = dict(zip(km.OFFSET_JOINTS, p[k:k + 3])); k += 3
            comp = dict(zip(km.OFFSET_JOINTS, p[k:k + 3])) if use_comp else {}; k += 3 if use_comp else 0
            scale = tuple(1 + p[k:k + 2] / 100) if use_scale else (1.0, 1.0); k += 2 if use_scale else 0
            X = km.make_T(p[k:k + 6]); k += 6
            boards = {}
            if level:
                z = p[k]; k += 1
                R = np.eye(4)
                if tilted:
                    R[:3, :3] = km.Rotation.from_rotvec([p[k], p[k + 1], 0]).as_matrix(); k += 2
                for pl in placements:
                    boards[pl] = R @ km.board_on_table(p[k], p[k + 1], p[k + 2], z); k += 3
            else:
                for pl in placements:
                    boards[pl] = km.make_T(p[k:k + 6]); k += 6
            return offs, comp, scale, X, boards

        p0 = [*[base['offsets_deg'].get(n, 0) for n in km.OFFSET_JOINTS]]
        p0 += [0, 0, 0] if use_comp else []
        p0 += [0, 0] if use_scale else []
        p0 += list(km.split_T(X0))
        if level:
            p0 += [np.mean([b[3] for b in b0.values()])] + ([0.0, 0.0] if tilted else []) + [x for pl in placements for x in b0[pl][:3]]
        else:
            p0 += [x for pl in placements for x in km.split_T(km.board_on_table(*b0[pl]))]
        tc = np.array([v['corners'] for v in train])
        sol = least_squares(lambda p: (chain(train, *unpack(p)[:2], unpack(p)[3], unpack(p)[4], unpack(p)[2]) - tc).ravel(),
                            np.array(p0, float), loss='soft_l1', f_scale=1.0, x_scale='jac', max_nfev=1500)
        offs, comp, scale, X, boards = unpack(sol.x)
        def err(vs):
            return np.linalg.norm(chain(vs, offs, comp, X, boards, scale) - np.array([v['corners'] for v in vs]), axis=-1)
        et, ev = err(train), err(val)
        tilts = {pl[-6:]: round(float(np.degrees(np.arccos(abs(B[2, 2])))), 2) for pl, B in boards.items()}
        zs = {pl[-6:]: round(1000 * float(B[2, 3]), 1) for pl, B in boards.items()}
        print(f"{name:34} train {np.sqrt(np.mean(et**2)):5.2f}px | VALIDATION rms {np.sqrt(np.mean(ev**2)):5.2f}px p95 {np.percentile(ev,95):5.1f} max {ev.max():5.1f}"
              f" | offs {[round(offs[n],2) for n in km.OFFSET_JOINTS]} comp(deg/Nm) {[round(float(np.degrees(comp[n])),2) for n in comp] if comp else '-'}"
              f" scale% {[round((s-1)*100,2) for s in scale] if use_scale else '-'} | tilt {tilts} z {zs}")
        return sol

    print(f'{len(train)} train / {len(val)} validation views over {len(placements)} placements;',
          'gravity torque range (Nm):', {n: (round(min(v["tau"][n] for v in views), 2), round(max(v["tau"][n] for v in views), 2)) for n in km.OFFSET_JOINTS})
    run('V1 level table')
    run('V2 free 6-DOF boards', level=False)
    run('V3 level + gravity compliance', use_comp=True)
    run('V4 level + compliance + link scale', use_comp=True, use_scale=True)
    run('V5 free boards + compliance', level=False, use_comp=True)
    run('V6 one shared TILTED table plane', tilted=True)


if __name__ == '__main__':
    main()
