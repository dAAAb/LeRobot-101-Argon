"""Software joint-zero offsets from a fitted kinematic model (scripts/kinematic_cal).

Kept out of geometry.py on purpose: Astra's geometry revision hashes geometry.py, so editing it would
invalidate every existing spatial/jaw/workspace calibration artifact.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .geometry import Kinematics

BODY_OFFSET_JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')


def model_offsets(path):
    model = json.loads(Path(path).read_text())
    offsets = model.get('offsets_ticks')
    if not isinstance(offsets, dict) or set(offsets) - set(BODY_OFFSET_JOINTS):
        raise ValueError('Invalid kinematic model offsets')
    out = {}
    for n, v in offsets.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or abs(v) > 200:
            raise ValueError('Invalid kinematic model offsets')
        if v:
            out[n] = float(v)
    return out


def runtime_offsets(root):
    """Offsets for every Astra FK: only with config `kinematic_model_global: true` (default: none).

    The kinematic grasp program always uses the model; the v2 spatial/jaw/workspace artifacts are bound to
    the offset-free FK revision."""
    config = json.loads((Path(root) / 'config/robot.json').read_text())
    selected = config.get('kinematic_model')
    if not selected or config.get('kinematic_model_global') is not True:
        return {}
    return model_offsets(Path(root) / selected)


class OffsetKinematics(Kinematics):
    """angle = (raw + offset - 2047) * 2pi/4096 for the body joints; raw limits and EEPROM untouched."""

    def __init__(self, urdf, offsets=None):
        super().__init__(urdf)
        offsets = dict(offsets or {})
        if set(offsets) - set(BODY_OFFSET_JOINTS) or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in offsets.values()):
            raise ValueError('Joint offsets apply to finite body joints only')
        self.offsets = {n: float(v) for n, v in offsets.items() if v}

    def transforms(self, raw):
        if not self.offsets:
            return super().transforms(raw)
        shifted = dict(raw)
        for n, v in self.offsets.items():
            if n in shifted and not isinstance(shifted[n], bool) and isinstance(shifted[n], (int, float)):
                shifted[n] = shifted[n] + v
        return super().transforms(shifted)
