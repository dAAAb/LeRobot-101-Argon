"""Select a reviewed, content-addressed jaw artifact without changing hardware."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .spatial_contract import JawCalibration, WristCalibration
from .spatial_runtime import _read_profile, geometry_revision


def reviewed_jaw_profile(root, profile_sha256, config, model):
    if not isinstance(profile_sha256, str) or not re.fullmatch('[0-9a-f]{64}', profile_sha256):
        raise ValueError('無效的夾爪版本')
    root = Path(root)
    directory = root / 'config/calibration'
    path = directory / ('jaw-' + profile_sha256 + '-profile.json')
    if directory.is_symlink() or path.is_symlink() or path.stat().st_size > 1024 * 1024:
        raise ValueError('夾爪設定必須是本機保存的完整版本')
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != profile_sha256:
        raise ValueError('夾爪設定內容與核對過的版本不同')
    profile = json.loads(payload)
    if profile.get('source') != 'wrist-only':
        raise ValueError('夾爪設定只能使用腕部相機資料')
    revision = geometry_revision(model)
    jaw = JawCalibration.from_mapping(profile, geometry_revision=revision)
    if not jaw.at(jaw.samples[0].raw).planning_eligible:
        raise ValueError('這份夾爪設定還不能用來規劃')
    if jaw.model_assisted_json is not None:
        optics = _read_profile(config, 'spatial_calibration_profile')
        if optics is None:
            raise ValueError('需要先選定此夾爪模型使用的相機校準')
        camera = WristCalibration.from_mapping(optics,
            native_image_size=optics['native_image_size'], camera_id=config['wrist_camera_id'],
            mount_revision=config.get('camera_mount_revision', 'existing-mount-unverified'),
            geometry_revision=revision, cad_handeye=np.eye(4))
        if (not camera.metric_calibrated or json.loads(jaw.model_assisted_json)[
                'camera_calibration_fingerprint'] != camera.fingerprint):
            raise ValueError('夾爪模型與目前相機校準不同，尚未切換設定')
    return str(path.relative_to(root)), jaw
