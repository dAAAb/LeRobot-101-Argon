"""Read-only, profile-bound empty-jaw opening targets; never scene approval."""
from copy import deepcopy
import json

import numpy as np

from .spatial_contract import WristCalibration
from .spatial_runtime import _profile_snapshot

LEGACY_TARGET = 1860
ENDPOINT_TOLERANCE = 6
PROFILE_RESERVE = 8


def empty_opening_contract(model, config):
    """Disclose one complete target. Missing v2 jaw selection keeps legacy scope.

    Increasing raw must actually increase the selected model's aperture. No
    actuator command or caller receives a wider target merely by loading v2.
    The returned contract must be part of the visually assessed candidate.
    """
    legacy = dict(version=1, mode='legacy', target_raw=LEGACY_TARGET,
        measured_raw_range=[1500, 1862], endpoint_tolerance_ticks=ENDPOINT_TOLERANCE,
        profile_fingerprint=None, target_geometry=None)
    if config.get('spatial_backend') != 'v2' or not config.get('jaw_calibration_profile'):
        return legacy
    revision, camera_data, jaw, _ = _profile_snapshot(model, config, include_jaws=True)
    if jaw is None:
        raise ValueError('缺少所選的夾爪開口模型')
    low, high = jaw.samples[0].raw, jaw.samples[-1].raw
    motor = model.calibration['gripper']
    target = min(high - PROFILE_RESERVE, motor['range_max'] - 12)
    if target - ENDPOINT_TOLERANCE < max(low, motor['range_min']):
        raise ValueError('所選夾爪範圍不足以保留張開終點餘量')
    # Squared interpolated aperture has nonnegative derivative throughout a
    # sample interval iff the derivative at its start is nonnegative.
    for a, b in zip(jaw.samples, jaw.samples[1:]):
        first = np.array(a.moving_pad_m) - a.fixed_pad_m
        last = np.array(b.moving_pad_m) - b.fixed_pad_m
        if float(first @ (last - first)) < -1e-12:
            raise ValueError('所選夾爪模型不能確認增加編碼值全程只會張開')
    geometry = jaw.at(target)
    if not geometry.planning_eligible:
        raise ValueError('所選夾爪模型尚不能用於開口規劃')
    camera_fingerprint = None
    if jaw.model_assisted_json is not None:
        if camera_data is None:
            raise ValueError('夾爪估計需要原本的腕部相機校準')
        camera = WristCalibration.from_mapping(camera_data,
            native_image_size=camera_data['native_image_size'],
            camera_id=config.get('wrist_camera_id', 'wrist'),
            mount_revision=config.get('camera_mount_revision', 'existing-mount-unverified'),
            geometry_revision=revision, cad_handeye=np.eye(4))
        camera_fingerprint = camera.fingerprint
        if (not camera.metric_calibrated or camera_fingerprint !=
                json.loads(jaw.model_assisted_json)['camera_calibration_fingerprint']):
            raise ValueError('張開模型與目前腕部相機校準不同')
    return dict(version=1, mode='profile', target_raw=target,
        measured_raw_range=[low, high], endpoint_tolerance_ticks=ENDPOINT_TOLERANCE,
        profile_reserve_ticks=PROFILE_RESERVE, profile_fingerprint=jaw.fingerprint,
        geometry_revision=revision, camera_fingerprint=camera_fingerprint,
        target_geometry=geometry.to_dict())


def require_opening_contract(contract, model, config):
    """Reject altered targets or a changed selected calibration before PWM."""
    expected = empty_opening_contract(model, config)
    if not isinstance(contract, dict) or contract != expected:
        raise ValueError('張開目標或所選夾爪校準已改變，請重新審核較大開口')
    return deepcopy(expected)
