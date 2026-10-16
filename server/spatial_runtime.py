"""Select a versioned wrist profile and adapt it to the existing depth engine.

No device access or motor commands. A profile selection cannot upgrade points
computed with another profile. Production never imports observer measurements.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re

from .spatial_contract import build_wrist_observation, JawCalibration, require_compatible
from .calibration_domain import boundary_reserve, effective_uncertainty

ROOT = Path(__file__).resolve().parents[1]


def settings(config):
    values = {'spatial_backend': config.get('spatial_backend', 'legacy'),
              'motion_backend': config.get('motion_backend', 'legacy'),
              'object_model_provider': config.get('object_model_provider', 'geometry')}
    if values['spatial_backend'] not in ('legacy', 'shadow', 'v2'):
        raise ValueError('Unknown spatial backend')
    if values['motion_backend'] not in ('legacy', 'task_space_v2'):
        raise ValueError('Unknown motion backend')
    if values['motion_backend'] == 'task_space_v2' and values['spatial_backend'] != 'v2':
        raise ValueError('Task-space motion requires spatial_backend=v2')
    if values['object_model_provider'] != 'geometry':
        raise ValueError('Boxer is not installed or validated; choose geometry')
    return values


def geometry_revision(model):
    """Bind actual onboard limits and the exact FK implementation/URDF."""
    urdf = Path(model.urdf_path).read_bytes()
    implementation = (ROOT / 'server/geometry.py').read_bytes()
    calibration = json.dumps(model.calibration, sort_keys=True, allow_nan=False).encode()
    offsets = getattr(model.fk, 'offsets', {})
    if not offsets:
        return hashlib.sha256(urdf + b'\0' + implementation + b'\0' + calibration).hexdigest()
    offsets = json.dumps(offsets, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(urdf + b'\0' + implementation + b'\0' + calibration + b'\0' + offsets).hexdigest()


def _read_profile(config, key):
    selected = config.get(key)
    if selected is None:
        return None
    if not isinstance(selected, str) or not selected.strip():
        raise ValueError('Calibration profile must be an explicit local path')
    path = Path(selected)
    if not path.is_absolute():
        path = ROOT / path
    if path.stat().st_size > 1024 * 1024:
        raise ValueError('Calibration profile is too large')
    content = path.read_bytes()
    # Reviewed jaw artifacts are content-addressed. Keep that binding after
    # activation too; an edited file must become a new reviewed version.
    identity = re.fullmatch(r'jaw-([0-9a-f]{64})-profile\.json', path.name)
    if key == 'jaw_calibration_profile' and identity and hashlib.sha256(content).hexdigest() != identity[1]:
        raise ValueError('Selected jaw calibration content differs from its reviewed version')
    data = json.loads(content)
    if not isinstance(data, dict) or data.get('source') != 'wrist-only':
        raise ValueError('Calibration profile must declare wrist-only provenance')
    # Content fingerprint is retained in the observation, not just a filename.
    return data


def _profile_snapshot(model, config, *, include_jaws):
    """Read each selected artifact once; one packet batch shares that content."""
    config = deepcopy(config)
    settings(config)
    revision = geometry_revision(model)
    profile = _read_profile(config, 'spatial_calibration_profile')
    jaw_data = _read_profile(config, 'jaw_calibration_profile') if include_jaws else None
    jaw = None if jaw_data is None else JawCalibration.from_mapping(jaw_data, geometry_revision=revision)
    return revision, profile, jaw, config


def _packet(model, frame, snapshot):
    revision, profile, jaw, config = snapshot
    packet = build_wrist_observation(frame, model.fk.transforms(frame['joints']),
        geometry_revision=revision, calibration=profile,
        camera_id=config.get('wrist_camera_id', 'wrist'),
        mount_revision=config.get('camera_mount_revision', 'existing-mount-unverified'),
        jaw_calibration=jaw)
    reserve = boundary_reserve(model, packet.calibration, frame['joints'])
    return replace(packet, scope_reserve_json=json.dumps(reserve, sort_keys=True, allow_nan=False))


def packet_uncertainty(packets):
    """Merge per-frame runtime reserves without modifying the calibrated artifact."""
    if not packets:
        raise ValueError('需要腕部觀察才能計算相機誤差')
    return effective_uncertainty(json.loads(packets[0].calibration.uncertainty_json),
        [json.loads(p.scope_reserve_json) for p in packets if p.scope_reserve_json])


def observation_packet(model, frame, config):
    return _packet(model, frame, _profile_snapshot(model, config, include_jaws=True))


def observation_pose_bounds(model, frame, config):
    """Use the selected camera domain while generating v2 observation paths.

    Looking does not depend on jaw calibration or its current opening. Legacy
    operation keeps its existing behavior; absent calibration is still handled
    by the normal setup report rather than inventing a workspace here.
    """
    if settings(config)['motion_backend'] != 'task_space_v2':
        return None
    packet = _packet(model, frame, _profile_snapshot(model, config, include_jaws=False))
    calibration = packet.calibration
    if not calibration.metric_calibrated or calibration.validation_scope_json is None:
        return None
    return {n: [max(lo, model.calibration[n]['range_min']), min(hi, model.calibration[n]['range_max'])]
            for n, (lo, hi) in calibration.accepted_pose_bounds().items()}


def depth_calibration(model, frames, config):
    """Return None for the original boardless path, or the selected exact model.

    Compatibility is checked for every contributing frame. Durable calibration
    binds camera/mount/model identity; the returned adapter is session-bound.
    """
    if settings(config)['spatial_backend'] != 'v2' or not config.get('spatial_calibration_profile'):
        return None
    # Jaw contact geometry is not an input to camera triangulation. In
    # particular a jaw opening outside its sampled aperture curve must not
    # prevent collecting new camera evidence to calibrate that very opening.
    snapshot = _profile_snapshot(model, config, include_jaws=False)
    packets = [_packet(model, frame, snapshot) for frame in frames]
    require_compatible(packets)
    current = packets[-1]
    c = current.calibration
    validated = all(p.metric_calibrated for p in packets)
    original = snapshot[1]
    return {**deepcopy(original), **packet_uncertainty(packets),
        'runtime_calibration_scopes': [json.loads(p.scope_reserve_json) for p in packets],
        'image_size': list(current.processing.output_size),
        'camera_matrix': [list(row) for row in current.camera_matrix],
        'distortion': list(current.distortion), 'handeye': c.T_wrist_camera.as_array().tolist(),
        'intrinsics_validated': validated, 'handeye_validated': validated,
        'robot_geometry_validated': validated, 'validation_id': c.validation_id,
        'metric_calibrated': validated,
        'camera_generation': current.camera_generation,
        'calibration_revision': c.calibration_revision,
        'calibration_fingerprint': c.fingerprint,
        'geometry_revision': c.geometry_revision}


def public_state(config, model=None, frame=None):
    result = {**settings(config), 'source': 'wrist-only', 'observer_in_production': False,
              'calibration_selected': bool(config.get('spatial_calibration_profile')),
              'jaw_calibration_selected': bool(config.get('jaw_calibration_profile')),
              'calibration_page': '/engineering/calibration',
              'metric_calibrated': False, 'jaw_calibrated': False, 'jaw_planning_eligible': False}
    if model is not None and frame is not None:
        packet = observation_packet(model, frame, config)
        result.update(metric_calibrated=packet.metric_calibrated,
                      jaw_calibrated=bool(packet.jaw_geometry and packet.jaw_geometry.metric_calibrated),
                      jaw_planning_eligible=bool(packet.jaw_geometry and getattr(packet.jaw_geometry,
                          'planning_eligible', packet.jaw_geometry.metric_calibrated)),
                      geometry_revision=packet.calibration.geometry_revision,
                      calibration_revision=packet.calibration.calibration_revision,
                      frame_id=frame['id'])
    return result
