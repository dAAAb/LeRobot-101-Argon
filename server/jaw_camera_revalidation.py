"""Offline revalidation of jaw evidence after an exactly frozen camera extension.

The original acquisitions and their camera snapshot stay immutable. This narrow
path accepts only an independently validated *direct* frozen-camera extension:
K, distortion, hand-eye, focal uncertainty and hardware identity cannot change.
It rederives every saved camera-relative contact section using the existing
jaw builder. It neither reacquires evidence nor relaxes the runtime contract.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .jaw_model_assisted import validate_silhouette_evidence, build_model_assisted_profile
from .spatial_contract import JawCalibration, WristCalibration, JOINTS

METHOD = 'jaw-revalidation-exact-frozen-camera-extension'
POSE_ERROR_FIELDS = ('relative_translation_uncertainty_m', 'base_position_uncertainty_m',
                     'relative_rotation_uncertainty_deg')
CAMERA_MUTABLE = {*POSE_ERROR_FIELDS, 'calibration_revision', 'validation_id',
    'board_session_id', 'validation_scope', 'extension_provenance',
    'uncertainty_interpretation', 'metric_accuracy'}
SCOPE_MUTABLE = {'joint_raw_bounds', 'sampled_joint_poses', 'image_points_bbox_px',
    'board_plane_base', 'interpretation', 'extrapolation_allowed',
    'individually_identified_joint_zeros', 'jaw_contact_geometry_calibrated'}


def _canonical_hash(value, *, ensure_ascii=False):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=ensure_ascii, allow_nan=False).encode()).hexdigest()


def _artifact(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError('Missing immutable evidence: ' + str(path))
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def _read(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError('Missing or oversized calibration artifact')
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('Expected a calibration object')
    json.dumps(value, allow_nan=False)
    return value


def _camera(value):
    camera = WristCalibration.from_mapping(value, native_image_size=value['native_image_size'],
        camera_id=value['camera_id'], mount_revision=value['mount_revision'],
        geometry_revision=value['geometry_revision'], cad_handeye=np.eye(4))
    if value.get('source') != 'wrist-only' or not camera.metric_calibrated or value.get('needs_more_views') is True:
        raise ValueError('Both camera artifacts must already be validated wrist-only profiles')
    return camera


def validate_frozen_camera_extension(old, new, extension_report):
    """Prove strict numeric equivalence and retained validation lineage."""
    previous, current = _camera(old), _camera(new)
    for key in set(old) | set(new):
        if key not in CAMERA_MUTABLE and old.get(key) != new.get(key):
            raise ValueError('Frozen-camera revalidation cannot change ' + key)
    if previous.fingerprint == current.fingerprint:
        raise ValueError('The camera artifact has not changed; no revalidation is needed')
    for key in POSE_ERROR_FIELDS:
        a, b = old.get(key), new.get(key)
        if (type(a) not in (int, float) or type(b) not in (int, float)
                or not math.isfinite(a) or not math.isfinite(b) or a < 0 or b < a):
            raise ValueError('Frozen-camera extension must retain or increase ' + key)
    lineage = new.get('extension_provenance', {})
    validation = extension_report.get('validation', {})
    # calibration_extension uses board_calibration._hash, whose canonical
    # JSON escapes non-ASCII text. Saved board detections include Chinese
    # messages, so the lineage hash is not our local UTF-8 artifact identity.
    parent_profile_hash = _canonical_hash(old, ensure_ascii=True)
    if (extension_report.get('status') != 'validated'
            or extension_report.get('method') != 'frozen-camera-workspace-extension'
            or extension_report.get('source') != 'wrist-only'
            or extension_report.get('calibration_profile') != new
            or lineage.get('parent_profile') != old
            or lineage.get('parent_profile_sha256') != parent_profile_hash
            or lineage.get('validation') != validation
            or validation.get('parent_profile_sha256') != parent_profile_hash
            or validation.get('frozen_parameters') != ['camera_matrix', 'distortion', 'handeye']
            or validation.get('fitted_parameters') != ['T_base_board']
            or validation.get('heldout_data_used_for_fitting') is not False
            or validation.get('heldout_used_for_shared_parameter_fitting') is not False
            or lineage.get('board_session_id') != new.get('board_session_id')
            or extension_report.get('board_session_id') != new.get('board_session_id')
            or old.get('board_session_id') == new.get('board_session_id')):
        raise ValueError('Need a validated direct frozen-parent lineage and its unchanged solve report')
    groups = [validation.get(k, []) for k in ('training_frame_ids', 'heldout_frame_ids')]
    if (any(not isinstance(g, list) or len(g) < 3 or len(set(g)) != len(g) for g in groups)
            or set(groups[0]) & set(groups[1])):
        raise ValueError('The extension needs separate training and held-out pose groups')
    old_scope, new_scope = old['validation_scope'], new['validation_scope']
    if any(old_scope.get(k) != new_scope.get(k) for k in set(old_scope) | set(new_scope)
           if k not in SCOPE_MUTABLE):
        raise ValueError('The frozen extension changed an unrelated workspace field')
    for key in ('extrapolation_allowed', 'individually_identified_joint_zeros', 'jaw_contact_geometry_calibrated'):
        if new_scope.get(key) is not False:
            raise ValueError('The extension must not claim extrapolation, identified zeros or jaw calibration')
    old_poses = old_scope['sampled_joint_poses']
    sampled = new_scope['sampled_joint_poses']
    frames = {f['id']: f for f in lineage.get('frames', [])}
    ids = groups[0] + groups[1]
    if (len(frames) != len(lineage.get('frames', [])) or not set(ids) <= frames.keys()
            or sampled[:len(old_poses)] != old_poses
            or sampled[len(old_poses):] != [{n: frames[i]['joints'][n] for n in JOINTS[:-1]} for i in ids]):
        raise ValueError('The extension must retain old poses and bind every added pose to its held-out report')
    expected_bounds = {n: [min(old_scope['joint_raw_bounds'][n][0], min(p[n] for p in sampled)),
                           max(old_scope['joint_raw_bounds'][n][1], max(p[n] for p in sampled))]
                       for n in JOINTS[:-1]}
    if new_scope['joint_raw_bounds'] != expected_bounds:
        raise ValueError('Extended joint bounds do not match the recorded pose union')
    if new_scope.get('board_plane_base') != validation.get('board_plane_base'):
        raise ValueError('Extended board plane differs from the validated report')
    return {'old_camera_fingerprint': previous.fingerprint,
        'new_camera_fingerprint': current.fingerprint,
        'numeric_projection_equivalent': True, 'focal_uncertainty_identical': True,
        'changed_top_level_fields': sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k)),
        'old_pose_uncertainty': {k: old[k] for k in POSE_ERROR_FIELDS},
        'new_pose_uncertainty': {k: new[k] for k in POSE_ERROR_FIELDS},
        'base_fk_error_is_separate_from_camera_relative_jaw_error': True}


def _same_jaw_geometry(a, b):
    if len(a['samples']) != len(b['samples']):
        raise ValueError('Saved sections do not cover the complete reviewed opening curve')
    for first, second in zip(a['samples'], b['samples']):
        if first['raw'] != second['raw']:
            raise ValueError('Saved section encoder sequence changed')
        for key in ('fixed_pad_m', 'moving_pad_m', 'approach_axis'):
            if not np.allclose(first[key], second[key], rtol=0, atol=1e-12):
                raise ValueError('Rederived jaw section differs from the reviewed geometry: ' + key)
    if (a['model_assisted']['uncertainty'] != b['model_assisted']['uncertainty']
            or a['position_uncertainty_m'] != b['position_uncertainty_m']):
        raise ValueError('Saved section uncertainty differs from the reviewed profile')


def revalidate_saved_jaw_camera(*, jaw_profile_path, jaw_report_path, old_camera_path,
                               new_camera_path, camera_extension_report_path,
                               manifest_path, output):
    """Write a NEW inactive report/profile pair; never select or edit originals.

    Held-out pixels are not reselected or refitted: the original evaluation is
    inherited under exact numeric projection equivalence. Every saved camera
    space contact section is rederived by the normal jaw profile builder.
    """
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new output directory; never overwrite reviewed evidence')
    paths = {'old_jaw_profile': jaw_profile_path, 'old_jaw_report': jaw_report_path,
        'old_camera_profile': old_camera_path, 'new_camera_profile': new_camera_path,
        'camera_extension_report': camera_extension_report_path, 'acquisition_manifest': manifest_path}
    artifacts = {name: _artifact(path) for name, path in paths.items()}
    old_jaw, old_report, old, new, extension = (_read(paths[n]) for n in
        ('old_jaw_profile', 'old_jaw_report', 'old_camera_profile', 'new_camera_profile', 'camera_extension_report'))
    comparison = validate_frozen_camera_extension(old, new, extension)
    parsed = JawCalibration.from_mapping(old_jaw, geometry_revision=old['geometry_revision'])
    if old_jaw.get('status') != 'model-assisted' or not parsed.at(parsed.samples[0].raw).planning_eligible:
        raise ValueError('Only a reviewed model-assisted jaw profile can be revalidated')
    metadata = old_jaw['model_assisted']
    for artifact in metadata['evidence_files']:
        if _artifact(artifact['path'])['sha256'] != artifact['sha256']:
            raise ValueError('Original jaw evidence changed: ' + artifact['path'])
    if artifacts['old_jaw_report']['sha256'] not in {a['sha256'] for a in metadata['evidence_files']}:
        raise ValueError('The supplied section report is not bound by the reviewed jaw profile')
    if artifacts['acquisition_manifest']['sha256'] != metadata['manifest']['sha256']:
        raise ValueError('The supplied manifest is not the reviewed jaw acquisition')
    provenance = validate_silhouette_evidence(manifest_path, metadata['annotations']['path'], old,
        expected_geometry_revision=old['geometry_revision'], cad_files=[a['path'] for a in metadata['cad_files']])
    if (old_report.get('provenance') != provenance
            or any(metadata.get(k) != v for k, v in provenance.items())
            or metadata['camera_calibration_fingerprint'] != comparison['old_camera_fingerprint']
            or old_report['conditional_covariance'] != metadata['conditional_covariance']
            or old_report['assumptions'] != metadata['assumptions']
            or old_report['heldout'] != metadata['validation']['heldout']):
        raise ValueError('Jaw acquisition, fit, reviewed report and camera provenance are inconsistent')
    original_files = [a['path'] for a in metadata['evidence_files']]
    common = dict(geometry_revision=old['geometry_revision'], sections=old_report['sections'],
        contact_section=metadata['contact_section'], assumptions=metadata['assumptions'],
        validation=metadata['validation'], covariance_evidence=metadata['conditional_covariance'])
    rebuilt_old = build_model_assisted_profile(provenance=provenance, camera_profile=old,
        calibration_revision=old_jaw['calibration_revision'], validation_id=old_jaw['validation_id'],
        evidence_files=original_files, **common)
    _same_jaw_geometry(old_jaw, rebuilt_old)
    # This is a new derived provenance object; the actual acquisition snapshot
    # remains unchanged and is retained in full beside the equivalence proof.
    chain = {'method': METHOD, 'inputs': artifacts, 'comparison': comparison,
        'acquisition_camera_fingerprint': comparison['old_camera_fingerprint'],
        'original_acquisition_provenance': provenance,
        'heldout_evaluation': 'retained under exact K/D/handeye/focal equivalence; not newly captured or refitted',
        'new_hardware_evidence': False, 'external_metric_measurement': False}
    new_provenance = {**deepcopy(provenance),
        'camera_calibration_fingerprint': comparison['new_camera_fingerprint'],
        'camera_profile_sha256': _canonical_hash(new, ensure_ascii=True),
        'camera_revalidation': chain}
    identity = _canonical_hash(chain)
    report = {'status': 'inactive-exact-projection-revalidated', 'method': METHOD,
        'source': 'wrist-only', 'provenance': new_provenance,
        'assumptions': metadata['assumptions'], 'conditional_covariance': metadata['conditional_covariance'],
        'heldout': metadata['validation']['heldout'], 'sections': old_report['sections'],
        'saved_sections_rederived': len(old_report['sections']),
        'camera_relative_uncertainty_unchanged': True, 'camera_base_fk_uncertainty_updated_separately': True,
        'activation_performed': False, 'motors_commanded': False, 'external_metric_measurement': False}
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / 'report.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    new_files = list(dict.fromkeys([*original_files, *[a['path'] for a in artifacts.values()], str(report_path)]))
    result = build_model_assisted_profile(provenance=new_provenance, camera_profile=new,
        calibration_revision='jaw-frozen-camera-' + identity[:20], validation_id='jaw-frozen-camera-' + identity,
        evidence_files=new_files, **common)
    _same_jaw_geometry(old_jaw, result)
    # Recheck all input bytes at the publication boundary, including images/CAD
    # already verified above. Nothing activates a partially produced artifact.
    for artifact in [*artifacts.values(), *metadata['evidence_files']]:
        if _artifact(artifact['path'])['sha256'] != artifact['sha256']:
            raise ValueError('Evidence changed during offline revalidation')
    (output / 'profile.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return {'status': report['status'], 'output': str(output), 'profile': str(output / 'profile.json'),
        'report': str(report_path), 'saved_sections_rederived': len(old_report['sections']),
        'activation_performed': False}
