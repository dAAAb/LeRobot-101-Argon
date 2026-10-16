"""Versioned, immutable wrist geometry; no camera, motor, file or network I/O.

T_A_B maps coordinates in B into A. Lengths are metres; camera optical axes
are right/down/forward. Encoder readings already include the motor's homing
calibration: this module never applies that offset a second time.

``build_wrist_observation(frame, model.fk.transforms(frame['joints']),
geometry_revision=...)`` adapts existing settled snapshots. Legacy camera
artifacts remain useful priors, never independent metric validation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from collections.abc import Mapping

import cv2
import numpy as np

SCHEMA_VERSION = 1
JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')
UNCERTAINTY_FIELDS = ('relative_translation_uncertainty_m', 'base_position_uncertainty_m',
                      'relative_rotation_uncertainty_deg', 'focal_relative_uncertainty')


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')
    return value


def _number(value, name, *, minimum=None):
    if isinstance(value, (bool, str)) or not np.isscalar(value) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    value = float(value)
    if minimum is not None and value < minimum:
        raise ValueError(f'{name} is out of range')
    return value


def _size(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2 or any(type(x) is not int or x <= 0 for x in value):
        raise ValueError('image size must contain two positive integers')
    return tuple(value)


def _matrix(value, shape, name):
    a = np.array(value, dtype=float, copy=True)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'{name} has invalid shape or nonfinite values')
    return a


def _tuple_matrix(value):
    return tuple(tuple(float(x) for x in row) for row in value)


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class RigidTransform:
    parent: str
    child: str
    matrix: tuple
    revision: str

    def __post_init__(self):
        _text(self.parent, 'parent frame'); _text(self.child, 'child frame')
        _text(self.revision, 'transform revision')
        t = _matrix(self.matrix, (4, 4), 'rigid transform')
        if (not np.allclose(t[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
                or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-6, rtol=0)
                or abs(np.linalg.det(t[:3, :3]) - 1) > 1e-6):
            raise ValueError('transform must contain a proper rotation and metre translation')
        object.__setattr__(self, 'matrix', _tuple_matrix(t))

    def as_array(self):
        """A fresh copy; callers cannot mutate the stored transform."""
        return np.array(self.matrix)

    def inverse(self):
        t = self.as_array(); result = np.eye(4)
        result[:3, :3] = t[:3, :3].T
        result[:3, 3] = -t[:3, :3].T @ t[:3, 3]
        return RigidTransform(self.child, self.parent, result, self.revision)

    def compose(self, other):
        if self.child != other.parent:
            raise ValueError('transform frames do not compose')
        return RigidTransform(self.parent, other.child, self.as_array() @ other.as_array(),
                              f'{self.revision}/{other.revision}')

    def apply(self, points):
        points = np.asarray(points, dtype=float)
        if points.shape[-1:] != (3,) or not np.isfinite(points).all():
            raise ValueError('points must contain finite XYZ metre coordinates')
        t = self.as_array()
        return points @ t[:3, :3].T + t[:3, 3]

    def to_dict(self):
        return {'parent': self.parent, 'child': self.child, 'matrix': [list(r) for r in self.matrix],
                'revision': self.revision, 'translation_unit': 'm'}


@dataclass(frozen=True)
class ImageProcessing:
    """Integer crop followed by OpenCV pixel-centre resize, with no hidden crop."""
    native_size: tuple
    output_size: tuple
    crop_xywh: tuple | None = None

    def __post_init__(self):
        native, output = _size(self.native_size), _size(self.output_size)
        crop = tuple(self.crop_xywh) if self.crop_xywh is not None else (0, 0, *native)
        if (len(crop) != 4 or any(type(v) is not int for v in crop)
                or min(crop[:2]) < 0 or min(crop[2:]) <= 0
                or crop[0] + crop[2] > native[0] or crop[1] + crop[3] > native[1]):
            raise ValueError('crop must lie inside the calibrated native image')
        object.__setattr__(self, 'native_size', native)
        object.__setattr__(self, 'output_size', output)
        object.__setattr__(self, 'crop_xywh', crop)

    @property
    def pixel_transform(self):
        x, y, w, h = self.crop_xywh
        sx, sy = self.output_size[0] / w, self.output_size[1] / h
        return np.array([[sx, 0, sx * (.5 - x) - .5],
                         [0, sy, sy * (.5 - y) - .5], [0, 0, 1]])

    def camera_matrix(self, native_matrix):
        return self.pixel_transform @ _matrix(native_matrix, (3, 3), 'camera matrix')

    def apply(self, image):
        if image.shape[:2][::-1] != self.native_size:
            raise ValueError('image resolution does not match native calibration')
        x, y, w, h = self.crop_xywh
        cropped = image[y:y+h, x:x+w]
        return cv2.resize(cropped, self.output_size, interpolation=cv2.INTER_LINEAR)

    def to_dict(self):
        return {'native_size': list(self.native_size), 'output_size': list(self.output_size),
                'crop_xywh': list(self.crop_xywh), 'pixel_centres': 'opencv-half-pixel',
                'pixel_transform': self.pixel_transform.tolist()}


@dataclass(frozen=True)
class WristCalibration:
    calibration_revision: str
    geometry_revision: str
    camera_id: str
    native_image_size: tuple
    camera_matrix: tuple
    distortion: tuple
    T_wrist_camera: RigidTransform
    validated_components: tuple = ()
    validation_id: str | None = None
    validation_kind: str = 'prior'
    camera_generation: str | None = None
    mount_revision: str = 'unverified-mount'
    schema_version: int = SCHEMA_VERSION
    validation_scope_json: str | None = None
    uncertainty_json: str = '{}'

    def __post_init__(self):
        for value in (self.calibration_revision, self.geometry_revision, self.camera_id, self.mount_revision):
            _text(value, 'calibration identity')
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError('unsupported spatial calibration schema')
        _text(self.validation_kind, 'calibration validation kind')
        if self.validation_id is not None:
            _text(self.validation_id, 'calibration validation ID')
        if self.camera_generation is not None:
            _text(self.camera_generation, 'calibration camera generation')
        if self.validation_scope_json is not None:
            scope = json.loads(self.validation_scope_json)
            bounds = scope.get('joint_raw_bounds') if isinstance(scope, dict) else None
            if (not isinstance(bounds, dict) or set(bounds) != set(JOINTS[:-1])
                    or any(not isinstance(v, list) or len(v) != 2
                           or any(type(x) is not int for x in v) or not 0 <= v[0] <= v[1] <= 4095
                           for v in bounds.values())):
                raise ValueError('scoped calibration requires measured five-joint raw bounds')
        uncertainty = json.loads(self.uncertainty_json)
        if not isinstance(uncertainty, dict) or set(uncertainty) - {*UNCERTAINTY_FIELDS, 'uncertainty_interpretation'}:
            raise ValueError('calibration uncertainty provenance must be an object')
        for name in UNCERTAINTY_FIELDS:
            if name in uncertainty:
                _number(uncertainty[name], name, minimum=0)
        size = _size(self.native_image_size)
        k = _matrix(self.camera_matrix, (3, 3), 'camera matrix')
        if (min(k[0, 0], k[1, 1]) <= 0 or not np.allclose(k[2], [0, 0, 1])
                or abs(k[0, 1]) > 1e-8 or abs(k[1, 0]) > 1e-8):
            raise ValueError('camera matrix must be OpenCV pinhole intrinsics')
        distortion = tuple(_number(v, 'Brown distortion') for v in self.distortion)
        if len(distortion) not in (4, 5, 8, 12, 14):
            raise ValueError('Brown distortion requires 4, 5, 8, 12 or 14 coefficients; not KB4')
        if (self.T_wrist_camera.parent, self.T_wrist_camera.child) != ('gripper_link', 'wrist_camera_optical'):
            raise ValueError('hand-eye transform must map wrist camera into gripper_link')
        if set(self.validated_components) - {'intrinsics', 'handeye', 'robot_geometry'}:
            raise ValueError('unknown validated calibration component')
        object.__setattr__(self, 'native_image_size', size)
        object.__setattr__(self, 'camera_matrix', _tuple_matrix(k))
        object.__setattr__(self, 'distortion', distortion)
        object.__setattr__(self, 'validated_components', tuple(sorted(set(self.validated_components))))

    @property
    def metric_calibrated(self):
        return (set(self.validated_components) == {'intrinsics', 'handeye', 'robot_geometry'}
                and self.validation_kind == 'independent-heldout'
                and isinstance(self.validation_id, str) and bool(self.validation_id.strip()))

    @property
    def fingerprint(self):
        return _fingerprint(self.to_dict())

    def contains_pose(self, joints):
        """Original bounds plus a fixed small runtime allowance, never cumulative."""
        return self.pose_scope(joints)['accepted']

    def pose_scope(self, joints):
        from .calibration_domain import pose_scope
        bounds = (json.loads(self.validation_scope_json)['joint_raw_bounds']
                  if self.validation_scope_json is not None else None)
        return pose_scope(bounds, joints)

    def accepted_pose_bounds(self):
        from .calibration_domain import BOUNDARY_ALLOWANCE_TICKS
        if self.validation_scope_json is None:
            return None
        return {n: [max(0, lo - BOUNDARY_ALLOWANCE_TICKS), min(4095, hi + BOUNDARY_ALLOWANCE_TICKS)]
                for n, (lo, hi) in json.loads(self.validation_scope_json)['joint_raw_bounds'].items()}

    def to_dict(self):
        return {'schema_version': self.schema_version, 'calibration_revision': self.calibration_revision,
                'geometry_revision': self.geometry_revision, 'camera_id': self.camera_id,
                'native_image_size': list(self.native_image_size),
                'camera_matrix': [list(r) for r in self.camera_matrix], 'distortion': list(self.distortion),
                'distortion_model': 'opencv-brown', 'handeye': [list(r) for r in self.T_wrist_camera.matrix],
                'validated_components': list(self.validated_components),
                'validation_id': self.validation_id, 'validation_kind': self.validation_kind,
                'camera_generation': self.camera_generation, 'mount_revision': self.mount_revision,
                'validation_scope': json.loads(self.validation_scope_json) if self.validation_scope_json else None,
                **json.loads(self.uncertainty_json)}

    @classmethod
    def from_mapping(cls, data, *, native_image_size, camera_id, geometry_revision, cad_handeye,
                     mount_revision='unverified-mount'):
        """Load either a versioned profile or an explicitly unvalidated legacy prior."""
        data = dict(data or {})
        if data.get('source', 'wrist-only') not in ('wrist', 'wrist-only'):
            raise ValueError('production spatial calibration accepts wrist data only')
        for key, expected in (('geometry_revision', geometry_revision), ('camera_id', camera_id),
                              ('mount_revision', mount_revision)):
            if key in data and data[key] != expected:
                raise ValueError(f'calibration {key} is incompatible')
        size = tuple(data.get('native_image_size', data.get('image_size', native_image_size)))
        if size != tuple(native_image_size):
            raise ValueError('camera calibration resolution changed; declare an explicit pixel mapping')
        if data.get('distortion_model', 'opencv-brown') not in ('opencv-brown', 'brown', 'pinhole'):
            raise ValueError('camera distortion model is not Brown/pinhole; KB4 is not interchangeable')
        w, h = size
        k = data.get('camera_matrix', [[w * .75, 0, (w - 1) / 2],
                                       [0, w * .75, (h - 1) / 2], [0, 0, 1]])
        x = data.get('handeye', cad_handeye)
        # Preserve revision labels while fingerprinting full content separately.
        revision = data.get('calibration_revision') or 'prior-' + _fingerprint({
            'K': np.asarray(k).tolist(), 'X': np.asarray(x).round(12).tolist(),
            'size': size, 'geometry': geometry_revision})[:16]
        versioned = type(data.get('schema_version')) is int and data['schema_version'] == SCHEMA_VERSION
        if 'schema_version' in data and not versioned:
            raise ValueError('unsupported spatial calibration schema')
        components = data.get('validated_components', ()) if versioned else ()
        if 'intrinsics' in components and not all(k in data for k in ('camera_matrix', 'distortion')):
            raise ValueError('validated intrinsics cannot use default focal length or distortion')
        if 'handeye' in components and 'handeye' not in data:
            raise ValueError('validated hand-eye cannot use a CAD mounting prior')
        if components and ('geometry_revision' not in data or 'camera_id' not in data
                           or 'mount_revision' not in data
                           or not any(k in data for k in ('native_image_size', 'image_size'))):
            raise ValueError('validated calibration needs explicit device, mount, geometry and resolution binding')
        if data.get('needs_more_views') or data.get('resolution_inferred_from_principal_point'):
            components = ()
        uncertainty = {n: _number(data[n], n, minimum=0) for n in UNCERTAINTY_FIELDS if n in data}
        if data.get('uncertainty_interpretation') is not None:
            uncertainty['uncertainty_interpretation'] = _text(data['uncertainty_interpretation'], 'uncertainty interpretation')
        return cls(revision, geometry_revision, camera_id, size, k, data.get('distortion', (0.,) * 5),
                   RigidTransform('gripper_link', 'wrist_camera_optical', x, revision),
                   tuple(components), data.get('validation_id'),
                   data.get('validation_kind', 'prior') if versioned else 'prior',
                   data.get('camera_generation'), mount_revision,
                   validation_scope_json=(json.dumps(data['validation_scope'], sort_keys=True, allow_nan=False)
                                          if data.get('validation_scope') is not None else None),
                   uncertainty_json=json.dumps(uncertainty, sort_keys=True, allow_nan=False))


@dataclass(frozen=True)
class JawSample:
    raw: int
    fixed_pad_m: tuple
    moving_pad_m: tuple
    approach_axis: tuple | None = None

    def __post_init__(self):
        if type(self.raw) is not int or not 0 <= self.raw <= 4095:
            raise ValueError('jaw sample raw must be an encoder reading in 0..4095')
        for name in ('fixed_pad_m', 'moving_pad_m'):
            values = _matrix(getattr(self, name), (3,), name)
            object.__setattr__(self, name, tuple(float(v) for v in values))
        if self.approach_axis is not None:
            axis = _matrix(self.approach_axis, (3,), 'sample approach axis')
            if abs(np.linalg.norm(axis) - 1) > 1e-6:
                raise ValueError('sample approach axis must be a unit vector')
            object.__setattr__(self, 'approach_axis', tuple(float(v) for v in axis))


@dataclass(frozen=True)
class JawGeometry:
    T_wrist_tcp: RigidTransform
    aperture_m: float
    position_uncertainty_m: float | None
    metric_calibrated: bool
    calibration_revision: str
    fingerprint: str
    jaw_raw: int
    geometry_revision: str
    landmark_definition: str = 'legacy'
    validation_kind: str = 'prior'
    model_assisted_json: str | None = None

    @property
    def planning_eligible(self):
        return self.metric_calibrated or self.model_assisted_json is not None

    def to_dict(self):
        result = {'T_wrist_tcp': self.T_wrist_tcp.to_dict(), 'aperture_m': self.aperture_m,
                'position_uncertainty_m': self.position_uncertainty_m,
                'metric_calibrated': self.metric_calibrated,
                'planning_eligible': self.planning_eligible,
                'validation_kind': self.validation_kind,
                'calibration_revision': self.calibration_revision, 'fingerprint': self.fingerprint,
                'jaw_raw': self.jaw_raw,
                'geometry_revision': self.geometry_revision,
                'landmark_definition': self.landmark_definition,
                'aperture_scope': ('width-at-matched-contact-cross-section; not the entire insertion corridor'
                    if self.landmark_definition == 'matched_contact_cross_section'
                    else 'distance-between-two-inner-pad-landmarks'),
                'reference': 'midpoint-of-two-inner-pad-landmarks',
                'scope': 'TCP-and-aperture; not a collision mesh or contact proof'}
        if self.model_assisted_json is not None:
            evidence = json.loads(self.model_assisted_json)
            result.update(accuracy_basis='conditional-cad-estimate',
                geometry_method=evidence['geometry_method'], external_metric_measurement=False,
                uncertainty_frame=evidence['uncertainty_frame'],
                aperture_uncertainty_m=evidence['uncertainty']['aperture_m'],
                camera_calibration_fingerprint=evidence['camera_calibration_fingerprint'],
                assumptions=evidence['assumptions'], contact_section=evidence['contact_section'],
                reference='midpoint-of-nominal-CAD-contact-section',
                scope='Estimated front contact section only; not independently measured pads, entire insertion corridor, or contact proof')
        return result


@dataclass(frozen=True)
class JawCalibration:
    calibration_revision: str
    geometry_revision: str
    samples: tuple
    approach_axis: tuple
    position_uncertainty_m: float | None = None
    status: str = 'prior'
    validation_id: str | None = None
    validation_kind: str = 'prior'
    landmark_definition: str = 'legacy'
    model_assisted_json: str | None = None

    def __post_init__(self):
        _text(self.calibration_revision, 'jaw revision'); _text(self.geometry_revision, 'geometry revision')
        if self.status not in ('prior', 'validated', 'model-assisted'):
            raise ValueError('jaw status must be prior, validated, or model-assisted')
        samples = tuple(self.samples)
        if (len(samples) < 2 or not all(isinstance(s, JawSample) for s in samples)
                or any(a.raw >= b.raw for a, b in zip(samples, samples[1:]))):
            raise ValueError('jaw profile needs ordered, distinct measured openings')
        if self.landmark_definition not in ('legacy', 'matched_contact_cross_section'):
            raise ValueError('unknown jaw landmark definition')
        supplied_axes = [s.approach_axis is not None for s in samples]
        if ((self.landmark_definition == 'matched_contact_cross_section' or any(supplied_axes))
                and not all(supplied_axes)):
            raise ValueError('per-opening contact sections require an approach axis at every sample')
        if all(supplied_axes) and any(np.dot(a.approach_axis, b.approach_axis) <= 0
                                      for a, b in zip(samples, samples[1:])):
            raise ValueError('neighboring sample approach axes must describe the same approach direction')
        axis = _matrix(self.approach_axis, (3,), 'TCP approach axis')
        if abs(np.linalg.norm(axis) - 1) > 1e-6:
            raise ValueError('TCP approach axis must be a unit vector')
        if self.position_uncertainty_m is not None:
            object.__setattr__(self, 'position_uncertainty_m',
                               _number(self.position_uncertainty_m, 'jaw position uncertainty', minimum=0))
        _text(self.validation_kind, 'jaw validation kind')
        if self.validation_id is not None:
            _text(self.validation_id, 'jaw validation ID')
        if self.status == 'validated' and (not self.validation_id or self.validation_kind != 'independent-heldout'
                                           or self.position_uncertainty_m is None):
            raise ValueError('validated jaw geometry needs independent validation and an error bound')
        if self.status == 'model-assisted':
            from .jaw_model_assisted import validate_model_assisted_metadata
            if (not self.validation_id or self.validation_kind != 'independent-silhouette-heldout'
                    or self.landmark_definition != 'matched_contact_cross_section'
                    or self.position_uncertainty_m is None or self.model_assisted_json is None):
                raise ValueError('model-assisted jaw needs silhouette validation and explicit conditional uncertainty')
            metadata = validate_model_assisted_metadata(json.loads(self.model_assisted_json))
            if (metadata['measured_raw_range'] != [samples[0].raw, samples[-1].raw]
                    or self.position_uncertainty_m < metadata['uncertainty']['tcp_m']):
                raise ValueError('model-assisted profile cannot expand its measured openings or reduce uncertainty')
            object.__setattr__(self, 'model_assisted_json', json.dumps(metadata, sort_keys=True, allow_nan=False))
        elif self.model_assisted_json is not None:
            raise ValueError('model-assisted evidence must not be relabeled as physical jaw validation')
        object.__setattr__(self, 'samples', samples)
        object.__setattr__(self, 'approach_axis', tuple(float(v) for v in axis))

    @property
    def fingerprint(self):
        value = {'revision': self.calibration_revision, 'geometry': self.geometry_revision,
            'samples': [(s.raw, s.fixed_pad_m, s.moving_pad_m) for s in self.samples],
            'approach_axis': self.approach_axis, 'position_uncertainty_m': self.position_uncertainty_m,
            'status': self.status, 'validation_id': self.validation_id, 'validation_kind': self.validation_kind}
        # Existing immutable profiles retain their original identity. New
        # per-opening geometry and its contact-section meaning are hash-bound.
        if self.landmark_definition != 'legacy' or any(s.approach_axis is not None for s in self.samples):
            value.update(landmark_definition=self.landmark_definition,
                         sample_approach_axes=[s.approach_axis for s in self.samples])
        if self.model_assisted_json is not None:
            value['model_assisted'] = json.loads(self.model_assisted_json)
        return _fingerprint(value)

    @classmethod
    def from_mapping(cls, data, *, geometry_revision):
        if (type(data.get('schema_version')) is not int or data['schema_version'] != SCHEMA_VERSION
                or data.get('geometry_revision') != geometry_revision):
            raise ValueError('jaw calibration schema or geometry revision is incompatible')
        if data.get('source', 'wrist-only') not in ('wrist', 'wrist-only'):
            raise ValueError('jaw profile must not use observer-camera calibration')
        return cls(data['calibration_revision'], geometry_revision,
                   tuple(JawSample(**s) for s in data['samples']), tuple(data['approach_axis']),
                   data.get('position_uncertainty_m'), data.get('status', 'prior'),
                   data.get('validation_id'), data.get('validation_kind', 'prior'),
                   data.get('landmark_definition', 'legacy'),
                   json.dumps(data['model_assisted'], sort_keys=True, allow_nan=False)
                   if data.get('model_assisted') is not None else None)

    def at(self, raw):
        if type(raw) is not int or not self.samples[0].raw <= raw <= self.samples[-1].raw:
            raise ValueError('jaw opening is outside measured calibration; extrapolation is disabled')
        a, b = next((a, b) for a, b in zip(self.samples, self.samples[1:]) if a.raw <= raw <= b.raw)
        fraction = (raw - a.raw) / (b.raw - a.raw)
        fixed = np.array(a.fixed_pad_m) * (1 - fraction) + np.array(b.fixed_pad_m) * fraction
        moving = np.array(a.moving_pad_m) * (1 - fraction) + np.array(b.moving_pad_m) * fraction
        closing = moving - fixed; aperture = float(np.linalg.norm(closing))
        if aperture < 1e-8:
            raise ValueError('coincident pad landmarks cannot determine a TCP closing axis')
        z = (np.array(self.approach_axis) if a.approach_axis is None else
             (1 - fraction) * np.array(a.approach_axis) + fraction * np.array(b.approach_axis))
        z /= np.linalg.norm(z)
        # These points must actually describe the same local contact section.
        # Do not project displaced pad centroids into a plane and call that a
        # measured gap. The small alignment allowance only constructs a rigid
        # TCP basis; neither 3D point nor the reported separation is changed.
        if abs(np.dot(closing / aperture, z)) > .05:
            raise ValueError('pad landmarks do not define an opening perpendicular to the approach axis')
        x = closing - z * np.dot(closing, z); x /= np.linalg.norm(x)
        t = np.eye(4); t[:3, :3] = np.column_stack((x, np.cross(z, x), z)); t[:3, 3] = (fixed + moving) / 2
        return JawGeometry(RigidTransform('gripper_link', 'grasp_tcp', t, self.calibration_revision),
                           aperture, self.position_uncertainty_m, self.status == 'validated',
                           self.calibration_revision, self.fingerprint, raw, self.geometry_revision,
                           self.landmark_definition, self.validation_kind, self.model_assisted_json)


@dataclass(frozen=True)
class TimingEvidence:
    mode: str
    bracket_monotonic: tuple | None = None
    joint_span_ticks: int | None = None
    timestamp_basis: str = 'unspecified'
    exposure_time_verified: bool = False

    @property
    def pose_bound(self):
        return self.mode == 'settled-readback-bracket'

    def to_dict(self):
        return {'mode': self.mode, 'pose_bound': self.pose_bound,
                'bracket_monotonic': list(self.bracket_monotonic) if self.bracket_monotonic else None,
                'joint_span_ticks': self.joint_span_ticks, 'timestamp_basis': self.timestamp_basis,
                'exposure_time_verified': self.exposure_time_verified,
                'moving_frame_fusion_verified': False}


def _timing(frame, joints):
    basis = _text(frame.get('timestamp_basis', 'unspecified'), 'image timestamp basis')
    bracket = frame.get('joint_observation_bracket_monotonic')
    if bracket is None:
        return TimingEvidence('unbound', timestamp_basis=basis)
    if len(bracket) != 2:
        raise ValueError('joint readback bracket must have start/end timestamps')
    start, end = (_number(v, 'joint timestamp', minimum=0) for v in bracket)
    if not start <= frame['capture_monotonic'] <= end:
        raise ValueError('image arrival is outside its measured joint bracket')
    window = frame.get('settled_joint_window', [])
    spans = (frame.get('joint_capture_span_ticks'), frame.get('image_wait_joint_span_ticks'))
    if any(type(v) is not int or v < 0 for v in spans):
        raise ValueError('settled readback requires measured joint spans')
    bound = len(window) >= 2 and max(spans) <= 2
    for sample in window:
        raw = sample.get('joints', {})
        if set(raw) != set(JOINTS) or any(type(raw[n]) is not int for n in JOINTS):
            raise ValueError('settled window requires actual six-joint readings')
        lo = _number(sample.get('read_start_monotonic'), 'joint read start', minimum=0)
        hi = _number(sample.get('read_end_monotonic'), 'joint read end', minimum=0)
        if hi < lo or hi > start:
            raise ValueError('settled window must precede the image readback bracket')
        bound = bound and all(abs(raw[n] - joints[n]) <= 2 for n in JOINTS)
    if window:
        bound = bound and all(max(s['joints'][n] for s in window) - min(s['joints'][n] for s in window) <= 2 for n in JOINTS)
    return TimingEvidence('settled-readback-bracket' if bound else 'unbound',
                          (start, end), max(spans), basis)


@dataclass(frozen=True)
class WristObservation:
    frame_id: str
    camera_generation: str
    sequence: int
    capture_monotonic: float
    measured_joints: tuple
    calibration: WristCalibration
    processing: ImageProcessing
    T_base_camera: RigidTransform
    timing: TimingEvidence
    camera_matrix: tuple
    distortion: tuple
    jaw_geometry: JawGeometry | None = None
    rectified: bool = False
    scope_reserve_json: str | None = None

    @property
    def metric_calibrated(self):
        scope = self.calibration.pose_scope(dict(self.measured_joints))
        return (self.calibration.metric_calibrated and self.timing.pose_bound
                and scope['accepted']
                and (scope['inside_measured_bounds'] or self.scope_reserve_json is not None))

    @property
    def compatibility_key(self):
        return (SCHEMA_VERSION, self.calibration.camera_id, self.camera_generation,
                self.calibration.geometry_revision, self.calibration.fingerprint)

    def to_dict(self):
        c = self.calibration
        scope = c.pose_scope(dict(self.measured_joints))
        in_bounds = scope['inside_measured_bounds']
        reserve = json.loads(self.scope_reserve_json) if self.scope_reserve_json else None
        from .calibration_domain import effective_uncertainty
        return {'schema_version': SCHEMA_VERSION, 'source': 'wrist', 'camera_id': c.camera_id,
                'camera_generation': self.camera_generation, 'frame_id': self.frame_id,
                'sequence': self.sequence, 'capture_monotonic': self.capture_monotonic,
                'measured_joints': dict(self.measured_joints), 'geometry_revision': c.geometry_revision,
                'calibration_revision': c.calibration_revision, 'calibration_fingerprint': c.fingerprint,
                'calibration_validation_id': c.validation_id, 'calibration_validation_kind': c.validation_kind,
                'mount_revision': c.mount_revision,
                'validation_scope': json.loads(c.validation_scope_json) if c.validation_scope_json else None,
                'inside_validation_joint_bounds': in_bounds,
                'validation_scope_status': scope['status'],
                'validation_scope_reason': '' if in_bounds else
                    ('位置只略超出原取樣邊界，已另計入誤差。' if reserve else
                     '位置略超出原取樣邊界，尚未計算邊界誤差。') if scope['accepted'] else
                    '目前手臂姿勢超出棋盤校準涵蓋的範圍；請補拍這個區域的棋盤視角來擴充校準。',
                'calibration_scope_policy': scope,
                'calibration_boundary_reserve': reserve,
                'calibration_uncertainty': effective_uncertainty(json.loads(c.uncertainty_json), [reserve] if reserve else []),
                'metric_calibrated': self.metric_calibrated,
                'camera_model_status': 'validated' if c.metric_calibrated else 'prior',
                'timing': self.timing.to_dict(), 'image_size': list(self.processing.output_size),
                'native_image_size': list(self.processing.native_size), 'processing': self.processing.to_dict(),
                'camera_matrix': [list(r) for r in self.camera_matrix], 'distortion': list(self.distortion),
                'distortion_model': 'pinhole' if self.rectified else 'opencv-brown',
                'T_base_camera': self.T_base_camera.to_dict(),
                'jaw_geometry': self.jaw_geometry.to_dict() if self.jaw_geometry else None}

    def unproject(self, pixels, depth_z_m):
        pixels = np.asarray(pixels, dtype=float)
        if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.isfinite(pixels).all():
            raise ValueError('pixels must be an Nx2 finite array in the processed image')
        depths = np.broadcast_to(np.asarray(depth_z_m, dtype=float), (len(pixels),))
        if not np.isfinite(depths).all() or np.any(depths <= 0):
            raise ValueError('depth is positive optical Z in metres, not radial range')
        normalized = cv2.undistortPoints(pixels.reshape(-1, 1, 2), np.array(self.camera_matrix),
                                         np.array(self.distortion)).reshape(-1, 2)
        return self.T_base_camera.apply(np.column_stack((normalized, np.ones(len(pixels)))) * depths[:, None])

    def project(self, base_points):
        local = self.T_base_camera.inverse().apply(base_points)
        if local.ndim != 2 or local.shape[1] != 3 or np.any(local[:, 2] <= 0):
            raise ValueError('projection needs Nx3 points in front of the camera')
        return cv2.projectPoints(local, np.zeros(3), np.zeros(3), np.array(self.camera_matrix),
                                 np.array(self.distortion))[0].reshape(-1, 2)


def build_wrist_observation(frame, transforms, *, geometry_revision, calibration=None,
                            camera_id='wrist', mount_revision='unverified-mount', processing=None,
                            jaw_calibration=None):
    """Bind a measured frame to explicit geometry, without making unknowns true."""
    _text(geometry_revision, 'geometry revision')
    if frame.get('source') != 'wrist':
        raise ValueError('spatial observations only accept the wrist camera, never D435')
    if frame.get('camera_id', camera_id) != camera_id:
        raise ValueError('captured wrist device differs from the selected calibration camera')
    if frame.get('mount_revision', mount_revision) != mount_revision:
        raise ValueError('captured wrist mount differs from the selected calibration mount')
    generation = _text(frame.get('camera_generation'), 'camera generation')
    if type(frame.get('sequence')) is not int or frame['sequence'] <= 0:
        raise ValueError('wrist sequence must be a positive integer')
    stamp = _number(frame.get('capture_monotonic'), 'image arrival timestamp', minimum=0)
    joints = frame.get('joints', {})
    if (set(joints) != set(JOINTS) or any(type(v) is not int or not 0 <= v <= 4095 for v in joints.values())):
        raise ValueError('spatial pose requires all six actual encoder readings')
    size = _size(frame.get('image_size', ()))
    processing = processing or ImageProcessing(size, size)
    if processing.native_size != size:
        raise ValueError('pixel processing was defined for a different acquired image')
    wrist = RigidTransform('base', 'gripper_link', transforms['gripper_link'], geometry_revision)
    if not isinstance(calibration, WristCalibration):
        optical = RigidTransform('base', 'wrist_camera_optical', transforms['wrist_camera_optical'], geometry_revision)
        cad = wrist.inverse().compose(optical).as_array().round(12)
        # FK inversion can produce either sign of numerical zero at different
        # poses. JSON distinguishes -0.0 from 0.0, but a fixed camera mount does
        # not change calibration when those signs change. Keep nonzero entries
        # and genuine mount changes in the content fingerprint.
        cad[cad == 0.] = 0.
        calibration = WristCalibration.from_mapping(calibration, native_image_size=size,
                           camera_id=camera_id, geometry_revision=geometry_revision, cad_handeye=cad,
                           mount_revision=mount_revision)
    if (calibration.geometry_revision != geometry_revision or calibration.camera_id != camera_id
            or calibration.native_image_size != size or calibration.mount_revision != mount_revision):
        raise ValueError('observation and calibration versions/camera/resolution are incompatible')
    if calibration.camera_generation is not None and calibration.camera_generation != generation:
        raise ValueError('calibration is bound to a different camera session')
    if frame.get('geometry_revision', geometry_revision) != geometry_revision:
        raise ValueError('frame was captured under another geometry revision')
    jaw = None
    if jaw_calibration is not None:
        if jaw_calibration.geometry_revision != geometry_revision:
            raise ValueError('jaw calibration and measured geometry revisions differ')
        if (jaw_calibration.model_assisted_json is not None and
                json.loads(jaw_calibration.model_assisted_json)['camera_calibration_fingerprint'] != calibration.fingerprint):
            raise ValueError('model-assisted jaw was fitted with a different wrist camera calibration')
        jaw = jaw_calibration.at(joints['gripper'])
    return WristObservation(_text(frame.get('id'), 'frame ID'), generation, frame['sequence'], stamp,
               tuple((n, joints[n]) for n in JOINTS), calibration, processing,
               wrist.compose(calibration.T_wrist_camera), _timing(frame, joints),
               _tuple_matrix(processing.camera_matrix(calibration.camera_matrix)), calibration.distortion, jaw)


def require_compatible(observations):
    """Reject stale sessions/revisions and out-of-order evidence before fusion."""
    if not observations:
        raise ValueError('need at least one wrist observation')
    first = observations[0]
    for before, after in zip(observations, observations[1:]):
        if after.compatibility_key != first.compatibility_key:
            raise ValueError('cannot fuse different camera sessions or calibration/geometry revisions')
        if after.sequence <= before.sequence or after.capture_monotonic <= before.capture_monotonic:
            raise ValueError('wrist evidence must have increasing sequence and timestamp')


def rectify_image(image, observation):
    """Native Brown image -> cropped/resized undistorted pinhole image and K.

    The returned observation retains the original calibration provenance. Brown
    coefficients are consumed here, never passed to a model as KB4 parameters.
    """
    if observation.rectified:
        raise ValueError('image has already been rectified')
    processed = observation.processing.apply(image)
    k = np.array(observation.camera_matrix)
    rectified = cv2.undistort(processed, k, np.array(observation.distortion), None, k)
    return rectified, replace(observation, distortion=(0.,) * 5, rectified=True)
