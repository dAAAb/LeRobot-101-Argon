"""Offline measured two-pad calibration; no devices or automatic activation.

Evidence hashes and independent acquisitions are checked, but a file hash is
not proof that a physical measurement was performed correctly. The operator
must survey both landmarks in gripper_link, not copy a CAD prior or two pixels.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .spatial_contract import JawCalibration, JawSample


class _Record(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)


class Evidence(_Record):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


class PadMeasurement(_Record):
    measurement_id: str = Field(min_length=1, max_length=120)
    acquisition_id: str = Field(min_length=1, max_length=120)
    split: Literal['training', 'heldout']
    raw: int = Field(ge=0, le=4095)
    fixed_pad_m: list[float] = Field(min_length=3, max_length=3)
    moving_pad_m: list[float] = Field(min_length=3, max_length=3)
    approach_axis: list[float] = Field(min_length=3, max_length=3)
    position_uncertainty_m: float = Field(gt=0, le=.02)
    axis_uncertainty_deg: float = Field(gt=0, le=5)
    evidence: list[Evidence] = Field(min_length=1, max_length=8)


class JawMeasurements(_Record):
    schema_version: Literal[1]
    source: Literal['wrist-only']
    measurement_origin: Literal['physical-measurement']
    measurement_method: Literal['mechanical-survey', 'calibrated-wrist-3d']
    coordinate_frame: Literal['gripper_link']
    landmark_definition: Literal['matched_contact_cross_section']
    units: Literal['m']
    geometry_revision: str = Field(pattern=r'^[0-9a-f]{64}$')
    jaw_mount_revision: str = Field(min_length=1, max_length=120)
    registration_description: str = Field(min_length=20, max_length=2000)
    registration_evidence: list[Evidence] = Field(min_length=1, max_length=8)
    wrist_calibration_fingerprint: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    measurements: list[PadMeasurement] = Field(min_length=5, max_length=200)


@dataclass(frozen=True)
class ValidationLimits:
    minimum_training: int = 3
    minimum_heldout: int = 2
    minimum_aperture_span_m: float = .005
    maximum_pad_error_m: float = .003
    maximum_tcp_error_m: float = .003
    maximum_aperture_error_m: float = .004
    maximum_axis_error_deg: float = 3.


LIMITS = ValidationLimits()


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _angle(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1., 1.))))


def _unit(value):
    axis = np.asarray(value, float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or abs(np.linalg.norm(axis) - 1.) > 1e-6:
        raise ValueError('Measured approach axis must be a unit 3D vector')
    return axis


def _verify_evidence(items, root):
    manifest = []
    for item in items:
        relative = Path(item.path)
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Evidence must be an existing file inside the measurement directory')
        if path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError('Evidence file exceeds 64 MiB')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != item.sha256:
            raise ValueError('Evidence fingerprint mismatch: ' + item.path)
        manifest.append({'path': item.path, 'sha256': digest})
    return manifest


def fit_jaw_calibration(data, *, evidence_root, expected_geometry_revision,
                        expected_wrist_calibration_fingerprint=None):
    """Validate declared measurements and return a report with an optional profile.

    Schema/provenance errors raise ValueError. Numerical validation failures
    return passed=False and calibration_profile=None. Training landmarks alone
    define the piecewise-linear model; held-out landmarks only evaluate it.
    """
    batch = JawMeasurements.model_validate(data)
    if batch.geometry_revision != expected_geometry_revision:
        raise ValueError('Measured geometry fingerprint differs from the current robot model')
    if batch.measurement_method == 'calibrated-wrist-3d' and (
            not expected_wrist_calibration_fingerprint
            or batch.wrist_calibration_fingerprint != expected_wrist_calibration_fingerprint):
        raise ValueError('Wrist 3D measurements require the matching validated calibration fingerprint')
    root = Path(evidence_root).resolve()
    registration = _verify_evidence(batch.registration_evidence, root)
    measurements = batch.measurements
    for key in ('measurement_id', 'acquisition_id'):
        if len({getattr(s, key) for s in measurements}) != len(measurements):
            raise ValueError('Each measurement requires an independent unique ' + key)
    training = sorted((s for s in measurements if s.split == 'training'), key=lambda s: s.raw)
    heldout = sorted((s for s in measurements if s.split == 'heldout'), key=lambda s: s.raw)
    if len(training) < LIMITS.minimum_training or len(heldout) < LIMITS.minimum_heldout:
        raise ValueError('At least three training and two independent held-out measurements are required')
    if len({s.raw for s in measurements}) != len(measurements):
        raise ValueError('Use distinct training and held-out encoder openings')
    if any(not training[0].raw < s.raw < training[-1].raw for s in heldout):
        raise ValueError('Held-out measurements must lie inside the measured training range')
    intervals = {next(i for i, (a, b) in enumerate(zip(training, training[1:]))
                      if a.raw < s.raw < b.raw) for s in heldout}
    if len(intervals) < 2:
        raise ValueError('Held-out measurements must cover at least two interpolation intervals')
    manifest = {s.measurement_id: _verify_evidence(s.evidence, root) for s in measurements}
    train_hashes = {f['sha256'] for s in training for f in manifest[s.measurement_id]}
    if any(f['sha256'] in train_hashes for s in heldout for f in manifest[s.measurement_id]):
        raise ValueError('Held-out evidence must not reuse training captures')
    heldout_hashes = set()
    for sample in heldout:
        hashes = {f['sha256'] for f in manifest[sample.measurement_id]}
        if hashes & heldout_hashes:
            raise ValueError('Independent held-out measurements must not reuse the same capture')
        heldout_hashes.update(hashes)
    for s in measurements:
        _unit(s.approach_axis)
        if np.max(np.abs([s.fixed_pad_m, s.moving_pad_m])) > .5:
            raise ValueError('Pad coordinates exceed this small gripper workspace; check meters/frame')
        # Validate each individually measured opening with the same production
        # axis convention; near-parallel or coincident pads cannot define TCP.
        closing = np.asarray(s.moving_pad_m) - s.fixed_pad_m
        aperture = float(np.linalg.norm(closing))
        if not .001 <= aperture <= .2 or abs(float(closing @ s.approach_axis) / aperture) > .05:
            raise ValueError('Measured pads must define a nonzero transverse opening')
    canonical = batch.model_dump()
    input_hash = _hash(canonical)
    revision = 'jaw-' + input_hash[:24]
    prior = JawCalibration(revision, batch.geometry_revision,
        tuple(JawSample(s.raw, tuple(s.fixed_pad_m), tuple(s.moving_pad_m), tuple(s.approach_axis)) for s in training),
        tuple(training[0].approach_axis), landmark_definition=batch.landmark_definition)
    # The production interpolation must remain defined for EVERY integer raw
    # opening in this range, not just the knots and validation points.
    apertures = [prior.at(raw).aperture_m for raw in range(training[0].raw, training[-1].raw + 1)]
    if max(apertures) - min(apertures) < LIMITS.minimum_aperture_span_m:
        raise ValueError('Measured aperture range is too narrow to calibrate opening')
    train_uncertainty = max(s.position_uncertainty_m for s in training)
    train_axis_uncertainty = max(s.axis_uncertainty_deg for s in training)
    rows = []
    for s in heldout:
        i = next(i for i, (a, b) in enumerate(zip(training, training[1:])) if a.raw < s.raw < b.raw)
        a, b = training[i:i + 2]
        fraction = (s.raw - a.raw) / (b.raw - a.raw)
        fixed = (1 - fraction) * np.asarray(a.fixed_pad_m) + fraction * np.asarray(b.fixed_pad_m)
        moving = (1 - fraction) * np.asarray(a.moving_pad_m) + fraction * np.asarray(b.moving_pad_m)
        actual_fixed, actual_moving = np.asarray(s.fixed_pad_m), np.asarray(s.moving_pad_m)
        delta = actual_moving - actual_fixed
        pad_error = max(float(np.linalg.norm(fixed - actual_fixed)), float(np.linalg.norm(moving - actual_moving)))
        tcp_error = float(np.linalg.norm((fixed + moving - actual_fixed - actual_moving) / 2))
        aperture_error = abs(prior.at(s.raw).aperture_m - float(np.linalg.norm(delta)))
        predicted_axis = prior.at(s.raw).T_wrist_tcp.as_array()[:3, 2]
        position_noise = train_uncertainty + s.position_uncertainty_m
        rows.append({'measurement_id': s.measurement_id, 'raw': s.raw, 'training_interval': i,
            'pad_residual_m': pad_error, 'tcp_residual_m': tcp_error, 'aperture_residual_m': aperture_error,
            'pad_error_envelope_m': pad_error + position_noise,
            'tcp_error_envelope_m': tcp_error + position_noise,
            'aperture_error_envelope_m': aperture_error + 2 * position_noise,
            'approach_axis_error_deg': _angle(predicted_axis, s.approach_axis) + train_axis_uncertainty + s.axis_uncertainty_deg,
            'closing_axis_residual_deg': _angle((moving - fixed) / np.linalg.norm(moving - fixed), delta / np.linalg.norm(delta))})
    maxima = {key: max(row[key] for row in rows) for key in rows[0]
              if key not in ('measurement_id', 'raw', 'training_interval')}
    blockers = []
    for key, limit in (('pad_error_envelope_m', LIMITS.maximum_pad_error_m),
                       ('tcp_error_envelope_m', LIMITS.maximum_tcp_error_m),
                       ('aperture_error_envelope_m', LIMITS.maximum_aperture_error_m),
                       ('approach_axis_error_deg', LIMITS.maximum_axis_error_deg),
                       ('closing_axis_residual_deg', LIMITS.maximum_axis_error_deg)):
        if maxima[key] > limit:
            blockers.append(key + ' exceeds validation limit')
    report = {'schema_version': 1, 'source': 'wrist-only', 'input_sha256': input_hash,
        'geometry_revision': batch.geometry_revision, 'jaw_mount_revision': batch.jaw_mount_revision,
        'passed': not blockers, 'blockers': blockers, 'limits': vars(LIMITS),
        'training_ids': [s.measurement_id for s in training], 'heldout_ids': [s.measurement_id for s in heldout],
        'raw_range': [training[0].raw, training[-1].raw], 'heldout_intervals': sorted(intervals),
        'registration_evidence': registration, 'measurement_evidence': manifest,
        'validation': {'heldout': rows, 'maxima': maxima},
        'interpretation': 'Independent declared measured acquisitions, with verified file hashes. Empirical errors in the sampled range only; not proof of evidence authenticity, contact force, collision clearance, or grasp success.',
        'calibration_profile': None, 'profile_fingerprint': None, 'automatically_activated': False}
    validation_id = 'jaw-holdout-' + _hash(report)
    if not blockers:
        profile = {'schema_version': 1, 'source': 'wrist-only', 'calibration_revision': revision,
            'geometry_revision': batch.geometry_revision, 'status': 'validated',
            'validation_kind': 'independent-heldout', 'validation_id': validation_id,
            'samples': [{'raw': s.raw, 'fixed_pad_m': s.fixed_pad_m, 'moving_pad_m': s.moving_pad_m,
                         'approach_axis': s.approach_axis} for s in training],
            'approach_axis': list(training[0].approach_axis),
            'landmark_definition': batch.landmark_definition,
            # One per-pad conservative envelope also covers center position and
            # half-aperture error consumed by the existing grasp TCP contract.
            'position_uncertainty_m': max(maxima['pad_error_envelope_m'], maxima['tcp_error_envelope_m'],
                                          maxima['aperture_error_envelope_m'] / 2),
            'measurement_provenance': {'input_sha256': input_hash,
                'jaw_mount_revision': batch.jaw_mount_revision, 'method': batch.measurement_method,
                'wrist_calibration_fingerprint': batch.wrist_calibration_fingerprint,
                'registration_description': batch.registration_description},
            'validation_scope': {'raw_range': report['raw_range'], 'extrapolation_allowed': False,
                'interpretation': report['interpretation']}}
        loaded = JawCalibration.from_mapping(profile, geometry_revision=expected_geometry_revision)
        report.update(calibration_profile=profile, profile_fingerprint=loaded.fingerprint)
    return report
