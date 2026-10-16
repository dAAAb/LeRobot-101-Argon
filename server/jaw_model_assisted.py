"""Wrist-silhouette/CAD evidence and conditional camera-relative uncertainty.

These estimates are not independent 3D pad measurements. This module does not
choose acceptance tolerances, activate profiles, or add camera/FK base error to
the camera-relative jaw gap. CAD/hinge tolerances may be explicit engineering
assumptions; an external mechanical survey is not required by this method.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .calibration_store import _validate_frame
from .gripper_context import BODY, require_matched_body
from .spatial_contract import RigidTransform, WristCalibration

METHOD = 'articulated-cad-wrist-silhouette'
VALIDATION_KIND = 'independent-silhouette-heldout'
DEFAULT_PROTOCOL = dict(name='empty-jaw-five-openings-v1',
    targets_raw=[1550, 1625, 1700, 1775, 1850], training_targets_raw=[1550, 1700, 1850],
    heldout_targets_raw=[1625, 1775])
PARAMETER_UNITS = ('degrees',) * 3 + ('millimeters',) * 3 + ('degrees',)


def _finite(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite with shape {shape}')
    return result


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_silhouette_evidence(manifest_path, annotations_path, camera_profile, *,
                                 expected_geometry_revision, cad_files):
    """Bind real openings and image annotations before a train-only fit.

    Returns immutable provenance and the predeclared 3/2 split. This verifies
    provenance/pose pairing, not the correctness of a human-drawn contour.
    """
    manifest_path, annotations_path = Path(manifest_path).resolve(), Path(annotations_path).resolve()
    manifest = json.loads(manifest_path.read_text()); annotations = json.loads(annotations_path.read_text())
    camera = WristCalibration.from_mapping(camera_profile,
        native_image_size=manifest['image_size'], camera_id=manifest['camera_id'],
        mount_revision=manifest['mount_revision'], geometry_revision=expected_geometry_revision,
        cad_handeye=np.eye(4))
    if (manifest.get('source') != 'wrist-only' or manifest.get('completed') is not True
            or manifest.get('empty_verified') is not True or not camera.metric_calibrated
            or manifest.get('geometry_revision') != expected_geometry_revision):
        raise ValueError('Need completed empty-jaw wrist evidence and the matching calibrated camera')
    profile_bytes = json.dumps(camera_profile, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if (manifest.get('camera_profile_sha256') != hashlib.sha256(profile_bytes).hexdigest()
            or manifest.get('camera_profile') != camera_profile):
        raise ValueError('Acquisition camera snapshot differs from the selected optical model')
    protocol = manifest.get('opening_protocol', DEFAULT_PROTOCOL)
    targets, training_targets, held_targets = (protocol[n] for n in
        ('targets_raw', 'training_targets_raw', 'heldout_targets_raw'))
    if (not isinstance(protocol.get('name'), str) or not protocol['name']
            or any(not isinstance(values, list) or any(type(v) is not int or not 0 <= v <= 4095 for v in values)
                   or len(set(values)) != len(values) for values in (targets, training_targets, held_targets))
            or len(targets) != 5 or len(training_targets) != 3 or len(held_targets) != 2
            or set(training_targets) & set(held_targets) or set(targets) != set(training_targets) | set(held_targets)):
        raise ValueError('Capture protocol must predeclare five targets with disjoint 3/2 groups')
    frames = manifest['frames']
    if len(frames) != 5 or {f['target_raw'] for f in frames} != set(targets):
        raise ValueError('Preserve the five predeclared opening groups; do not resplit after fitting')
    seen_ids, seen_images, seen_raw = set(), set(), set()
    before = None
    for frame in frames:
        _validate_frame(frame)
        if (frame['camera_id'] != camera.camera_id or frame['camera_generation'] != manifest['camera_generation']
                or frame['mount_revision'] != camera.mount_revision
                or frame['geometry_revision'] != expected_geometry_revision
                or tuple(frame['image_size']) != camera.native_image_size):
            raise ValueError('Frame camera/mount/geometry/resolution differs from acquisition')
        raw = frame['actual_gripper_raw']
        if type(raw) is not int or raw != frame['joints']['gripper']:
            raise ValueError('Articulation requires actual encoder values, never target labels')
        if frame['id'] in seen_ids or frame['image_sha256'] in seen_images or raw in seen_raw:
            raise ValueError('Training and held-out openings need distinct acquisitions and measured raw values')
        seen_ids.add(frame['id']); seen_images.add(frame['image_sha256']); seen_raw.add(raw)
        require_matched_body({n: frame['joints'][n] for n in BODY}, manifest['body_reference'])
        if before and (frame['sequence'] <= before['sequence']
                       or frame['capture_monotonic'] <= before['capture_monotonic']):
            raise ValueError('Acquisition frames must remain in camera order')
        before = frame
        path = Path(frame['path']).resolve()
        if not path.is_relative_to(manifest_path.parent / 'frames') or _digest(path) != frame['image_sha256']:
            raise ValueError('Image path or hash differs from the immutable capture')
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or list(image.shape[1::-1]) != frame['image_size']:
            raise ValueError('Image does not decode at its native resolution')
        entry = annotations['frames'][frame['id']]
        if entry['image_sha256'] != frame['image_sha256'] or entry['target_raw'] != frame['target_raw']:
            raise ValueError('Annotation is not bound to this captured opening')
        groups = entry['groups']
        if not groups or {g['part'] for g in groups.values()} != {'fixed', 'moving'}:
            raise ValueError('Both physical jaw silhouettes must contribute visible annotations')
        for group in groups.values():
            if not group['polylines']:
                raise ValueError('Empty silhouette annotation')
            for line in group['polylines']:
                pixels = np.asarray(line, float)
                if (pixels.ndim != 2 or pixels.shape[1] != 2 or len(pixels) < 2
                        or not np.isfinite(pixels).all() or np.any(pixels < 0)
                        or np.any(pixels >= np.asarray(frame['image_size']))):
                    raise ValueError('Contour coordinates must be finite native-image pixels')
    training = [f for f in frames if f['target_raw'] in training_targets]
    held = [f for f in frames if f['target_raw'] in held_targets]
    knots = sorted(f['actual_gripper_raw'] for f in training)
    intervals = {next((i for i, (a, b) in enumerate(zip(knots, knots[1:]))
                       if a < f['actual_gripper_raw'] < b), None) for f in held}
    if intervals != {0, 1}:
        raise ValueError('Held-out actual openings must span both measured training intervals')
    if not cad_files:
        raise ValueError('Nominal CAD/hinge sources must be identified')
    provenance = dict(geometry_method=METHOD, validation_kind=VALIDATION_KIND,
        source='wrist-only', camera_calibration_fingerprint=camera.fingerprint,
        camera_profile_sha256=hashlib.sha256(profile_bytes).hexdigest(),
        manifest=dict(path=str(manifest_path), sha256=_digest(manifest_path)),
        annotations=dict(path=str(annotations_path), sha256=_digest(annotations_path)),
        cad_files=[dict(path=str(Path(p).resolve()), sha256=_digest(p)) for p in cad_files],
        training_frame_ids=[f['id'] for f in training], heldout_frame_ids=[f['id'] for f in held],
        opening_protocol=deepcopy(protocol),
        measured_raw_range=[min(seen_raw), max(seen_raw)],
        heldout_meaning='Whole encoder openings test articulation projection. Fixed camera/body pose does not independently validate camera depth or hidden contact surfaces.',
        external_metric_measurement=False, activation_performed=False)
    return provenance


@dataclass(frozen=True)
class ArticulatedJawModel:
    """Nominal CAD material landmarks under the fitter's seven parameters.

    These rotate with the part; they need not stay in one matched contact
    cross-section. A contact-section adapter must recompute mesh intersections
    for each raw/parameter perturbation before its width can be used for grasp.
    This small model is useful for registration/hinge sensitivity, not a profile.
    """
    camera_from_cad_prior: object
    hinge_transform: object
    hinge_axis: object
    fixed_pad_cad_m: object
    moving_pad_link_m: object
    encoder_sign: int
    reference_raw: int = 1700

    def __post_init__(self):
        for name in ('camera_from_cad_prior', 'hinge_transform'):
            RigidTransform('parent', 'child', getattr(self, name), 'nominal-cad')
        axis = _finite(self.hinge_axis, (3,), 'hinge axis')
        if abs(np.linalg.norm(axis) - 1.) > 1e-6 or type(self.encoder_sign) is not int or self.encoder_sign not in (-1, 1):
            raise ValueError('Need a unit CAD hinge axis and train-selected encoder sign')
        _finite(self.fixed_pad_cad_m, (3,), 'nominal fixed pad')
        _finite(self.moving_pad_link_m, (3,), 'nominal moving pad')
        if type(self.reference_raw) is not int or not 0 <= self.reference_raw <= 4095:
            raise ValueError('Invalid reference encoder')

    def points(self, parameters, raw):
        p = _finite(parameters, (7,), 'registration/zero-angle parameters')
        if type(raw) not in (int, float) or not math.isfinite(raw) or not 0 <= raw <= 4095:
            raise ValueError('Invalid actual encoder')
        delta = np.eye(4); delta[:3, :3] = cv2.Rodrigues(np.radians(p[:3]))[0]; delta[:3, 3] = p[3:6] / 1000
        camera = delta @ np.asarray(self.camera_from_cad_prior)
        angle = p[6] + self.encoder_sign * (raw - self.reference_raw) * 360 / 4096
        rotation = np.eye(4); rotation[:3, :3] = cv2.Rodrigues(np.asarray(self.hinge_axis) * math.radians(angle))[0]
        moving = camera @ np.asarray(self.hinge_transform) @ rotation
        return np.array([camera[:3, :3] @ self.fixed_pad_cad_m + camera[:3, 3],
                         moving[:3, :3] @ self.moving_pad_link_m + moving[:3, 3]])


@dataclass(frozen=True)
class MatchedSectionJawModel:
    """Facing CAD mesh intersections at a fixed gripper-CAD y/z section."""
    camera_from_cad_prior: object
    hinge_transform: object
    hinge_axis: object
    fixed_vertices: object
    fixed_faces: object
    moving_vertices: object
    moving_faces: object
    section_yz_m: object
    encoder_sign: int
    reference_raw: int = 1700

    def __post_init__(self):
        # Reuse the common articulation/SE3 validation without calling its
        # material-point geometry as a contact section.
        ArticulatedJawModel(self.camera_from_cad_prior, self.hinge_transform,
            self.hinge_axis, [0., 0., 0.], [0., 0., 0.], self.encoder_sign, self.reference_raw)
        _finite(self.section_yz_m, (2,), 'section CAD y/z')
        object.__setattr__(self, '_section_cache', {})
        for name in ('fixed', 'moving'):
            vertices = np.asarray(getattr(self, name + '_vertices'), float)
            faces = np.asarray(getattr(self, name + '_faces'))
            if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
                    or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
                    or not np.issubdtype(faces.dtype, np.integer) or np.any(faces < 0) or np.any(faces >= len(vertices))):
                raise ValueError('Need finite nominal CAD triangles for both jaws')
        for name in ('camera_from_cad_prior', 'hinge_transform', 'hinge_axis', 'fixed_vertices',
                     'fixed_faces', 'moving_vertices', 'moving_faces', 'section_yz_m'):
            value = np.array(getattr(self, name), copy=True); value.setflags(write=False)
            object.__setattr__(self, name, value)

    def points(self, parameters, raw):
        p = _finite(parameters, (7,), 'articulation parameters')
        if type(raw) not in (int, float) or not math.isfinite(raw) or not 0 <= raw <= 4095:
            raise ValueError('Invalid encoder for contact section')
        angle = p[6] + self.encoder_sign * (raw - self.reference_raw) * 360 / 4096
        if angle in self._section_cache:
            camera = self.camera_transform(p)
            return self._section_cache[angle] @ camera[:3, :3].T + camera[:3, 3]
        rotation = np.eye(4)
        rotation[:3, :3] = cv2.Rodrigues(np.asarray(self.hinge_axis) * math.radians(angle))[0]
        moving_transform = np.asarray(self.hinge_transform) @ rotation
        moving = np.einsum('ij,kj->ki', moving_transform[:3, :3], np.asarray(self.moving_vertices),
                           optimize=False) + moving_transform[:3, 3]
        y, z = self.section_yz_m
        def hits(vertices, faces):
            triangles = np.asarray(vertices)[np.asarray(faces)]
            a, u, v = triangles[:, 0], triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            determinant = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
            good = np.abs(determinant) > 1e-14
            beta, gamma = np.zeros(len(a)), np.zeros(len(a))
            beta[good] = ((y-a[good, 1])*v[good, 2]-(z-a[good, 2])*v[good, 1])/determinant[good]
            gamma[good] = (u[good, 1]*(z-a[good, 2])-u[good, 2]*(y-a[good, 1]))/determinant[good]
            good &= (beta >= -1e-7) & (gamma >= -1e-7) & (beta + gamma <= 1 + 1e-7)
            return (a[:, 0] + beta*u[:, 0] + gamma*v[:, 0])[good]
        left, right = hits(self.fixed_vertices, self.fixed_faces), hits(moving, self.moving_faces)
        if not len(left) or not len(right) or min(right) <= max(left):
            raise ValueError('Both facing CAD surfaces must intersect the same nonoverlapping section')
        points = np.array([[max(left), y, z], [min(right), y, z]])
        self._section_cache[angle] = points
        camera = self.camera_transform(p)
        return points @ camera[:3, :3].T + camera[:3, 3]

    def camera_transform(self, parameters):
        p = _finite(parameters, (7,), 'articulation parameters')
        delta = np.eye(4); delta[:3, :3] = cv2.Rodrigues(np.radians(p[:3]))[0]; delta[:3, 3] = p[3:6]/1000
        return delta @ np.asarray(self.camera_from_cad_prior)

    def approach_camera(self, parameters):
        return self.camera_transform(parameters)[:3, :3] @ np.array([0., 0., -1.])


def residual_covariance(jacobian, residuals):
    """Conditional local covariance; correlated contour pixels are not truth.

    Full numerical rank is necessary. No hand-picked condition-number cutoff is
    used; poor conditioning increases the propagated uncertainty instead.
    """
    j, r = np.asarray(jacobian, float), np.asarray(residuals, float)
    if (j.ndim != 2 or r.shape != (len(j),) or j.shape[1] != 7
            or len(j) <= 7 or not np.isfinite(j).all() or not np.isfinite(r).all()):
        raise ValueError('Need finite training residuals and their seven-parameter Jacobian')
    scales = np.linalg.norm(j, axis=0)
    normalized = j / np.where(scales > 0, scales, 1.)
    rank = int(np.linalg.matrix_rank(normalized))
    singular = np.linalg.svd(normalized, compute_uv=False)
    result = dict(parameter_rank=rank, normalized_singular_values=singular.tolist(),
        identifiable=rank == 7, parameter_units=list(PARAMETER_UNITS), covariance=None,
        interpretation='Conditional training-residual linearization. Contour correspondence, CAD accuracy and correlated pixels require separate declared assumptions/sensitivity checks; this is not external metric accuracy.')
    if rank == 7:
        variance = float(r @ r / (len(r) - 7))
        inverse = np.linalg.pinv(normalized) / scales[:, None]
        covariance = np.einsum('ik,jk->ij', inverse, inverse, optimize=False) * variance
        if not np.isfinite(covariance).all():
            raise ValueError('Conditional covariance is not finite')
        result.update(covariance=covariance.tolist(), residual_variance=variance)
    return result


def propagate_camera_relative_uncertainty(model, parameters, covariance, raw, *,
        sigma_multiplier, cad_pad_tolerance_m, encoder_tolerance_ticks, assumption_description,
        alternative_parameters=(), independent_parameter_families=None):
    """Propagate shared parameter correlation, CAD assumptions and fit spread.

    ``model.points(parameters, raw)`` supplies the relevant two 3D points. A
    matched-contact-section model must update its intersections at EVERY call;
    fixed material landmarks alone do not measure a rotary jaw's usable gap.
    All numeric engineering assumptions are explicit inputs. Shared translation
    affects camera-relative TCP but cancels out of pad separation/aperture. The
    base-camera/FK uncertainty is intentionally not an input to this calculation.
    """
    p = _finite(parameters, (7,), 'parameters'); cov = _finite(covariance, (7, 7), 'covariance')
    if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-14) or np.min(np.linalg.eigvalsh(cov)) < -1e-12 * max(1., np.linalg.norm(cov)):
        raise ValueError('Parameter covariance must be symmetric positive semidefinite')
    values = (sigma_multiplier, cad_pad_tolerance_m, encoder_tolerance_ticks)
    if (any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values)
            or sigma_multiplier <= 0 or not isinstance(assumption_description, str) or len(assumption_description.strip()) < 20):
        raise ValueError('Declare finite nonnegative engineering assumptions and their basis')

    def output(v, encoder=raw):
        fixed, moving = model.points(v, encoder); delta = moving - fixed
        return np.r_[fixed, moving, (fixed + moving) / 2, delta, np.linalg.norm(delta)]

    nominal = output(p)
    steps = np.cbrt(np.finfo(float).eps) * np.maximum(np.abs(p), 1.)
    jac = np.column_stack([(output(p + np.eye(7)[i] * steps[i]) -
                            output(p - np.eye(7)[i] * steps[i])) / (2 * steps[i]) for i in range(7)])
    out_cov = np.einsum('ik,kl,jl->ij', jac, cov, jac, optimize=False)
    if not np.isfinite(out_cov).all():
        raise ValueError('Forward-propagated covariance is not finite')
    slices = dict(fixed_pad=slice(0, 3), moving_pad=slice(3, 6), tcp=slice(6, 9), separation=slice(9, 12))
    parameter_error = {name: float(sigma_multiplier * np.sqrt(max(0., np.linalg.eigvalsh(out_cov[s, s])[-1])))
                       for name, s in slices.items()}
    parameter_error['aperture'] = float(sigma_multiplier * np.sqrt(max(0., out_cov[12, 12])))
    alternatives = [output(_finite(v, (7,), 'alternative fit')) for v in alternative_parameters]
    encoder_samples = [output(p, raw + offset) for offset in (-encoder_tolerance_ticks, encoder_tolerance_ticks)]
    spread, encoder = {}, {}
    for name, section in {**slices, 'aperture': slice(12, 13)}.items():
        spread[name] = max((float(np.linalg.norm(v[section] - nominal[section])) for v in alternatives), default=0.)
        encoder[name] = max(float(np.linalg.norm(v[section] - nominal[section])) for v in encoder_samples)
    independent = {}
    for family, parameters_list in (independent_parameter_families or {}).items():
        predictions = [output(_finite(v, (7,), 'independent sensitivity fit')) for v in parameters_list]
        if not isinstance(family, str) or not family or not predictions:
            raise ValueError('Independent uncertainty families need a name and fitted parameters')
        independent[family] = {name: max(float(np.linalg.norm(v[section] - nominal[section])) for v in predictions)
                              for name, section in {**slices, 'aperture': slice(12, 13)}.items()}
    # The two pad tolerances may disagree in opposite directions. Their midpoint
    # retains one pad tolerance; separation/aperture require twice that amount.
    combined = {n: max(parameter_error[n], spread[n]) + sum(v[n] for v in independent.values())
                + encoder[n] + cad_pad_tolerance_m * (2 if n in ('separation', 'aperture') else 1)
                for n in parameter_error}
    return dict(geometry_method=METHOD, uncertainty_frame='wrist_camera_optical',
        metric_accuracy='model-assisted-estimate', external_metric_measurement=False,
        nominal=dict(fixed_pad_m=nominal[:3].tolist(), moving_pad_m=nominal[3:6].tolist(),
                     tcp_m=nominal[6:9].tolist(), separation_m=nominal[9:12].tolist(), aperture_m=float(nominal[12])),
        parameter_error_m=parameter_error, alternative_fit_spread_m=spread, encoder_error_m=encoder,
        independent_parameter_error_m=independent,
        conditional_error_m=combined,
        assumptions=dict(sigma_multiplier=sigma_multiplier, cad_pad_tolerance_m=cad_pad_tolerance_m,
            encoder_tolerance_ticks=encoder_tolerance_ticks, description=assumption_description),
        interpretation='Local conditional estimate plus explicit engineering CAD/encoder assumptions and alternative-fit spread. Shared registration error is not independent pad noise; base/FK error is separate. This is not a guarantee, contact measurement, or grasp clearance.',
        calibration_profile=None, activation_performed=False)


def validate_model_assisted_metadata(metadata):
    """Validate the distinct estimate contract; no physical-survey relabeling.

    The offline builder binds actual files. The immutable runtime contract binds
    their hashes and the reviewed acceptance/assumption record, without file I/O.
    """
    value = json.loads(json.dumps(metadata, allow_nan=False))
    def digest(v):
        return isinstance(v, str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v)
    if (value.get('geometry_method') != METHOD or value.get('validation_kind') != VALIDATION_KIND
            or value.get('accuracy_basis') != 'conditional-cad-estimate'
            or value.get('source') != 'wrist-only' or value.get('external_metric_measurement') is not False
            or value.get('uncertainty_frame') != 'wrist_camera_optical'
            or not digest(value.get('camera_calibration_fingerprint'))):
        raise ValueError('Need explicit wrist-camera-relative CAD estimate provenance')
    groups = [value.get(n) for n in ('training_frame_ids', 'heldout_frame_ids')]
    if (any(not isinstance(g, list) or len(g) < size or any(not isinstance(x, str) or not x for x in g)
            or len(set(g)) != len(g) for g, size in zip(groups, (3, 2)))
            or set(groups[0]) & set(groups[1])):
        raise ValueError('Need separate training and whole-opening held-out frame IDs')
    bounds = value.get('measured_raw_range')
    if (not isinstance(bounds, list) or len(bounds) != 2
            or any(type(v) is not int or not 0 <= v <= 4095 for v in bounds) or bounds[0] >= bounds[1]):
        raise ValueError('Need the actual measured encoder range')
    section = value.get('contact_section', {})
    _finite(section.get('cad_yz_m'), (2,), 'matched CAD contact section')
    if not isinstance(section.get('description'), str) or len(section['description']) < 20:
        raise ValueError('Describe the local contact section and its limited scope')
    assumptions = value.get('assumptions', {})
    for name in ('sigma_multiplier', 'cad_pad_tolerance_m', 'encoder_tolerance_ticks'):
        v = assumptions.get(name)
        if type(v) not in (int, float) or not math.isfinite(v) or v < 0 or name == 'sigma_multiplier' and v == 0:
            raise ValueError('Model-assisted engineering assumptions must be explicit and finite')
    if not isinstance(assumptions.get('description'), str) or len(assumptions['description']) < 20:
        raise ValueError('Explain the CAD/hinge/encoder assumption basis')
    for name in ('tcp_m', 'aperture_m', 'pad_m'):
        v = value.get('uncertainty', {}).get(name)
        if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
            raise ValueError('Need separate finite camera-relative geometry uncertainties')
    validation = value.get('validation', {})
    if validation.get('parameter_rank') != 7 or validation.get('accepted') is not True:
        raise ValueError('An unobservable or unaccepted fit cannot be planning eligible')
    limit = validation.get('heldout_silhouette_p95_limit_px')
    if (type(limit) not in (int, float) or not math.isfinite(limit) or limit <= 0
            or not isinstance(validation.get('acceptance_basis'), str) or len(validation['acceptance_basis']) < 20):
        raise ValueError('Need an explicit reviewed silhouette criterion; no default tolerance is invented')
    rows = validation.get('heldout', [])
    if {r.get('frame_id') for r in rows} != set(groups[1]) or len(rows) != len(groups[1]):
        raise ValueError('Acceptance must evaluate every held-out opening exactly once')
    for row in rows:
        error = row.get('silhouette_p95_px')
        if type(error) not in (int, float) or not math.isfinite(error) or not 0 <= error <= limit:
            raise ValueError('Held-out silhouette exceeds the declared criterion')
    artifacts = value.get('evidence_files', [])
    if len(artifacts) < 4 or any(not isinstance(a.get('path'), str) or not digest(a.get('sha256')) for a in artifacts):
        raise ValueError('Bind acquisition, annotations, fit and sensitivity evidence hashes')
    return value


def build_model_assisted_profile(*, provenance, camera_profile, calibration_revision,
        validation_id, geometry_revision, sections, contact_section, assumptions,
        validation, covariance_evidence, evidence_files):
    """Make an inactive estimate from forward-propagated matched sections.

    Each integer opening in the measured range is evaluated offline: the runtime
    table is the articulated CAD model, not a straight chord between five poses.
    Acceptance values and mechanical assumptions are mandatory caller inputs.
    """
    from .spatial_contract import JawCalibration
    if not covariance_evidence.get('identifiable') or covariance_evidence.get('parameter_rank') != 7:
        raise ValueError('Unobservable articulation cannot produce an eligible profile')
    for artifact in [provenance['manifest'], provenance['annotations'], *provenance['cad_files']]:
        if _digest(artifact['path']) != artifact['sha256']:
            raise ValueError('Evidence changed after silhouette provenance validation')
    _finite(covariance_evidence.get('covariance'), (7, 7), 'conditional covariance')
    lo, hi = provenance['measured_raw_range']
    if [s['raw'] for s in sections] != list(range(lo, hi + 1)):
        raise ValueError('Evaluate each actual encoder tick in the measured range; no curve extrapolation')
    camera = WristCalibration.from_mapping(camera_profile,
        native_image_size=camera_profile['native_image_size'], camera_id=camera_profile['camera_id'],
        mount_revision=camera_profile['mount_revision'], geometry_revision=geometry_revision, cad_handeye=np.eye(4))
    if not camera.metric_calibrated or camera.fingerprint != provenance['camera_calibration_fingerprint']:
        raise ValueError('Profile geometry requires the same calibrated wrist camera as the evidence')
    files = []
    for path in evidence_files:
        path = Path(path).resolve(); files.append(dict(path=str(path), sha256=_digest(path)))
    maxima = dict(tcp_m=0., aperture_m=0., pad_m=0.)
    samples = []
    for row in sections:
        estimate = row['estimate']
        if (estimate.get('geometry_method') != METHOD or estimate.get('uncertainty_frame') != 'wrist_camera_optical'
                or estimate.get('assumptions') != assumptions):
            raise ValueError('Every sample must retain the same conditional uncertainty assumptions')
        bounds = estimate['conditional_error_m']
        for key, values in dict(tcp_m=[bounds['tcp']], aperture_m=[bounds['aperture']],
                                pad_m=[bounds['fixed_pad'], bounds['moving_pad']]).items():
            if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
                raise ValueError('Nonfinite forward geometry error')
            maxima[key] = max(maxima[key], *values)
        nominal = estimate['nominal']
        pads = camera.T_wrist_camera.apply([nominal['fixed_pad_m'], nominal['moving_pad_m']])
        approach = _finite(row['approach_camera'], (3,), 'section approach axis')
        axis = camera.T_wrist_camera.as_array()[:3, :3] @ approach
        samples.append(dict(raw=row['raw'], fixed_pad_m=pads[0].tolist(), moving_pad_m=pads[1].tolist(),
                            approach_axis=axis.tolist()))
    metadata = validate_model_assisted_metadata({**deepcopy(provenance),
        'accuracy_basis': 'conditional-cad-estimate', 'uncertainty_frame': 'wrist_camera_optical',
        'contact_section': deepcopy(contact_section), 'assumptions': deepcopy(assumptions),
        'uncertainty': maxima, 'validation': deepcopy(validation), 'evidence_files': files,
        'uncertainty_combination': 'At each raw: max(conditional parameter spread, same-image refit sensitivity) + separately identified independent parameter-family sensitivity + encoder sensitivity + declared CAD/hinge pad assumption; maximum over measured encoder ticks. Base/FK uncertainty is separate.',
        'conditional_covariance': deepcopy(covariance_evidence)})
    profile = dict(schema_version=1, source='wrist-only', calibration_revision=calibration_revision,
        geometry_revision=geometry_revision, status='model-assisted', validation_id=validation_id,
        validation_kind=VALIDATION_KIND, landmark_definition='matched_contact_cross_section',
        approach_axis=samples[0]['approach_axis'], position_uncertainty_m=maxima['tcp_m'],
        samples=samples, model_assisted=metadata)
    JawCalibration.from_mapping(profile, geometry_revision=geometry_revision)
    return profile
