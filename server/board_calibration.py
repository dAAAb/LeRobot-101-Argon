"""Offline eye-in-hand calibration against a stationary, printed metric board.

The board must stay rigid and stationary for the complete session. A 9 x 6
internal-corner board has 10 x 7 squares; the first printed square is black.
Twenty millimetres is the printed square size, not an assumed object size.

Training estimates K/Brown distortion and T_gripper_link_camera. Entire
measured poses are withheld, then tested against ONE board-in-base transform
fitted using training views only. Passing validates the combined camera/FK
chain in the sampled workspace; it does not identify individual joint zeros,
prove global robot geometry, or calibrate jaw contact surfaces.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import hashlib
import json
import math

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .geometry import observation_families
from .spatial_contract import RigidTransform, build_wrist_observation, require_compatible


@dataclass(frozen=True)
class BoardSpec:
    columns: int = 9
    rows: int = 6
    square_size_m: float = .020

    def __post_init__(self):
        if (type(self.columns) is not int or type(self.rows) is not int
                or min(self.columns, self.rows) < 3 or max(self.columns, self.rows) > 30
                or type(self.square_size_m) not in (int, float)
                or not math.isfinite(self.square_size_m) or not .001 <= self.square_size_m <= .1):
            raise ValueError('invalid metric chessboard specification')
        if self.columns % 2 == self.rows % 2:
            raise ValueError('board needs opposite row/column parity for corner-polarity orientation')

    def to_dict(self):
        return {'columns': self.columns, 'rows': self.rows, 'square_size_m': self.square_size_m,
                'first_square': 'black', 'coordinate_origin': 'first-inner-corner'}

    @property
    def fingerprint(self):
        return _hash(self.to_dict())

    def object_points(self):
        result = np.zeros((self.rows * self.columns, 3), np.float64)
        result[:, :2] = np.mgrid[0:self.columns, 0:self.rows].T.reshape(-1, 2) * self.square_size_m
        return result


@dataclass(frozen=True)
class ValidationLimits:
    """Declared workspace-calibration acceptance, not generic motion thresholds."""
    minimum_views: int = 12
    minimum_training_views: int = 8
    minimum_heldout_views: int = 3
    minimum_rotation_span_deg: float = 12.
    minimum_second_axis_ratio: float = .08
    training_rms_px: float = 1.5
    heldout_reprojection_px: float = 6.
    heldout_translation_m: float = .015
    heldout_rotation_deg: float = 3.


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _inverse(t):
    result = np.eye(4)
    result[:3, :3] = t[:3, :3].T
    result[:3, 3] = -t[:3, :3].T @ t[:3, 3]
    return result


def _transform(rvec, tvec):
    result = np.eye(4)
    result[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3))[0]
    result[:3, 3] = np.asarray(tvec).reshape(3)
    return result


def _pack(t):
    return np.r_[cv2.Rodrigues(t[:3, :3])[0].reshape(3), t[:3, 3]]


def _project(points, camera_board, k, distortion):
    local = points @ camera_board[:3, :3].T + camera_board[:3, 3]
    if np.any(local[:, 2] <= 0):
        raise ValueError('board lies behind the fitted camera')
    return cv2.projectPoints(points, cv2.Rodrigues(camera_board[:3, :3])[0], camera_board[:3, 3],
                             k, distortion)[0].reshape(-1, 2)


def _gray(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError('board detection requires an 8-bit wrist image')
    if image.ndim == 3:
        if image.shape[2] != 3:
            raise ValueError('colour board image must contain BGR channels')
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _canonical_corners(gray, corners, board):
    """Resolve the 180-degree chessboard ordering with local black/white parity.

    Opposite row/column parity makes reversing all corners invert this colour
    pattern. No screen-left heuristic is used as the wrist can roll.
    """
    p = corners[0]
    dx, dy = corners[1] - p, corners[board.columns] - p
    shades = []
    for sx, sy in ((-1, -1), (1, 1), (-1, 1), (1, -1)):
        q = p + .4 * (sx * dx + sy * dy)
        if not (2 <= q[0] < gray.shape[1] - 2 and 2 <= q[1] < gray.shape[0] - 2):
            raise ValueError('完整棋盤邊緣不在畫面內，請退後或重新對準。')
        shades.append(float(cv2.getRectSubPix(gray, (3, 3), tuple(map(float, q))).mean()))
    contrast = (shades[2] + shades[3] - shades[0] - shades[1]) / 2
    if abs(contrast) < 12:
        raise ValueError('棋盤黑白對比不夠清楚，暫時無法固定角點方向。')
    return corners if contrast > 0 else corners[::-1].copy(), abs(contrast)


def detect_board(image, *, frame=None, board=BoardSpec()):
    """Find a complete board at bounded resolution, then refine native pixels.

    Exhaustive full-resolution SB search can spend tens of seconds exploring
    an occluded board. A 960-pixel search keeps capture responsive; its corners
    are mapped back with OpenCV's pixel-centre convention and refined against
    the original image. A single classic detector fallback at the same bounded
    resolution handles complete boards that SB misses. Both detector paths use
    the same native refinement and validation; no missing corners are inferred.
    Calibration never receives reduced-image coordinates.
    """
    gray = _gray(image)
    height, width = gray.shape
    result = {'found': False, 'corners_px': [], 'image_size': [width, height],
              'board_spec': board.to_dict(), 'board_fingerprint': board.fingerprint,
              'coverage_fraction': 0., 'reason': ''}
    if frame is not None:
        if frame.get('source') != 'wrist' or tuple(frame.get('image_size', ())) != (width, height):
            raise ValueError('board detection must bind the exact wrist image resolution')
        result.update(frame_id=frame['id'], camera_generation=frame['camera_generation'],
                      sequence=frame['sequence'], capture_monotonic=frame['capture_monotonic'])
    scale = min(1., 960. / max(width, height))
    search_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    search = (cv2.resize(gray, search_size, interpolation=cv2.INTER_AREA)
              if search_size != (width, height) else gray)
    result['detector'] = {'method': 'bounded-SB-and-native-subpixel',
                          'search_image_size': list(search_size),
                          'corner_coordinate_image_size': [width, height]}
    found, corners = cv2.findChessboardCornersSB(search, (board.columns, board.rows),
                                                flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    result['detector']['classic_fallback_attempted'] = not found
    if not found:
        found, corners = cv2.findChessboardCorners(search, (board.columns, board.rows),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        result['detector']['method'] = 'bounded-classic-fallback-and-native-subpixel'
    if not found:
        result['reason'] = f'尚未看到完整的 {board.columns} × {board.rows} 棋盤內角點；請讓整張棋盤入鏡。'
        return result
    corners = corners.reshape(-1, 2).astype(np.float32)
    corners = np.ascontiguousarray((corners + .5) * np.array(
        [width / search_size[0], height / search_size[1]], np.float32) - .5)
    search_corners = corners.copy()
    grid = corners.reshape(board.rows, board.columns, 2)
    spacing = np.r_[np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel(),
                    np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel()]
    radius = max(2, min(5, int(np.percentile(spacing, 5) * .2)))
    cv2.cornerSubPix(gray, corners, (radius, radius), (-1, -1),
                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, .001))
    refinement = np.linalg.norm(corners - search_corners, axis=1)
    result['detector']['native_refinement_p95_px'] = float(np.percentile(refinement, 95))
    if not np.isfinite(corners).all() or np.max(refinement) > max(3., np.percentile(spacing, 5) * .25):
        result['reason'] = '棋盤角點不夠清楚，請停穩後再拍一張。'
        return result
    try:
        corners, contrast = _canonical_corners(gray, corners, board)
    except ValueError as exc:
        result['reason'] = str(exc)
        return result
    coverage = float(cv2.contourArea(cv2.convexHull(corners.astype(np.float32))) / (width * height))
    if coverage < .02:
        result['reason'] = '棋盤在畫面中太小，請讓它佔更多畫面後再拍。'
        return result
    result.update(found=True, corners_px=corners.tolist(), coverage_fraction=coverage,
                  black_white_contrast=contrast, reason='已看見完整棋盤。')
    return result


def _detection(frame, board):
    detection = frame.get('board_detection')
    if detection is None:
        image = cv2.imread(frame.get('path', ''), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError('calibration wrist image file is missing')
        return detect_board(image, frame=frame, board=board)
    for key, value in (('frame_id', frame['id']), ('camera_generation', frame['camera_generation']),
                       ('sequence', frame['sequence']), ('capture_monotonic', frame['capture_monotonic']),
                       ('image_size', frame['image_size']), ('board_fingerprint', board.fingerprint)):
        if detection.get(key) != value:
            raise ValueError(f'cached board detection has stale or mismatched {key}')
    if type(detection.get('found')) is not bool:
        raise ValueError('cached board detection must declare whether corners were found')
    return detection


def _observability(poses):
    vectors = []
    for a, b in combinations(poses, 2):
        vectors.append(cv2.Rodrigues(b[:3, :3].T @ a[:3, :3])[0].reshape(3))
    vectors = np.array(vectors)
    singular = np.linalg.svd(vectors, compute_uv=False)
    return {'rotation_span_deg': float(np.degrees(np.max(np.linalg.norm(vectors, axis=1)))),
            'rotation_singular_values_rad': singular.tolist(),
            'second_axis_ratio': float(singular[1] / max(singular[0], 1e-12))}


def _polynomial_minimum(coefficients, upper):
    """Exact extrema of a low-degree polynomial on the squared-radius interval."""
    coefficients = np.asarray(coefficients, float)
    derivative = np.polynomial.polynomial.polyder(coefficients)
    roots = np.polynomial.polynomial.polyroots(derivative) if np.any(derivative) else []
    samples = [0., upper] + [float(r.real) for r in roots
        if abs(r.imag) < 1e-10 and 0 < r.real < upper]
    return float(np.min(np.polynomial.polynomial.polyval(samples, coefficients)))


def _brown_image_domain(k, distortion, size):
    """Check a one-to-one Brown map covering every pixel ray in the image.

    On a disc of radius R the radial Jacobian eigenvalues are f and f+2r²f'.
    Tangential distortion lowers either eigenvalue by at most 6R*|p|. A
    positive lower bound makes the full map strictly monotone on that disc.
    Its boundary radius is at least R*f(R²)-3R²*|p|; covering the image's
    farthest normalized corner therefore establishes an inverse over the
    image, rather than merely checking a few locally well-fitted corners.
    """
    k = np.asarray(k, float); d = np.asarray(distortion, float).reshape(-1)
    if d.shape != (5,) or not np.isfinite(d).all():
        return {'valid': False, 'reason': 'expected finite five-coefficient Brown model'}
    width, height = size
    grid = np.array([(x, y) for y in np.linspace(0, height - 1, 9)
                     for x in np.linspace(0, width - 1, 15)], dtype=np.float64)
    target_radius = float(np.max(np.linalg.norm((grid - k[:2, 2]) / np.diag(k)[:2], axis=1)))
    k1, k2, p1, p2, k3 = d
    radial = [1., k1, k2, k3]
    radial_derivative = [1., 3 * k1, 5 * k2, 7 * k3]
    tangential = float(np.hypot(p1, p2))
    radius = target_radius * 1.01
    domain = None
    # Expand only to establish coverage. Stop on a folded radial map; larger
    # discs cannot remove an already negative Jacobian inside the domain.
    for _ in range(30):
        lower = min(_polynomial_minimum(radial, radius * radius),
                    _polynomial_minimum(radial_derivative, radius * radius)) - 6 * radius * tangential
        coverage = (radius * np.polynomial.polynomial.polyval(radius * radius, radial)
                    - 3 * radius * radius * tangential)
        domain = {'normalized_radius': float(radius), 'image_corner_radius': target_radius,
                  'jacobian_eigenvalue_lower_bound': float(lower),
                  'boundary_radius_lower_bound': float(coverage)}
        if lower <= 1e-9:
            return {'valid': False, 'reason': 'Brown distortion folds within the required image domain', **domain}
        if coverage >= target_radius:
            break
        radius *= 1.05
    else:
        return {'valid': False, 'reason': 'Brown model cannot establish coverage of the full image', **domain}
    # Verify the actual OpenCV inverse used by production consumers, including
    # its default iteration budget, not just the mathematical inverse.
    rays = cv2.undistortPoints(grid.reshape(-1, 1, 2), k, d).reshape(-1, 2)
    projected = cv2.projectPoints(np.c_[rays, np.ones(len(rays))], np.zeros(3), np.zeros(3), k, d)[0].reshape(-1, 2)
    inverse_error = float(np.max(np.linalg.norm(projected - grid, axis=1)))
    valid = bool(np.isfinite(rays).all() and math.isfinite(inverse_error) and inverse_error <= .25)
    return {'valid': valid, 'reason': '' if valid else 'OpenCV inverse is inaccurate at image edges',
            **domain, 'inverse_roundtrip_max_px': inverse_error,
            'inverse_roundtrip_limit_px': .25,
            'interpretation': 'Mathematical image-domain validity, not empirical calibration of unseen image regions.'}


def _fit_intrinsics(points, pixels, size):
    """Select an admissible lens model using training pixels exclusively."""
    fixed_radial = cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
    candidates = [('brown-k1-k2-tangential', cv2.CALIB_FIX_K3, 8),
                  ('brown-k1-tangential', fixed_radial, 7),
                  ('brown-k1', fixed_radial | cv2.CALIB_ZERO_TANGENT_DIST, 5),
                  ('pinhole', fixed_radial | cv2.CALIB_FIX_K1 | cv2.CALIB_ZERO_TANGENT_DIST, 4)]
    diagnostics, admissible = [], []
    coordinate_count = len(pixels) * len(points) * 2
    for name, flags, intrinsic_parameters in candidates:
        row = {'model': name, 'free_intrinsic_parameters': intrinsic_parameters}
        diagnostics.append(row)
        try:
            fit = cv2.calibrateCameraExtended([points.astype(np.float32)] * len(pixels),
                [np.asarray(p, np.float32) for p in pixels], size, None, None, flags=flags)
            rms, k, distortion = fit[:3]
            if (not math.isfinite(rms) or not np.isfinite(k).all() or not np.isfinite(distortion).all()
                    or min(k[0, 0], k[1, 1]) < .25 * size[0]
                    or max(k[0, 0], k[1, 1]) > 3 * size[0]
                    or not 0 <= k[0, 2] < size[0] or not 0 <= k[1, 2] < size[1]):
                row.update(admissible=False, reason='implausible or nonfinite intrinsics')
                continue
            domain = _brown_image_domain(k, distortion, size)
            # OpenCV RMS uses Euclidean point residuals; divide its variance by
            # two for the scalar x/y residual likelihood used by BIC.
            bic = coordinate_count * math.log(max(float(rms * rms / 2), 1e-20))
            bic += (intrinsic_parameters + 6 * len(pixels)) * math.log(coordinate_count)
            row.update(training_rms_px=float(rms), training_bic=float(bic),
                       image_domain=domain, admissible=domain['valid'])
            if domain['valid']:
                admissible.append((bic, name, fit, domain))
        except (cv2.error, ValueError, np.linalg.LinAlgError) as exc:
            row.update(admissible=False, reason=str(exc))
    selection = {'method': 'training-only BIC among image-domain-admissible models',
                 'heldout_used_for_selection': False, 'training_view_count': len(pixels),
                 'candidates': diagnostics, 'selected_model': None}
    if not admissible:
        return None, selection
    _, name, fit, domain = min(admissible, key=lambda item: item[0])
    selection.update(selected_model=name, selected_image_domain=domain)
    return fit, selection


def _mean_pose(poses):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_matrix(np.array([x[:3, :3] for x in poses])).mean().as_matrix()
    t[:3, 3] = np.mean([x[:3, 3] for x in poses], axis=0)
    return t


def _fit_handeye(grippers, camera_boards, points, pixels, k, distortion):
    rotation, translation = cv2.calibrateHandEye(
        [x[:3, :3] for x in grippers], [x[:3, 3] for x in grippers],
        [x[:3, :3] for x in camera_boards], [x[:3, 3] for x in camera_boards],
        method=cv2.CALIB_HAND_EYE_PARK)
    x = np.eye(4); x[:3, :3] = rotation; x[:3, 3] = translation.reshape(3)
    x = RigidTransform('gripper_link', 'wrist_camera_optical', x, 'candidate').as_array()
    y = _mean_pose([g @ x @ c for g, c in zip(grippers, camera_boards)])

    def residual(parameters):
        xx, yy = _transform(parameters[:3], parameters[3:6]), _transform(parameters[6:9], parameters[9:12])
        errors = []
        for g, measured in zip(grippers, pixels):
            try:
                predicted = _project(points, _inverse(g @ xx) @ yy, k, distortion)
                errors.extend((predicted - measured).reshape(-1))
            except ValueError:
                errors.extend(np.full(len(points) * 2, 1e5))
        return errors

    fit = least_squares(residual, np.r_[_pack(x), _pack(y)], loss='soft_l1', f_scale=1., max_nfev=240)
    if not fit.success:
        raise ValueError('手眼位置尚未收斂；需要更多不同方向的棋盤視角。')
    x, y = _transform(fit.x[:3], fit.x[3:6]), _transform(fit.x[6:9], fit.x[9:12])
    if np.linalg.norm(x[:3, 3]) > .30:
        raise ValueError('推算的相機安裝位置不合理；請检查棋盤尺寸、角度和位置是否固定。')
    return x, y


def fit_board_calibration(frames, model, *, camera_id, mount_revision, geometry_revision,
                          board_session_id, board=BoardSpec(), holdout_indices=None,
                          limits=ValidationLimits()):
    """Fit recorded wrist observations; return a profile candidate, never activate it.

    Frame ``board_session_id`` identifies one unchanged placement. Cached
    ``board_detection`` must be bound to the exact frame. Whole near-duplicate
    poses stay in one split, preventing return visits from leaking into holdout.
    """
    report = {'status': 'insufficient', 'source': 'wrist-only', 'calibration_profile': None,
              'board_session_id': board_session_id, 'board_spec': board.to_dict(),
              'validation': {}, 'blockers': [], 'detections': [], 'activation_performed': False}
    if not isinstance(board_session_id, str) or not board_session_id.strip():
        raise ValueError('a stationary-board session ID is required')
    if not frames:
        report['blockers'] = ['需要至少 12 個停穩後、方向不同的完整棋盤視角。']
        return report
    observations, grippers, pixels = [], [], []
    accepted_frames, original_indices = [], []
    try:
        all_observations = []
        for index, frame in enumerate(frames):
            if frame.get('board_session_id') != board_session_id:
                raise ValueError('棋盤已更換位置或來自不同採樣；不能混合校準。')
            transforms = model.fk.transforms(frame['joints'])
            observation = build_wrist_observation(frame, transforms, geometry_revision=geometry_revision,
                                                  camera_id=camera_id, mount_revision=mount_revision)
            all_observations.append(observation)
            if not observation.timing.pose_bound:
                raise ValueError('校準照片缺少停穩後的實測關節配對，請重新擷取。')
            detection = _detection(frame, board)
            report['detections'].append({'frame_id': frame['id'], 'found': detection['found'],
                                         'reason': detection.get('reason', '')})
            if not detection['found']:
                continue
            corners = np.asarray(detection['corners_px'], dtype=float)
            w, h = frame['image_size']
            if (corners.shape != (board.rows * board.columns, 2) or not np.isfinite(corners).all()
                    or np.any(corners < 0) or np.any(corners[:, 0] >= w) or np.any(corners[:, 1] >= h)):
                raise ValueError('棋盤角點不符合目前完整影像。')
            observations.append(observation); grippers.append(np.asarray(transforms['gripper_link']))
            pixels.append(corners); accepted_frames.append(frame); original_indices.append(index)
        require_compatible(all_observations)
        groups = observation_families(accepted_frames)
        report['validation'].update(accepted_views=len(observations), independent_pose_groups=len(groups))
        if len(groups) < limits.minimum_views:
            report['blockers'] = [f'目前有 {len(groups)} 個不同棋盤視角，還需要 {limits.minimum_views - len(groups)} 個；原地重拍不算新的方向。']
            return report
        if holdout_indices is None:
            held_groups = groups[3::4]
        else:
            if (not isinstance(holdout_indices, (list, tuple))
                    or any(type(i) is not int or i not in original_indices for i in holdout_indices)):
                raise ValueError('heldout indices must refer to accepted captured frames')
            requested = {original_indices.index(i) for i in holdout_indices}
            held_groups = [g for g in groups if set(g) & requested]
        held = sorted({i for group in held_groups for i in group})
        train = [i for i in range(len(observations)) if i not in held]
        train_groups = [g for g in groups if not set(g).intersection(held)]
        if len(train_groups) < limits.minimum_training_views or len(held_groups) < limits.minimum_heldout_views:
            report['blockers'] = ['校準需要至少 8 個擬合姿勢及 3 個獨立驗證姿勢；請再加入不同角度。']
            return report
        size = observations[0].processing.native_size
        observable = _observability([grippers[i] for i in train])
        report['validation']['observability'] = observable
        if (observable['rotation_span_deg'] < limits.minimum_rotation_span_deg
                or observable['second_axis_ratio'] < limits.minimum_second_axis_ratio):
            report['blockers'] = ['這些照片的旋轉方向太相似；需要左右與俯仰兩種不同轉向，不能只沿同一軸轉動。']
            return report
        points = board.object_points()
        fit, selection = _fit_intrinsics(points, [pixels[i] for i in train], size)
        report['validation']['intrinsics_model_selection'] = selection
        if fit is None:
            raise ValueError('相機畸變在畫面邊緣沒有穩定解；請補拍棋盤位於不同畫面區域的照片。')
        rms, k, distortion, rvecs, tvecs, intrinsic_std, _, _ = fit
        if (not np.isfinite(k).all() or not np.isfinite(distortion).all()
                or min(k[0, 0], k[1, 1]) < .25 * size[0] or max(k[0, 0], k[1, 1]) > 3 * size[0]
                or not 0 <= k[0, 2] < size[0] or not 0 <= k[1, 2] < size[1]):
            raise ValueError('相機內參尚未得到合理解；需要更多不同傾斜與遠近的棋盤視角。')
        camera_boards = [_transform(r, t) for r, t in zip(rvecs, tvecs)]
        x, y = _fit_handeye([grippers[i] for i in train], camera_boards, points,
                            [pixels[i] for i in train], k, distortion)
        validation_rows = []
        training_chain_errors = []
        for index in train + held:
            predicted_board = _inverse(grippers[index] @ x) @ y
            errors = np.linalg.norm(_project(points, predicted_board, k, distortion) - pixels[index], axis=1)
            if index in train:
                training_chain_errors.extend(errors)
                continue
            ok, r, t = cv2.solvePnP(points, pixels[index], k, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                raise ValueError('獨立驗證照片無法辨認棋盤姿態。')
            c = _transform(r, t); measured_board = grippers[index] @ x @ c
            rotation_error = float(np.degrees(np.linalg.norm(cv2.Rodrigues(y[:3, :3].T @ measured_board[:3, :3])[0])))
            validation_rows.append({'frame_id': accepted_frames[index]['id'],
                'reprojection_p95_px': float(np.percentile(errors, 95)),
                'board_translation_error_m': float(np.linalg.norm(measured_board[:3, 3] - y[:3, 3])),
                'board_rotation_error_deg': rotation_error})
        max_translation = max(r['board_translation_error_m'] for r in validation_rows)
        max_rotation = max(r['board_rotation_error_deg'] for r in validation_rows)
        max_pixel = max(r['reprojection_p95_px'] for r in validation_rows)
        train_chain_p95 = float(np.percentile(training_chain_errors, 95))
        report['validation'].update(training_frame_ids=[accepted_frames[i]['id'] for i in train],
            heldout_frame_ids=[accepted_frames[i]['id'] for i in held], heldout_views=validation_rows,
            training_intrinsics_rms_px=float(rms), training_chain_p95_px=train_chain_p95,
            heldout_max_translation_m=max_translation, heldout_max_rotation_deg=max_rotation,
            heldout_max_reprojection_p95_px=max_pixel,
            empirical_bounds_only=True, heldout_data_used_for_fitting=False,
            acceptance_limits={'training_rms_px': limits.training_rms_px,
                'heldout_reprojection_px': limits.heldout_reprojection_px,
                'heldout_translation_m': limits.heldout_translation_m,
                'heldout_rotation_deg': limits.heldout_rotation_deg})
        passed = (rms <= limits.training_rms_px and train_chain_p95 <= limits.heldout_reprojection_px
                  and max_pixel <= limits.heldout_reprojection_px
                  and max_translation <= limits.heldout_translation_m
                  and max_rotation <= limits.heldout_rotation_deg)
        body_names = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')
        scope = {'kind': 'combined-camera-fk-sampled-workspace',
            'joint_raw_bounds': {n: [min(f['joints'][n] for f in accepted_frames),
                                      max(f['joints'][n] for f in accepted_frames)] for n in body_names},
            'sampled_joint_poses': [{n: f['joints'][n] for n in body_names} for f in accepted_frames],
            'extrapolation_allowed': False, 'individually_identified_joint_zeros': False,
            'jaw_contact_geometry_calibrated': False,
            'interpretation': 'Combined camera/FK verified on withheld board views, not global robot geometry; joint bounds alone do not prove all combinations.',
            'board_print_scale': 'declared square size; print at 100 percent and check a physical ruler',
            'image_points_bbox_px': [np.min(np.concatenate(pixels), axis=0).tolist(),
                                    np.max(np.concatenate(pixels), axis=0).tolist()],
            'image_domain_validity': selection['selected_image_domain'],
            'image_domain_interpretation': 'Lens mapping is mathematically invertible over the image; empirical accuracy is only checked on sampled board views.',
            'board_plane_base': y.tolist()}
        identity = _hash({'camera_id': camera_id, 'mount_revision': mount_revision,
            'geometry_revision': geometry_revision, 'board': board.to_dict(),
            'frames': [(f['id'], f['camera_generation'], f['sequence'], f['capture_monotonic']) for f in accepted_frames],
            'K': k.tolist(), 'handeye': x.tolist(), 'validation': report['validation']})
        focal_relative = float(max(intrinsic_std.reshape(-1)[0] / k[0, 0],
                                   intrinsic_std.reshape(-1)[1] / k[1, 1]))
        profile = {'schema_version': 1, 'source': 'wrist-only', 'calibration_revision': f'board-{identity[:20]}',
            'camera_id': camera_id, 'mount_revision': mount_revision, 'geometry_revision': geometry_revision,
            'native_image_size': list(size), 'image_size': list(size), 'camera_matrix': k.tolist(),
            'distortion': distortion.reshape(-1).tolist(), 'distortion_model': 'opencv-brown', 'handeye': x.tolist(),
            'validated_components': ['intrinsics', 'handeye', 'robot_geometry'] if passed else [],
            'validation_id': f'board-holdout-{identity}', 'validation_kind': 'independent-heldout' if passed else 'prior',
            'validation_scope': scope, 'board_fingerprint': board.fingerprint, 'board_session_id': board_session_id,
            'relative_translation_uncertainty_m': max_translation, 'base_position_uncertainty_m': max_translation,
            'relative_rotation_uncertainty_deg': max_rotation, 'focal_relative_uncertainty': focal_relative,
            'uncertainty_interpretation': 'Empirical heldout maxima within sampled workspace; focal uncertainty is training linearized standard deviation; not guaranteed error bounds or print-scale verification.',
            'needs_more_views': not passed, 'metric_calibrated': passed, 'camera_generation': None}
        report.update(status='validated' if passed else 'rejected', calibration_profile=profile)
        if not passed:
            report['blockers'] = ['獨立驗證的棋盤位置還不一致；請確認棋盤沒有移動，再补不同方向的照片。']
        return report
    except (ValueError, KeyError, cv2.error, np.linalg.LinAlgError) as exc:
        report.update(status='rejected', blockers=[str(exc)])
        return report
