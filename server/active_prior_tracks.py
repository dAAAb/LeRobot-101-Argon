"""Calibration-only feature identities, independent of the camera/FK being fit.

Tracks form a mutual-descriptor graph across all view pairs. A component may
contain at most one keypoint per image. Target IDs are split before fitting any
image geometry; the held-out IDs never train either a fundamental matrix or
the robot camera model. Image-only filtering can inspect the held-out pose,
but its FK and pixels are absent from the later camera-parameter fit.
"""
from itertools import combinations

import cv2
import numpy as np


def _check(cancel):
    if cancel is not None and cancel.is_set():
        raise RuntimeError('相機參數更新已停止或超過本次計算時間')


def _fundamental_errors(matrix, first, second):
    a = np.c_[first, np.ones(len(first))]
    b = np.c_[second, np.ones(len(second))]
    fa, fb = a @ matrix.T, b @ matrix
    denominator = np.sqrt(np.maximum(np.sum(fa[:, :2] ** 2 + fb[:, :2] ** 2, axis=1), 1e-30))
    return np.abs(np.sum(b * fa, axis=1)) / denominator


def _partition_target_tracks(target, reference):
    # Do not bucket by rounded pixel coordinates: a pixel-center convention
    # shift must not reshuffle the train/test assignment around bucket edges.
    ordered = sorted(target, key=lambda t: (float(t[reference][1]), float(t[reference][0])))
    return ([t for i, t in enumerate(ordered) if not i % 2],
            {i: t for i, t in enumerate(ordered) if i % 2 and len(t) >= 3})


def image_consensus(training, validation, count, width, cancel=None):
    """Remove entire inconsistent identities using TRAIN-only image models.

All thresholds scale with image width; no camera prior, encoder or fitted 3D
residual participates in selecting the held-out identities.
"""
    scale = width / 960.
    fundamental = {}
    for i, j in combinations(range(count), 2):
        _check(cancel)
        shared = [t for t in training if i in t and j in t]
        if len(shared) < 12:
            continue
        a, b = np.array([t[i] for t in shared]), np.array([t[j] for t in shared])
        matrix, mask = cv2.findFundamentalMat(a, b, cv2.USAC_MAGSAC, .75 * scale, .999, 2000)
        if matrix is not None and matrix.shape == (3, 3) and np.isfinite(matrix).all() and mask is not None and mask.sum() >= 12:
            fundamental[i, j] = matrix

    def valid(track):
        _check(cancel)
        errors = [_fundamental_errors(matrix, np.array([track[i]]), np.array([track[j]]))[0]
                  for (i, j), matrix in fundamental.items() if i in track and j in track]
        return bool(errors) and max(errors) <= scale

    clean_training = [t for t in training if valid(t)]
    clean_validation = {identity: t for identity, t in validation.items() if valid(t)}
    return clean_training, clean_validation, {
        'image_geometry_pairs': len(fundamental),
        'training_identities_before_filter': len(training),
        'training_identities_after_filter': len(clean_training),
        'heldout_identities_before_filter': len(validation),
        'heldout_identities_after_filter': len(clean_validation),
        'image_geometry_trained_on_heldout_identities': False,
        'heldout_pose_used_for_image_correspondence_filter_only': True,
        'image_geometry_threshold_px': scale,
    }


def multi_anchor_tracks(images, roi, reference, *, stable_jaw, cancel=None, max_feature_width=960, features=None):
    from .active_depth import _mutual_matches, GRIPPER_IMAGE_BOUNDARY
    width, height = images[0].shape[::-1]
    scale = width / 960.
    if type(max_feature_width) is not int or not 480 <= max_feature_width <= 960:
        raise ValueError('Feature detection width must be between 480 and 960 pixels')
    detection_scale = min(1., max_feature_width / width)
    reused_features = features is not None
    if features is None:
        detector = cv2.SIFT_create(nfeatures=2000, contrastThreshold=.012)
        features = []
        for image in images:
            _check(cancel)
            mask = np.full(image.shape, 255, np.uint8)
            if not stable_jaw:
                mask[int(height * GRIPPER_IMAGE_BOUNDARY) - round(8 * scale):] = 0
            if detection_scale < 1:
                small_size = (round(width * detection_scale), round(height * detection_scale))
                detection_image = cv2.resize(image, small_size, interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, small_size, interpolation=cv2.INTER_NEAREST)
            else:
                detection_image = image
            keys, descriptors = detector.detectAndCompute(detection_image, mask)
            if detection_scale < 1:
                sx, sy = detection_image.shape[1] / width, detection_image.shape[0] / height
                keys = [cv2.KeyPoint((k.pt[0] + .5) / sx - .5, (k.pt[1] + .5) / sy - .5,
                                    k.size / min(sx, sy), k.angle, k.response, k.octave, k.class_id)
                        for k in keys]
            if descriptors is None:
                features.append(([], None))
                continue
            keep, seen = [], set()
            for k in sorted(range(len(keys)), key=lambda n: keys[n].response, reverse=True):
                point = keys[k]
                pixel = tuple(round(v) for v in point.pt)
                if pixel not in seen and 6 * point.size <= point.pt[0] < width - 6 * point.size and 6 * point.size <= point.pt[1] < height - 6 * point.size:
                    keep.append(k)
                    seen.add(pixel)
            features.append(([keys[k] for k in keep], descriptors[keep]))
    elif len(features) != len(images):
        raise ValueError('Prepared image features do not match the observation count')
    _check(cancel)
    parent = {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for i, j in combinations(range(len(images)), 2):
        _check(cancel)
        if features[i][1] is None or features[j][1] is None:
            continue
        for a, b in _mutual_matches(features[i][1], features[j][1]).items():
            first, second = find((i, a)), find((j, b))
            if first != second:
                parent[first] = second
    components = {}
    for node in parent:
        components.setdefault(find(node), []).append(node)
    target, background = [], []
    conflicts = fixed = unanchored = 0
    x, y, w, h = roi
    for nodes in components.values():
        _check(cancel)
        if len({i for i, _ in nodes}) != len(nodes):
            conflicts += 1
            continue
        track = {i: np.array(features[i][0][k].pt) for i, k in nodes}
        if reference not in track:
            unanchored += 1
            continue
        if max(np.linalg.norm(a - b) for a, b in combinations(track.values(), 2)) <= 1.5 * scale:
            fixed += 1
            continue
        u, v = track[reference]
        inside = x * width <= u <= (x + w) * width and y * height <= v <= (y + h) * height
        clear_of_fingers = all(features[i][0][k].pt[1] + 6 * features[i][0][k].size < height * GRIPPER_IMAGE_BOUNDARY - 8 * scale for i, k in nodes)
        if inside and (stable_jaw or clear_of_fingers):
            target.append(track)
        elif not inside and clear_of_fingers:
            background.append(track)
    # Every observation of an identity remains on one side of this fixed split.
    training, validation = _partition_target_tracks(target, reference)
    training = training + background
    training, validation, consensus = image_consensus(training, validation, len(images), width, cancel)
    _check(cancel)
    return training, validation, {
        'method': 'mutual-descriptor-multi-anchor-components',
        'feature_detection_width': round(width * detection_scale),
        'reused_same_call_features': reused_features,
        'target_used_for_fit': True, 'target_identities': len(target),
        'background_identities': len(background), 'conflicting_components_rejected': conflicts,
        'camera_fixed_identities_rejected': fixed, 'unanchored_identities_rejected': unanchored,
        **consensus,
    }
