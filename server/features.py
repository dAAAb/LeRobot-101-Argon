"""Static-scene wrist correspondences, filtered without a fitted camera model."""
import itertools
import cv2
import numpy as np


def wrist_tracks(observations, cancel=None):
    sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=.012)
    features, images = [], []
    for obs in observations:
        if cancel is not None and cancel.is_set():
            raise RuntimeError('已停止觀察')
        if obs.get('source') != 'wrist' or not obs.get('joints'):
            raise ValueError('Need wrist images with measured joint angles')
        im = cv2.imread(obs['path'], cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise ValueError('Observation expired')
        if images and im.shape != images[0].shape:
            raise ValueError('Image geometry changed')
        mask = np.full_like(im, 255)
        # The gripper is attached to the camera: its pixels are not world features.
        mask[int(im.shape[0] * .595):] = 0
        keypoints, descriptors = sift.detectAndCompute(im, mask)
        ids, seen = [], set()
        for n in sorted(range(len(keypoints)), key=lambda n: keypoints[n].response, reverse=True):
            # OpenCV masks only the keypoint center. SIFT descriptor support is
            # ~5.303 * keypoint.size; exclude the full patch, not just its center.
            px, py = keypoints[n].pt
            radius = 6 * keypoints[n].size
            if not (radius <= px < im.shape[1]-radius and radius <= py < int(im.shape[0]*.595)-radius):
                continue
            cell = tuple(round(v) for v in keypoints[n].pt)
            if cell not in seen:
                seen.add(cell)
                ids.append(n)
        features.append(([keypoints[n] for n in ids], descriptors[ids] if descriptors is not None else None))
        images.append(im)
    if not images:
        raise ValueError('No wrist observations')
    maps, matcher = {}, cv2.BFMatcher()
    for i, j in itertools.combinations(range(len(images)), 2):
        if cancel is not None and cancel.is_set():
            raise RuntimeError('已停止觀察')
        ki, di = features[i]
        kj, dj = features[j]
        if di is None or dj is None or min(len(di), len(dj)) < 2:
            continue
        def ratio_map(a, b):
            return {m.queryIdx: m.trainIdx for pair in matcher.knnMatch(a, b, k=2)
                    if len(pair) == 2 for m, n in [pair] if m.distance < .7 * n.distance}
        forward, reverse = ratio_map(di, dj), ratio_map(dj, di)
        pairs = [(a, b) for a, b in forward.items() if reverse.get(b) == a]
        if len(pairs) < 8:
            continue
        p1 = np.float64([ki[a].pt for a, b in pairs])
        p2 = np.float64([kj[b].pt for a, b in pairs])
        try:
            F, inliers = cv2.findFundamentalMat(p1, p2, cv2.USAC_MAGSAC, 1., .9999, 10000)
        except cv2.error:
            # Near-identical home/return images can have no fundamental model.
            continue
        if F is None or inliers is None:
            continue
        pairs = [p for p, keep in zip(pairs, inliers.ravel()) if keep]
        maps[i, j] = dict(pairs)
        maps[j, i] = {b: a for a, b in pairs}
    matches = []
    for i, j in itertools.combinations(range(len(images)), 2):
        # A match must close a triangle through a third view. Filtering is based
        # on descriptor identity, not on residuals from the calibration under test.
        pairs = [(a, b) for a, b in maps.get((i, j), {}).items()
                 if any(maps.get((i, k), {}).get(a, -1) >= 0
                        and maps.get((i, k), {}).get(a, -1) == maps.get((j, k), {}).get(b, -2)
                        for k in range(len(images)) if k not in (i, j))]
        if len(pairs) < 12:
            continue
        p1 = np.float64([features[i][0][a].pt for a, b in pairs])
        p2 = np.float64([features[j][0][b].pt for a, b in pairs])
        moving = np.linalg.norm(p2 - p1, axis=1) > 2.5
        if moving.sum() >= 12:
            matches.append((i, j, p1[moving], p2[moving]))
    return matches, (images[0].shape[1], images[0].shape[0])
