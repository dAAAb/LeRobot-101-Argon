"""Task acceptance policy, separate from every physical motion safeguard.

A useful empty-hand step may finish short of its target. This module neither
authorizes motion nor weakens the caller's screened corridor, motor bounds,
temperature, camera freshness, cancellation, contact or stall checks.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from .feedback import EndpointFeedback
from .local_cartesian import ALL, NAMES


LOCAL_MOVE_MIN_CONFIDENCE = .7
EMPTY_JAW_OPEN_MIN_CONFIDENCE = .7
WAVE_MIN_CONFIDENCE = .8
CONTACT_MIN_CONFIDENCE = .9
VERIFICATION_MIN_CONFIDENCE = .9
REQUIRED_SETTLED_SAMPLES = EndpointFeedback.REQUIRED_STABLE_SAMPLES
SETTLED_SAMPLE_PERIOD_S = EndpointFeedback.PERIOD
MAX_SETTLED_RANGE_TICKS = 2
MIN_MEANINGFUL_MOVEMENT_TICKS = int(EndpointFeedback.MIN_MOVEMENT)


def minimum_visual_confidence(action):
    """Scores alone never replace the action's explicit clearance flags."""
    thresholds = {'local_move': LOCAL_MOVE_MIN_CONFIDENCE, 'wave': WAVE_MIN_CONFIDENCE,
                  'empty_jaw_open': EMPTY_JAW_OPEN_MIN_CONFIDENCE,
                  'close': CONTACT_MIN_CONFIDENCE, 'grasp_verify': VERIFICATION_MIN_CONFIDENCE}
    if action not in thresholds:
        raise ValueError('Unknown visual action policy')
    return thresholds[action]


def _pose(value):
    T = np.asarray(value, dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError('Expected a finite rigid tip transform')
    R = T[:3, :3]
    if (not np.allclose(T[3], [0., 0., 0., 1.], atol=1e-9, rtol=0)
            or not np.allclose(R.T @ R, np.eye(3), atol=1e-9, rtol=0)
            or not np.isclose(np.linalg.det(R), 1., atol=1e-9, rtol=0)):
        raise ValueError('Expected a rigid tip transform')
    return T


def _joints(value):
    if not isinstance(value, Mapping) or set(value) != set(ALL):
        raise ValueError('All six measured encoder positions are required')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           or int(v) != v or not 0 <= v <= 4095 for v in value.values()):
        raise ValueError('Expected finite integer encoder positions')
    return {n: int(value[n]) for n in ALL}


def _angle(a, b):
    return float(np.degrees(Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).magnitude()))


def classify_translation_progress(start_tip, actual_tip, requested_delta_m, *, target_tip,
                                  origin_joints, actual_joints, empty_hand, corridor_pass):
    """Classify ONE measured sample; caller still requires a settled streak.

    ``empty_hand`` must come from fresh, explicit visual evidence, never merely
    idle state or lack of an existing grasp receipt. ``corridor_pass`` is the
    result of the caller's existing independently approved path checks for this
    measured pose; True here does not perform those checks or certify clearance.
    Partial acceptance only ends the current step and requests new observation.
    Contact, lifting a held object, and grasp verification retain separate policy.
    """
    if type(empty_hand) is not bool or type(corridor_pass) is not bool:
        raise ValueError('Explicit boolean empty-hand evidence and corridor result are required')
    start = _pose(start_tip); actual = _pose(actual_tip); target = _pose(target_tip)
    origin = _joints(origin_joints); current = _joints(actual_joints)
    request = np.asarray(requested_delta_m, dtype=float)
    if request.shape != (3,) or not np.isfinite(request).all():
        raise ValueError('Expected a finite translation request')
    requested_mm = float(np.linalg.norm(request) * 1000)
    if not .5 <= requested_mm <= 5.0000001:
        raise ValueError('A local translation request must be 0.5..5 mm')
    direction = request / np.linalg.norm(request)
    delta_mm = (actual[:3, 3] - start[:3, 3]) * 1000
    along = float(delta_mm @ direction)
    cross = float(np.linalg.norm(delta_mm - along * direction))
    distance = float(np.linalg.norm(delta_mm))
    error = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]) * 1000)
    initial_error = float(np.linalg.norm(start[:3, 3] - target[:3, 3]) * 1000)
    target_angle = _angle(actual, target); start_angle = _angle(actual, start)
    tick_change = max(abs(current[n] - origin[n]) for n in NAMES)
    minimum_progress = max(.5, requested_mm * .5)
    cross_limit = min(1.5, max(.5, requested_mm * .5))
    # Keep the larger approved path tube separate from useful task progress:
    # a point far beyond the small requested step is not a good partial result.
    overshoot_limit = requested_mm + 1.5
    meaningful = tick_change >= MIN_MEANINGFUL_MOVEMENT_TICKS and along >= minimum_progress - 1e-9
    bounded = corridor_pass and distance <= overshoot_limit + 1e-9 and along <= overshoot_limit + 1e-9
    exact = bool(bounded and meaningful and error <= 1. + 1e-9 and target_angle <= 1. + 1e-9)
    partial = bool(not exact and empty_hand and bounded and meaningful
                   and cross <= cross_limit + 1e-9 and start_angle <= 2. + 1e-9
                   and target_angle <= 2. + 1e-9 and error < initial_error - 1e-9)
    accepted = exact or partial
    if not corridor_pass: reason = 'outside_screened_corridor'
    elif not meaningful: reason = 'insufficient_forward_progress_or_encoder_change'
    elif not bounded: reason = 'excessive_overshoot'
    elif exact: reason = 'exact_endpoint'
    elif partial: reason = 'useful_empty_hand_progress_reobserve'
    elif not empty_hand: reason = 'partial_progress_requires_explicit_empty_hand'
    else: reason = 'cross_track_orientation_or_target_error_not_useful'
    return {
        'accepted': accepted,
        'completion': 'exact_target' if exact else 'partial_progress' if partial else 'not_accepted',
        'reason': reason, 'exact_target_reached': exact,
        'requested_mm': requested_mm, 'along_progress_mm': along,
        'progress_fraction': along / requested_mm, 'cross_track_mm': cross,
        'actual_distance_mm': distance, 'remaining_target_error_mm': error,
        'target_orientation_error_deg': target_angle,
        'start_orientation_change_deg': start_angle,
        'maximum_body_tick_change': tick_change,
        'minimum_progress_mm': minimum_progress, 'cross_track_limit_mm': cross_limit,
        'maximum_step_distance_mm': overshoot_limit,
        'requires_settled_samples': REQUIRED_SETTLED_SAMPLES,
        'maximum_settled_range_ticks': MAX_SETTLED_RANGE_TICKS,
        'requires_reobservation': accepted,
        'task_success': False, 'grasp_verified': False,
        'model_displacement_not_external_metrology': True,
    }


class SettledProgressWindow:
    """Pure reference for the existing three-sample controller contract.

    The live driver already owns its .12 s sampling and three-second deadline;
    use its existing acceptance window rather than nesting another controller.
    This helper makes that contract independently testable and reusable offline.
    It does not stop a motor or retain a terminal success across later samples.
    """
    def __init__(self):
        self.samples = []
        self.last_seen = None
        self.last_counted = None

    def observe(self, classification, actual_joints, now):
        if not isinstance(classification, Mapping) or type(classification.get('accepted')) is not bool:
            raise ValueError('A translation classification with boolean acceptance is required')
        actual = _joints(actual_joints)
        if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now):
            raise ValueError('A finite monotonic time is required')
        if self.last_seen is not None and now < self.last_seen:
            self.samples = []
            raise ValueError('Monotonic time moved backwards')
        self.last_seen = now
        if not classification['accepted']:
            self.samples = []
            return False
        if self.samples and any(max([p[n] for p in self.samples] + [actual[n]]) -
                                min([p[n] for p in self.samples] + [actual[n]]) > MAX_SETTLED_RANGE_TICKS
                                for n in ALL):
            self.samples = []
        if self.last_counted is not None and now - self.last_counted < SETTLED_SAMPLE_PERIOD_S - 1e-9:
            return False
        self.last_counted = now
        self.samples.append(actual)
        self.samples = self.samples[-REQUIRED_SETTLED_SAMPLES:]
        return len(self.samples) >= REQUIRED_SETTLED_SAMPLES
