"""Small, explicit extrapolation allowance around immutable measured bounds.

This is a runtime policy, never new calibration data. The FK difference is a
local sensitivity reserve, not an independently measured error bound.
"""
import math
import numpy as np

BOUNDARY_ALLOWANCE_TICKS = 8
POLICY = 'fixed-measured-boundary-eight-ticks-v1'


def pose_scope(bounds, joints):
    if bounds is None:
        return dict(accepted=True, inside_measured_bounds=True, status='unscoped',
                    excess_ticks={}, allowance_ticks=0, policy=POLICY)
    excess = {n: float(joints[n] - min(max(joints[n], lo), hi))
              for n, (lo, hi) in bounds.items()}
    inside = not any(excess.values())
    accepted = all(abs(v) <= BOUNDARY_ALLOWANCE_TICKS for v in excess.values())
    return dict(accepted=accepted, inside_measured_bounds=inside, excess_ticks=excess,
        status='within-measured-joint-bounds' if inside else
               'near-boundary-extrapolation' if accepted else 'outside-measured-joint-bounds',
        allowance_ticks=BOUNDARY_ALLOWANCE_TICKS, policy=POLICY)


def boundary_reserve(model, calibration, joints, *, endpoint_transform=None):
    """Compare FK at the actual pose with the ORIGINAL boundary, never clamp FK."""
    scope = calibration.pose_scope(joints)
    result = {**scope, 'translation_m': 0., 'rotation_deg': 0.,
              'basis': 'local-FK-extrapolation-sensitivity-not-measured-accuracy'}
    if scope['inside_measured_bounds'] or not scope['accepted']:
        return result
    clipped = {**joints, **{n: joints[n] - delta for n, delta in scope['excess_ticks'].items()}}
    mount = (calibration.T_wrist_camera.as_array() if endpoint_transform is None
             else np.asarray(endpoint_transform, float))
    actual = model.fk.transforms(joints)['gripper_link'] @ mount
    boundary = model.fk.transforms(clipped)['gripper_link'] @ mount
    relative = actual[:3, :3] @ boundary[:3, :3].T
    result.update(translation_m=float(np.linalg.norm(actual[:3, 3] - boundary[:3, 3])),
        rotation_deg=math.degrees(math.acos(float(np.clip((np.trace(relative) - 1) / 2, -1., 1.)))))
    return result


def effective_uncertainty(original, reserves):
    """One common-frame reserve; two per-view reserves for relative geometry."""
    result = dict(original)
    translation = max((r['translation_m'] for r in reserves), default=0.)
    rotation = max((r['rotation_deg'] for r in reserves), default=0.)
    if translation:
        result['base_position_uncertainty_m'] = result.get('base_position_uncertainty_m', 0.) + translation
        result['relative_translation_uncertainty_m'] = result.get('relative_translation_uncertainty_m', 0.) + 2 * translation
    if rotation:
        result['relative_rotation_uncertainty_deg'] = result.get('relative_rotation_uncertainty_deg', 0.) + 2 * rotation
    if translation or rotation:
        result['uncertainty_interpretation'] = (result.get('uncertainty_interpretation', '') +
            ' Runtime near-boundary FK sensitivity is added separately: one per-view translation reserve'
            ' for base position and twice the maximum per-view reserve for relative translation/rotation.'
            ' This is local extrapolation, not additional measured calibration coverage.').strip()
    return result
