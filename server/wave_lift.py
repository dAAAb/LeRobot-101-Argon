"""A bounded, empty-hand reposition endpoint; never grasp/contact acceptance."""
import numpy as np
from scipy.spatial.transform import Rotation


def lift_endpoint(origin,actual,requested_delta,*,verified_max_up_mm=None,orientation_origin=None):
    delta=np.asarray(requested_delta,float)
    if (delta.shape!=(3,) or not np.isfinite(delta).all() or
            abs(delta[0])>1e-10 or abs(delta[1])>1e-10 or not 0<delta[2]<=.005000001):
        raise ValueError('Wave lift must be a positive vertical step <=5mm')
    movement=np.asarray(actual)[:3,3]-np.asarray(origin)[:3,3]
    progress=float(movement[2]);lateral=float(np.linalg.norm(movement[:2]))
    angle=float(np.degrees(Rotation.from_matrix(actual[:3,:3]@origin[:3,:3].T).magnitude()))
    maximum=delta[2]+.001
    if verified_max_up_mm is not None:
        if type(verified_max_up_mm) not in (int,float) or not np.isfinite(verified_max_up_mm) or not delta[2]*1000<=verified_max_up_mm<=12:
            raise ValueError('Lift endpoint must stay inside the independently assessed corridor')
        maximum=verified_max_up_mm/1000
    reference=origin if orientation_origin is None else orientation_origin
    attempt_angle=float(np.degrees(Rotation.from_matrix(actual[:3,:3]@reference[:3,:3].T).magnitude()))
    accepted=(max(.0005,delta[2]*.5)<=progress<=maximum and lateral<=.001 and angle<=1. and attempt_angle<=1.)
    return {'accepted':bool(accepted),'progress_mm':progress*1000,
            'lateral_mm':lateral*1000,'orientation_error_deg':angle,
            'reference_orientation_error_deg':attempt_angle}


def lift_attempt(origin,actual):
    """Whole-attempt corridor, including drift between individual steps."""
    movement=np.asarray(actual)[:3,3]-np.asarray(origin)[:3,3]
    angle=float(np.degrees(Rotation.from_matrix(actual[:3,:3]@origin[:3,:3].T).magnitude()))
    if not (-.001<=movement[2]<=.060 and np.linalg.norm(movement[:2])<=.005 and angle<=3.):
        raise RuntimeError('抬手已到這次可確認的範圍，先停下來')
    return max(0.,float(movement[2]*1000))


def lift_corridor(origin,actual):
    """Measured excursion of the elbow-leading local step, not its endpoint."""
    movement=np.asarray(actual)[:3,3]-np.asarray(origin)[:3,3]
    angle=float(np.degrees(Rotation.from_matrix(actual[:3,:3]@origin[:3,:3].T).magnitude()))
    if not (-.001<=movement[2]<=.012 and np.linalg.norm(movement[:2])<=.003 and angle<=3.):
        raise RuntimeError('抬手關節沒有沿著確認的路徑配合，先停下來')
