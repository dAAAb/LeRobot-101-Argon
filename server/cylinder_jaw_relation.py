"""Camera-relative cylinder/jaw error analysis, with no motion authorization.

Absolute camera/FK uncertainty remains in the original cylinder estimate.
Only a coherent change of the shared base coordinate frame cancels here. Each
relative pose/edge/mount perturbation is refitted before transforming both the
object and reference camera; marginal world-center errors are never subtracted.
This describes the CURRENT observed jaw, not accuracy at a future motor pose.
"""
from copy import deepcopy
import math

import cv2
import numpy as np

from .cylinder_silhouette import (_basis, _number, _plane, _solve, _views,
                                  fit_cylinder_silhouettes)


def _jaw(value, reference_record):
    raw = reference_record.get('measured_joints', {}).get('gripper')
    if (not isinstance(value, dict) or value.get('source') != 'wrist-only'
            or value.get('camera_calibration_fingerprint') != reference_record['calibration_fingerprint']
            or value.get('reference_frame_id') != reference_record['frame_id']
            or value.get('camera_generation') != reference_record['camera_generation']
            or type(raw) is not int or not 0 <= raw <= 4095
            or type(value.get('jaw_raw')) is not int or value['jaw_raw'] != raw
            or not isinstance(value.get('jaw_calibration_fingerprint'), str)
            or not value['jaw_calibration_fingerprint']):
        raise ValueError('Bind jaw opening to the same current wrist frame, generation and calibration')
    fixed, moving = (np.asarray(value[k], float) for k in ('fixed_pad_camera_m','moving_pad_camera_m'))
    if (fixed.shape != (3,) or moving.shape != (3,) or not np.isfinite([fixed,moving]).all()
            or np.linalg.norm(moving-fixed) < 1e-6):
        raise ValueError('Need two distinct finite pad points in the reference camera frame')
    tcp = _number(value['tcp_position_uncertainty_m'], 'jaw TCP uncertainty')
    aperture = _number(value['aperture_uncertainty_m'], 'jaw aperture uncertainty')
    return fixed, moving, tcp, aperture


def _relation(solution, axis, reference, fixed, moving):
    """Six coupled outputs: relative center(3), radius, left/right side gaps."""
    rotation, translation = reference[:3,:3], reference[:3,3]
    center = rotation.T @ (solution[:3]-translation)
    upright = rotation.T @ axis
    midpoint = (fixed+moving)/2
    # The cylinder centerline's axial anchor is arbitrary. Compare its point
    # nearest the pad midpoint; unknown height remains a separate constraint.
    center = center+upright*float((midpoint-center)@upright)
    closing = moving-fixed; closing /= np.linalg.norm(closing)
    transverse = closing-upright*float(closing@upright)
    length = np.linalg.norm(transverse)
    if length < .1:
        raise ValueError('Jaw closing axis cannot resolve the cylinder lateral gap')
    transverse /= length
    left = float((center-fixed)@transverse)-solution[3]
    right = float((moving-center)@transverse)-solution[3]
    return np.r_[center-midpoint, solution[3], left, right]


def analyze_cylinder_jaw_relation(records, *, reference_frame_id, jaw_camera,
                                 axis_base, uncertainty, support_plane=None):
    """Retain full absolute fit; derive a separately bound local error budget.

    The returned clearance numbers are conditional linearized sensitivities,
    not confidence intervals or contact permission. A caller must keep absolute
    path accuracy, insertion depth/height, tracking and obstacle checks.
    """
    absolute = fit_cylinder_silhouettes(records, shape_intent='upright_cylinder',
        axis_base=axis_base, uncertainty=uncertainty, support_plane=support_plane)
    views = _views(records); axis, _ = _basis(axis_base); plane = _plane(support_plane)
    indices = [i for i,row in enumerate(records) if row['frame_id'] == reference_frame_id]
    if len(indices) != 1 or indices[0] != len(records)-1:
        raise ValueError('The reference must be the current/last supplied wrist frame')
    reference_index = indices[0]
    fixed,moving,tcp_error,aperture_error = _jaw(jaw_camera,records[reference_index])
    scales = absolute['assumptions']
    nominal = np.r_[absolute['centerline_point_base_m'], absolute['radius_m']]
    reference = views[reference_index]['transform']
    value = _relation(nominal, axis, reference, fixed, moving)

    def evaluate(change, step, *, plane_change=None):
        values=[]
        for sign in (-1.,1.):
            modified=deepcopy(views); changed_plane=deepcopy(plane)
            changed_axis=change(modified,axis.copy(),sign*step)
            if plane_change is not None:
                changed_axis=plane_change(changed_plane,changed_axis,sign*step)
            fitted=_solve(modified,changed_axis,changed_plane)[0]
            # The reference pose receives the SAME perturbation as its view.
            # Holding this transform fixed would recreate the false base error.
            values.append(_relation(fitted,changed_axis,modified[reference_index]['transform'],fixed,moving))
        return (values[1]-values[0])/(2*step)

    shared={}; independent={}; mount={}
    def add(destination,name,columns,scale):
        if scale:
            jacobian=np.column_stack(columns)
            destination[name]=jacobian@jacobian.T*scale**2

    for i,view in enumerate(views):
        for kind,scale in (('translation',scales['relative_translation_m']),
                           ('rotation',math.radians(scales['relative_rotation_deg']))):
            if not scale:continue
            columns=[]
            for j in range(3):
                def change(v,ax,amount,i=i,j=j,kind=kind):
                    if kind=='translation':v[i]['transform'][j,3]+=amount
                    else:v[i]['transform'][:3,:3]=cv2.Rodrigues(np.eye(3)[j]*amount)[0]@v[i]['transform'][:3,:3]
                    return ax
                columns.append(evaluate(change,1e-5))
            add(independent,f'view_{i}_{kind}',columns,scale)
        if scales['pixel_edge_mislocalization_px']:
            for side_index,side in enumerate(view['sides']):
                direction=side[-1]-side[0];direction/=np.linalg.norm(direction)
                normal=np.array([-direction[1],direction[0]])
                for mode in ('shift','tilt'):
                    weights=np.ones(len(side)) if mode=='shift' else np.linspace(-1.,1.,len(side))
                    def change(v,ax,amount,i=i,side_index=side_index,normal=normal,weights=weights):
                        v[i]['sides'][side_index]+=amount*weights[:,None]*normal
                        return ax
                    add(shared,f'view_{i}_side_{side_index}_{mode}',[evaluate(change,.01)],
                        scales['pixel_edge_mislocalization_px'])
            if plane is not None and len(view['foot']):
                columns=[]
                for j in range(2):
                    def change(v,ax,amount,i=i,j=j):
                        v[i]['foot'][:,j]+=amount
                        return ax
                    columns.append(evaluate(change,.01))
                add(shared,f'view_{i}_bottom_contour',columns,scales['pixel_edge_mislocalization_px'])
    if scales['focal_relative']:
        def focal(v,ax,amount):
            for view in v:
                view['k'][0,0]*=1+amount;view['k'][1,1]*=1+amount
            return ax
        add(shared,'shared_focal_object_response',[evaluate(focal,1e-5)],scales['focal_relative'])
    if plane is not None:
        for kind,scale in (('translation',scales['support_plane_translation_m']),
                           ('rotation',math.radians(scales['support_plane_rotation_deg']))):
            if not scale:continue
            columns=[]
            for j in range(3):
                def change(p,ax,amount,j=j,kind=kind):
                    if kind=='translation':p['point_base_m'][j]+=amount
                    else:
                        rotation=cv2.Rodrigues(np.eye(3)[j]*amount)[0]
                        p['normal_base']=rotation@p['normal_base']
                        ax=rotation@ax
                    return ax
                columns.append(evaluate(lambda v,ax,a:ax,1e-5,plane_change=change))
            add(shared,'relative_support_plane_'+kind,columns,scale)
    for kind,scale in (('translation',scales['relative_translation_m']),
                       ('rotation',math.radians(scales['relative_rotation_deg']))):
        if not scale:continue
        columns=[]
        for j in range(3):
            def change(v,ax,amount,j=j,kind=kind):
                delta=np.eye(3)[j]*amount
                for view in v:
                    rotation=view['transform'][:3,:3]
                    if kind=='translation':view['transform'][:3,3]+=rotation@delta
                    else:view['transform'][:3,:3]=rotation@cv2.Rodrigues(delta)[0]
                return ax
            columns.append(evaluate(change,1e-5))
        add(mount,'shared_camera_local_'+kind,columns,scale)
    # Pure shared SE(3) gauge has exactly zero relative derivative. The full
    # base empirical magnitude is retained in `absolute_estimate`, not erased
    # or reclassified as a measured, cancellable FK error at future poses.
    modes={}
    for name,contributions in (('independent_views',{**shared,**independent}),
                                ('shared_camera_local_mount',{**shared,**mount})):
        covariance=sum(contributions.values(),np.zeros((6,6)))
        if not np.isfinite(covariance).all():raise ValueError('Nonfinite relative relation sensitivity')
        modes[name]=dict(covariance=covariance.tolist(),
            left_gap_sensitivity_m=float(np.sqrt(max(0.,covariance[4,4]))),
            right_gap_sensitivity_m=float(np.sqrt(max(0.,covariance[5,5]))),
            contributions={k:v.tolist() for k,v in contributions.items()})
    alternatives=[_relation(_solve(views[:i]+views[i+1:],axis,plane)[0],axis,reference,fixed,moving)
                  for i in range(len(views))]
    spread=np.max(np.abs(np.asarray(alternatives)-value),axis=0)
    left=max(spread[4],*(m['left_gap_sensitivity_m'] for m in modes.values()))
    right=max(spread[5],*(m['right_gap_sensitivity_m'] for m in modes.values()))
    # Preserve both original jaw allowances. Their correlation with object
    # focal/edge errors is not identified, so no cancellation is claimed.
    jaw_allowance=tcp_error+aperture_error/2
    return dict(status='conditional-relative-analysis',source='wrist-only',
        reference_frame='wrist_camera_optical',reference_frame_id=reference_frame_id,
        camera_generation=records[0]['camera_generation'],
        camera_calibration_fingerprint=records[0]['calibration_fingerprint'],
        jaw_calibration_fingerprint=jaw_camera['jaw_calibration_fingerprint'],
        jaw_raw=jaw_camera['jaw_raw'],
        absolute_estimate=absolute,absolute_path_budget_unchanged=True,
        relative_center_nearest_jaw_camera_m=value[:3].tolist(),radius_m=float(value[3]),
        left_gap_m=float(value[4]),right_gap_m=float(value[5]),
        left_object_gap_sensitivity_m=float(left),right_object_gap_sensitivity_m=float(right),
        retained_jaw_allowance_per_side_m=jaw_allowance,
        left_conditional_gap_lower_m=float(value[4]-left-jaw_allowance),
        right_conditional_gap_lower_m=float(value[5]-right-jaw_allowance),
        uncertainty_modes=modes,leave_one_view_out_relation_spread=spread.tolist(),
        cancelled_mode='shared rigid base coordinate change applied to object, support plane and reference camera',
        cancellation_scope='Only the explicit shared-coordinate nuisance cancels. An aggregate empirical base/FK error is not evidence that all physical calibration error is this gauge.',
        retained_errors=['relative view pose','shared camera-local mount','body/support pixels',
                         'focal object response','independent support plane','jaw TCP','jaw aperture'],
        unmodelled_joint_correlations='Jaw focal/contour correlation is not identified; its original allowance remains.',
        limitation='Current observed jaw only. Absolute FK/path error, insertion depth, axial extent, tracking and obstacles still need separate checks.',
        planning_eligible=False,execution_authorized=False,grasp_ready=False)
