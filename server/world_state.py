"""Wrist-only, parameterized world state; serialization never grants motion.

Encoder readings, CAD-derived poses, visual labels and experimental surface
reconstruction are distinct evidence. Unknown depth stays null. Nothing in
this module opens a camera, reads serial, infers hidden geometry or accepts an
engineering observer measurement as production input.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import math
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .local_cartesian import ALL

MAX_AGE_S = 5.
MAX_OBJECTS = 8


def finite(value, low=-1e6, high=1e6):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError('Expected a bounded finite measurement')
    return float(value)


def _text(value, maximum=120):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError('Expected a bounded evidence identifier')
    return value.strip()


def _pose(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(matrix[:3,:3].T @ matrix[:3,:3], np.eye(3), atol=1e-8, rtol=0)
            or not np.isclose(np.linalg.det(matrix[:3,:3]), 1, atol=1e-8, rtol=0)
            or np.linalg.norm(matrix[:3,3]) > 2):
        raise ValueError('Invalid robot model pose')
    return {'position_m': matrix[:3,3].round(6).tolist(),
            'quaternion_xyzw': Rotation.from_matrix(matrix[:3,:3]).as_quat().round(7).tolist(),
            'coordinate_frame': 'robot_base_model', 'evidence': 'encoder_and_cad_prior',
            'metric_accuracy_verified': False}


def _box(value):
    if not isinstance(value, dict) or set(value) != {'x','y','w','h'}:
        raise ValueError('Expected normalized image bounds')
    box = {k: finite(v, 0, 1) for k,v in value.items()}
    if box['w'] <= 0 or box['h'] <= 0 or box['x']+box['w'] > 1.01 or box['y']+box['h'] > 1.01:
        raise ValueError('Invalid normalized image bounds')
    box['w']=min(box['w'],1-box['x']);box['h']=min(box['h'],1-box['y'])
    if not box['w'] or not box['h']:raise ValueError('Object bounds are outside the image')
    return box


def _surface(spatial, index, name, frame):
    """Only accept reconstruction explicitly bound to this observation/object.

    Old spatial-latest files have no such binding and deliberately cannot be
    promoted into the current scene. Even accepted values remain experimental
    visible-surface estimates, never object centers or collision volumes.
    """
    if not spatial:return None
    if spatial.get('source') != 'wrist-only':
        raise ValueError('Production state rejects observer geometry')
    binding = spatial.get('evidence', {})
    frames = binding.get('observations', [])
    if any(f.get('camera_generation') != frame['camera_generation'] for f in frames):
        raise ValueError('Reconstruction spans different camera sessions')
    if not any(f.get('id') == frame['id'] and f.get('camera_generation') == frame['camera_generation']
               and f.get('capture_monotonic') == frame['capture_monotonic'] for f in frames):
        return None
    matches = [o for o in spatial.get('objects', []) if o.get('object_index') == index
               and o.get('scene_frame_id') == frame['id'] and o.get('name') == name
               and o.get('status') == 'estimated']
    if len(matches) != 1:return None
    value = matches[0]
    if value.get('position_reference') != 'median-visible-surface-not-object-center':return None
    xyz = value.get('position_m')
    if not isinstance(xyz, list) or len(xyz) != 3:raise ValueError('Invalid surface estimate')
    xyz = [finite(v, -2, 2) for v in xyz]
    disagreement = value.get('same_point_p95_disagreement_m')
    if disagreement is not None:disagreement = finite(disagreement, 0, 2)
    return {'position_m': xyz, 'coordinate_frame': 'robot_base_model',
            'reference': 'median_visible_surface', 'evidence': 'wrist_multiview_estimate',
            'consistency_p95_m': disagreement, 'consistency_is_accuracy': False,
            'repeatable': spatial.get('stability') == 'repeatable' and spatial.get('needs_more_views') is False,
            'metric_accuracy_verified': False, 'collision_volume_known': False}


def build_world_state(frame, kinematics, *, objects=None, object_frames=None, spatial=None, now=None):
    if not isinstance(frame, dict) or frame.get('source') != 'wrist':
        raise ValueError('World state only accepts wrist observations')
    if spatial is not None and spatial.get('source') != 'wrist-only':
        raise ValueError('Production state rejects observer geometry')
    fid = _text(frame.get('id'));generation = _text(frame.get('camera_generation'))
    stamp = finite(frame.get('capture_monotonic'), 0, 1e12)
    now = time.monotonic() if now is None else finite(now, 0, 1e12)
    if stamp > now:raise ValueError('Observation timestamp is in the future')
    raw = frame.get('joints')
    if not isinstance(raw, dict) or set(raw) != set(ALL):raise ValueError('All measured joints are required')
    raw = {k: finite(v, 0, 4095) for k,v in raw.items()}
    if any(int(v) != v for v in raw.values()):raise ValueError('Expected integer encoder positions')
    raw = {k: int(v) for k,v in raw.items()}
    objects_analyzed=objects is not None
    if objects is None:objects=()
    if not isinstance(objects, (list,tuple)) or len(objects) > MAX_OBJECTS:raise ValueError('Too many observed objects')
    if object_frames is None:object_frames = [frame]
    if any(f.get('source') != 'wrist' or f.get('camera_generation') != generation for f in object_frames):
        raise ValueError('Objects must refer to the same wrist camera session')
    transforms = kinematics.transforms(raw)
    poses = {'fixed_finger_pad': _pose(transforms['gripper_frame_link']),
             'wrist_camera': _pose(transforms['wrist_camera_optical'])}
    observed = []
    for index,obj in enumerate(objects):
        fi = obj.get('frame_index', 0)
        if type(fi) is not int or not 0 <= fi < len(object_frames):raise ValueError('Invalid object frame')
        ref = object_frames[fi]
        # A label in a prior view is history, not a current pose for an object.
        if ref.get('id') != fid or ref.get('capture_monotonic') != stamp:continue
        original_name=obj.get('name')
        if not isinstance(original_name,str):raise ValueError('Invalid visual object label')
        name=original_name.strip()[:120] or 'unlabeled object'
        observed.append({'id': f'{fid}:object_{index}', 'name': name,
            'image_bbox': _box(obj.get('bbox')), 'identification_confidence': finite(obj.get('confidence'), 0, 1),
            'evidence': 'astra_visual_observation', 'surface_3d': _surface(spatial,index,original_name,frame),
            'object_center_m': None, 'dimensions_3d_m': None, 'collision_volume': None})
    relations = []
    for i,a in enumerate(observed):
        for b in observed[i+1:]:
            A=a['image_bbox'];B=b['image_bbox']
            if A['x']+A['w'] <= B['x']:relations.append({'subject':a['id'],'predicate':'image_left_of','object':b['id']})
            elif B['x']+B['w'] <= A['x']:relations.append({'subject':b['id'],'predicate':'image_left_of','object':a['id']})
            if A['y']+A['h'] <= B['y']:relations.append({'subject':a['id'],'predicate':'image_above','object':b['id']})
            elif B['y']+B['h'] <= A['y']:relations.append({'subject':b['id'],'predicate':'image_above','object':a['id']})
    return {'version':1, 'domain':'production', 'source':'wrist', 'observation_id':fid,
        'camera_generation':generation, 'captured_monotonic':stamp, 'age_s':round(now-stamp,3),
        'fresh':now-stamp <= MAX_AGE_S, 'motion_authorized':False,
        'robot':{'measured_joint_ticks':raw, 'poses_model':poses,
                 'jaw_aperture_m':None, 'jaw_center_m':None, 'held_object_verified':None},
        'objects':observed, 'objects_analyzed':objects_analyzed, 'relations':relations,
        'unknowns':['external_obstacle_clearance','hidden_object_geometry','calibrated_hand_eye_accuracy'],
        'table':{'evidence':'model_prior','plane_in_robot_base_model':[0.,0.,1.,0.],
                 'observed_plane':False},
        'provenance':{'inputs':['wrist_image','measured_encoders','robot_cad_prior'],
                      'observer_used':False, 'object_positions_are_surface_estimates':True}}


def policy_state(state, *, now=None):
    """Bounded English-key summary for Astra/Jev, without raw ticks or paths."""
    if state.get('domain') != 'production' or state.get('source') != 'wrist' or state.get('provenance',{}).get('observer_used') is not False:
        raise ValueError('Policy state rejects engineering observer data')
    poses = state['robot']['poses_model']
    now=time.monotonic() if now is None else finite(now,0,1e12)
    age=now-state['captured_monotonic']
    objects=state['objects'][:4];ids={o['id'] for o in objects}
    return {'source':'wrist', 'observation_id':state['observation_id'], 'age_s':round(max(0,age),3),
        'fresh':bool(state['fresh'] and 0<=age<=MAX_AGE_S), 'motion_authorized':False,
        'tip_model_position_m':poses['fixed_finger_pad']['position_m'],
        'camera_model_position_m':poses['wrist_camera']['position_m'],
        'pose_accuracy':'uncalibrated_encoder_and_cad_prior',
        'object_observation_status':'analyzed' if state['objects_analyzed'] else 'not_yet_analyzed',
        'object_count':len(state['objects']) if state['objects_analyzed'] else None,
        'objects_truncated':len(state['objects'])>4,
        'objects':[{'id':o['id'],'name':o['name'],'image_bbox':o['image_bbox'],
                    'surface_position_m':o['surface_3d']['position_m'] if o['surface_3d'] else None,
                    'depth_evidence':'experimental_wrist_multiview' if o['surface_3d'] else 'unknown',
                    'surface_consistency_p95_m':o['surface_3d']['consistency_p95_m'] if o['surface_3d'] else None,
                    'surface_repeatable':o['surface_3d']['repeatable'] if o['surface_3d'] else None,
                    'reference':'visible_surface_not_object_center',
                    'metric_accuracy_verified':False,'dimensions_3d_m':None} for o in objects],
        'relations':[deepcopy(r) for r in state['relations'] if r['subject'] in ids and r['object'] in ids],
        'unknowns':list(state['unknowns'])}


class WorldStateMemory:
    """Short local memory. Never carry object locations into a newer capture."""
    def __init__(self, kinematics, capacity=24):
        self.kinematics=kinematics;self._states=deque(maxlen=capacity);self._lock=threading.RLock()

    def remember(self, frame, **kwargs):
        state=build_world_state(frame,self.kinematics,**kwargs)
        with self._lock:
            if self._states and self._states[-1]['camera_generation'] != state['camera_generation']:
                if state['captured_monotonic'] <= self._states[-1]['captured_monotonic']:
                    state['fresh']=False;state['stale_reason']='superseded_camera_generation'
                    return deepcopy(state)
                self._states.clear()
            # A delayed inference may enrich its old observation, but must not
            # overwrite a more recent capture at the head of memory.
            by_id={s['observation_id']:s for s in self._states}
            old=by_id.get(state['observation_id'])
            if old and (old['captured_monotonic'] != state['captured_monotonic'] or
                        old['robot']['measured_joint_ticks'] != state['robot']['measured_joint_ticks']):
                raise ValueError('Observation identity was reused for different measurements')
            if old and old['objects_analyzed'] and not state['objects_analyzed']:
                state['objects']=deepcopy(old['objects']);state['relations']=deepcopy(old['relations'])
                state['objects_analyzed']=True
            by_id[state['observation_id']]=state
            ordered=sorted(by_id.values(),key=lambda s:s['captured_monotonic'])[-self._states.maxlen:]
            self._states=deque(ordered,maxlen=self._states.maxlen)
        return deepcopy(state)

    def latest(self, *, camera_generation=None, camera_fresh=False, now=None):
        now=time.monotonic() if now is None else finite(now,0,1e12)
        with self._lock:
            if not self._states:return {'version':1,'domain':'production','source':'wrist','available':False,'reason':'尚未取得可用的腕部狀態','motion_authorized':False}
            value=deepcopy(self._states[-1])
        age=now-value['captured_monotonic']
        value['age_s']=round(max(0,age),3)
        value['fresh']=bool(0<=age<=MAX_AGE_S and camera_fresh and camera_generation==value['camera_generation'])
        value['available']=True
        value['current_observation_available']=value['fresh']
        if not value['fresh']:value['stale_reason']='需要新的腕部畫面，舊狀態僅供回顧'
        return value
