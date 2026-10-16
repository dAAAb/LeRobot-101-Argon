"""Pure SO101 local Cartesian proposals plus collision-mesh/table-plane screening.
No hardware, camera, controller, network or known object-size inputs.
The endpoint is gripper_frame_link: fixed printed pad, NOT two-jaw center.
"""
from __future__ import annotations
from collections.abc import Mapping
import hashlib,itertools,json,math,struct,sys,time,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from .geometry import Kinematics
NAMES=('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll')
ALL=NAMES+('gripper',)
RAD_PER_TICK=2*math.pi/4096

def inward_departure_bounds(model, start, end, envelope, joints):
    """Explicit one-way escape from a hard-valid start into normal reserve.

    This changes only limit checking at the start. The geometric uncertainty
    box remains symmetric, including its hypothetical beyond-limit portion.
    """
    if (not isinstance(joints,(tuple,list,set,frozenset)) or len(set(joints))!=len(joints)
            or any(n not in NAMES for n in joints)):
        raise ValueError('Invalid inward limit joints')
    result={}
    for n in joints:
        c=model.calibration[n];lo,hi=c['range_min'],c['range_max']
        safe_lo,safe_hi=lo+12+envelope[n],hi-12-envelope[n]
        if hasattr(model,'joint_limits'):
            a,b=model.joint_limits[n]
            lo=max(lo,math.ceil(2047+a/RAD_PER_TICK-1e-9))
            hi=min(hi,math.floor(2047+b/RAD_PER_TICK+1e-9))
            safe_lo=max(safe_lo,math.ceil(2047+a/RAD_PER_TICK+envelope[n]-1e-9))
            safe_hi=min(safe_hi,math.floor(2047+b/RAD_PER_TICK-envelope[n]+1e-9))
        direction=1 if start[n]<safe_lo else -1 if start[n]>safe_hi else 0
        if (not lo<=start[n]<=hi or safe_lo>safe_hi or not direction
                or not safe_lo<=end[n]<=safe_hi or direction*(end[n]-start[n])<=0):
            raise ValueError('向內退開必須從合法端點回到完整行程餘量內：'+n)
        result[n]={'direction':direction,'hard_bounds':[lo,hi],
                   'target_bounds':[safe_lo,safe_hi]}
    return result

def tracking_envelope(value):
    """Normalize a body-joint raw box; the jaw already has a full360 envelope.

    Scalars apply to each body axis. Mappings must name every body axis and
    may explicitly include gripper=0. No missing-axis or unknown-axis defaults.
    """
    if type(value) is int:
        bounds={n:value for n in NAMES}
    elif isinstance(value,Mapping):
        if set(value) not in (set(NAMES),set(ALL)):
            raise ValueError('Tracking envelope must name all five body joints')
        bounds=dict(value)
    else:
        raise ValueError('Tracking envelope must be an integer or joint mapping')
    if any(type(v) is not int or not 0<=v<=60 for v in bounds.values()):
        raise ValueError('Tracking envelope bounds must be integers in 0..60')
    if bounds.get('gripper',0)!=0:
        raise ValueError('Gripper raw envelope must be zero; geometry uses full360 jaw')
    return {**bounds,'gripper':0}

def xyz_transform(origin):
    T=np.eye(4)
    if origin is not None:
        T[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
        T[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
    return T

def stl_vertices(path):
    b=Path(path).read_bytes()
    if len(b)<84:raise ValueError('Truncated STL')
    count=struct.unpack('<I',b[80:84])[0]
    if len(b)!=84+count*50:raise ValueError('Expected binary STL: '+str(path))
    dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attr','<u2')])
    return np.frombuffer(b[84:],dtype=dtype)['vertices'].reshape(-1,3).astype(float)

class LocalCartesianModel:
    def __init__(self,urdf_path,mesh_root,calibration,offsets=None):
        self.urdf_path=Path(urdf_path);self.mesh_root=Path(mesh_root)
        self.calibration=calibration
        if offsets:
            from .kinematic_offsets import OffsetKinematics
            self.fk=OffsetKinematics(self.urdf_path,offsets)
        else:self.fk=Kinematics(self.urdf_path)
        self.tree=ET.parse(self.urdf_path).getroot();self.mesh_files=[];vertices={}
        for link in self.tree.findall('link'):
            for collision in link.findall('collision'):
                mesh=collision.find('geometry/mesh')
                if mesh is None:continue
                path=self.mesh_root/mesh.get('filename');self.mesh_files.append(path)
                v=stl_vertices(path)*np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
                T=xyz_transform(collision.find('origin'))
                vertices.setdefault(link.get('name'),[]).append(np.einsum('ij,kj->ki',T[:3,:3],v)+T[:3,3])
        self.vertices={}
        for link,parts in vertices.items():
            v=np.unique(np.concatenate(parts),axis=0)
            self.vertices[link]=v[ConvexHull(v).vertices]
        # Visual-only camera geometries are enclosed by boxes, marked as priors.
        camera_parts=[]
        camera=self.tree.find("link[@name='wrist_camera_link']")
        if camera is not None:
            for vis in camera.findall('visual'):
                box=vis.find('geometry/box');cyl=vis.find('geometry/cylinder')
                if box is not None:size=np.fromstring(box.get('size'),sep=' ')
                elif cyl is not None:size=np.array([2*float(cyl.get('radius'))]*2+[float(cyl.get('length'))])
                else:raise ValueError('Unscreened camera visual geometry')
                corners=np.array(list(itertools.product([-1,1],repeat=3)))*size/2
                T=xyz_transform(vis.find('origin'))
                camera_parts.append(np.einsum('ij,kj->ki',T[:3,:3],corners)+T[:3,3])
            self.vertices['wrist_camera_link']=np.concatenate(camera_parts)
        jawjoint=self.tree.find("joint[@name='gripper']")
        if not np.allclose(np.fromstring(jawjoint.find('axis').get('xyz'),sep=' '),[0,0,1]):raise ValueError('Full360 envelope assumes URDF jaw +Z axis')
        self.jaw=self.vertices.pop('moving_jaw_so101_v1_link')
        self.jaw_origin=xyz_transform(jawjoint.find('origin'))
        self.jaw_label='moving_jaw_full360_envelope'
        self.radii={link:self.radius_bounds(link,v) for link,v in self.vertices.items()}
        self.radii[self.jaw_label]=self.radius_bounds('moving_jaw_so101_v1_link',self.jaw)
        self.joint_limits={j.get('name'):(float(j.find('limit').get('lower')),float(j.find('limit').get('upper'))) for j in self.tree.findall('joint') if j.get('name') in NAMES}
    def radius_bounds(self,link,vertices):
        """Global downstream-chain radius bound, independent of joint angles."""
        bychild={j[3]:j for j in self.fk.joints}
        radius=float(np.max(np.linalg.norm(vertices,axis=1)));bounds={}
        while link in bychild:
            name,kind,parent,child,T,axis=bychild[link]
            if name in NAMES:bounds[name]=radius
            radius+=float(np.linalg.norm(T[:3,3]));link=parent
        return bounds
    def validate_raw(self,raw,*,margin_ticks=12,uncertainty_ticks=0):
        if set(raw)!=set(ALL):raise ValueError('All six measured joints required')
        envelope=tracking_envelope(uncertainty_ticks)
        for n in ALL:
            v=raw[n]
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v):raise ValueError('Bad joint '+n)
            c=self.calibration[n]
            if not c['range_min']+margin_ticks<=v-envelope[n]<=v+envelope[n]<=c['range_max']-margin_ticks:raise ValueError('Onboard range/envelope exceeded: '+n)
    def tip(self,raw):return self.fk.transforms(raw)['gripper_frame_link']
    def local_ik(self,raw,delta_base_m,*,max_joint_step_ticks=30,margin_ticks=12,orientation_origin=None):
        self.validate_raw(raw,margin_ticks=margin_ticks)
        delta=np.asarray(delta_base_m,float)
        if delta.shape!=(3,) or not np.isfinite(delta).all() or np.linalg.norm(delta)>.0050000001:raise ValueError('Local translation must be finite and <=5mm')
        if type(max_joint_step_ticks) is not int or not 1<=max_joint_step_ticks<=30:raise ValueError('Joint step must be <=30ticks')
        q0=np.array([raw[n] for n in NAMES],float);T0=self.tip(raw);pgoal=T0[:3,3]+delta
        # Tool +Y direction has two independent orientation constraints. This
        # leaves the unavoidable third rotation free for a five-DOF body. An
        # optional attempt-start reference prevents accumulated endpoint drift
        # from becoming each next step's desired orientation. Position, trust
        # region and all local error checks remain anchored at the measured raw.
        if orientation_origin is not None:
            self.validate_raw(orientation_origin,margin_ticks=margin_ticks)
        Tref=T0 if orientation_origin is None else self.tip(orientation_origin)
        direction=Tref[:3,1]
        lo=np.maximum(q0-max_joint_step_ticks,[self.calibration[n]['range_min']+margin_ticks for n in NAMES])
        hi=np.minimum(q0+max_joint_step_ticks,[self.calibration[n]['range_max']-margin_ticks for n in NAMES])
        # Respect geometric URDF body limits too; they remain a model convention.
        lo=np.maximum(lo,[2047+self.joint_limits[n][0]/RAD_PER_TICK for n in NAMES])
        hi=np.minimum(hi,[2047+self.joint_limits[n][1]/RAD_PER_TICK for n in NAMES])
        if np.any(lo>=hi) or np.any(q0<lo) or np.any(q0>hi):raise ValueError('Origin outside guarded model joint limits')
        def residual(q):
            T=self.tip({**raw,**dict(zip(NAMES,map(float,q)))})
            return np.r_[T[:3,3]-pgoal,.05*np.cross(T[:3,1],direction),1e-6*(q-q0)]
        sol=least_squares(residual,q0,bounds=(lo,hi),max_nfev=80,xtol=1e-10,ftol=1e-10,gtol=1e-10)
        if not sol.success:raise ValueError('Local IK did not converge')
        q=np.rint(sol.x).astype(int)
        if np.any(q<lo-1e-10) or np.any(q>hi+1e-10):raise ValueError('Rounded IK exceeds local joint limit')
        target={**raw,**dict(zip(NAMES,q.tolist()))};self.validate_raw(target,margin_ticks=margin_ticks)
        T1=self.tip(target);pred=T1[:3,3]-T0[:3,3]
        if np.linalg.norm(pred)>.0050000001:raise ValueError('Predicted rounded translation exceeds5mm')
        if np.linalg.norm(delta)>1e-9 and pred@delta<=0:raise ValueError('No forward Cartesian progress in trust region')
        result={'proposal_only':True,'endpoint_frame':'gripper_frame_link (fixed pad, not jaw center)',
          'measured_origin':dict(raw),'joint_targets':target,'body_joint_targets':{n:target[n] for n in NAMES},'delta_ticks':{n:float(q[i]-q0[i]) for i,n in enumerate(NAMES)},
          'requested_delta_m':delta.tolist(),'predicted_delta_m':pred.tolist(),
          'predicted_position_error_mm':float(np.linalg.norm(pred-delta)*1000),
          'full_orientation_error_deg':float(np.degrees(Rotation.from_matrix(T1[:3,:3]@T0[:3,:3].T).magnitude())),
          'held_tool_axis_error_deg':float(np.degrees(np.arccos(np.clip(T1[:3,1]@direction,-1,1)))),
          'model_bound_hit':bool(np.any(np.isclose(sol.x,lo,atol=.01)) or np.any(np.isclose(sol.x,hi,atol=.01)))}
        if orientation_origin is not None:
            result.update(orientation_reference_origin=dict(orientation_origin),
                          orientation_reference_error_deg=float(np.degrees(Rotation.from_matrix(T1[:3,:3]@Tref[:3,:3].T).magnitude())))
        return result
    def floor_minimum(self,raw,table_z_m=0.):
        T=self.fk.transforms(raw);out={}
        for link,v in self.vertices.items():
            out[link]=float(np.min(np.einsum('ij,j->i',v,T[link][2,:3]))+T[link][2,3]-table_z_m)
        H=T['gripper_link']@self.jaw_origin;r=H[2,:3]
        A=r[0]*self.jaw[:,0]+r[1]*self.jaw[:,1]
        B=-r[0]*self.jaw[:,1]+r[1]*self.jaw[:,0]
        C=r[2]*self.jaw[:,2]+H[2,3]-table_z_m
        out[self.jaw_label]=float(np.min(C-np.hypot(A,B)))
        return out
    def screen_segment(self,start,end,*,uncertainty_ticks=12,table_z_m=0.,min_clearance_m=.005,path_step_ticks=1,margin_ticks=12,stationary_limit_joints=(),inward_limit_joints=()):
        """Rigid-model infinite-plane lower bound for one segment and raw-error box.
        uncertainty_ticks is an int or a complete per-body-axis mapping (0..60),
        with optional gripper=0. It is a caller-chosen raw envelope, NOT proven
        by this helper. Caller must enforce it from measured poses continuously.
        Only base_link is exempt (base mesh overlaps its reference plane -2.4mm).
        No object, human, cable, compliance or nonadjacent self-collision model.
        margin_ticks defaults to the existing 12-tick onboard reserve. A caller
        using zero must separately enforce any moving-axis reserve; this does
        not change the uncertainty box or the mesh/table clearance calculation.
        stationary_limit_joints may name only joints with identical start/end
        values. For those unchanged holds, range validation checks the nominal
        position; the caller must enforce a measured interval clipped to actual
        onboard/URDF limits. The FULL uncertainty box still bounds geometry,
        including hypothetical points beyond those runtime position limits.
        inward_limit_joints explicitly permits a hard-valid boundary start to
        move only inward until its endpoint restores the normal moving reserve.
        The caller must enforce clipped hard bounds and monotonic inward goals;
        this method retains the FULL symmetric geometry box throughout.
        """
        envelope=tracking_envelope(uncertainty_ticks)
        if (not isinstance(stationary_limit_joints,(tuple,list,set,frozenset))
                or any(n not in NAMES for n in stationary_limit_joints)
                or len(set(stationary_limit_joints))!=len(stationary_limit_joints)
                or any(start[n]!=end[n] for n in stationary_limit_joints)):
            raise ValueError('Only unchanged body joints may use clipped stationary limits')
        stationary=tuple(n for n in NAMES if n in stationary_limit_joints)
        inward=inward_departure_bounds(self,start,end,envelope,inward_limit_joints)
        if set(stationary)&set(inward):raise ValueError('Inward joint must move')
        limit_envelope={n:0 if n in stationary else envelope[n] for n in ALL}
        if type(margin_ticks) is not int or not 0<=margin_ticks<=12:raise ValueError('Bad onboard margin')
        if not math.isfinite(table_z_m) or not math.isfinite(min_clearance_m) or min_clearance_m<0 or not 0<path_step_ticks<=2:raise ValueError('Bad table guard')
        for raw in [start,end]:
            current_envelope={n:0 if raw is start and n in inward else limit_envelope[n] for n in ALL}
            # Moving-axis reserve is enforced by inward_departure_bounds at
            # the endpoint; only those explicit starts are hard-limit checked.
            if inward:
                self.validate_raw(raw,margin_ticks=0,uncertainty_ticks=current_envelope)
                for n in ALL:
                    if raw is start and n in inward:continue
                    c=self.calibration[n]
                    if not c['range_min']+margin_ticks<=raw[n]-current_envelope[n]<=raw[n]+current_envelope[n]<=c['range_max']-margin_ticks:
                        raise ValueError('Onboard range/envelope exceeded: '+n)
            else:self.validate_raw(raw,margin_ticks=margin_ticks,uncertainty_ticks=current_envelope)
        maximum=max(abs(start[n]-end[n]) for n in NAMES)
        steps=max(1,int(math.ceil(maximum/path_step_ticks)))
        offsets=[0]  # Runtime fast path: bound the entire error box analytically.
        minima={};where={}
        for t in np.linspace(0,1,steps+1):
            center={n:float(start[n]+t*(end[n]-start[n])) for n in ALL}
            for errors in itertools.product(offsets,repeat=5):
                raw={**center,**{n:center[n]+e for n,e in zip(NAMES,errors)}}
                for link,z in self.floor_minimum(raw,table_z_m).items():
                    if z<minima.get(link,float('inf')):minima[link]=z;where[link]={'fraction':float(t),'pose_raw':raw}
        # Every actual coordinate lies within its own envelope of its nominal
        # path. The nearest nominal sample adds <=path_step/2 ticks. Sum the
        # globally bounded chord displacement of each independent joint; no
        # combinatorial uncertainty sampling is needed for this lower bound.
        residual={link:sum(2*r*math.sin((envelope[n]+path_step_ticks/2)*RAD_PER_TICK/2) for n,r in rs.items()) for link,rs in self.radii.items()}
        lower={link:z-residual[link] for link,z in minima.items()}
        failures={link:z for link,z in lower.items() if link!='base_link' and z<min_clearance_m}
        ranges={n:[min(start[n],end[n])-envelope[n],max(start[n],end[n])+envelope[n]] for n in ALL}
        return {'model_table_pass':not failures,'not_environment_collision_certificate':True,
          'table_z_m':table_z_m,'minimum_required_clearance_mm':min_clearance_m*1000,
          'actual_tracking_envelope_ticks':envelope,'envelope_is_assumed_not_measured':True,
          'limit_validation_uncertainty_ticks':limit_envelope,'stationary_limit_joints':list(stationary),
          **({'inward_limit_departures':inward} if inward else {}),
          'stationary_limit_scope':'nominal held joints only; caller must enforce clipped measured/goal bounds; geometry retains full envelope',
          'sampled_body_states':(steps+1)*len(offsets)**5,'method':'nominal-path plus analytic downstream-radius error-box bound',
          'sampled_min_clearance_mm':{l:z*1000 for l,z in minima.items()},
          'continuous_residual_bound_mm':{l:z*1000 for l,z in residual.items()},
          'continuous_model_lower_bound_mm':{l:z*1000 for l,z in lower.items()},
          'violations_mm':{l:z*1000 for l,z in failures.items()},'where':where,
          'onboard_margin_including_envelope_ticks':{n:min(a-self.calibration[n]['range_min'],self.calibration[n]['range_max']-b) for n,(a,b) in ranges.items()}}
    def propose(self,raw,delta_base_m,*,held_goals=None,uncertainty_ticks=12,min_clearance_m=.005,orientation_origin=None):
        envelope=tracking_envelope(uncertainty_ticks)
        proposal=self.local_ik(raw,delta_base_m,orientation_origin=orientation_origin)
        clearance=self.screen_segment(raw,proposal['joint_targets'],uncertainty_ticks=envelope,min_clearance_m=min_clearance_m)
        if not clearance['model_table_pass']:raise ValueError('Proposed path fails model table guard: '+str(clearance['violations_mm']))
        proposal['clearance']=clearance
        if held_goals is not None:
            # Preserve existing load compensation without silently resetting idle
            # joints. Existing physical offset must fit the screened raw box.
            self.validate_raw({**raw,**{n:held_goals[n] for n in NAMES}})
            if any(abs(held_goals[n]-raw[n])>envelope[n] for n in NAMES):raise ValueError('Existing physical hold offset exceeds screened envelope')
            physical={n:round(held_goals[n]+proposal['delta_ticks'][n]) for n in NAMES}
            self.validate_raw({**raw,**physical})
            proposal['physical_goal_targets']=physical
        return proposal
    def propose_adaptive(self,raw,delta_base_m,*,held_goals=None,uncertainty_ticks=12,
                         min_clearance_m=.005,max_position_error_mm=1.,
                         max_orientation_error_deg=2.,min_step_mm=.5,shrink_factor=.75,orientation_origin=None):
        """Propose one feasible shorter step along exactly the requested ray.

        No trajectory or remaining-distance plan is generated. Every candidate
        retains the 30tick IK trust region, onboard bounds, held-goal offsets and
        the same table/error envelope. Reobserve after the accepted single step.
        Existing propose/local_ik behavior is unchanged; this method is opt-in.
        """
        delta=np.asarray(delta_base_m,float)
        if delta.shape!=(3,) or not np.isfinite(delta).all():raise ValueError('Invalid local translation')
        requested_mm=float(np.linalg.norm(delta)*1000)
        if not 0<requested_mm<=5.0000001:raise ValueError('Adaptive translation must be positive and <=5mm')
        params=(max_position_error_mm,max_orientation_error_deg,min_step_mm,shrink_factor)
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in params):
            raise ValueError('Invalid adaptive proposal bounds')
        if not (0<max_position_error_mm<=1 and 0<max_orientation_error_deg<=2 and .1<=min_step_mm<=5 and .5<=shrink_factor<=.9):
            raise ValueError('Adaptive proposal bounds would relax existing local limits')
        envelope=tracking_envelope(uncertainty_ticks)
        # Reject invalid starting state once. Shortening must not mask an invalid
        # measured origin, stale physical hold offset or existing table violation.
        self.validate_raw(raw,uncertainty_ticks=envelope)
        if orientation_origin is not None:self.validate_raw(orientation_origin)
        if held_goals is not None:
            self.validate_raw({**raw,**{n:held_goals[n] for n in NAMES}})
            if any(abs(held_goals[n]-raw[n])>envelope[n] for n in NAMES):
                raise ValueError('Existing physical hold offset exceeds screened envelope')
        origin_clearance=self.screen_segment(raw,raw,uncertainty_ticks=envelope,min_clearance_m=min_clearance_m)
        if not origin_clearance['model_table_pass']:
            raise ValueError('Current pose fails model table guard; shortening cannot repair the origin')
        floor_mm=min(requested_mm,min_step_mm);lengths=[requested_mm]
        while lengths[-1]>floor_mm+1e-9:
            lengths.append(max(floor_mm,lengths[-1]*shrink_factor))
        attempts=[]
        for distance_mm in lengths:
            scale=distance_mm/requested_mm
            try:
                p=self.propose(raw,delta*scale,held_goals=held_goals,
                               uncertainty_ticks=envelope,min_clearance_m=min_clearance_m,
                               orientation_origin=orientation_origin)
            except ValueError as exc:
                attempts.append({'requested_distance_mm':distance_mm,'accepted':False,'reason':str(exc)})
                continue
            reason=None
            if p['predicted_position_error_mm']>max_position_error_mm:reason='position_error'
            elif p['full_orientation_error_deg']>max_orientation_error_deg:reason='orientation_error'
            elif max(abs(v) for v in p['delta_ticks'].values())>30:reason='joint_trust_region'
            attempts.append({'requested_distance_mm':distance_mm,'accepted':reason is None,
                             'predicted_position_error_mm':p['predicted_position_error_mm'],
                             'full_orientation_error_deg':p['full_orientation_error_deg'],
                             'reason':reason})
            if reason is not None:continue
            p.update(original_requested_delta_m=delta.tolist(),original_requested_distance_mm=requested_mm,
                     accepted_requested_distance_mm=distance_mm,shortening_factor=scale,
                     adaptive_shortened=scale<1-1e-12,
                     predicted_distance_mm=float(np.linalg.norm(p['predicted_delta_m'])*1000),
                     proposal_attempts=attempts,requires_new_observation_after_step=True)
            return p
        raise ValueError('No feasible same-direction local step: '+json.dumps(attempts))
    def propose_rotation(self,raw,axis,angle_deg,*,held_goals=None,uncertainty_ticks=12,
                         min_clearance_m=.005,actual_margin_ticks=12):
        """One measured-origin wrist rotation proposal, never a motor command.

        roll changes only wrist_roll; pitch changes only wrist_flex. The angle
        request is <=3deg and quantization retains the existing 30tick trust cap.
        This is rotation about the wrist joint, NOT about the fixed-pad endpoint;
        pitch can sweep that endpoint by more than the Cartesian 5mm allowance.
        Whole robot/table screening includes the unknown full360 jaw envelope.
        """
        axes={'roll':'wrist_roll','pitch':'wrist_flex'}
        if axis not in axes:raise ValueError('Rotation axis must be roll or pitch')
        if isinstance(angle_deg,bool) or not isinstance(angle_deg,(int,float)) or not math.isfinite(angle_deg) or not 0<abs(angle_deg)<=3:
            raise ValueError('One wrist rotation request must be nonzero and <=3deg')
        envelope=tracking_envelope(uncertainty_ticks)
        if type(actual_margin_ticks) is not int or not 0<=actual_margin_ticks<=12 or any(actual_margin_ticks>envelope[n] for n in NAMES):
            raise ValueError('Actual motion tube must fit the screened per-axis envelope')
        self.validate_raw(raw,margin_ticks=0)
        joint=axes[axis];requested_ticks=angle_deg/(RAD_PER_TICK*180/math.pi)
        ticks=int(np.clip(round(requested_ticks),-30,30))
        if ticks==0:raise ValueError('Requested wrist rotation is below one encoder tick')
        target={**raw,joint:raw[joint]+ticks}
        physical_hold=None
        if held_goals is not None:
            physical_hold={n:held_goals[n] for n in NAMES}
            if any(abs(physical_hold[n]-raw[n])>envelope[n] for n in NAMES):
                raise ValueError('Existing physical hold offset exceeds screened envelope')
        else:physical_hold={n:raw[n] for n in NAMES}
        physical={joint:round(physical_hold[joint]+ticks)}
        # Use the same contract as complete motion segments. Only this wrist
        # joint is commanded; unchanged holds near a hard limit are legal.
        # Their runtime intervals are clipped, but their FULL symmetric error
        # box still participates in the mesh/table screen. Moving joints keep
        # the original 12-tick reserve around both their box and physical goals.
        from .motion_program import _segment_bounds, _validate_segment_limits
        from .local_feedback import check_bounds
        held_pose={**raw,**physical_hold}
        physical_pose={**held_pose,**physical}
        active=[joint];stationary=[n for n in NAMES if n!=joint]
        _validate_segment_limits(self,raw,held_pose,target,physical_pose,active,envelope)
        physical_bounds,measured_bounds,stationary_limits=_segment_bounds(
            self,raw,target,active,envelope,{n:actual_margin_ticks for n in NAMES})
        check_bounds(held_pose,physical_bounds)
        check_bounds(physical_pose,physical_bounds)
        clearance=self.screen_segment(raw,target,uncertainty_ticks=envelope,min_clearance_m=min_clearance_m,
                                      margin_ticks=0,stationary_limit_joints=stationary)
        if not clearance['model_table_pass']:
            raise ValueError('Rotation path fails model table guard: '+str(clearance['violations_mm']))
        first=self.fk.transforms(raw);last=self.fk.transforms(target)
        tip_delta=last['gripper_frame_link'][:3,3]-first['gripper_frame_link'][:3,3]
        camera_delta=last['wrist_camera_optical'][:3,3]-first['wrist_camera_optical'][:3,3]
        achieved=ticks*RAD_PER_TICK*180/math.pi
        return {'proposal_only':True,'motion_kind':'wrist_rotation','axis':axis,'active_joint':joint,
            'measured_origin':dict(raw),'joint_targets':target,
            'body_joint_targets':{n:target[n] for n in NAMES},'active_joint_targets':{joint:target[joint]},
            'physical_goal_targets':physical,'preserved_physical_hold_goals':{n:physical_hold[n] for n in NAMES if n!=joint},
            'delta_ticks':{n:ticks if n==joint else 0 for n in NAMES},
            'requested_angle_deg':float(angle_deg),'accepted_angle_deg':achieved,
            'angle_quantization_error_deg':achieved-angle_deg,'limited_by_30tick_trust_region':abs(requested_ticks)>30,
            'full_orientation_change_deg':float(np.degrees(Rotation.from_matrix(last['gripper_frame_link'][:3,:3]@first['gripper_frame_link'][:3,:3].T).magnitude())),
            'endpoint_frame':'gripper_frame_link (fixed pad, not jaw center)',
            'predicted_fixed_pad_delta_mm':(tip_delta*1000).tolist(),
            'predicted_fixed_pad_travel_mm':float(np.linalg.norm(tip_delta)*1000),
            'predicted_camera_delta_mm':(camera_delta*1000).tolist(),
            'predicted_camera_baseline_mm':float(np.linalg.norm(camera_delta)*1000),
            'measured_bounds':measured_bounds,'physical_goal_bounds':physical_bounds,
            'stationary_limit_joints':stationary,'stationary_joint_limits':stationary_limits,
            'stationary_goal_resends':False,
            'clearance':clearance,'gripper_command_included':False,'requires_new_observation_after_step':True,
            'not_environment_collision_certificate':True}

def default_model(project_root=None,mesh_root=None,offsets=None):
    root=Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    mesh_root=Path(mesh_root) if mesh_root is not None else root/'assets/so101'
    if offsets is None:
        from .kinematic_offsets import runtime_offsets
        offsets=runtime_offsets(root)
    return LocalCartesianModel(root/'assets/so101_follower_cam.urdf',mesh_root,json.loads((root/'config/follower-calibration.json').read_text()),
                               offsets)
