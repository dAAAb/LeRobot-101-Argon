"""Pure proposal for a staged observation trajectory. NEVER accesses hardware.
Parent controller must validate live camera, fresh measured pose, physical biased
motor goals, and actual modeled/environmental clearance before each movement.
"""
from math import isfinite
JOINTS=('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper')
LOCAL_CENTER=dict(zip(JOINTS,[2161,1118,2932,2330,986,1523]))
GAZE_AXES=('shoulder_pan','wrist_flex')

def require_local_gaze_region(pose,*,moving=False,active_axis=None):
    """Specific engineering-screened region, not a generic collision check."""
    if set(pose)!=set(JOINTS):raise ValueError('Missing measured local-gaze pose')
    if moving and active_axis not in (*GAZE_AXES,'saved'):raise ValueError('Missing screened path segment')
    for n in JOINTS:
        v=pose[n];limit=4
        if moving and n==active_axis:limit=24
        if moving and active_axis=='saved' and n in GAZE_AXES:limit=8
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not isfinite(v) or abs(v-LOCAL_CENTER[n])>limit:
            raise ValueError('Pose outside the screened local-gaze region: '+n)

def build_local_gaze_plan(origin,calibration,*,return_target,bias=None):
    """Nine observations: center and signed pan/wrist excursions, ±20 ticks.
    No roll, body-translation, gripper command or object approach is proposed.
    """
    from math import ceil
    validate_pose(origin,calibration,bias=bias);require_local_gaze_region(origin)
    if set(return_target)!=set(GAZE_AXES) or any(abs(return_target[n]-origin[n])>4 for n in GAZE_AXES):
        raise ValueError('Saved gaze hold differs from the measured center')
    saved={**origin,**return_target};previous=dict(origin)
    # The first interpolated command starts at saved physical hold goals.
    # Both axes must already be in the strict center corridor before probing.
    require_local_gaze_region(validate_pose(saved,calibration,bias=bias))
    stages=[{'label':'center_initial','validation_group':'center','targets':dict(origin),'waypoints':[],'waypoint_regions':[]}]
    for axis in GAZE_AXES:
        for sign in (1,-1):
            target={**origin,axis:origin[axis]+sign*20}
            for label,end in [(axis+('+' if sign>0 else '-'),target),('center_return',dict(origin))]:
                segments=[(end,axis)]
                if label=='center_return' and axis==GAZE_AXES[-1] and sign==-1:segments.append((saved,'saved'))
                waypoints=[];regions=[]
                for end,region in segments:
                    physical=validate_pose(end,calibration,bias=bias);require_local_gaze_region(physical,moving=True,active_axis=region)
                    steps=max(1,ceil(max(abs(end[n]-previous[n]) for n in GAZE_AXES)/20))
                    waypoints.extend({n:round(previous[n]+(end[n]-previous[n])*k/steps) for n in GAZE_AXES} for k in range(1,steps+1))
                    regions.extend([region]*steps);previous=end
                stages.append({'label':label,'validation_group':'center' if label=='center_return' else label,
                               'targets':end,'waypoints':waypoints,'waypoint_regions':regions})
    return stages

def validate_pose(pose,calibration,margin=12,bias=None):
    if set(pose)!=set(JOINTS):raise ValueError('Every target must explicitly contain all six joints')
    if not isinstance(margin,int) or isinstance(margin,bool) or margin<0:raise ValueError('Invalid range margin')
    physical={}
    for name in JOINTS:
        value=pose[name]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not isfinite(value) or int(value)!=value:
            raise ValueError('Expected finite integer raw encoder target: '+name)
        extra=0 if bias is None else bias.get(name,0)
        if isinstance(extra,bool) or not isinstance(extra,(int,float)) or not isfinite(extra):raise ValueError('Invalid command bias')
        physical[name]=round(value+extra)
        c=calibration[name]
        if not c['range_min']+margin<=value<=c['range_max']-margin:raise ValueError('Logical target outside guarded range: '+name)
        if not c['range_min']+margin<=physical[name]<=c['range_max']-margin:raise ValueError('Biased motor target outside guarded range: '+name)
    return physical

def build_plan(origin,calibration,framing_ticks=70,return_target=None):
    """Offsets always use the SINGLE fresh measured origin. Captures are required
    at every settled stage. Return paths retrace the screened intermediate poses.
    return_target may be the saved logical hold goals; if distinct, its final
    segment must also be checked against the current geometry before execution.
    """
    origin=dict(origin);validate_pose(origin,calibration)
    if isinstance(framing_ticks,bool) or not isinstance(framing_ticks,int) or not 0<=framing_ticks<=70:raise ValueError('Framing exceeds screened range')
    finish=dict(origin if return_target is None else return_target);validate_pose(finish,calibration)
    framed={**origin,'wrist_flex':origin['wrist_flex']+framing_ticks}
    def at(amount):
        target=dict(framed)
        target['shoulder_lift']+=amount;target['elbow_flex']-=2*amount;target['wrist_flex']+=amount
        validate_pose(target,calibration)
        return target
    stages=[]
    def add(label,target,group,motion=True):
        validate_pose(target,calibration)
        stages.append({'label':label,'targets':dict(target),'validation_group':group,'capture_after_settling':True,'motion':motion})
    add('reference',origin,'unframed',False);add('framed',at(0),'a0')
    for n in (50,100,150):add('outward_a'+str(n),at(n),'a'+str(n))
    for n in (100,50,0):add('return_a'+str(n),at(n),'a'+str(n))
    # A saved hold goal can settle several ticks away because of hysteresis.
    # It is still a return visit, not an independent calibration validation view.
    add('restore_saved_pose',finish,'unframed')
    return {'origin':origin,'framing_ticks':framing_ticks,'framed_origin':framed,'stages':stages,
            'capture_source':'wrist-only','all_targets_are_absolute':True,
            'model_clearance_scope':'Current collision-mesh URDF with raw-minus-2047 joint convention and table z=0; not an environmental collision certificate',
            'validation':'Withhold all visits in a validation_group together; do not use planned targets as measured camera poses',
            'grasp_ready':False}

def orientation_views(framed,calibration,bias=None):
    """Distinct rotation axes, each excursion via its common center."""
    from math import ceil
    views=[];previous=dict(framed)
    for label,goal in [('roll_plus',dict(framed,wrist_roll=framed['wrist_roll']+100)),
                       ('roll_minus',dict(framed,wrist_roll=framed['wrist_roll']-100)),
                       ('pan_minus',dict(framed,shoulder_pan=framed['shoulder_pan']-70)),
                       ('pan_plus',dict(framed,shoulder_pan=framed['shoulder_pan']+70)),
                       ('framed_repeat',dict(framed))]:
        waypoints=[]
        for end in (framed,goal):
            count=max(ceil(abs(end['wrist_roll']-previous['wrist_roll'])/50),
                      ceil(abs(end['shoulder_pan']-previous['shoulder_pan'])/35))
            start=dict(previous)
            for k in range(1,count+1):
                p={n:round(start[n]+(end[n]-start[n])*k/count) for n in JOINTS}
                validate_pose(p,calibration,bias=bias);waypoints.append(p)
            previous=dict(end)
        views.append({'label':label,'targets':goal,'waypoints':waypoints,
                      'validation_group':'a0' if label=='framed_repeat' else label,'motion':True})
    return views
