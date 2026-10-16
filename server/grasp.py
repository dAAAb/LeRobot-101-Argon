"""Wrist-only incremental manipulation. Model proposals never write motors.

Measured poses anchor every short step; approximate CAD geometry only screens
the table. Images decide object clearance except for one explicitly requested
manual wrist wave, which retains the controller's motion and camera bounds.
"""
from __future__ import annotations
import hashlib,json,time,threading
from pathlib import Path
import cv2,numpy as np
from .local_cartesian import NAMES,default_model
from .gripper_power import SessionCancelled
from .motion_policy import LOCAL_MOVE_MIN_CONFIDENCE,WAVE_MIN_CONFIDENCE,classify_translation_progress

# Existing visual-servo policy for one <=5 mm move followed by a new image.
# Empty-hand steps, waving and contact use separate task-acceptance policies.
# Only a retreat fallback preference; never suppress a screened upward candidate
# or treat this CAD height as object clearance.
WAVE_PREFERRED_HEIGHT_M=.18
# Greetings and visual scans may finish within ~0.88 degrees. Image poses
# remain measured, never replaced by requested angles. The tolerance stays
# inside the existing 12-tick screened tracking margin.
WAVE_ENDPOINT_TOLERANCE_TICKS=10
OBSERVATION_ENDPOINT_TOLERANCE_TICKS=10
# Recorded rates project about 33 s for two cycles plus one observation
# prefix, including endpoint settling/capture. This is one finite budget;
# it neither changes motor speed nor resets at each leg or manual handoff.
PLANNED_WAVE_EXECUTION_SECONDS=45

class WristSceneChangedError(ValueError):
    """Fresh same-camera images changed at an otherwise comparable pose."""

class PreMotionSceneChangedError(WristSceneChangedError):
    """A move's final image comparison changed before any motion dispatch."""
    def __init__(self,message,frame):
        super().__init__(message)
        self.frame=frame

class _ManualWaveRequested(Exception):
    """Switch this job's planner to its explicitly requested bounded gesture."""
    def __init__(self,progress):
        self.progress=dict(progress)

class _WaveDecisionCancel:
    """Read-only cancellation view for inference, never for motor execution."""
    def __init__(self,cancel,manual_override):
        self.cancel=cancel;self.manual_override=manual_override
    def is_set(self):
        return self.cancel.is_set() or self.manual_override.is_set()
    def wait(self,timeout=None):
        deadline=None if timeout is None else time.monotonic()+timeout
        while not self.is_set():
            remaining=None if deadline is None else deadline-time.monotonic()
            if remaining is not None and remaining<=0:return False
            self.cancel.wait(.02 if remaining is None else min(.02,remaining))
        return True

# Jaw contact pushes back on the wrist a few ticks (5 measured at PWM cap 150); a collision or drag is larger.
CONTACT_BODY_DRIFT_TICKS=8

def unchanged_wrist(before,after,*,max_joint_drift=3):
    if any(f.get('source')!='wrist' for f in (before,after)):
        raise ValueError('抓取僅接受腕部相機')
    if not before.get('camera_generation') or before['camera_generation']!=after.get('camera_generation'):
        raise ValueError('腕部相機已重新連接，請重新觀察')
    if after['capture_monotonic']<=before['capture_monotonic']:
        raise ValueError('需要更新的腕部影像')
    if max(abs(before['joints'][n]-after['joints'][n]) for n in (*NAMES,'gripper'))>max_joint_drift:
        raise ValueError('判斷期間手臂姿勢已改變，請重新觀察')
    a=cv2.imread(before['path'],cv2.IMREAD_GRAYSCALE);b=cv2.imread(after['path'],cv2.IMREAD_GRAYSCALE)
    if a is None or b is None or a.shape!=b.shape:raise ValueError('腕部影像無法比較')
    # At a stationary pose, compensate only global illumination. No spatial
    # registration may hide a person/object entering the planned path.
    a=cv2.resize(a,(480,270)).astype(float);b=cv2.resize(b,(480,270)).astype(float)
    diff=np.abs((b-a)-np.median(b-a));changed=float(np.mean(diff>24))
    if changed>.015 or float(np.percentile(diff,99))>40:
        raise WristSceneChangedError('眼前場景已改變，先重新觀察再移動')
    return {'changed_fraction':changed,'p99_difference':float(np.percentile(diff,99))}


class WristGrasp:
    def __init__(self,robot,brain,snapshot,runtime):
        self.robot=robot;self.brain=brain;self.snapshot=snapshot;self.runtime=Path(runtime)
        self._model=None;self.state='idle';self.latest=None
        self.attempt_start=None;self.attempt_history=[]
        self.tactician=None
        self.state_memory=None
    @property
    def model(self):
        if self._model is None:self._model=default_model()
        return self._model
    def capture(self):
        frame=self.robot.settled_snapshot(self.snapshot)
        if getattr(self,'state_memory',None) is not None:self.state_memory.remember(frame)
        return frame
    def policy_context(self,frame):
        if getattr(self,'state_memory',None) is None:return None
        from .world_state import policy_state
        return policy_state(self.state_memory.remember(frame))
    def record(self,event):
        with (self.runtime/'grasp-steps.jsonl').open('a') as log:
            log.write(json.dumps({'time':time.time(),**event},ensure_ascii=False)+'\n')
    def vector(self,direction,distance_mm,raw):
        if direction not in ('left','right','up','down','forward','backward') or not 0<distance_mm<=5:
            raise ValueError('每次只允許 5 mm 以內的單一方向')
        transforms=self.model.fk.transforms(raw)
        if direction in ('up','down'):v=np.array([0.,0.,1. if direction=='up' else -1.])
        elif direction in ('forward','backward'):
            v=-transforms['gripper_link'][:3,2]*(1 if direction=='forward' else -1)
        else:
            v=transforms['wrist_camera_optical'][:3,0].copy();v[2]=0
            if np.linalg.norm(v)<.5:raise ValueError('目前姿勢無法可靠對應影像左右')
            v/=np.linalg.norm(v);v*=1 if direction=='right' else -1
        return v*distance_mm/1000
    def roll_sweep(self,goal,reference,cancel,*,body_origin=None,camera_generation=None,endpoint_tolerance_ticks=3,wrist_roll_bounds=None,program_guard=None):
        """A prepared roll leg; fresh measured <=30-tick proposals.

        Only the leg endpoints are settled wrist observations. Intermediate
        controller reads are live, not falsely labelled as settled images.
        """
        from .local_feedback import adopt_goal_anchor
        if type(endpoint_tolerance_ticks) is not int or endpoint_tolerance_ticks not in (3,WAVE_ENDPOINT_TOLERANCE_TICKS):
            raise ValueError('揮腕到位容差只能使用觀察或揮手設定')
        if wrist_roll_bounds is not None and (not isinstance(wrist_roll_bounds,(tuple,list)) or
                len(wrist_roll_bounds)!=2 or any(type(v) is not int for v in wrist_roll_bounds) or
                not 0<=wrist_roll_bounds[0]<=goal<=wrist_roll_bounds[1]<=4095):
            raise ValueError('揮腕範圍必須包含這次目標')
        r=self.robot;r.begin(allow_rearm=False);joint='wrist_roll'
        origin=reference['joints'];generation=camera_generation or reference['camera_generation']
        body_origin=body_origin or origin
        if reference['camera_generation']!=generation:raise RuntimeError('揮腕期間相機重新連接')
        if abs(goal-origin[joint])>round(21*4096/360):raise ValueError('單次揮腕幅度超出限制')
        for _ in range(12):
            if cancel.is_set():raise SessionCancelled('揮手已停止')
            raw=r.read();remaining=goal-raw[joint]
            if program_guard is not None:program_guard(raw)
            if wrist_roll_bounds is not None and not wrist_roll_bounds[0]<=raw[joint]<=wrist_roll_bounds[1]:
                raise RuntimeError('手腕超出這次揮手範圍，已停止')
            if abs(remaining)<=endpoint_tolerance_ticks:
                after=self.capture()
                if program_guard is not None:program_guard(after['joints'])
                if after.get('source')!='wrist':raise ValueError('揮腕僅接受腕部相機')
                if after['camera_generation']!=generation:raise RuntimeError('揮腕期間相機重新連接')
                if after['capture_monotonic']<=reference['capture_monotonic']:raise RuntimeError('揮腕需要更新的腕部影像')
                if any(abs(after['joints'][n]-body_origin[n])>4 for n in NAMES if n!=joint):raise RuntimeError('揮腕時其他關節偏移')
                if abs(after['joints']['gripper']-body_origin['gripper'])>4:raise RuntimeError('揮腕時夾爪開口改變')
                if abs(after['joints'][joint]-goal)>endpoint_tolerance_ticks:raise RuntimeError('揮腕終點未穩定')
                if wrist_roll_bounds is not None and not wrist_roll_bounds[0]<=after['joints'][joint]<=wrist_roll_bounds[1]:
                    raise RuntimeError('手腕超出這次揮手範圍，已停止')
                return after
            ticks=max(-30,min(30,remaining))
            envelope={n:max(12,abs(r.goals[n]-raw[n])+6) for n in NAMES}
            proposal=self.model.propose_rotation(raw,'roll',ticks*360/4096,held_goals=r.goals,uncertainty_ticks=envelope)
            target=proposal['active_joint_targets']
            if wrist_roll_bounds is not None:
                # Reanchoring after a tolerated endpoint error must not grow
                # the whole gesture beyond its original screened roll arc.
                for key in ('physical_goal_bounds','measured_bounds'):
                    lo,hi=proposal[key][joint]
                    proposal[key][joint]=[max(lo,wrist_roll_bounds[0]),min(hi,wrist_roll_bounds[1])]
            def guard(actual):
                if cancel.is_set():raise SessionCancelled('揮手已停止')
                if r.camera.generation!=generation:raise RuntimeError('揮腕期間相機重新連接')
                if abs(actual['gripper']-body_origin['gripper'])>4:raise RuntimeError('揮腕時夾爪開口改變')
                clearance=self.model.floor_minimum({**raw,**actual})
                if any(z<.005 for n,z in clearance.items() if n!='base_link'):raise RuntimeError('揮腕已接近模型桌面界線')
            measured_bounds=proposal['measured_bounds']
            for n in NAMES:
                if n!=joint:
                    # Preserve clipped hardware/model limits from the proposal
                    # when applying the tighter whole-gesture hold tolerance.
                    lo,hi=measured_bounds[n]
                    measured_bounds[n]=[max(lo,body_origin[n]-4),min(hi,body_origin[n]+4)]
            with r.io:
                if r.stop.is_set():raise SessionCancelled('揮腕已停止')
                anchor=adopt_goal_anchor(raw,r.goals)
                if target[joint]+anchor['bias'][joint]!=proposal['physical_goal_targets'][joint]:
                    raise RuntimeError('揮腕保持補償不一致')
                r.bias=anchor['bias']
            self.record({'type':'rotation_proposal','measured':raw,'proposal':proposal})
            # Fresh position checks at every write and a <=0.7s endpoint wait;
            # stationary image capture is reserved for the whole leg endpoint.
            r.glide(target,speed=70,precision=endpoint_tolerance_ticks,guard=guard,allow_rearm=False,
                    raw_bounds=proposal['physical_goal_bounds'],measured_bounds=measured_bounds,
                    **({'measured_guard':program_guard} if program_guard is not None else {}))
            after=r.read();angle=(after[joint]-raw[joint])*360/4096
            if angle*ticks<=0:raise RuntimeError('揮腕沒有足夠進展')
            self.record({'type':'rotation_result','actual':after,'actual_angle_deg':angle})
        raise RuntimeError('揮腕角度沒有足夠進展')

    def _screen_roll_sweep(self,first,offsets,cancel):
        """Screen the entire candidate without touching motors."""
        raw=dict(first['joints']);held=dict(self.robot.goals);center=raw['wrist_roll']
        if not offsets or any(type(v) is not int or abs(v)>int(20*4096/360) for v in offsets):
            raise ValueError('腕部觀察路徑只能使用 20 度以內的整數位置')
        for offset in offsets:
            while raw['wrist_roll']!=center+offset:
                if cancel.is_set():raise SessionCancelled('腕部觀察已停止')
                ticks=max(-30,min(30,center+offset-raw['wrist_roll']))
                envelope={n:max(12,abs(held[n]-raw[n])+6) for n in NAMES}
                proposal=self.model.propose_rotation(raw,'roll',ticks*360/4096,held_goals=held,uncertainty_ticks=envelope)
                raw=proposal['joint_targets'];held.update(proposal['physical_goal_targets'])
        return {'full_mesh_table_guard':True,'max_roll_degrees':max(abs(v) for v in offsets)*360/4096,
                'tracking_margin_degrees':12*360/4096}

    def _wave_options(self,frames,**kwargs):
        from .wave_planner import assess_wave_options
        return assess_wave_options(self.brain,frames,**kwargs)

    def _manual_wave_checkpoint(self,cancel,manual_override,progress):
        if manual_override is None or not manual_override.is_set():return
        if cancel.is_set() or self.robot.stop.is_set():raise SessionCancelled('揮手已停止')
        raise _ManualWaveRequested(progress)

    def _wave_decision(self,decide,cancel,manual_override,progress):
        """Withdraw pending AI work without cancelling an in-flight motor step."""
        self._manual_wave_checkpoint(cancel,manual_override,progress)
        signal=cancel if manual_override is None else _WaveDecisionCancel(cancel,manual_override)
        try:answer=decide(signal)
        except Exception:
            self._manual_wave_checkpoint(cancel,manual_override,progress)
            raise
        # Providers that finish concurrently with the click cannot authorize a
        # lift/retreat with their now-obsolete answer.
        self._manual_wave_checkpoint(cancel,manual_override,progress)
        return answer

    def _prepare_manual_wave(self,report,cancel,offsets,progress):
        """Human-requested wrist-only wave; no claim of visual clearance."""
        if self.state not in ('idle','approaching') or self.robot.pwm_owner is not None:
            raise RuntimeError('先完成目前的夾爪工作，再揮手')
        if cancel.is_set() or self.robot.stop.is_set():raise SessionCancelled('揮手已停止')
        self.robot.begin(allow_rearm=False)
        current=self.capture();reference=progress['reference']
        generation=current.get('camera_generation')
        if current.get('source')!='wrist':raise ValueError('揮手僅接受腕部相機')
        if (not generation or generation!=reference.get('camera_generation')
                or generation!=self.robot.camera.generation):
            raise RuntimeError('揮手期間相機重新連接')
        if current['capture_monotonic']<=reference['capture_monotonic']:
            raise RuntimeError('揮手需要更新的腕部影像')
        if not self.robot.camera.fresh():raise RuntimeError('腕部畫面已過期，先停止揮手')
        geometry=self._screen_roll_sweep(current,offsets,cancel)
        authorization={'source':'human_request','manual_override':True,
                       'visual_clearance_checked':False}
        report('依你的要求，跳過影像障礙判斷，在目前位置小幅揮手')
        self.record({'type':'manual_wave_override','authorization':authorization,
                     'frame':current,'controller_geometry':geometry,
                     'completed_raise_steps':len(progress.get('raise_steps',[])),
                     'completed_retreat_steps':len(progress.get('retreat_steps',[]))})
        deadline_fields={}
        if 'execution_remaining_seconds' in progress:
            deadline_fields['execution_deadline']=time.monotonic()+progress['execution_remaining_seconds']
        elif 'execution_deadline' in progress:
            deadline_fields['execution_deadline']=progress['execution_deadline']
        return {'initial_frame':reference,'start_frame':current,
                'initial_hold_goals':dict(self.robot.goals),'body_origin':dict(current['joints']),
                'camera_generation':generation,'clearance':authorization,
                'freshness':{'visual_comparison_checked':False,'fresh_wrist_capture':True},
                'manual_override':True,'raise_steps':progress.get('raise_steps',[]),
                'lifted_model_mm':progress.get('lifted_model_mm',0.),
                'retreat_steps':progress.get('retreat_steps',[]),
                'retreated_model_mm':progress.get('retreated_model_mm',0.),**deadline_fields}

    def _translation_proposal(self,direction,distance_mm,raw,*,stationary_axes=(),orientation_origin=None):
        envelope={n:max(12 if n in stationary_axes else 30,abs(self.robot.goals[n]-raw[n])+6) for n in NAMES}
        if max(envelope.values())>60:raise RuntimeError('目前保持補償超過局部模型可檢查範圍')
        orientation={} if orientation_origin is None else {'orientation_origin':orientation_origin}
        proposal=self.model.propose_adaptive(raw,self.vector(direction,distance_mm,raw),
                    held_goals=self.robot.goals,uncertainty_ticks=envelope,**orientation)
        if any(proposal['joint_targets'][n]!=raw[n] for n in stationary_axes):
            raise RuntimeError('這個抬手姿勢需要額外轉動，請重新規劃')
        return proposal,envelope

    def _wave_lift_proposal(self,distance_mm,raw,attempt_origin=None):
        from .wave_arc import propose_raise
        return propose_raise(self.model,raw,self.robot.goals,attempt_origin or raw,max_distance_mm=distance_mm)

    def _wave_raise_proposal(self,raw,attempt_origin):
        """Prefer one larger screened arc; geometry may select the small one.

        This choice precedes visual assessment. A rejected visual verdict is
        never recycled as permission for either this macro or its fallback.
        """
        from .wave_arc import propose_raise,MACRO_NOMINAL_LIMITS,NOMINAL_LIMITS
        try:
            return propose_raise(self.model,raw,self.robot.goals,attempt_origin,
                max_distance_mm=MACRO_NOMINAL_LIMITS['maximum_distance_mm'],motion_kind='elbow_raise_macro')
        except (ValueError,RuntimeError):
            return self._wave_lift_proposal(NOMINAL_LIMITS['maximum_distance_mm'],raw,attempt_origin)

    def _wave_cartesian_lift_proposal(self,distance_mm,raw,attempt_origin=None):
        """Screen the elbow-leading start as well as the final IK endpoint."""
        from scipy.spatial.transform import Rotation
        for _ in range(5):
            proposal,envelope=self._translation_proposal('up',distance_mm,raw,
                    stationary_axes=('shoulder_pan','wrist_roll'),orientation_origin=attempt_origin)
            middle={**proposal['joint_targets'],**{n:raw[n] for n in ('wrist_flex','shoulder_lift')}}
            T0=self.model.tip(raw);T1=self.model.tip(middle);delta=T1[:3,3]-T0[:3,3]
            angle=float(np.degrees(Rotation.from_matrix(T1[:3,:3]@T0[:3,:3].T).magnitude()))
            checks=[self.model.screen_segment(a,b,uncertainty_ticks=envelope)
                    for a,b in ((raw,middle),(middle,proposal['joint_targets']))]
            attempt_ok=True
            if attempt_origin is not None:
                from .wave_lift import lift_attempt
                try:
                    for pose in (middle,proposal['joint_targets']):
                        lift_attempt(self.model.tip(attempt_origin),self.model.tip(pose))
                    target_rotation=self.model.tip(proposal['joint_targets'])[:3,:3]
                    if np.degrees(Rotation.from_matrix(target_rotation@self.model.tip(attempt_origin)[:3,:3].T).magnitude())>1.:
                        attempt_ok=False
                except RuntimeError:attempt_ok=False
            if (0<=delta[2] and np.linalg.norm(delta)<=.012 and np.linalg.norm(delta[:2])<=.003 and angle<=3.
                    and attempt_ok and all(c['model_table_pass'] for c in checks)):
                proposal['coordination']={'intermediate_delta_m':delta.tolist(),
                    'intermediate_orientation_deg':angle,'model_table_pass':True,
                    'actual_tip_corridor_mm':{'minimum_up':-1.,'maximum_up':12.,'maximum_lateral':3.,'maximum_rotation_degrees':3.}}
                proposal['coordination_table_checks']=checks
                return proposal,envelope
            distance_mm*=.75
        raise RuntimeError('這個姿勢還找不到合適的抬手路徑')

    def _prepare_roll_sweep(self,report,cancel,offsets,*,allow_raise=False,manual_override=None):
        """Shared no-motion path screen and external-clearance assessment.

        The returned start frame follows inference and the unchanged-scene
        check. All later legs retain this one body corridor and generation.
        """
        if self.state not in ('idle','approaching') or self.robot.pwm_owner is not None:
            raise RuntimeError('先完成目前的夾爪工作，再轉動手腕觀察')
        if cancel.is_set():raise SessionCancelled('腕部觀察已停止')
        self.robot.begin(allow_rearm=False)
        first=self.capture();attempt_start=first;raises=[];lifted=0.;scene_refreshes=0
        def progress():
            return {'reference':first,'raise_steps':raises,'lifted_model_mm':lifted}
        def manual_checkpoint():
            self._manual_wave_checkpoint(cancel,manual_override,progress())
        def refresh_scene(exc,current):
            nonlocal scene_refreshes
            if cancel.is_set() or self.robot.stop.is_set():raise SessionCancelled('抬手已停止')
            if not allow_raise or scene_refreshes>=2:raise exc
            if not self.robot.camera.fresh():raise RuntimeError('腕部畫面中斷，先停止觀察')
            scene_refreshes+=1
            report('周圍有變化，先用新的畫面重新觀察')
            self.record({'type':'wave_scene_refresh','refresh_count':scene_refreshes,
                         'previous_frame':first,'frame':current,'reason':str(exc),
                         'step_count':len(raises),'lifted_model_mm':lifted})
            # Discard the old verdict/proposal, retaining the original attempt
            # and every completed step. This helper never dispatches motion.
            return current
        while True:
            if allow_raise:
                from .wave_arc import check_arc_attempt
                lifted=check_arc_attempt(self.model,attempt_start['joints'],first['joints'])
            manual_checkpoint()
            report('看一下手邊空間，確認原地揮手或先抬手')
            geometry=self._screen_roll_sweep(first,offsets,cancel)
            initial_goals=dict(self.robot.goals);lift_proposal=None;lift_envelope=None;lift_error=''
            prefer_retreat=allow_raise and self.state=='idle' and self.model.tip(first['joints'])[2,3]>=WAVE_PREFERRED_HEIGHT_M
            if allow_raise:
                # Select the full bounded arc before asking Astra once. Local
                # interpolation runs without further inference inside that arc.
                if self.state=='idle' and len(raises)<20 and lifted<=48.:
                    try:lift_proposal,lift_envelope=self._wave_raise_proposal(first['joints'],attempt_start['joints'])
                    except (ValueError,RuntimeError) as exc:lift_error=str(exc)
                views=[first] if not raises else [attempt_start,raises[-1]['before'],first]
                options=self._wave_decision(lambda signal:self._wave_options(views,
                    roll_geometry=geometry,lift_proposal=lift_proposal,
                    lifted_mm=lifted,step_count=len(raises),cancel=signal),cancel,manual_override,progress())
                check=options['wave'];lift=options['lift']
                self.record({'type':'wave_options','frame':first,'lifted_model_mm':lifted,
                             'assessment':options,'lift_proposal_error':lift_error})
            else:
                check=self.brain.wave_clearance([first],cancel,controller_geometry=geometry)
            self.record({'type':'roll_clearance','frame':first,'requested_roll_degrees':geometry['max_roll_degrees'],'assessment':check})
            if cancel.is_set():raise SessionCancelled('腕部觀察已停止')
            roll_clear=check['confidence']>=(WAVE_MIN_CONFIDENCE if allow_raise else .9) and all(check[n] for n in ('gripper_empty','clear_sweep','human_clear'))
            # Keep an independently approved local lift even after several
            # views or above a preferred CAD height. Only its availability,
            # fresh visual verdict and unchanged attempt bounds decide it.
            if not roll_clear and (not allow_raise or lift_proposal is None or
                    lift['confidence']<LOCAL_MOVE_MIN_CONFIDENCE or not all(lift[n] for n in ('gripper_empty','clear_path','human_clear'))):
                # A roll-specific obstruction does not forbid assessing a
                # different path. Every retreat still needs its own verdict.
                if allow_raise and (raises or prefer_retreat):
                    kwargs={} if manual_override is None else {'manual_override':manual_override}
                    return self._prepare_retreat(report,cancel,offsets,attempt_start,raises,lifted,**kwargs)
                reason=check['reason']
                if allow_raise:
                    reason+=' '+(lift_error or lift['reason'])
                    if lift['clear_path'] and lift['confidence']<LOCAL_MOVE_MIN_CONFIDENCE:
                        reason+=' 上方看似有空間，但目前還看不夠清楚，先不抬手。'
                manual_checkpoint()
                raise RuntimeError('揮腕空間尚未確認：'+reason)
            current=self.capture()
            if cancel.is_set():raise SessionCancelled('抬手已停止')
            manual_checkpoint()
            try:comparison=unchanged_wrist(first,current)
            except WristSceneChangedError as exc:
                manual_checkpoint()
                first=refresh_scene(exc,current)
                continue
            if allow_raise:
                try:lifted=check_arc_attempt(self.model,attempt_start['joints'],current['joints'])
                except RuntimeError as exc:
                    if raises:
                        self.robot.error=str(exc);self.robot.stop.set();self.robot.hold()
                    raise
            if roll_clear:
                return {'initial_frame':first,'start_frame':current,'initial_hold_goals':initial_goals,
                        'body_origin':dict(current['joints']),'camera_generation':current['camera_generation'],
                        'clearance':check,'freshness':comparison,'raise_steps':raises,'lifted_model_mm':lifted}
            report('前方物件可能擋到揮手，先抬高一段，再看一次')
            manual_checkpoint()
            distance=float(np.linalg.norm(lift_proposal['requested_delta_m'])*1000)
            try:
                moved=self.move('up',distance,current,report,free_space_lift=True,
                                lift_origin=attempt_start['joints'],cancel=cancel,
                                **({'before_dispatch':manual_checkpoint} if manual_override is not None else {}),
                                assessed_lift={'proposal':lift_proposal,'envelope':lift_envelope,
                                               'origin':dict(first['joints']),'hold_goals':initial_goals})
            except PreMotionSceneChangedError as exc:
                manual_checkpoint()
                first=refresh_scene(exc,exc.frame)
                continue
            raises.append(moved)
            if cancel.is_set():raise SessionCancelled('抬手已停止')
            first=moved['after']

    def _prepare_retreat(self,report,cancel,offsets,attempt_start,raises,lifted,*,manual_override=None):
        """One separately assessed lateral phase after raising was insufficient."""
        from .wave_retreat import propose_retreat,check_retreat_attempt
        from .wave_backward import propose_backward,check_backward_attempt
        from .wave_planner import assess_retreat_options
        from .wave_arc import check_arc_attempt
        first=raises[-1]['after'] if raises else attempt_start
        steps=[];distance=0.
        def progress():
            return {'reference':first,'raise_steps':raises,'lifted_model_mm':lifted,
                    'retreat_steps':steps,'retreated_model_mm':distance}
        def manual_checkpoint():
            self._manual_wave_checkpoint(cancel,manual_override,progress())
        manual_checkpoint()
        try:
            first=self.capture();manual_checkpoint()
            unchanged_wrist(raises[-1]['after'] if raises else attempt_start,first)
            lifted=check_arc_attempt(self.model,attempt_start['joints'],first['joints'])
        except _ManualWaveRequested:raise
        except Exception as exc:
            self.robot.error=str(exc);self.robot.stop.set();self.robot.hold()
            self.record({'type':'retreat_transition_failed','error':str(exc)});raise
        phase_start=first;steps=[];side=None
        while True:
            if cancel.is_set():raise SessionCancelled('側移已停止')
            phase_guard=check_backward_attempt if side=='backward' else check_retreat_attempt
            distance=phase_guard(self.model,phase_start['joints'],first['joints'])
            manual_checkpoint()
            geometry=self._screen_roll_sweep(first,offsets,cancel)
            initial_goals=dict(self.robot.goals);candidates={'left':None,'right':None,'backward':None};envelopes={}
            if len(steps)<20 and distance<=28.:
                for sign in (() if side=='backward' else (-1,1)):
                    try:
                        proposal,envelope=propose_retreat(self.model,first['joints'],initial_goals,phase_start['joints'],sign)
                        label=proposal['image_side']
                        if side is None or label==side:
                            candidates[label]=proposal;envelopes[label]=envelope
                    except (ValueError,RuntimeError):continue
                if side in (None,'backward'):
                    try:
                        candidates['backward'],envelopes['backward']=propose_backward(
                            self.model,first['joints'],initial_goals,phase_start['joints'])
                    except (ValueError,RuntimeError):pass
            report('看看往後或往旁邊退，哪一條路能避開物件')
            views=[attempt_start,phase_start,first] if not steps else [phase_start,steps[-1]['before'],first]
            options=self._wave_decision(lambda signal:assess_retreat_options(self.brain,views,
                roll_geometry=geometry,retreat_proposals=candidates,retreated_mm=distance,
                step_count=len(steps),cancel=signal),cancel,manual_override,progress())
            self.record({'type':'wave_retreat_options','frame':first,'assessment':options,
                         'retreated_model_mm':distance,'selected_side':side})
            if cancel.is_set():raise SessionCancelled('側移已停止')
            check=options['wave']
            # The original pose already received a blocked-roll verdict. Do not
            # overturn it by re-asking with different side-movement candidates.
            roll_clear=bool(steps) and check['confidence']>=WAVE_MIN_CONFIDENCE and all(check[n] for n in ('gripper_empty','clear_sweep','human_clear'))
            eligible=[label for label,p in candidates.items() if p is not None
                      and options[label]['confidence']>=LOCAL_MOVE_MIN_CONFIDENCE
                      and all(options[label][n] for n in ('gripper_empty','clear_path','human_clear'))]
            if not roll_clear and not eligible:
                reasons=' '.join(options[label]['reason'] for label in candidates)
                manual_checkpoint()
                raise RuntimeError('揮手空間尚未確認：仍找不到能避開物件的退讓路線。 '+reasons)
            current=self.capture();manual_checkpoint()
            try:comparison=unchanged_wrist(first,current)
            except WristSceneChangedError:
                manual_checkpoint()
                raise
            if cancel.is_set():raise SessionCancelled('側移已停止')
            try:phase_guard(self.model,phase_start['joints'],current['joints'])
            except RuntimeError as exc:
                if steps:
                    self.robot.error=str(exc);self.robot.stop.set();self.robot.hold()
                raise
            if roll_clear:
                return {'initial_frame':first,'start_frame':current,'initial_hold_goals':initial_goals,
                        'body_origin':dict(current['joints']),'camera_generation':current['camera_generation'],
                        'clearance':check,'freshness':comparison,'raise_steps':raises,'lifted_model_mm':lifted,
                        'retreat_steps':steps,'retreated_model_mm':distance}
            chosen=max(eligible,key=lambda label:options[label]['confidence'])
            if self.tactician is not None and len(eligible)>1 and len(steps)<12:
                # Jev ranks only already-cleared candidates. It receives no
                # pixels, motor targets, key, or authority to change a path.
                choice=self._wave_decision(lambda signal:self.tactician.choose({'goal':'create room for a wave',
                    'completed_retreat_steps':len(steps),'model_retreat_mm':distance,
                    'world_state':self.policy_context(current),
                    'candidate_geometry':{label:{
                        'delta_base_model_m':candidates[label]['predicted_delta_m'],
                        'delta_camera_model_m':candidates[label]['camera_relative_delta_m'],
                        'motion_kind':candidates[label]['motion_kind'],
                        'visual_reason':options[label]['reason'][:160],
                        'geometry_accuracy':'cad_prior_not_measured_obstacle_distance'} for label in eligible},
                    'last_progress':None if not steps else {
                        'direction':steps[-1]['direction'],
                        'actual_delta_model_mm':steps[-1]['actual_delta_model_mm'],
                        'completion':steps[-1]['completion']},
                    'evidence':{label:{k:options[label][k] for k in
                        ('gripper_empty','human_clear','clear_path','confidence')} for label in eligible}},
                    {label:{'left':'One cleared small step left, then reobserve.',
                            'right':'One cleared small step right, then reobserve.',
                            'backward':'One cleared backward withdrawal, then reobserve.'}[label] for label in eligible},cancel=signal),
                    cancel,manual_override,progress())
                self.record({'type':'jev_retreat_choice','decision':choice,'eligible':eligible})
                if cancel.is_set():raise SessionCancelled('退讓已停止')
                if choice['status']=='selected' and choice['selected'] in eligible:chosen=choice['selected']
            proposal=candidates[chosen];side=chosen
            report('把手往後縮一點，再看能不能揮手' if chosen=='backward' else '向旁邊小步退讓，讓物件離開手掌前方')
            manual_checkpoint()
            try:
                moved=self.move(chosen,float(np.linalg.norm(proposal['requested_delta_m'])*1000),current,report,
                    free_space_retreat=True,lift_origin=phase_start['joints'],cancel=cancel,empty_hand_progress=True,
                    **({'before_dispatch':manual_checkpoint} if manual_override is not None else {}),
                    assessed_lift={'proposal':proposal,'envelope':envelopes[chosen],
                                   'origin':dict(first['joints']),'hold_goals':initial_goals})
            except PreMotionSceneChangedError:
                manual_checkpoint()
                raise
            steps.append(moved);first=moved['after']

    def _roll_view_evidence(self,frames,origin):
        """Encoder/FK evidence only; neither depth nor successful manipulation."""
        angles=[(f['joints']['wrist_roll']-origin)*360/4096 for f in frames]
        camera_positions=[self.model.fk.transforms(f['joints'])['wrist_camera_optical'][:3,3] for f in frames]
        baseline=max(float(np.linalg.norm(a-b))*1000 for a in camera_positions for b in camera_positions)
        return {'measured_roll_range_deg':[min(angles),max(angles)],
                'return_error_deg':angles[-1],'camera_baseline_model_mm':baseline,
                'not_external_metrology':True}

    def wave(self,report,cancel,*,manual_override:threading.Event | None=None):
        """Two compact +/-10 degree cycles, with one assessment of this arc.

        A blocked roll can select an independently assessed upward arc. Its
        motor interpolation runs locally; only completed arcs are reassessed.
        A per-request human override abandons that planner for the same bounded
        wrist-only gesture; it does not override camera or controller checks.
        """
        amplitude=int(10*4096/360);angle=amplitude*360/4096;tolerance=WAVE_ENDPOINT_TOLERANCE_TICKS*360/4096
        offsets=(amplitude,0,-amplitude,0)
        kwargs={} if manual_override is None else {'manual_override':manual_override}
        try:
            preparation=self._prepare_roll_sweep(report,cancel,offsets,allow_raise=True,**kwargs)
            self._manual_wave_checkpoint(cancel,manual_override,
                {**preparation,'reference':preparation['start_frame']})
        except _ManualWaveRequested as request:
            preparation=self._prepare_manual_wave(report,cancel,offsets,request.progress)
        return self._complete_wave(preparation,report,cancel,manual_override=manual_override)

    def _complete_wave(self,preparation,report,cancel,*,manual_override=None):
        amplitude=int(10*4096/360);angle=amplitude*360/4096;tolerance=WAVE_ENDPOINT_TOLERANCE_TICKS*360/4096
        offsets=(amplitude,0,-amplitude,0)
        # The API closes its override window on this report. Sample once more
        # afterwards, so a click accepted during the handoff is not lost.
        report('向你揮揮手 · 1 / 2')
        if not preparation.get('manual_override'):
            try:self._manual_wave_checkpoint(cancel,manual_override,
                {**preparation,'reference':preparation['start_frame']})
            except _ManualWaveRequested as request:
                previous=preparation
                preparation=self._prepare_manual_wave(report,cancel,offsets,request.progress)
                for key in ('planning_calls','view_recovery','view_recovery_kind','view_recovery_direction','view_recovery_program_id',
                            'view_recovery_clearance','view_recovery_execution_seconds'):
                    if key in previous:preparation[key]=previous[key]
        current=preparation['start_frame'];body_origin=preparation['body_origin']
        generation=preparation['camera_generation'];origin=body_origin['wrist_roll']
        program=preparation.get('program')
        if program is not None:origin=program['origin']['wrist_roll']
        wrist_roll_bounds=[origin-amplitude-12,origin+amplitude+12]
        program_bounds=None
        if program is not None:
            from .motion_program import _segment_bounds
            expected=program['raise_target'] or program['origin']
            raised_joint=program['raise_joint']
            margins={n:10 if n==raised_joint else 4 for n in NAMES}
            _,program_bounds,_=_segment_bounds(self.model,expected,expected,['wrist_roll'],
                                               program['roll_tracking_envelope_ticks'],margins)
            program_bounds['wrist_roll']=wrist_roll_bounds
        def program_guard(actual):
            from .local_feedback import check_bounds
            if cancel.is_set():raise SessionCancelled('揮手已停止')
            if time.monotonic()>=preparation['execution_deadline']:raise RuntimeError('這段動作已到時間，保持目前姿勢')
            if program is None:return
            if not wrist_roll_bounds[0]<=actual['wrist_roll']<=wrist_roll_bounds[1]:raise RuntimeError('手腕偏離這次完整動作的範圍')
            check_bounds(actual,program_bounds)
            self.model.validate_raw(actual,margin_ticks=0)
            expected=program['raise_target'] or program['origin']
            for n in (*NAMES,'gripper'):
                if n=='wrist_roll':continue
                margin=10 if n==program['raise_joint'] else 4
                if abs(actual[n]-expected[n])>margin:raise RuntimeError('手臂偏離這次完整動作的路徑，已停止')
        samples=[current];extremes=[]
        for cycle in range(2):
            for offset in (amplitude,0,-amplitude,0):
                goal=origin+offset
                if cycle or offset!=amplitude:report(f'向你揮揮手 · {cycle+1} / 2')
                current=self.roll_sweep(goal,current,cancel,body_origin=body_origin,camera_generation=generation,
                    endpoint_tolerance_ticks=WAVE_ENDPOINT_TOLERANCE_TICKS,wrist_roll_bounds=wrist_roll_bounds,
                    **({'program_guard':program_guard} if program is not None or 'execution_deadline' in preparation else {}));samples.append(current)
                extremes.append(current)
                if cancel.wait(.15):raise SessionCancelled('揮手已停止')
            angles=[(f['joints']['wrist_roll']-origin)*360/4096 for f in extremes[-4:]]
            if (abs(angles[0]-angle)>tolerance or abs(angles[2]+angle)>tolerance or
                    abs(angles[1])>tolerance or abs(angles[3])>tolerance):
                raise RuntimeError('這一輪揮手的實測幅度或回程不足')
        measured_angles=[(f['joints']['wrist_roll']-origin)*360/4096 for f in samples]
        if min(measured_angles)>-angle+tolerance or max(measured_angles)<angle-tolerance:raise RuntimeError('實際揮手幅度不足')
        receipt={'wave_completed':True,'cycles':2,'requested_roll_range_deg':[-angle,angle],**self._roll_view_evidence(samples,origin),
                 'endpoint_tolerance_deg':tolerance,
                 'frames':extremes,'clearance':preparation['clearance'],
                 'raise_steps':preparation['raise_steps'],'lifted_model_mm':preparation['lifted_model_mm'],
                 'retreat_steps':preparation.get('retreat_steps',[]),
                 'retreated_model_mm':preparation.get('retreated_model_mm',0.)}
        if preparation.get('manual_override'):receipt['manual_override']=True
        if program is not None:receipt.update(execution_mode='planned_program',program_id=program['program_id'],planning_calls=preparation.get('planning_calls',1))
        for key in ('planning_calls','view_recovery','view_recovery_kind','view_recovery_direction','view_recovery_program_id',
                    'view_recovery_clearance','view_recovery_execution_seconds'):
            if key in preparation:receipt[key]=preparation[key]
        self.record({'type':'wave_result',**receipt});return receipt

    def _recover_planned_wave_view(self,first,programs,verdict,report,cancel,manual_override,progress):
        """One separately cleared observation move, then in-place reassessment.

        Prefix approval never authorizes a roll. The second full-wave decision
        can only select an in-place wave; neither recursion nor another raise is
        available. Stationary cloud time does not consume the shared wave budget
        of execution time, but the completed prefix is deducted from it.
        """
        from .wave_program import assess_wave_programs,build_in_place_wave_program,wave_observation_motion
        from .wave_observation import (build_wave_observation_prefixes,
                                       assess_wave_observation_prefixes,check_wave_observation_prefix)
        from .motion_program import execute_prepared_segment
        if not verdict['gripper_empty'] or not verdict['human_clear']:
            raise RuntimeError('目前還不能完成揮手：'+verdict['reason'])
        self._manual_wave_checkpoint(cancel,manual_override,progress)
        report('目前角度看不清楚；看看能不能向旁邊移開或抬高相機')
        reference=self.capture();unchanged_wrist(first,reference)
        candidates=build_wave_observation_prefixes(self.model,reference['joints'],dict(self.robot.goals),
            cancel=cancel,wave_programs=programs)
        if not candidates:raise RuntimeError('目前找不到能換個角度觀察的路徑，先保持目前姿勢')
        progress['reference']=reference;progress['planning_calls']+=1
        prefix=self._wave_decision(lambda signal:assess_wave_observation_prefixes(
            self.brain,[reference],candidates,cancel=signal),cancel,manual_override,progress)
        self.record({'type':'wave_view_prefix_assessment','frame':reference,
                     'assessment':prefix,'program_ids':[p['program_id'] for p in candidates],
                     'prefixes':candidates})
        if (prefix['program_id'] is None or prefix['confidence']<LOCAL_MOVE_MIN_CONFIDENCE
                or not all(prefix[k] for k in ('gripper_empty','human_clear','clear_prefix'))):
            raise RuntimeError('目前還找不到可換角度觀察的空間：'+prefix['reason'])
        selected=next(p for p in candidates if p['program_id']==prefix['program_id'])
        current=self.capture();unchanged_wrist(reference,current)
        self._manual_wave_checkpoint(cancel,manual_override,progress)
        direction=selected['spatial_direction']
        report({'sideways':'先向旁邊移開一點，再確認揮手空間',
                'backward':'手臂一起往後退，換個角度再確認揮手空間',
                'up':'先抬高手臂換個視角，完成後再確認揮手空間'}[direction])
        execution_started=time.monotonic()
        moved=execute_prepared_segment(self,selected['segment'],current,cancel,report,
            deadline=execution_started+PLANNED_WAVE_EXECUTION_SECONDS,
            measured_guard=lambda actual:check_wave_observation_prefix(self.model,selected,actual))
        consumed=time.monotonic()-execution_started
        current=moved['after']
        delta=(self.model.tip(current['joints'])[:3,3]-self.model.tip(first['joints'])[:3,3])*1000
        lifted=max(0.,float(delta[2]))
        if direction in ('sideways','backward'):
            progress.setdefault('retreat_steps',[]).append(moved)
            progress['retreated_model_mm']=float(np.linalg.norm(delta[:2]))
        else:progress['raise_steps'].append(moved)
        progress.update(reference=current,lifted_model_mm=lifted,view_recovery=True,
                        view_recovery_kind=selected['kind'],view_recovery_direction=direction,
                        view_recovery_program_id=selected['program_id'],view_recovery_clearance=prefix,
                        view_recovery_execution_seconds=consumed,
                        execution_remaining_seconds=PLANNED_WAVE_EXECUTION_SECONDS-consumed)
        self.record({'type':'wave_view_prefix_completed','frame':current,
                     'program_id':selected['program_id'],'kind':selected['kind'],'execution_seconds':consumed,
                     'wave_authorized':False})
        if consumed>=PLANNED_WAVE_EXECUTION_SECONDS:raise RuntimeError('這次動作已到時間，先保持目前姿勢')
        self._manual_wave_checkpoint(cancel,manual_override,progress)
        if getattr(self.robot,'moving',False):raise RuntimeError('手腕還沒停穩，先停止揮手')
        # Only the measured endpoint and its still-held goals anchor this plan.
        # Do not regenerate any raise candidates after using the single prefix.
        program=build_in_place_wave_program(self.model,current['joints'],dict(self.robot.goals),cancel=cancel)
        if program is None:raise RuntimeError('已換角度重新觀察，但這個姿勢仍沒有可用的原地揮手路徑')
        fresh=self.capture();unchanged_wrist(current,fresh)
        # Retain the two actual captured views and the achieved encoder/FK
        # motion. The CAD displacement is evidence of viewpoint change, not a
        # measurement of the cylinder's position or clearance.
        views=[moved['before'],fresh]
        observation_motion=wave_observation_motion(self.model,views[0],views[1],
            prefix_id=selected['program_id'],kind=selected['kind'])
        progress['reference']=fresh;progress['planning_calls']+=1
        report('已換好視角，現在確認原地揮手的整段空間')
        final_verdict=self._wave_decision(lambda signal:assess_wave_programs(
            self.brain,views,[program],cancel=signal,observation_motion=observation_motion),
            cancel,manual_override,progress)
        self.record({'type':'wave_program_assessment','frame':fresh,'programs':[program],
                     'assessment':final_verdict,'after_view_recovery':True,
                     'frames':views,'observation_motion':observation_motion})
        if (final_verdict['program_id'] is None or final_verdict['confidence']<WAVE_MIN_CONFIDENCE
                or not all(final_verdict[k] for k in ('gripper_empty','human_clear','clear_path'))):
            raise RuntimeError('已換角度重新觀察，但目前仍不能揮手：'+final_verdict['reason'])
        current=self.capture();comparison=unchanged_wrist(fresh,current)
        self._manual_wave_checkpoint(cancel,manual_override,progress)
        return {'initial_frame':first,'start_frame':current,'initial_hold_goals':dict(self.robot.goals),
                'body_origin':dict(current['joints']),'camera_generation':current['camera_generation'],
                'clearance':final_verdict,'freshness':comparison,'raise_steps':progress['raise_steps'],
                'retreat_steps':progress.get('retreat_steps',[]),'retreated_model_mm':progress.get('retreated_model_mm',0.),
                'lifted_model_mm':lifted,'program':program,
                'execution_deadline':time.monotonic()+PLANNED_WAVE_EXECUTION_SECONDS-consumed,
                **{key:progress[key] for key in ('planning_calls','view_recovery','view_recovery_kind','view_recovery_direction','view_recovery_program_id',
                    'view_recovery_clearance','view_recovery_execution_seconds')}}

    def planned_wave(self,report,cancel,*,manual_override=None):
        """Plan a whole wave, with at most one separately cleared view recovery."""
        from .wave_program import build_wave_programs,assess_wave_programs
        from .motion_program import prepare_joint_segment,execute_prepared_segment
        r=self.robot
        if self.state not in ('idle','approaching') or r.pwm_owner is not None:
            raise RuntimeError('先完成目前的夾爪工作，再揮手')
        if cancel.is_set() or r.stop.is_set():raise SessionCancelled('揮手已停止')
        r.begin(allow_rearm=False)
        first=self.capture();current=first;raises=[]
        progress={'reference':first,'raise_steps':raises,'lifted_model_mm':0.,'planning_calls':0,'view_recovery':False}
        offsets=(int(10*4096/360),0,-int(10*4096/360),0)
        try:
            self._manual_wave_checkpoint(cancel,manual_override,progress)
            report('正在規劃整段揮手：需要時先抬高，再連續揮動')
            programs=build_wave_programs(self.model,first['joints'],dict(r.goals),cancel=cancel)
            if not programs:raise RuntimeError('目前姿勢還找不到完整的揮手路徑')
            segments={};executable=[]
            for p in programs:
                if p['raise_target'] is not None:
                    try:
                        segments[p['program_id']]=prepare_joint_segment(self.model,p['origin'],p['initial_hold_goals'],p['raise_target'],
                            approved_envelope=p['raise_tracking_envelope_ticks'])
                    except ValueError:
                        continue
                executable.append(p)
            programs=executable
            if not programs:raise RuntimeError('目前姿勢還找不到完整的揮手路徑')
            # Only candidates whose exact local executor path also passed the
            # mesh screen are offered to Astra; no per-step approval recycling.
            fresh=self.capture();unchanged_wrist(first,fresh);first=fresh;progress['reference']=fresh
            progress['planning_calls']+=1
            verdict=self._wave_decision(lambda signal:assess_wave_programs(self.brain,[fresh],programs,cancel=signal),
                                        cancel,manual_override,progress)
            self.record({'type':'wave_program_assessment','frame':fresh,'programs':programs,'assessment':verdict})
            if (verdict['program_id'] is None or verdict['confidence']<WAVE_MIN_CONFIDENCE or
                    not all(verdict[k] for k in ('gripper_empty','human_clear','clear_path'))):
                preparation=self._recover_planned_wave_view(first,programs,verdict,
                    report,cancel,manual_override,progress)
                return self._complete_wave(preparation,report,cancel,manual_override=manual_override)
            program=next(p for p in programs if p['program_id']==verdict['program_id'])
            current=self.capture();comparison=unchanged_wrist(fresh,current)
            self._manual_wave_checkpoint(cancel,manual_override,progress)
            deadline=time.monotonic()+PLANNED_WAVE_EXECUTION_SECONDS
            if program['raise_target'] is not None:
                from .wave_program import check_wave_raise_attempt
                report('路徑已規劃好，先抬高，再接著揮手')
                moved=execute_prepared_segment(self,segments[program['program_id']],current,cancel,report,deadline=deadline,
                    measured_guard=lambda actual:check_wave_raise_attempt(self.model,program,actual))
                raises.append(moved);current=moved['after']
            lifted=max(0.,float((self.model.tip(current['joints'])[2,3]-self.model.tip(first['joints'])[2,3])*1000))
            preparation={'initial_frame':first,'start_frame':current,'initial_hold_goals':dict(r.goals),
                'body_origin':dict(current['joints']),'camera_generation':current['camera_generation'],
                'clearance':verdict,'freshness':comparison,'raise_steps':raises,'lifted_model_mm':lifted,
                'program':program,'execution_deadline':deadline,'planning_calls':progress['planning_calls'],
                'view_recovery':False}
            progress={**preparation,'reference':current}
            self._manual_wave_checkpoint(cancel,manual_override,progress)
        except _ManualWaveRequested as request:
            preparation=self._prepare_manual_wave(report,cancel,offsets,request.progress)
            for key in ('planning_calls','view_recovery','view_recovery_kind','view_recovery_direction','view_recovery_program_id',
                        'view_recovery_clearance','view_recovery_execution_seconds'):
                if key in request.progress:preparation[key]=request.progress[key]
        return self._complete_wave(preparation,report,cancel,manual_override=manual_override)

    def run_program(self,query,report,cancel,*,task='grasp',report_stage=None):
        from .object_program import run_object_program
        return run_object_program(self,query,report,cancel,task=task,report_stage=report_stage)

    def observe_views(self,report,cancel):
        """Seven wrist views over one cleared roll arc, ending at its center.

        Stops are 0,+10,+20,0,-10,-20,0 degrees, rounded inward to encoder
        ticks. Actual positions and the model's off-axis camera baseline are
        reported, without claiming measured object depth or grasp success.
        """
        half=int(10*4096/360);amplitude=int(20*4096/360)
        offsets=(0,half,amplitude,0,-half,-amplitude,0)
        preparation=self._prepare_roll_sweep(report,cancel,offsets[1:])
        current=preparation['start_frame'];body_origin=preparation['body_origin']
        generation=preparation['camera_generation'];origin=body_origin['wrist_roll']
        wrist_roll_bounds=[origin-amplitude-12,origin+amplitude+12]
        frames=[current]
        for index,offset in enumerate(offsets[1:],start=2):
            report(f'轉動手腕，從不同角度觀察 · {index} / 7')
            current=self.roll_sweep(origin+offset,current,cancel,
                                    body_origin=body_origin,camera_generation=generation,
                                    endpoint_tolerance_ticks=OBSERVATION_ENDPOINT_TOLERANCE_TICKS,
                                    wrist_roll_bounds=wrist_roll_bounds)
            frames.append(current)
            if cancel.wait(.15):raise SessionCancelled('腕部觀察已停止')
        # Validate the returned evidence, rather than infer completion from
        # dispatched goals or from the requested camera baseline.
        if any(abs(f['joints']['wrist_roll']-(origin+offset))>OBSERVATION_ENDPOINT_TOLERANCE_TICKS for f,offset in zip(frames,offsets)):
            raise RuntimeError('腕部觀察的實測角度或回程不足')
        receipt={'frames':frames,**self._roll_view_evidence(frames,origin),
                 'endpoint_tolerance_deg':OBSERVATION_ENDPOINT_TOLERANCE_TICKS*360/4096,
                 'requested_roll_offsets_deg':[v*360/4096 for v in offsets],
                 'initial_frame':preparation['initial_frame'],
                 'initial_hold_goals':preparation['initial_hold_goals'],
                 'body_origin':body_origin,'camera_generation':generation,
                 'clearance':preparation['clearance'],'freshness':preparation['freshness']}
        self.record({'type':'observation_views',**receipt});return receipt
    def move(self,direction,distance_mm,reference,report,*,free_space_lift=False,free_space_retreat=False,lift_origin=None,cancel=None,assessed_lift=None,empty_hand_progress=False,assessed_motion=None,before_dispatch=None):
        if type(empty_hand_progress) is not bool:raise ValueError('需要明確的空手移動判斷')
        if empty_hand_progress and self.state not in ('idle','approaching'):
            raise RuntimeError('持有或接觸物件時不能使用空手進展容差')
        if assessed_motion is not None:
            if assessed_lift is not None or not empty_hand_progress or free_space_lift or free_space_retreat:
                raise ValueError('一般核准提案只適用空手局部移動')
            assessed_lift=assessed_motion
        if free_space_retreat and (free_space_lift or direction not in ('left','right','backward') or self.state!='idle'
                                   or self.robot.pwm_owner is not None or assessed_lift is None):
            raise RuntimeError('側移僅限空手待機，並需已確認的路徑')
        if free_space_lift and (direction!='up' or self.state!='idle' or self.robot.pwm_owner is not None):
            raise RuntimeError('抬手僅限空手待機時向上移動')
        assessed_arc=(free_space_lift and assessed_lift is not None
                      and assessed_lift['proposal'].get('motion_kind') in ('elbow_raise','elbow_raise_macro'))
        if assessed_arc:
            from .wave_arc import arc_profile
            nominal,_=arc_profile(assessed_lift['proposal']['motion_kind'])
            if (isinstance(distance_mm,bool) or not isinstance(distance_mm,(int,float))
                    or not np.isfinite(distance_mm) or not 0<distance_mm<=nominal['maximum_distance_mm']):
                raise ValueError('抬手距離超出已定義的單肘步幅')
            if assessed_lift['proposal']['motion_kind']=='elbow_raise_macro':
                from .wave_planner import _lift_context
                # A small-step approval cannot be relabelled as a macro: the
                # entire sampled path and exact macro corridor must be present.
                _lift_context(assessed_lift['proposal'])
                if lift_origin is None:
                    raise ValueError('整段抬手需要保留本次開始姿勢')
        r=self.robot;r.begin(allow_rearm=False)
        def pre_motion_comparison(older,current):
            try:return unchanged_wrist(older,current)
            except WristSceneChangedError as exc:
                raise PreMotionSceneChangedError(str(exc),current) from exc
        before=self.capture();pre_motion_comparison(reference,before)
        # Only the exact elbow arc already shown to the visual assessor may
        # use its own nominal distance cap. Cartesian/grasp/retreat and direct
        # unassessed moves retain the common 5 mm vector contract.
        raw=before['joints'];delta=None if assessed_arc else self.vector(direction,distance_mm,raw)
        # A command can lead the loaded joint while measured motion remains in
        # the tighter path tube. Screen that wider possible command geometry too.
        if assessed_lift is not None:
            if not (free_space_lift or free_space_retreat or assessed_motion is not None):raise ValueError('重新定位規劃只能用於空手動作')
            if raw!=assessed_lift['origin'] or dict(r.goals)!=assessed_lift['hold_goals']:
                raise RuntimeError('判斷後手臂位置有變，請重新觀察再移動')
            # Execute the exact path shown to the visual assessor. Do not solve
            # a different IK candidate after the clearance decision.
            proposal,envelope=assessed_lift['proposal'],assessed_lift['envelope']
            arc=proposal.get('motion_kind') in ('elbow_raise','elbow_raise_macro','pan_retreat','cartesian_backward')
            same=(abs(distance_mm-np.linalg.norm(proposal['requested_delta_m'])*1000)<=1e-9 if arc
                  else np.allclose(delta,proposal['requested_delta_m'],rtol=0,atol=1e-12))
            if not same:
                raise ValueError('移動距離與已確認的路徑不同')
        else:
            proposal,envelope=(self._wave_lift_proposal(distance_mm,raw,lift_origin) if free_space_lift
                               else self._translation_proposal(direction,distance_mm,raw))
        retreat=free_space_retreat and proposal.get('motion_kind')=='pan_retreat'
        backward=free_space_retreat and proposal.get('motion_kind')=='cartesian_backward'
        if free_space_retreat and not (retreat or backward):raise ValueError('退讓需要明確的受限路徑')
        arc=free_space_lift and proposal.get('motion_kind') in ('elbow_raise','elbow_raise_macro')
        reposition=free_space_lift or free_space_retreat
        if retreat:
            from .wave_retreat import check_retreat_attempt
            check_retreat_attempt(self.model,lift_origin or raw,proposal['joint_targets'])
        if backward:
            from .wave_backward import check_backward_attempt
            check_backward_attempt(self.model,lift_origin or raw,proposal['joint_targets'])
        if arc:
            from .wave_arc import check_arc_attempt
            check_arc_attempt(self.model,lift_origin or raw,proposal['joint_targets'])
        elif free_space_lift and lift_origin is not None:
            from .wave_lift import lift_attempt
            lift_attempt(self.model.tip(lift_origin),self.model.tip(proposal['joint_targets']))
        delta=np.asarray(proposal['requested_delta_m'],float)
        accepted_mm=float(np.linalg.norm(delta)*1000)
        fresh=self.capture();comparison=pre_motion_comparison(before,fresh)
        target=proposal['joint_targets'];generation=before['camera_generation']
        bounds={n:(min(raw[n],target[n])-envelope[n],max(raw[n],target[n])+envelope[n]) for n in NAMES}
        measured_bounds={n:(min(raw[n],target[n])-12,max(raw[n],target[n])+12) for n in NAMES}
        def guard(actual):
            if cancel is not None and cancel.is_set():raise SessionCancelled('抬手已停止')
            if r.camera.generation!=generation:raise RuntimeError('移動期間腕部相機重新連接')
            if (reposition or empty_hand_progress) and abs(actual['gripper']-raw['gripper'])>4:raise RuntimeError('移動時夾爪開口改變')
            clearance=self.model.floor_minimum({**raw,**actual})
            if any(z<.005 for n,z in clearance.items() if n!='base_link'):
                raise RuntimeError('已接近模型桌面界線')
        def measured_guard(actual):
            # Measured-space limits must never mistake a gravity-compensating
            # motor goal for an observed Cartesian pose. The command geometry
            # still passes the separate full-mesh/table and raw-box guard.
            if backward:
                from .wave_backward import check_backward_attempt,check_backward_corridor
                check_backward_corridor(self.model,raw,actual)
                check_backward_attempt(self.model,lift_origin or raw,actual)
            elif retreat:
                from .wave_retreat import check_retreat_attempt,check_retreat_corridor
                check_retreat_corridor(self.model,raw,actual)
                check_retreat_attempt(self.model,lift_origin or raw,actual)
            elif arc:
                from .wave_arc import check_arc_attempt,check_arc_corridor
                check_arc_corridor(self.model,raw,actual,motion_kind=proposal['motion_kind'])
                check_arc_attempt(self.model,lift_origin or raw,actual)
            elif free_space_lift:
                from .wave_lift import lift_attempt,lift_corridor
                lift_corridor(self.model.tip(raw),self.model.tip(actual))
                if lift_origin is not None:
                    lift_attempt(self.model.tip(lift_origin),self.model.tip(actual))
            elif empty_hand_progress:
                from scipy.spatial.transform import Rotation
                a=self.model.tip(raw);b=self.model.tip(actual)
                if (np.linalg.norm(b[:3,3]-a[:3,3])>.007 or
                        np.degrees(Rotation.from_matrix(b[:3,:3]@a[:3,:3].T).magnitude())>3):
                    raise RuntimeError('這一步離開已檢查的局部移動範圍')
        def observed(event):
            actual=r.read()
            if any(not min(raw[n],target[n])-12<=actual[n]<=max(raw[n],target[n])+12 for n in NAMES):
                raise RuntimeError('實測姿勢離開這一步的局部路徑')
            guard(actual)
            if reposition or empty_hand_progress:measured_guard(actual)
            telemetry={}
            with r.io:
                for n in NAMES:
                    load=r._read(n,60);current=r._read(n,69)
                    telemetry[n]={'position':actual[n],'goal':r.goals[n],
                        'signed_pwm':-(load&0x3ff) if load&0x400 else load,
                        'current_milliamps':(-(current&0x7fff) if current&0x8000 else current)*6.5,
                        'temperature_c':r._read(n,63,1),'status':r._read(n,65,1)}
            self.record({'type':'motor_feedback','phase':event.get('phase','trajectory'),'actual':actual,
                         'telemetry':telemetry,'logical_targets':logical,'bias':dict(r.bias),'decision':event})
        report(f'抬高約 {accepted_mm:.1f} mm，連續核對馬達位置' if proposal.get('motion_kind')=='elbow_raise_macro'
               else f'小步移動 {accepted_mm:g} mm，核對實際位置')
        # The IK target is a MEASURED endpoint. A previous accepted Cartesian
        # step may finish with nonzero scalar error; its old logical anchor must
        # not become this step's measured goal. Rebase bookkeeping on the new
        # settled encoder pose without rewriting any physical hold goal.
        from .local_feedback import adopt_goal_anchor
        with r.io:
            if r.stop.is_set():raise SessionCancelled('移動已停止')
            anchor=adopt_goal_anchor(raw,r.goals)
            active=(tuple(proposal['active_joint_targets']) if backward else
                    ('shoulder_pan',) if retreat else ('elbow_flex',) if arc else NAMES)
            logical={n:target[n] for n in active}
            if any(logical[n]+anchor['bias'][n]!=proposal['physical_goal_targets'][n] for n in logical):
                raise RuntimeError('量測目標與既有保持補償不一致')
            self.record({'type':'measured_step_anchor','measured':raw,'old_bias':dict(r.bias),
                         'physical_goals':dict(r.goals),'new_bias':anchor['bias'],'logical_targets':logical})
            r.bias=anchor['bias']
        goal_tip=self.model.tip(target)
        def exact_endpoint(actual):
            from scipy.spatial.transform import Rotation
            T=self.model.tip(actual)
            position_error=np.linalg.norm(T[:3,3]-goal_tip[:3,3])
            angle=np.degrees(Rotation.from_matrix(T[:3,:3]@goal_tip[:3,:3].T).magnitude())
            progress=(T[:3,3]-self.model.tip(raw)[:3,3])@delta/np.linalg.norm(delta)
            return bool(position_error<=.001 and angle<=1 and progress>=max(.0005,np.linalg.norm(delta)*.5))
        def endpoint_accept(actual):
            if empty_hand_progress and not (arc or retreat):
                measured_guard(actual)
                if backward:
                    # Keep the phase orientation budget as well as the local
                    # corridor; useful progress only replaces target accuracy.
                    from scipy.spatial.transform import Rotation
                    if np.degrees(Rotation.from_matrix(self.model.tip(actual)[:3,:3]@
                            self.model.tip(lift_origin or raw)[:3,:3].T).magnitude())>1:return False
                return progress_result(actual)['accepted']
            if backward:
                from .wave_backward import backward_endpoint
                return backward_endpoint(self.model,raw,actual,proposal,attempt_origin=lift_origin or raw)
            if retreat:
                from .wave_retreat import retreat_endpoint
                return retreat_endpoint(self.model,raw,actual,proposal)
            if arc:
                from .wave_arc import arc_endpoint
                return arc_endpoint(self.model,raw,actual,proposal)
            if free_space_lift:
                from .wave_lift import lift_endpoint
                # For empty-hand repositioning, a settled point within the
                # separately assessed local lift corridor is useful progress.
                # Never use this completion region for grasp/lift verification.
                maximum=(proposal.get('coordination') or {}).get('actual_tip_corridor_mm',{}).get('maximum_up')
                return lift_endpoint(self.model.tip(raw),self.model.tip(actual),delta,
                                     verified_max_up_mm=maximum,
                                     orientation_origin=self.model.tip(lift_origin) if lift_origin is not None else None)['accepted']
            return exact_endpoint(actual)
        def progress_result(actual):
            return classify_translation_progress(self.model.tip(raw),self.model.tip(actual),delta,
                target_tip=goal_tip,origin_joints=raw,actual_joints=actual,empty_hand=True,corridor_pass=True)
        start=time.monotonic()
        self.record({'type':'proposal','direction':direction,'before':before,'proposal':proposal,'freshness':comparison})
        # An override arriving during the final camera/model checks can still
        # discard this unstarted reposition. Once glide starts, only the real
        # job cancellation and existing controller guards can interrupt it.
        if before_dispatch is not None:before_dispatch()
        try:
            r.glide(logical,speed=40,precision=4,feedback=True,guard=guard,raw_bounds=bounds,
                    endpoint_accept=endpoint_accept,endpoint_stability_ticks=2 if reposition or empty_hand_progress else None,
                    measured_guard=measured_guard if reposition or empty_hand_progress else None,
                    stall_response_wait=bool(arc),
                    deferred_joints=('wrist_flex','shoulder_lift') if free_space_lift and not arc else (),
                    measured_bounds=measured_bounds,observe=observed,feedback_observe=observed,allow_rearm=False)
        except Exception as exc:
            failure={'type':'movement_failed','error':str(exc),'actual':r.read(),'goals':dict(r.goals)}
            if r.camera.fresh():failure['frame']=self.snapshot()
            self.record(failure);raise
        try:
            after=self.capture();measured=self.model.tip(after['joints'])[:3,3]-self.model.tip(raw)[:3,3]
            if reposition or empty_hand_progress:
                # A motor acknowledgement is insufficient: settled image encoders
                # must still satisfy the full bounded lift predicate.
                from .local_feedback import check_bounds
                check_bounds(after['joints'],measured_bounds)
                guard(after['joints'])
                measured_guard(after['joints'])
                if not endpoint_accept(after['joints']):raise RuntimeError('移動後位置未穩定，先停下來')
        except Exception as exc:
            if reposition or empty_hand_progress:
                r.error=str(exc);r.stop.set();r.hold()
                self.record({'type':'lift_settling_failed','error':str(exc)})
            raise
        result={'direction':direction,'requested_mm':distance_mm,'accepted_mm':accepted_mm,'actual_delta_model_mm':(measured*1000).tolist(),
                'before':before,'after':after,'elapsed':time.monotonic()-start,
                'model_displacement_not_external_metrology':True}
        if reposition:
            exact=exact_endpoint(after['joints'])
            result.update(purpose='wave_retreat' if free_space_retreat else 'wave_raise',exact_target_reached=exact,
                          motion_kind=proposal.get('motion_kind','cartesian'),
                          completion='exact_target' if exact else 'partial_progress')
        if empty_hand_progress and not (arc or retreat):
            progress=progress_result(after['joints'])
            result.update(completion=progress['completion'],exact_target_reached=progress['exact_target_reached'],
                          progress=progress,requires_reobservation=True)
        self.record({'type':'movement_result',**result});self.latest=result
        if not reposition and not empty_hand_progress and (np.linalg.norm(measured)>.007 or measured@delta/np.linalg.norm(delta)<max(.0005,np.linalg.norm(delta)*.5)):
            r.stop.set();r.hold();raise RuntimeError('實際移動與提案不同，已停止並保留量測')
        return result
    def advice(self,query,report,cancel):
        self.robot.begin(allow_rearm=False);frame=self.capture()
        if self.attempt_start is None:self.attempt_start=frame
        if self.attempt_start['camera_generation']!=frame['camera_generation']:
            raise RuntimeError('腕部相機已重新連接，請重新開始抓取觀察')
        previous=[self.attempt_start]
        if self.latest and self.latest['before'].get('camera_generation')==frame['camera_generation']:
            previous.append(self.latest['before'])
        views=list({f['id']:f for f in previous+[frame]}.values())
        report('Astra 正在判斷夾爪與物件的相對位置')
        advice=self.brain.grasp_advice(query,views,{
            'stage':self.state,'jaw_position_raw':frame['joints']['gripper'],
            'frame_roles':['current' if f['id']==frame['id'] else 'attempt_start' if f['id']==self.attempt_start['id'] else 'latest_before_move' for f in views],
            'completed_steps':self.attempt_history[-8:],
            'last_measured_step':None if not self.latest else {k:v for k,v in self.latest.items() if k not in ('before','after')},
            'single_step_limit_mm':5},cancel)
        fresh=self.capture();comparison=unchanged_wrist(frame,fresh)
        self.record({'type':'advice','frame':frame,'advice':advice,'freshness':comparison})
        return advice,fresh

    def open_empty(self,reference,report,*,cancel=None,guard=None,opening_contract=None):
        """Open an empty jaw under low PWM, retaining the session for its caller.

        An optional observation guard sees measured poses only, including while
        lowering the RAM PWM cap. Failure never finishes or restores the cap.
        """
        if guard is not None and not callable(guard):
            raise ValueError('Empty-jaw guard must be callable')
        # Existing advice/open callers have reviewed only the exercised 1860
        # opening. A wider v2 target requires the exact assessed contract.
        target=1860;raw_range=(1500,1862);endpoint_tolerance=6
        if opening_contract is not None:
            from .jaw_opening import require_opening_contract
            opening_contract=require_opening_contract(opening_contract,self.model,self.robot.config)
            target=opening_contract['target_raw'];raw_range=opening_contract['measured_raw_range']
            endpoint_tolerance=opening_contract['endpoint_tolerance_ticks']
        r=self.robot;generation=reference.get('camera_generation')
        if reference.get('source')!='wrist' or not generation:
            raise ValueError('張開夾爪需要腕部影像')
        def live():
            if (cancel is not None and cancel.is_set()) or r.stop.is_set():
                raise SessionCancelled('張開夾爪已停止')
            if not r.camera.fresh():raise RuntimeError('張開夾爪期間腕部畫面中斷')
            if r.camera.generation!=generation:raise RuntimeError('張開夾爪期間腕部相機重新連接')
        live();r.begin(allow_rearm=False);live()
        fresh=self.capture();live();unchanged_wrist(reference,fresh)
        if not raw_range[0]<=fresh['joints']['gripper']<=raw_range[1]:
            raise ValueError('目前夾爪開口不在已驗證的張開區間')
        origin=fresh['joints'];start=time.monotonic()
        if opening_contract is not None:
            require_opening_contract(opening_contract,self.model,r.config)
        if opening_contract is not None and target < origin['gripper']:
            raise ValueError('這個空爪動作只能增加開口，不能縮回較小目標')
        def body_guard(actual):
            live()
            if any(abs(actual[n]-origin[n])>4 for n in NAMES):
                raise RuntimeError('張開夾爪時手臂姿勢改變')
            if opening_contract is not None and not max(raw_range[0],origin['gripper']-endpoint_tolerance)<=actual['gripper']<=min(raw_range[1],target+endpoint_tolerance):
                raise RuntimeError('張開夾爪偏離這次審核的開口範圍')
        def measured_guard(actual):
            body_guard(actual)
            if guard is not None:guard(actual)
            live()
        measured_guard(origin)
        power=r.gripper_power;old_read=power.read;old_write=power.write
        def power_check():
            live();measured_guard(r.read())
        def checked_read(*args):
            power_check();value=old_read(*args);power_check();return value
        def checked_write(*args):
            power_check();value=old_write(*args);power_check();return value
        transaction_started=False
        # Restore callback identity even if journal I/O, cancellation, or a
        # partially acknowledged PWM write aborts entry. No cleanup can rearm
        # the jaw or restore its high cap on this failure path.
        try:
            with r.io:
                power_check();power.read=checked_read;power.write=checked_write
            transaction_started=True
            if r.pwm_owner is None:r.pwm_owner=power.enter()
            power_check()
            report('輕輕張開夾爪，準備對準物件')
            live()
            r.glide({'gripper':target},speed=50,precision=endpoint_tolerance,guard=body_guard,
                    measured_guard=measured_guard,observe=lambda event:live(),allow_rearm=False)
            live();after=self.capture();live()
            if (after.get('source')!='wrist' or after.get('camera_generation')!=generation or
                    after['capture_monotonic']<=fresh['capture_monotonic']):
                raise RuntimeError('張開夾爪後缺少同一相機的新影像')
            measured_guard(after['joints'])
            if opening_contract is not None and abs(after['joints']['gripper']-target)>endpoint_tolerance:
                raise RuntimeError('張開後尚未到達這次審核的目標開口')
            self.record({'type':'opened','before':fresh,'after':after,'elapsed':time.monotonic()-start,
                         'opening_contract':opening_contract,'target_raw':target})
            return after
        except Exception:
            if transaction_started:
                power.leave_pending();r.stop.set();r.hold()
            raise
        finally:
            with r.io:
                power.read=old_read;power.write=old_write

    def close_candidate(self,reference,report,cancel,*,squeeze_ticks=0):
        """Short goal lead; obstruction stops closing, never asserts grasp/force.

        squeeze_ticks>0: after the obstruction is detected, advance the goal that much further past the
        stalled position (5 ticks/100 ms, same low PWM cap, same drift/temperature checks) for a firmer hold."""
        if type(squeeze_ticks) is not int or not 0<=squeeze_ticks<=40:raise ValueError('Squeeze must be 0..40 ticks')
        r=self.robot;power=r.gripper_power;r.begin(allow_rearm=False)
        frame=self.capture();unchanged_wrist(reference,frame)
        if r.pwm_owner is None:r.pwm_owner=power.enter()
        power.mark_contact_possible(r.pwm_owner);self.state='closing'
        origin=frame['joints'];window=[];started=time.monotonic();last_goal=origin['gripper']
        report('以低出力輕合夾爪，讀取馬達回饋')
        r.moving=True
        try:
            while time.monotonic()-started<12:
                if cancel.is_set() or r.stop.is_set():raise SessionCancelled('合爪已停止')
                if not r.camera.fresh():raise RuntimeError('合爪期間腕部影像中斷')
                s=r.gripper_telemetry();self.record({'type':'closing_sample',**s})
                if s['pwm_limit']!=power.LOW or s['temperature_c']>55:raise RuntimeError('夾爪出力或溫度不符限制')
                if any(abs(s['body_position'][n]-origin[n])>CONTACT_BODY_DRIFT_TICKS for n in NAMES):raise RuntimeError('合爪期間手臂偏移')
                window.append(s);window=[x for x in window if s['capture_monotonic']-x['capture_monotonic']<=.6]
                if (len(window)>=4 and s['capture_monotonic']-window[0]['capture_monotonic']>=.35
                        and max(x['position'] for x in window)-min(x['position'] for x in window)<=2
                        and s['position']-s['goal']>=8 and abs(s['load_signed_pwm'])>=20):
                    self.state='contact_candidate'
                    contact=s
                    if squeeze_ticks:
                        report('偵測到接觸，在低出力上限內夾緊')
                        target=max(1525,s['position']-squeeze_ticks)
                        while last_goal>target:
                            if cancel.is_set() or r.stop.is_set():raise SessionCancelled('合爪已停止')
                            if not r.camera.fresh():raise RuntimeError('合爪期間腕部影像中斷')
                            goal=max(target,last_goal-5)
                            with r.io:
                                power.assert_motion_allowed(r.pwm_owner)
                                if r._read('gripper',48)!=power.LOW:raise RuntimeError('夾爪低出力設定改變')
                                r._write('gripper',42,goal);r.goals['gripper']=goal;last_goal=goal
                            if cancel.wait(.1):raise SessionCancelled('合爪已停止')
                            s=r.gripper_telemetry();self.record({'type':'squeeze_sample',**s})
                            if s['pwm_limit']!=power.LOW or s['temperature_c']>55:raise RuntimeError('夾爪出力或溫度不符限制')
                            if any(abs(s['body_position'][n]-origin[n])>CONTACT_BODY_DRIFT_TICKS for n in NAMES):raise RuntimeError('夾緊期間手臂偏移')
                            if s['position']<=1525:raise RuntimeError('夾緊時夾爪接近空夾位置，物件可能已滑出')
                        if cancel.wait(.3):raise SessionCancelled('合爪已停止')
                        s=r.gripper_telemetry()
                    # Retain the bounded low-power goal.
                    result={'obstruction_candidate':True,'not_force_measurement':True,'sample':s,'contact_sample':contact,
                            'squeeze_ticks':squeeze_ticks,'before':frame}
                    self.record({'type':'closing_stopped',**result});return result
                if s['position']<=1525:
                    self.state='contact_candidate';raise RuntimeError('夾爪已接近空夾位置，尚無可靠抓住證據')
                # At most 5 new commanded ticks/100ms and 12 ticks ahead of actual.
                goal=max(1520,last_goal-5,s['position']-12)
                if goal<last_goal:
                    with r.io:
                        if cancel.is_set() or r.stop.is_set():raise SessionCancelled('合爪已停止')
                        power.assert_motion_allowed(r.pwm_owner)
                        if r._read('gripper',48)!=power.LOW:raise RuntimeError('夾爪低出力設定改變')
                        r._write('gripper',42,goal);r.goals['gripper']=goal;last_goal=goal
                if cancel.wait(.1):raise SessionCancelled('合爪已停止')
            raise RuntimeError('合爪等待逾時，保留低出力')
        except SessionCancelled:
            try:power.stop_contact(r.pwm_owner)
            finally:r.stop.set();r.hold()
            raise
        except Exception:
            # Any I/O error keeps the contact barrier and never restores power.
            power.leave_pending();r.stop.set();r.hold();raise
        finally:r.moving=False

    def release(self,reference,report,cancel,*,support_verified=False,empty_verified=False,retreat=None):
        """Explicit, visually reviewed release; never used as automatic cleanup.

        retreat: optional callable that moves the BODY away from a placed object after the jaw is measured
        open (the gripper power session authorizes exactly that); the empty-jaw image is then taken there."""
        if retreat is not None and (not callable(retreat) or not support_verified):
            raise ValueError('Retreat after release requires a supported placement and a callable')
        r=self.robot;power=r.gripper_power
        if not (support_verified or empty_verified):raise ValueError('先確認物件已有支撐，或夾爪空置')
        # Stopped recovery may read a new image/pose; it never rearms body motors.
        def recovery_frame():
            before=r.read();frame=self.snapshot();after=r.read()
            if max(abs(before[n]-after[n]) for n in before)>2:raise RuntimeError('釋放取像時手臂仍在移動')
            frame['joints']=after;return frame
        fresh=recovery_frame();unchanged_wrist(reference,fresh)
        record=power._record()
        if not record or record['phase']=='completed':raise RuntimeError('沒有待完成的低出力夾爪工作')
        if not record.get('contact_possible'):
            if not empty_verified:raise RuntimeError('空夾工作需要新的空爪確認')
            result=power.finish() if power.state=='active' else power.recover()
            r.goals['gripper']=result.get('neutral_goal',r.goals.get('gripper'));r.pwm_owner=None
            self.state='idle';return result
        if power.state in ('active','holding'):
            challenge=power.begin_release(r.pwm_owner,camera_generation=fresh['camera_generation'],frame_sequence=fresh['sequence'])
        else:
            challenge=power.recover_for_release(camera_generation=fresh['camera_generation'],frame_sequence=fresh['sequence'])
            r.pwm_owner=challenge['owner_token']
        origin=fresh['joints'];report('確認物件有支撐，輕輕張開夾爪')
        try:
            deadline=time.monotonic()+12
            while time.monotonic()<deadline:
                if cancel.is_set():raise SessionCancelled('釋放已停止')
                if not r.camera.fresh() or r.camera.generation!=fresh['camera_generation']:raise RuntimeError('釋放期間腕部畫面中斷')
                s=r.gripper_telemetry()
                if any(abs(s['body_position'][n]-origin[n])>4 for n in NAMES):raise RuntimeError('釋放期间手臂偏移')
                if s['pwm_limit']!=power.LOW or s['temperature_c']>55:raise RuntimeError('釋放出力或溫度異常')
                if s['goal']==1860 and power._open_enough(s['position'],1860):break
                goal=min(1860,max(s['position'],s['goal'])+5)
                power.release_goal(r.pwm_owner,goal);r.goals['gripper']=goal
                if cancel.wait(.1):raise SessionCancelled('釋放已停止')
            else:raise RuntimeError('夾爪張開等待逾時')
            challenge=power.verify_release_open(r.pwm_owner,allow_retreat=retreat is not None)
            if cancel.wait(.25):raise SessionCancelled('釋放已停止')
            if retreat is not None:
                report('夾爪已張開，先退開再核對空爪')
                retreat()
                if cancel.wait(.4):raise SessionCancelled('釋放已停止')
            view=recovery_frame();report('核對兩側夾爪之間已空置')
            focus=('只確認兩指之間的空間是否為空；夾爪外側的其他物體及其支撐狀態不影響這個判斷，也不應降低信心。'
                   if empty_verified or not support_verified or retreat is not None else
                   '確認剛放下的物件是否由桌面支撐、已不再被夾住；它站在張開的兩指之間是預期的。')
            check=self.brain.jaw_view(focus,[view],cancel)
            # After a supported put-down the released object still stands between the open jaws;
            # then the evidence is: both fingers visible and the object visibly supported by the table.
            required=(('both_fingers_visible','target_supported_on_table')
                      if support_verified and not empty_verified and retreat is None
                      else ('both_fingers_visible','gap_visible','gripper_empty'))
            self.record({'type':'release_view','check':check,'required':list(required),'frame':view.get('id'),
                         'support_verified':support_verified,'empty_verified':empty_verified})
            if check['confidence']<.9 or not all(check[n] for n in required):
                raise RuntimeError('夾爪已張開，影像證據仍不足；保留低出力：'+str(check.get('reason',''))[:200])
            latest=recovery_frame();unchanged_wrist(view,latest)
            evidence={**{k:latest[k] for k in ('source','camera_generation','sequence','capture_monotonic')},
                'session_id':challenge['session_id'],'release_id':challenge['release_id'],'empty_verified':True,
                'frame_sha256':hashlib.sha256(Path(latest['path']).read_bytes()).hexdigest()}
            result=power.confirm_released(r.pwm_owner,evidence)
            r.pwm_owner=None;r.goals['gripper']=result['neutral_goal'];r.bias['gripper']=0;self.state='idle'
            self.record({'type':'released','frame':latest,'visual':check,'receipt':result});return result
        except SessionCancelled:
            try:power.stop_contact(r.pwm_owner)
            finally:r.hold()
            raise
        except Exception:power.leave_pending();r.hold();raise

    def run(self,query,report,cancel,*,max_steps=24):
        if self.state not in ('idle','approaching'):raise RuntimeError('請先處理目前夾爪中的物件')
        self.attempt_start=None;self.attempt_history=[];self.latest=None
        self.state='approaching';report('先確認物件與兩側夾爪的位置')
        try:
            for index in range(max_steps):
                if cancel.is_set():raise RuntimeError('抓取已停止')
                advice,frame=self.advice(query,report,cancel)
                minimum_confidence=LOCAL_MOVE_MIN_CONFIDENCE if advice['step']=='move' else .9 if advice['step']=='close' else .85
                if not advice['target_visible'] or not advice['clear_path'] or advice['confidence']<minimum_confidence:
                    raise RuntimeError(advice['warning'] or advice['reason'] or '目前看不清楚可行的抓取位置')
                if advice['step']=='open' and advice['gripper_empty']:
                    self.open_empty(frame,report)
                elif advice['step']=='move':
                    motion=self.move(advice['direction'],advice['distance_mm'],frame,report,
                                     cancel=cancel,empty_hand_progress=bool(advice['gripper_empty']))
                    self.attempt_history.append({'direction':motion['direction'],'actual_delta_model_mm':motion['actual_delta_model_mm'],
                        'purpose':advice['reason'],'remaining_uncertainty':advice['warning']})
                elif advice['step']=='close' and advice['target_between_fingers']:
                    contact=self.close_candidate(frame,report,cancel)
                    self.state='verifying';before=self.capture()
                    motion=self.move('up',3,before,report);after=motion['after']
                    report('檢查物件是否真的離桌、跟著夾爪抬起')
                    verification=self.brain.verify_grasp(query,[before,after],{
                        'actual_delta_model_mm':motion['actual_delta_model_mm'],
                        'closing_obstruction_candidate':contact['obstruction_candidate']},cancel)
                    fresh=self.capture();unchanged_wrist(after,fresh)
                    confirmed=(verification['confidence']>=.9 and all(verification[n] for n in
                        ('same_object','between_fingers','clear_of_table','follows_gripper')) and not verification['gripper_empty'])
                    self.robot.gripper_power.defer_holding(self.robot.pwm_owner)
                    self.state='holding' if confirmed else 'holding_unverified'
                    result={'grasp_confirmed':confirmed,'verification':verification,'state':self.state,
                            'pwm_limit':100,'pre_lift_frame':before,'held_frame':fresh}
                    self.record({'type':'grasp_result',**result});return result
                else:raise RuntimeError(advice['warning'] or advice['reason'] or '需要更清楚的物件與夾爪關係')
            raise RuntimeError('已達本次小步觀察上限，保持目前姿勢')
        except Exception as exc:
            power=self.robot.gripper_power
            if self.state not in ('approaching','idle'):
                if isinstance(exc,SessionCancelled) and power.state in ('active','holding'):
                    try:power.stop_contact(self.robot.pwm_owner)
                    except Exception:power.leave_pending()
                else:power.leave_pending()
                self.state='holding_unverified';self.robot.stop.set();self.robot.hold()
            if self.state in ('approaching','idle') and power.state=='active':
                # No closing/contact was authorized. Preserve opening goal under
                # low cap until the next fresh inspection confirms an empty jaw.
                self.state='approaching'
            raise
