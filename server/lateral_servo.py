"""Pure local image-error servo proposals; caller owns all hardware and guards.

After an independently controlled <=3mm pilot demonstrates the local response,
use target correspondence plus aperture edges to propose subsequent lateral
steps. No metric object size, depth, camera extrinsic, force or grasp inference.
All outputs are proposals. Lost evidence stops this state instance permanently.
"""
from __future__ import annotations
import math
import numpy as np

def _finite(value,name):
    if isinstance(value,bool) or not isinstance(value,(int,float,np.number)) or not math.isfinite(float(value)):
        raise ValueError('Nonfinite '+name)
    return float(value)

def _frame(frame,now,generation=None):
    if frame.get('source')!='wrist':raise ValueError('Only wrist frames accepted')
    g=frame.get('camera_generation');seq=frame.get('sequence')
    if not isinstance(g,str) or not g or (generation is not None and g!=generation):raise ValueError('Camera generation changed')
    if type(seq) is not int or seq<0:raise ValueError('Frame sequence required')
    timestamp=_finite(frame.get('capture_monotonic'),'frame time');now=_finite(now,'current time')
    if not 0<=now-timestamp<=1.:raise ValueError('Current wrist frame is stale or from future')
    size=frame.get('image_size')
    if not isinstance(size,(tuple,list)) or len(size)!=2 or any(type(v) is not int or v<64 for v in size):raise ValueError('Image size required')
    return {'source':'wrist','camera_generation':g,'sequence':seq,'capture_monotonic':timestamp,'image_size':list(size)}

def _aperture(value,size):
    if value.get('status')!='ok' or value.get('image_size')!=size:raise ValueError('Aperture landmark unavailable')
    p=np.asarray(value.get('aperture_center_px'),float)
    if p.shape!=(2,) or not np.isfinite(p).all() or np.any(p<0) or np.any(p>=size):raise ValueError('Bad aperture point')
    for side in ('left','right'):
        e=value.get(side,{})
        if e.get('status')!='ok' or _finite(e.get('inlier_fraction'),'edge fraction')<.55:
            raise ValueError('Jaw edge identity insufficient')
        if _finite(e.get('residual_p95_px'),'edge residual')>3 or _finite(e.get('alternating_rows_fit_disagreement_p95_px'),'edge repeatability')>3:
            raise ValueError('Jaw edge fit inconsistent')
    gap=_finite(value.get('visible_gap_px'),'jaw image gap')
    if gap<=8:raise ValueError('Jaw image aperture too narrow to localize')
    return p,gap

class LateralServo:
    """Bounded local controller state, with one pending proposal at a time.

    Construct from a fresh selected target material point and aperture fit.
    observe_step(..., proposal_id=None) ingests the first external pilot; later
    calls must consume the exact pending proposal_id. Never use the whole bbox
    center if it does not represent the selected grasp-region material.
    """
    def __init__(self,frame,target_point_px,aperture,*,target_token,now_monotonic,
                 max_step_mm=3.,total_travel_mm=40.,image_tolerance_px=4.):
        self.max_step=_finite(max_step_mm,'step');self.budget=_finite(total_travel_mm,'budget')
        self.tolerance=_finite(image_tolerance_px,'image tolerance')
        if not 0<self.max_step<=3 or not self.max_step<=self.budget<=40 or not 1<=self.tolerance<=12:
            raise ValueError('Invalid local servo limits')
        if not isinstance(target_token,str) or not target_token:raise ValueError('Selected target identity token required')
        self.frame=_frame(frame,now_monotonic);self.aperture,self.gap=_aperture(aperture,self.frame['image_size'])
        p=np.asarray(target_point_px,float)
        if p.shape!=(2,) or not np.isfinite(p).all() or np.any(p<0) or np.any(p>=self.frame['image_size']):raise ValueError('Invalid selected target material point')
        self.point=p;self.target_token=target_token;self.error=float(p[0]-self.aperture[0])
        self.gain=None;self.noise=.5;self.measured_travel=0.;self.commanded_travel=0.;self.steps=0
        self.pending=None;self.next_id=1;self.target_box=None;self.state='needs_pilot';self.reason='Need a measured lateral pilot with retained target identity'
        if abs(self.error)<=self.tolerance:self.state='aligned';self.reason='Selected material point is horizontally aligned in this image only'
    def status(self):
        return {'state':self.state,'reason':self.reason,'target_token':self.target_token,
            'error_x_px':self.error,'target_point_px':self.point.tolist(),'aperture_point_px':self.aperture.tolist(),
            'observed_response_px_per_mm':self.gain,'measured_lateral_travel_mm':self.measured_travel,
            'commanded_lateral_travel_mm':self.commanded_travel,'observed_steps':self.steps,
            'image_tolerance_px':max(self.tolerance,3*self.noise),'proposal_only':True,
            'calibrated_contact_tcp':False,'metric_object_dimensions_used':False,'confirmed_grasp':False,
            'frame_sequence':self.frame['sequence'],'camera_generation':self.frame['camera_generation']}
    def stop(self,reason):
        self.state='stopped';self.reason=str(reason);self.pending=None
        return self.status()
    def observe_step(self,frame,pair_evidence,aperture,*,target_token,
                     actual_camera_right_mm,now_monotonic,proposal_id=None):
        """Consume actual observed motion, never commanded motion as response.

        Positive camera-right translation must reduce positive image error.
        Caller computes the measured projection from actual body encoder FK;
        this helper neither reads those encoders nor asserts external metrology.
        """
        if self.state in ('stopped','aligned'):return self.status()
        try:
            if target_token!=self.target_token:raise ValueError('Target identity token changed')
            pending=self.pending
            if self.state=='needs_pilot':
                if proposal_id is not None:raise ValueError('External pilot must have no proposal id')
            elif self.state!='awaiting_observation' or pending is None or proposal_id!=pending['proposal_id']:
                raise ValueError('Observation does not match one pending lateral proposal')
            current=_frame(frame,now_monotonic,self.frame['camera_generation'])
            if current['image_size']!=self.frame['image_size']:raise ValueError('Image geometry changed')
            if current['sequence']<=self.frame['sequence'] or current['capture_monotonic']<=self.frame['capture_monotonic']:
                raise ValueError('Need a new ordered wrist frame')
            ev=pair_evidence
            if ev.get('source')!='wrist-only' or ev.get('camera_generation')!=self.frame['camera_generation'] or ev.get('image_size')!=current['image_size']:
                raise ValueError('Pair evidence camera identity/geometry mismatch')
            sequences=ev.get('frame_sequences',[]);times=ev.get('frame_capture_monotonic',[])
            if len(sequences)<2 or len(times)!=len(sequences) or sequences[0]!=self.frame['sequence'] or sequences[-1]!=current['sequence']:
                raise ValueError('Pair evidence endpoints do not match state frames')
            if any(type(s) is not int for s in sequences) or any(b<=a for a,b in zip(sequences,sequences[1:])):
                raise ValueError('Pair frame sequence is not ordered')
            times=[_finite(t,'evidence time') for t in times]
            if any(b<=a for a,b in zip(times,times[1:])) or abs(times[0]-self.frame['capture_monotonic'])>1e-6 or abs(times[-1]-current['capture_monotonic'])>1e-6:
                raise ValueError('Pair evidence timestamps do not match frames')
            target=ev.get('target',{})
            if target.get('status')!='ok' or target.get('track_count',0)<12 or _finite(target.get('retained_fraction'),'retained identity')<.6:
                raise ValueError('Target tracks lost or insufficient identity retention')
            fb=_finite(target.get('fb_p95_px'),'FB error');ncc=_finite(target.get('ncc_median'),'NCC')
            residual=_finite(target.get('similarity_residual_p95_px'),'target consistency')
            scale=_finite(target.get('median_scale'),'target scale')
            if not 0<=fb<=.5 or not .85<=ncc<=1.000001 or not 0<=residual<=1.5 or not .9<=scale<=1.1:
                raise ValueError('Target correspondence failed local lateral quality checks')
            box=np.asarray(ev.get('target_bbox_input_px'),float)
            if box.shape!=(4,) or not np.isfinite(box).all() or np.any(box[2:]<=0) or np.any(self.point<box[:2]) or np.any(self.point>box[:2]+box[2:]):
                raise ValueError('Tracked box does not contain the selected target material point')
            M=np.asarray(target.get('similarity_matrix'),float)
            if M.shape!=(2,3) or not np.isfinite(M).all():raise ValueError('Target transform unavailable')
            newpoint=M[:,:2]@self.point+M[:,2]
            if np.any(newpoint<0) or np.any(newpoint>=current['image_size']):raise ValueError('Selected target material left the image')
            x,y,w,h=box;corners=np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
            mapped=corners@M[:,:2].T+M[:,2]
            low=np.maximum(0,mapped.min(axis=0));high=np.minimum(current['image_size'],mapped.max(axis=0))
            propagated_box=np.r_[low,high-low]
            center,gap=_aperture(aperture,current['image_size'])
            if abs(center[0]-self.aperture[0])>3 or abs(gap-self.gap)>max(4,.08*self.gap):
                raise ValueError('Jaw aperture changed; reacquire its geometry before lateral continuation')
            actual=_finite(actual_camera_right_mm,'measured lateral displacement')
            allowance=self.max_step if pending is None else pending['distance_mm']+1.
            if not .15<=abs(actual)<=min(4.,allowance):raise ValueError('Measured lateral displacement outside observable local step')
            expected_sign=math.copysign(1.,self.error) if pending is None else pending['sign']
            if actual*expected_sign<=0:raise ValueError('Actual lateral displacement opposed the planned correction')
            self.measured_travel+=abs(actual)
            if pending is None:self.commanded_travel+=abs(actual)
            if self.measured_travel>self.budget+1e-9:raise ValueError('Measured local lateral travel budget exceeded')
            newerror=float(newpoint[0]-center[0]);delta_error=newerror-self.error
            gain=-delta_error/actual
            noise=max(.5,fb,residual)
            if not math.isfinite(gain) or gain<=0:raise ValueError('Measured image response has no correcting lateral sign')
            tolerance=max(self.tolerance,3*noise)
            progress=abs(self.error)-abs(newerror)
            if progress<max(.75,2*fb) and abs(newerror)>tolerance:
                raise ValueError('Lateral step did not positively reduce target-aperture error')
            if self.gain is not None and not .4<=gain/self.gain<=2.5:
                raise ValueError('Observed local image response changed; re-evaluate direction')
            self.gain=gain if self.gain is None else .5*self.gain+.5*gain
            self.noise=noise;self.point=newpoint;self.error=newerror;self.aperture=center;self.gap=gap
            self.frame=current;self.steps+=1;self.pending=None
            self.target_box=propagated_box.tolist()
            if abs(newerror)<=tolerance:
                self.state='aligned';self.reason='Selected material point is horizontally aligned in this image only'
            elif newerror*expected_sign<0:
                self.state='stopped';self.reason='Crossed the target image center; reacquire before reversing'
            else:
                self.state='ready';self.reason='Retained target identity and positive measured lateral progress'
            return self.status()
        except (ValueError,TypeError,KeyError,IndexError) as exc:
            return self.stop(str(exc))
    def propose(self,*,now_monotonic):
        """No hardware operation. Repeated calls cannot enqueue a second step."""
        if self.state!='ready':return {**self.status(),'proposal':None}
        try:_frame(self.frame,now_monotonic,self.frame['camera_generation'])
        except ValueError as exc:return {**self.stop(str(exc)),'proposal':None}
        # Reserve1mm for the caller's actual endpoint tolerance so a final step
        # does not intentionally consume the entire measured-travel allowance.
        remaining=self.budget-max(self.measured_travel,self.commanded_travel)-1.
        if remaining<.1:return {**self.stop('Local lateral travel budget reached'),'proposal':None}
        distance=min(self.max_step,.65*abs(self.error)/self.gain,remaining)
        if distance<.1:return {**self.stop('Image response requests a sub-.1mm step; reacquire before continuing'),'proposal':None}
        sign=math.copysign(1.,self.error)
        proposal={'proposal_id':self.next_id,'direction':'right' if sign>0 else 'left','sign':sign,
                  'distance_mm':float(distance),'source_frame_sequence':self.frame['sequence'],
                  'target_token':self.target_token,'predicted_error_reduction_px':distance*self.gain,
                  'tracking_bbox_px':self.target_box,
                  'requires_fresh_image_and_caller_motion_guards':True,'proposal_only':True}
        self.next_id+=1;self.pending=proposal;self.commanded_travel+=distance
        self.state='awaiting_observation';self.reason='Await the exact proposed step and fresh target observation'
        return {**self.status(),'proposal':dict(proposal)}
