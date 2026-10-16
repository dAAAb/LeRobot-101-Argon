"""Offline-only JSONL validation / empty-jaw baseline fitting. Never imports robot.

Expected file: one metadata row, sample rows, and one end row.
metadata: type=metadata, empty_verified=true, body_reference={5 joints: ticks},
  original_pwm_limit=500, pwm_limit=100, speed_ticks_per_second=50,
  sweep_min=1520, sweep_max=1860, camera_source=wrist, feedback=false.
sample: type=sample, cycle=0..2, direction=opening|closing, phase=cruise|...,
  capture_start_monotonic, capture_end_monotonic, capture_monotonic (midpoint),
  position, goal (observed register42), load_raw, load_signed_pwm,
  current_raw, status, model_number=777, pwm_limit=100,
  speed_ticks_per_second=50, communication_ok=true,
  frame_sequence, frame_capture_monotonic, camera_source=wrist,
  leg_elapsed_seconds, goal_distance_to_endpoint.
end: type=end, complete=true, restored_pwm_limit=500,
  restore_verified=true, outstanding_closing_goal=false.

Cruise samples within20ticks of either sweep endpoint, first0.4s per leg,
>0.10s read spans, or frame age>0.5s are excluded/fail validation. The source
reader must be locked across the entire position/goal/load snapshot. A midpoint
is not a simultaneity guarantee: acquire span is retained and bounded.
"""
import argparse,json,math
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from . import contact as mod
from .gripper_context import measured_body, require_matched_body, body_context
BODY=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll']
def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def sm(x,bit):return -(x&((1<<bit)-1)) if x&(1<<bit) else x

def analyze(rows):
 errors=[];warnings=[];reject=Counter(); metas=[r for r in rows if r.get('type')=='metadata']; ends=[r for r in rows if r.get('type')=='end'];ss=[r for r in rows if r.get('type')=='sample']
 if len(metas)!=1 or len(ends)!=1: return {'fit_allowed':False,'errors':['Exactly one metadata and end row required']}
 m=metas[0];end=ends[0]
 required={'empty_verified':True,'original_pwm_limit':500,'pwm_limit':100,'speed_ticks_per_second':50,'sweep_min':1520,'sweep_max':1860,'camera_source':'wrist','feedback':False}
 for k,v in required.items():
  if m.get(k)!=v:errors.append(f'metadata {k}: expected {v!r}')
 try:reference=measured_body(m.get('body_reference'))
 except ValueError as exc:return {'fit_allowed':False,'errors':[str(exc)]}
 if not(end.get('complete') is True and end.get('restore_verified') is True and end.get('restored_pwm_limit')==500 and end.get('outstanding_closing_goal') is False):errors.append('Acquisition incomplete or PWM restore/quiescence not verified; no usable baseline')
 prev=None;polarity=[];polarity_by_direction=defaultdict(list);usable=[];cycle_directions=defaultdict(set);spans=[];frameages=[];current=[]
 for i,s in enumerate(ss):
  prefix=f'sample{i}: ';bad=[]
  try:require_matched_body(s.get('body_position'),reference)
  except ValueError as exc:bad.append(str(exc))
  fields=['capture_start_monotonic','capture_end_monotonic','capture_monotonic','position','goal','load_raw','load_signed_pwm','current_raw','pwm_limit','speed_ticks_per_second','frame_sequence','frame_capture_monotonic','leg_elapsed_seconds','goal_distance_to_endpoint']
  if any(not finite(s.get(k)) for k in fields):errors.append(prefix+'missing/nonfinite numerical field');continue
  t=s['capture_monotonic'];t0=s['capture_start_monotonic'];t1=s['capture_end_monotonic'];span=t1-t0;age=t-s['frame_capture_monotonic'];spans.append(span);frameages.append(age)
  if not(t0<=t<=t1 and 0<=span<=.10 and abs(t-(t0+t1)/2)<=.002):bad.append('snapshot bracket/span/midpoint')
  if prev and t<=prev['capture_monotonic']:bad.append('non-increasing timestamp')
  if not isinstance(s.get('cycle'),int) or isinstance(s.get('cycle'),bool) or s['cycle'] not in range(3):bad.append('cycle identity')
  if s.get('direction') not in ('opening','closing'):bad.append('direction')
  if s.get('model_number')!=777 or type(s.get('status')) is not int or s.get('status')!=0 or s.get('communication_ok') is not True:bad.append('hardware/transport fault')
  if s.get('camera_source')!='wrist' or not(0<=age<=.50):bad.append('wrist frame unavailable/stale')
  if s['pwm_limit']!=100 or s['speed_ticks_per_second']!=50:bad.append('unexpected cap/speed; do not pool contexts')
  if not(1456<=s['position']<=2983 and 1456<=s['goal']<=2983):bad.append('outside onboard range')
  load=s['load_raw'];cur=s['current_raw']
  if not(float(load).is_integer() and 0<=load<=2047 and abs(sm(int(load),10))<=1000):bad.append('invalid encoded load')
  elif s['load_signed_pwm']!=sm(int(load),10):bad.append('signed load differs from raw sign-magnitude decode')
  if not(float(cur).is_integer() and 0<=cur<=65535):bad.append('invalid current encoding')
  if bad:errors.append(prefix+', '.join(bad));prev=s;continue
  current.append(sm(int(cur),15));cycle_directions[s['cycle']].add(s['direction'])
  if prev and prev.get('cycle')==s['cycle'] and prev.get('direction')==s['direction']:
   dt=t-prev['capture_monotonic'];dp=s['position']-prev['position']
   if 0<dt<=.30 and abs(dp)/dt>=10 and abs(s['load_signed_pwm'])>=3:
    pol=np.sign(dp*s['load_signed_pwm']);polarity.append(pol);polarity_by_direction[s['direction']].append(pol)
  phase=s.get('phase')
  if phase=='cruise' and (s['leg_elapsed_seconds']<.4 or min(s['position']-1520,1860-s['position'])<=20 or s['goal_distance_to_endpoint']<=20):
   reject['trajectory acceleration/endpoint exclusion']+=1;phase='excluded'
  if prev and t-prev['capture_monotonic']>.30:reject['telemetry gap >0.30s']+=1
  usable.append((s,phase));prev=s
 if any(cycle_directions.get(i)!=set(['opening','closing']) for i in range(3)):errors.append('Three separate cycles must each contain opening and closing')
 pp=np.array(polarity);fraction=max(float(np.mean(pp==1)),float(np.mean(pp==-1))) if len(pp) else 0; sign=1 if len(pp) and np.mean(pp==1)>=.5 else -1
 direction_polarity_ok=all(len(polarity_by_direction[d])>=15 and np.mean(np.array(polarity_by_direction[d])==sign)>=.95 for d in ['opening','closing'])
 if len(pp)<30 or fraction<.95 or not direction_polarity_ok:errors.append('PWM polarity not empirically consistent with motion: cannot use directional PWM helper')
 report={'fit_allowed':False,'errors':errors,'warnings':warnings,'raw_sample_count':len(ss),'excluded_reasons':dict(reject),'pwm_polarity':{'valid':len(pp)>=30 and fraction>=.95 and direction_polarity_ok,'counts_by_direction':{d:len(v) for d,v in polarity_by_direction.items()},'moving_evidence_samples':len(pp),'agreement_fraction':fraction,'multiply_signed_raw_by':sign,'meaning':'Separate empirical PWM polarity, never change closing_sign=-1'},'current_evidence':{'unique_values':sorted(set(current)),'varying':len(set(current))>1,'interpretation':'varying raw electrical reading only' if len(set(current))>1 else 'flat or unavailable; not evidence of zero contact/force'},'snapshot_span_p95':float(np.percentile(spans,95)) if spans else None,'frame_age_p95':float(np.percentile(frameages,95)) if frameages else None}
 if errors:return report
 report['body_context']=body_context(reference)
 cfg=mod.Config(position_min=1456,position_max=2983,closing_sign=-1,angle_bin_width=40,endpoint_margin=24)
 cycles=[[] for _ in range(3)]; reason=Counter()
 for s,phase in usable:
  v=mod.Sample(timestamp=s['capture_monotonic'],direction=s['direction'],goal=s['goal'],position=s['position'],load_signed_pwm=s['load_signed_pwm']*sign,pwm_limit=s['pwm_limit'],speed=s['speed_ticks_per_second'],servo_status=s['status'],phase=phase,valid=True,current_raw=s['current_raw'],current_valid=False,body_position=s['body_position'])
  why=mod._reason(v,cfg)
  if why:reason[why]+=1
  cycles[s['cycle']].append(v)
 try:b=mod.fit_empty_baseline(cycles,cfg,body_reference=reference)
 except ValueError as e:report['errors'].append(str(e));return report
 report['helper_excluded_reasons']=dict(reason);report['config']=cfg.__dict__;report['bins']=[{'angle_bin':k[0],'position_interval':[1456+k[0]*40,1456+(k[0]+1)*40],'direction':k[1],'speed_ticks_per_second':k[2],'pwm_limit':k[3],**v} for k,v in sorted(b.bins.items())];report['fit_allowed']=bool(b.bins)
 report['supported_direction_counts']=dict(Counter(k[1] for k in b.bins));report['contact_effort_discrimination_bins']=sum(v['pwm_effort_upper']<k[3] for k,v in b.bins.items());report['cap_saturated_or_unreachable_effort_bins']=sum(v['pwm_effort_upper']>=k[3] for k,v in b.bins.items());
 if report['cap_saturated_or_unreachable_effort_bins']:report['warnings'].append('Some baseline PWM envelopes reach/exceed cap; contact_candidate cannot be supported there. Do not increase cap or lower threshold.')
 report['interpretation']='Only listed bins supported; no interpolation/extrapolation, true grasp, object width or force claim.'
 if not b.bins:report['errors'].append('No supported bins after helper cruise/progress/context requirements')
 return report

def synthetic():
 rows=[dict(type='metadata',empty_verified=True,original_pwm_limit=500,pwm_limit=100,speed_ticks_per_second=50,sweep_min=1520,sweep_max=1860,camera_source='wrist',feedback=False,body_reference={n:2000 for n in BODY})]; t=1000
 for c in range(3):
  for direction,start,sgn,end in [('opening',1520,1,1860),('closing',1860,-1,1520)]:
   for j in range(137):
    pos=start+sgn*min(340,j*2.5);goal=max(1520,min(1860,pos+sgn*4));load=30*sgn;raw=load if load>=0 else (-load)|1024
    rows.append(dict(type='sample',body_position={n:2000 for n in BODY},cycle=c,direction=direction,phase='cruise',capture_start_monotonic=t-.004,capture_end_monotonic=t+.004,capture_monotonic=t,position=pos,goal=goal,load_raw=raw,load_signed_pwm=load,current_raw=0,status=0,model_number=777,pwm_limit=100,speed_ticks_per_second=50,communication_ok=True,frame_sequence=int(t*8),frame_capture_monotonic=t-.05,camera_source='wrist',leg_elapsed_seconds=j*.05,goal_distance_to_endpoint=abs(end-goal)));t+=.05
   t+=.4
 rows.append(dict(type='end',complete=True,restored_pwm_limit=500,restore_verified=True,outstanding_closing_goal=False));return rows

def test():
 import copy
 rows=synthetic();good=analyze(rows);assert good['fit_allowed'] and len(good['bins'])>=10,good
 assert all(b['cycles']==3 and b['samples']>=12 for b in good['bins'])
 bad=copy.deepcopy(rows);bad[50]['load_signed_pwm']=-bad[50]['load_signed_pwm'];assert not analyze(bad)['fit_allowed']
 bad=copy.deepcopy(rows);bad[-1]['restore_verified']=False;assert not analyze(bad)['fit_allowed']
 bad=copy.deepcopy(rows);bad[50]['status']=8;assert not analyze(bad)['fit_allowed']
 bad=copy.deepcopy(rows);bad[50]['capture_end_monotonic']+=1;assert not analyze(bad)['fit_allowed']
 bad=copy.deepcopy(rows)
 for s in bad:
  if s.get('type')=='sample':s['load_signed_pwm']*=-1;s['load_raw']=int(abs(s['load_signed_pwm']))|(1024 if s['load_signed_pwm']<0 else 0)
 inv=analyze(bad);assert inv['fit_allowed'] and inv['pwm_polarity']['multiply_signed_raw_by']==-1
 print(json.dumps({'tests':'6 passed','synthetic_supported_bins':len(good['bins']),'note':'Synthetic data only; no physical baseline collected'}))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('jsonl',nargs='?');a.add_argument('--output',default='.runtime/gripper-empty-baseline.json');a.add_argument('--self-test',action='store_true');args=a.parse_args()
 if args.self_test:test()
 elif args.jsonl:
  report=analyze([json.loads(s) for s in Path(args.jsonl).read_text().splitlines() if s.strip()]);Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='bins'},indent=2));print('baseline bins',len(report.get('bins',[])),'output',args.output)
 else:a.error('JSONL or --self-test required')
