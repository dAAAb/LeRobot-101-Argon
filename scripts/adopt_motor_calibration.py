"""User-authorized read-only adoption of newest onboard calibration. No motor writes."""
from pathlib import Path
import fcntl, json, time
from scservo_sdk import PortHandler, PacketHandler, COMM_SUCCESS
root=Path(__file__).resolve().parents[1]
cfg=json.loads((root/'config/robot.json').read_text())
(root/'.runtime').mkdir(exist_ok=True)
port_lock=(root/'.runtime/follower.lock').open('a+')
try:fcntl.flock(port_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise SystemExit('Follower is already owned by the controller; stop it before reading calibration.')
p=PortHandler(cfg['follower_port']);assert p.openPort();p.setBaudRate(1000000);h=PacketHandler(0)
regs={'min':(9,2),'max':(11,2),'offset':(31,2),'mode':(33,1),'torque':(40,1),'goal':(42,2),'position':(56,2),'voltage':(62,1),'temperature':(63,1),'torque_limit':(48,2),'max_torque':(16,2)}
snapshot={};calibration={}
try:
 for i,n in enumerate(['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper'],1):
  r={}
  for label,(addr,size) in regs.items():
   v,res,e=(h.read2ByteTxRx if size==2 else h.read1ByteTxRx)(p,i,addr)
   assert res==COMM_SUCCESS and not e,(n,label,res,e)
   r[label]=v
  snapshot[n]=r
  assert 0<=r['min']<r['max']<=4095 and r['max']-r['min']>100,(n,r)
  assert r['min']-10<=r['position']<=r['max']+10,(n,r)
  assert r['mode']==0,(n,'not position mode')
  offset=-(r['offset']&0x7ff) if r['offset']&0x800 else r['offset']
  calibration[n]={'id':i,'drive_mode':0,'homing_offset':offset,'range_min':r['min'],'range_max':r['max']}
finally:p.closePort()
(root/'.runtime').mkdir(exist_ok=True)
(root/'.runtime'/f'motor-snapshot-{int(time.time())}.json').write_text(json.dumps(snapshot,indent=2))
(root/'config/follower-calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
print(json.dumps({'calibration':calibration,'positions':{n:r['position'] for n,r in snapshot.items()}},indent=2))
