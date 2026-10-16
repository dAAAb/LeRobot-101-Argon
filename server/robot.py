"""Single-owner STS3215 driver. Explicit units, finite targets, cancellation and hold.
Never writes EEPROM or changes calibration. All motion is bounded and observed.
"""
from __future__ import annotations
import fcntl, json, math, threading, time
from pathlib import Path
try:
    from scservo_sdk import COMM_SUCCESS, COMM_PORT_BUSY, COMM_RX_TIMEOUT, COMM_RX_CORRUPT, PacketHandler, PortHandler
except ImportError:
    COMM_SUCCESS, COMM_PORT_BUSY, COMM_RX_TIMEOUT, COMM_RX_CORRUPT = 0, -1, -6, -7
    class PortHandler:  # type: ignore[no-redef]
        def __init__(self, port): self.port = port
        def openPort(self): return False
        def setBaudRate(self, baud): return False
        def closePort(self): pass
    class PacketHandler:  # type: ignore[no-redef]
        def __init__(self, proto): self.proto = proto
from .gripper_power import SessionCancelled
from .local_feedback import BODY_JOINTS, LocalEndpoint, adopt_goal_anchor, check_bounds, tick
from .feedback import EndpointFeedback
from .stop_latch import PersistentStopEvent

JOINTS=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']
HOME_POSE={'shoulder_pan':2030,'shoulder_lift':1103,'elbow_flex':2915,'wrist_flex':2318,'wrist_roll':998}
READ_ATTEMPTS=3
READ_RETRY_DELAY_S=.004
SETTLED_IMAGE_WAIT_SECONDS=1.5

class MaintenanceRejected(RuntimeError):
    """Motion was never admitted; preserve the existing maintenance hold."""


class MotorReadError(RuntimeError):
    def __init__(self,joint,result,error):
        super().__init__(f'{joint} 馬達讀取失敗 ({result}/{error})')
        self.retryable=not error and result in (COMM_RX_TIMEOUT,COMM_RX_CORRUPT)

class Robot:
    def __init__(self, config, calibration, runtime, camera):
        self.config=config;self.calibration=calibration;self.runtime=Path(runtime);self.camera=camera
        self.io=threading.RLock();self.stop=PersistentStopEvent(self.runtime/'software-stop');self.port=None;self.ph=None;self.error='';self.position={};self.models={};self.connected=False;self.moving=False;self.revision=0;self.lockfile=None;self.armed=False;self.goals={};self.bias={n:0 for n in JOINTS}
        self._sdk_transaction_owner=None
        self.failure_path=self.runtime/'last-motion-failure.json'
        self.maintenance_latched=False;self.maintenance_receipt=None
        if self.stop.is_set():
            try:
                saved=json.loads(self.failure_path.read_text())
                if isinstance(saved.get('error'),str):self.error=saved['error']
            except (OSError,ValueError,AttributeError):pass
        from .gripper_power import GripperPWMSession,AtomicJournal,RecoveryRequired
        if calibration['gripper']['id']!=GripperPWMSession.SERVO_ID:
            raise ValueError('PWM recovery is verified only for gripper motor ID 6')
        self.pwm_owner=None
        self.gripper_power=GripperPWMSession(
            read=lambda servo,address,width:self._read('gripper',address,width),
            write=lambda servo,address,value,width:self._write('gripper',address,value,width),
            lock=self.io,stop_latch=self.stop.set,
            journal=AtomicJournal(self.runtime/'gripper-pwm-session.json'),
            device_key=self.config['follower_port']+':6',
            position_bounds=(calibration['gripper']['range_min'],calibration['gripper']['range_max']),
            position_tolerance=2,verified_open_position=1860,opening_direction=1,
            release_tolerance=int(self.config.get('gripper_release_tolerance_ticks',2)),low_cap=int(self.config.get('gripper_low_pwm_cap',100)))
        try:self.gripper_power.assert_motion_allowed()
        except RecoveryRequired as e:self.error=str(e)
    def connect(self):
        with self.io:
            if self.connected:return
            self.lockfile=open(self.runtime/'follower.lock','w')
            try: fcntl.flock(self.lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('另一個 Astra 控制器正在使用這隻手臂')
            # Advisory exclusion complements the application owner lock.
            self.port=PortHandler(self.config['follower_port'])
            if not self.port.openPort(): raise RuntimeError('Follower USB 連線失敗')
            self.port.setBaudRate(1_000_000)
            self.port.ser.exclusive=True
            self.ph=PacketHandler(0)
            try:
                current=self.read()
                for n in JOINTS:
                    self.models[n]=self._read(n,3)
                    if self.models[n]!=777:raise RuntimeError(f'{n} 馬達型號未驗證：{self.models[n]}')
                    raw=self._read(n,31,2); offset=-(raw&0x7ff) if raw&0x800 else raw
                    if self._read(n,33,1)!=0:raise RuntimeError(f'{n} 不是位置控制模式')
                    if self._read(n,9)!=self.calibration[n]['range_min'] or self._read(n,11)!=self.calibration[n]['range_max']:raise RuntimeError(f'{n} 行程校準已改變，請重新讀取')
                    if offset!=self.calibration[n]['homing_offset']:
                        raise RuntimeError(f'{n} 機內校準與目前快照不同；請重新讀取機內校準並核對座標設定')
                self.validate(current,margin=0)
                self.connected=True
                if not self.stop.is_set():self.error=''
            except Exception:
                self.port.closePort();self.port=None;raise
    def _sdk_transaction(self,fn,*args,busy_result):
        """Serialize one SDK call and release only its own host-side busy flag.

        scservo_sdk sets is_using before serial I/O but omits finally cleanup
        when serial flush/write/read raises. That would poison later calls with
        COMM_PORT_BUSY. Do not clear a flag already owned on entry, or enter a
        nested SDK transaction even though the application's lock is reentrant.
        This never retries a packet, resets a servo, or clears a stop/fault.
        """
        with self.io:
            port=self.port
            if (getattr(self,'_sdk_transaction_owner',None) is not None
                    or getattr(port,'is_using',False)):
                return busy_result
            owner=object();self._sdk_transaction_owner=owner
            try:
                return fn(port,*args)
            except Exception:
                self._read_failed();raise
            finally:
                # Capture the exact port, not a potentially replaced self.port.
                # No other Robot transaction can own it while self.io is held.
                if self._sdk_transaction_owner is owner:
                    try:port.is_using=False
                    finally:self._sdk_transaction_owner=None
    def _read_once(self,n,address,size=2):
        """One locked SDK READ; enclosing batches may retain the same RLock."""
        fn=self.ph.read2ByteTxRx if size==2 else self.ph.read1ByteTxRx
        value,result,error=self._sdk_transaction(fn,self.calibration[n]['id'],address,
                                                busy_result=(0,COMM_PORT_BUSY,0))
        if result!=COMM_SUCCESS or error:raise MotorReadError(n,result,error)
        return -(value&0x7fff) if address==56 and value&0x8000 else value
    def _read_failed(self):
        if hasattr(self,'gripper_power') and self.gripper_power.state in ('active','holding','releasing'):
            self.gripper_power.leave_pending()
    def _read(self,n,address,size=2):
        # SDK clearPort() only flushes TX. Explicitly discard queued RX before
        # retrying the same READ; never retry servo errors or any WRITE.
        with self.io:
            for attempt in range(READ_ATTEMPTS):
                try:return self._read_once(n,address,size)
                except MotorReadError as exc:
                    if not exc.retryable or attempt==READ_ATTEMPTS-1:
                        self._read_failed();raise
                    time.sleep(READ_RETRY_DELAY_S)
                    try:self.port.ser.reset_input_buffer()
                    except Exception:
                        self._read_failed();raise
    def _write(self,n,address,value,size=2):
        # A position update can clear a servo's overcurrent/overload protection.
        # This guard also applies to hold() and torque enable; leave a faulted
        # servo's firmware protection intact instead of rearming it implicitly.
        with self.io:
            if address==42 or (address==40 and value):
                fault=self._read(n,65,1)
                if fault:raise RuntimeError(f'{n} 馬達保護狀態 {fault:#04x}，禁止發送新位置或啟用扭力')
            fn=self.ph.write2ByteTxRx if size==2 else self.ph.write1ByteTxRx
            result,error=self._sdk_transaction(fn,self.calibration[n]['id'],address,int(value),
                                               busy_result=(COMM_PORT_BUSY,0))
            if result!=COMM_SUCCESS or error:
                if hasattr(self,'gripper_power') and self.gripper_power.state in ('active','holding','releasing'):self.gripper_power.leave_pending()
                raise RuntimeError(f'{n} 馬達寫入失敗 ({result}/{error})')
    def read(self):
        with self.io:
            self.position={n:self._read(n,56) for n in JOINTS}
            return dict(self.position)
    def pose(self):
        self.connect()
        return self.read()
    def _stream_segment_to(self,target_body,*,speed_ticks_per_s=240.0,max_tracking_error=220):
        """Stream 25 Hz interpolated goals to target_body without intermediate settling delay."""
        body_names=[n for n in ('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll') if n in target_body]
        if not body_names:return
        self.validate(target_body,margin=12)
        with self.io:
            start_goals={n:int(self.goals.get(n,self.position.get(n,target_body[n]))) for n in body_names}
        max_delta=max((abs(int(target_body[n])-start_goals[n]) for n in body_names),default=0)
        if max_delta<=3:return
        duration=max(0.20,max_delta/float(speed_ticks_per_s))
        t0=time.monotonic();step=0
        while True:
            with self.io:
                if self.stop.is_set():raise SessionCancelled('已停止')
                if not self.camera.fresh():raise RuntimeError('腕部畫面中斷，已停止移動')
                u=min(1.0,(time.monotonic()-t0)/duration)
                # Smooth cosine ease-in-out
                s=0.5*(1.0-math.cos(math.pi*u))
                for n in body_names:
                    v=int(round(start_goals[n]+(int(target_body[n])-start_goals[n])*s))
                    if abs(v-int(self.goals.get(n,v)))>=2 or (u>=1.0 and v!=int(self.goals.get(n,v))):
                        self._write(n,42,v);self.goals[n]=v
                if step%6==0:
                    actual=self.read()
                    for n in body_names:
                        if abs(actual[n]-self.goals[n])>max_tracking_error:
                            raise RuntimeError(f'{n} 跟隨誤差過大，停止並保持姿勢')
                        if self._read(n,63,1)>60:raise RuntimeError(f'{n} 溫度過高，停止移動')
                step+=1
            if u>=1.0:break
            if self.stop.wait(0.04):raise SessionCancelled('動作已停止')
    def go_home(self,report=None,*,speed_ticks_per_s=240.0,precision=38):
        """Return the arm to its canonical forward-facing HOME_POSE (shoulder_pan=2030)."""
        self._require_not_maintenance()
        self.moving=True
        try:
            self.begin()
            start=self.read()
            if abs(start['shoulder_pan']-HOME_POSE['shoulder_pan'])>110 and start['elbow_flex']>2680:
                if report:report('先抬起前臂確保桌面淨空，再轉回正前方…')
                lift_pose=dict(start)
                lift_pose['elbow_flex']=min(start['elbow_flex'],2640)
                lift_pose['wrist_flex']=min(start['wrist_flex'],2140)
                self._stream_segment_to({k:lift_pose[k] for k in ('elbow_flex','wrist_flex')},speed_ticks_per_s=speed_ticks_per_s)
            if report:report('正在歸回正前方預設原位 (Home)…')
            self._stream_segment_to(HOME_POSE,speed_ticks_per_s=speed_ticks_per_s)
            deadline=time.monotonic()+0.65
            while time.monotonic()<deadline:
                if self.stop.wait(0.05):raise SessionCancelled('動作已停止')
                actual=self.read()
                if all(abs(actual[n]-HOME_POSE[n])<=precision for n in HOME_POSE):break
            self.revision+=1
            end=self.read()
            return {'completed':True,'home_pose':dict(HOME_POSE),'measured_start':start,'measured_end':end}
        except MaintenanceRejected:
            raise
        except Exception as e:
            self.error=str(e);self.stop.set();self.hold();raise
        finally:
            self.moving=False
    def stream_trajectory(self,frames_body,duration_s,*,progress=None,title_zh='自然動作',end_pose=None):
        """Stream a 20–25 Hz body trajectory in real time matching the 3D simulation speed."""
        self._require_not_maintenance()
        if not frames_body:raise ValueError('空的動作軌跡')
        target_end=dict(end_pose or HOME_POSE)
        body_names=('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll')
        for f in frames_body:self.validate({k:int(f[k]) for k in body_names if k in f},margin=12)
        self.validate(target_end,margin=12)
        self.moving=True
        try:
            self.begin()
            measured_start=self.read()
            # Ensure we start from canonical forward-facing direction if the arm was left turned sideways
            if abs(measured_start['shoulder_pan']-HOME_POSE['shoulder_pan'])>90:
                if progress:progress('moving','先將手臂對齊正前方原位…')
                if measured_start['elbow_flex']>2680:
                    self._stream_segment_to({'elbow_flex':2640,'wrist_flex':2140},speed_ticks_per_s=260.0)
                self._stream_segment_to(HOME_POSE,speed_ticks_per_s=260.0)
            first_frame={k:int(frames_body[0][k]) for k in body_names}
            self._stream_segment_to(first_frame,speed_ticks_per_s=260.0)
            peak_pan=[measured_start['shoulder_pan'],measured_start['shoulder_pan']]
            peak_roll=[measured_start['wrist_roll'],measured_start['wrist_roll']]
            peak_lift=[measured_start['elbow_flex'],measured_start['elbow_flex']]
            total_dur=max(0.8,float(duration_s))
            n_frames=len(frames_body)
            t0=time.monotonic();step=0;max_tracking_error=250
            while True:
                with self.io:
                    if self.stop.is_set():raise SessionCancelled('已停止')
                    if not self.camera.fresh():raise RuntimeError('腕部畫面中斷，已停止移動')
                    u=min(1.0,(time.monotonic()-t0)/total_dur)
                    f_pos=u*max(0,n_frames-1)
                    i0=int(math.floor(f_pos));i1=min(n_frames-1,i0+1);alpha=f_pos-i0
                    for n in body_names:
                        v0=int(frames_body[i0][n]);v1=int(frames_body[i1][n])
                        desired=int(round(v0+(v1-v0)*alpha))
                        prev_goal=int(self.goals.get(n,desired))
                        # Rate-limit per 40ms step to 22 ticks (~550 ticks/s) for smooth servo tracking
                        delta=max(-22,min(22,desired-prev_goal))
                        v=prev_goal+delta
                        if abs(v-prev_goal)>=2 or (u>=1.0 and v!=prev_goal):
                            self._write(n,42,v);self.goals[n]=v
                    peak_pan[0]=min(peak_pan[0],self.goals['shoulder_pan'])
                    peak_pan[1]=max(peak_pan[1],self.goals['shoulder_pan'])
                    peak_roll[0]=min(peak_roll[0],self.goals['wrist_roll'])
                    peak_roll[1]=max(peak_roll[1],self.goals['wrist_roll'])
                    peak_lift[0]=min(peak_lift[0],self.goals['elbow_flex'])
                    peak_lift[1]=max(peak_lift[1],self.goals['elbow_flex'])
                    if step%6==0:
                        actual=self.read()
                        peak_pan[0]=min(peak_pan[0],actual['shoulder_pan'])
                        peak_pan[1]=max(peak_pan[1],actual['shoulder_pan'])
                        peak_roll[0]=min(peak_roll[0],actual['wrist_roll'])
                        peak_roll[1]=max(peak_roll[1],actual['wrist_roll'])
                        peak_lift[0]=min(peak_lift[0],actual['elbow_flex'])
                        peak_lift[1]=max(peak_lift[1],actual['elbow_flex'])
                        for n in body_names:
                            if abs(actual[n]-self.goals[n])>max_tracking_error:
                                raise RuntimeError(f'{n} 跟隨誤差過大，停止並保持姿勢')
                            if self._read(n,63,1)>60:raise RuntimeError(f'{n} 溫度過高，停止移動')
                        if progress:
                            pct=int(round(u*100.0))
                            progress('moving',f'正在執行單手重映射動作：{title_zh} ({pct}%)…')
                    step+=1
                if u>=1.0:break
                if self.stop.wait(0.04):raise SessionCancelled('動作已停止')
            if progress:progress('verifying','動作完成，正在平穩歸回正前方原位…')
            self._stream_segment_to(target_end,speed_ticks_per_s=240.0)
            deadline=time.monotonic()+0.55
            while time.monotonic()<deadline:
                if self.stop.wait(0.05):raise SessionCancelled('動作已停止')
                actual=self.read()
                if all(abs(actual[n]-target_end[n])<=38 for n in body_names):break
            self.revision+=1
            measured_end=self.read()
            return {
                'measured_start':measured_start,
                'measured_end':measured_end,
                'peak_pan_ticks':peak_pan,
                'peak_roll_ticks':peak_roll,
                'peak_lift_ticks':peak_lift,
            }
        except MaintenanceRejected:
            raise
        except Exception as e:
            self.error=str(e);self.stop.set();self.hold();raise
        finally:
            self.moving=False
    def gripper_telemetry(self):
        self.connect()
        with self.io:
            # Preserve fail-fast contact telemetry: a failure at any register
            # discards the entire sample, with no retries across register IDs.
            try:return self._gripper_telemetry_once()
            except MotorReadError:
                self._read_failed();raise
    def _gripper_telemetry_once(self):
        read=self._read_once;read_start=time.monotonic();n='gripper';fault=read(n,65,1)
        if fault:
            self._read_failed();raise RuntimeError(f'夾爪馬達保護狀態 {fault:#04x}')
        load=read(n,60);current=read(n,69)
        signed_load=-(load&0x3ff) if load&0x400 else load
        signed_current=-(current&0x7fff) if current&0x8000 else current
        sample={'model_number':self.models[n],
                'position':read(n,56),'goal':read(n,42),'status':fault,
                'load_raw':load,'load_signed_pwm':signed_load,'drive_duty_percent':signed_load/10,
                'current_raw':current,'current_milliamps':signed_current*6.5,
                'current_validity':'transport-valid; not calibrated as contact or force',
                'pwm_limit':read(n,48),'voltage':read(n,62,1)/10,
                'temperature_c':read(n,63,1),'moving':bool(read(n,66,1)),
                'body_position':{j:read(j,56) for j in JOINTS if j!=n}}
        fault=read(n,65,1)
        if fault:
            self._read_failed();raise RuntimeError(f'夾爪馬達保護狀態 {fault:#04x}')
        read_end=time.monotonic()
        sample.update(read_start_monotonic=read_start,read_end_monotonic=read_end,
                      capture_monotonic=(read_start+read_end)/2)
        return sample
    def joint_diagnostics(self):
        """Read onboard settings while idle; never move, rearm, or clear a fault.

        Keep serial ownership in this controller. Do not queue a slow diagnostic
        read behind a trajectory or hold up its safety observations.
        """
        if not self.io.acquire(blocking=False):
            raise RuntimeError('控制器忙碌中，請在動作停止後讀取診斷')
        try:
            if self.moving:
                raise RuntimeError('控制器忙碌中，請在動作停止後讀取診斷')
            previous_error=self.error
            try:self.connect()
            finally:self.error=previous_error
            started=time.monotonic();samples={}
            registers={'position':(56,2),'goal':(42,2),'torque_enabled':(40,1),
                'status':(65,1),'moving':(66,1),'load_raw':(60,2),'current_raw':(69,2),
                'voltage_raw':(62,1),'temperature_c':(63,1),'torque_limit':(48,2),
                'p_coefficient':(21,1),'d_coefficient':(22,1),'i_coefficient':(23,1),
                'minimum_startup_force':(24,2),'cw_dead_zone':(26,1),'ccw_dead_zone':(27,1),
                'acceleration':(41,1),'goal_time':(44,2),'goal_velocity':(46,2)}
            for n in JOINTS:
                read_started=time.monotonic()
                s={key:self._read(n,address,width) for key,(address,width) in registers.items()}
                load=s['load_raw'];current=s['current_raw']
                s.update(goal_error_ticks=s['goal']-s['position'],
                    signed_pwm=-(load&0x3ff) if load&0x400 else load,
                    current_milliamps=(-(current&0x7fff) if current&0x8000 else current)*6.5,
                    voltage=s['voltage_raw']/10,
                    read_start_monotonic=read_started,read_end_monotonic=time.monotonic())
                samples[n]=s
            return {'read_only':True,'stopped':self.stop.is_set(),
                'read_start_monotonic':started,'read_end_monotonic':time.monotonic(),
                'joints':samples,'interpretation':'Register snapshot only; not a clearance, contact, or force measurement'}
        finally:
            self.io.release()
    def read_onboard_calibration(self):
        """Report actual motor registers without adopting or writing calibration.

        The startup file is a comparison snapshot. This read never replaces
        it, refreshes kinematics, moves a joint, or clears a stopped fault.
        """
        if not self.io.acquire(blocking=False):
            raise RuntimeError('控制器忙碌中，請在動作停止後讀取馬達校準')
        try:
            if (self.moving or self.pwm_owner is not None
                    or self.gripper_power.state in ('active','holding','releasing')
                    or self._sdk_transaction_owner is not None
                    or getattr(self.port,'is_using',False)):
                raise RuntimeError('控制器忙碌中，請在動作停止後讀取馬達校準')
            previous_error=self.error
            try:self.connect()
            finally:self.error=previous_error
            started=time.monotonic();samples={}
            for n in JOINTS:
                read_started=time.monotonic()
                raw_offset=self._read(n,31,2)
                sample={'id':self.calibration[n]['id'],
                    'model_number':self._read(n,3,2),
                    'drive_mode':self._read(n,33,1),
                    'homing_offset':-(raw_offset&0x7ff) if raw_offset&0x800 else raw_offset,
                    'range_min':self._read(n,9,2),'range_max':self._read(n,11,2)}
                expected={key:self.calibration[n][key] for key in
                    ('id','drive_mode','homing_offset','range_min','range_max')}
                expected['model_number']=self.models.get(n,777)
                mismatches=[key for key,value in expected.items() if sample[key]!=value]
                sample.update(matches_startup_snapshot=not mismatches,mismatched_fields=mismatches,
                    read_start_monotonic=read_started,read_end_monotonic=time.monotonic())
                samples[n]=sample
            return {'read_only':True,'source':'motor_registers','stopped':self.stop.is_set(),
                'matches_startup_snapshot':all(s['matches_startup_snapshot'] for s in samples.values()),
                'read_start_monotonic':started,'read_end_monotonic':time.monotonic(),
                'joints':samples}
        finally:
            self.io.release()
    def validate(self,targets,margin=12):
        if not targets or set(targets)-set(JOINTS):raise ValueError('未知的關節')
        for n,v in targets.items():
            c=self.calibration[n]
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v):raise ValueError('關節目標必須是有限數值')
            if not c['range_min']+margin<=v<=c['range_max']-margin:raise ValueError(f'{n} 超出校準行程')
    def adopt_stationary_hold(self):
        """Adopt six stable measured samples read-only, never rearm or resend goals.

        Moving66 may stay set on a loaded position servo. Its validated raw value
        is diagnostic; the whole encoder window establishes stationarity.
        """
        from .feedback import EndpointFeedback
        self.gripper_power.assert_motion_allowed();self.connect()
        if self.stop.is_set() or not self.camera.fresh():raise RuntimeError('目前無法讀取穩定保持姿勢')
        with self.io:
            samples=[];previous_physical=None
            for index in range(6):
                if self.stop.is_set():raise SessionCancelled('已停止')
                if not self.camera.fresh():raise RuntimeError('目前無法讀取穩定保持姿勢')
                t0=time.monotonic();state=self.read();goals={n:self._read(n,42) for n in JOINTS}
                self.validate(state,margin=0);self.validate(goals,margin=0)
                health={n:{'torque':self._read(n,40,1),'status':self._read(n,65,1),
                           'moving':self._read(n,66,1)} for n in JOINTS}
                if any(v['torque']!=1 or v['status']!=0 or v['moving'] not in (0,1) for v in health.values()):
                    raise RuntimeError('小幅觀察需要所有關節已正常保持，不自動啟用扭力')
                samples.append({'read_start_monotonic':t0,'read_end_monotonic':time.monotonic(),
                                'actual':dict(state),'physical_goals':dict(goals),'servo_health':health})
                if (any(abs(state[n]-goals[n])>(EndpointFeedback.MAX_BIAS if n in BODY_JOINTS else 16) for n in JOINTS) or
                        any(max(s['actual'][n] for s in samples)-min(s['actual'][n] for s in samples)>2 for n in JOINTS)):
                    raise RuntimeError('既有保持姿勢不穩定，不能開始小幅觀察')
                if previous_physical is not None and goals!=previous_physical:
                    raise RuntimeError('既有保持目標仍在改變，不能以靜止姿勢接續')
                previous_physical=goals
                if index<5 and self.stop.wait(.12):raise SessionCancelled('已停止')
            duration=samples[-1]['read_end_monotonic']-samples[0]['read_end_monotonic']
            if duration<.5:raise RuntimeError('既有保持姿勢的量測時間不足')
            anchor=adopt_goal_anchor(state,goals,
                previous_goals=self.goals if self.armed else None,
                previous_bias=self.bias if self.armed else None)
            anchor.update(stationary_samples=samples,stationary_duration_s=duration)
            self.goals=anchor['physical_goals'];self.bias=anchor['bias'];self.armed=True
            return anchor
    def begin(self,*,allow_rearm=True):
        self._require_not_maintenance()
        self.gripper_power.assert_motion_allowed(self.pwm_owner)
        self.connect()
        with self.io:
            self._require_not_maintenance()
            if self.stop.is_set():raise SessionCancelled('已停止；請按恢復後再試')
            if not self.camera.fresh():raise RuntimeError('沒有即時腕部畫面，無法移動')
            state=self.read();self.validate(state,margin=0)
            torque_held=all(self._read(n,40,1)==1 for n in JOINTS)
            if self.armed and not torque_held:self.armed=False
            if not self.armed and torque_held:
                # A restarted process adopts the existing stable hold read-only.
                # Physical goal minus measured anchor becomes the retained bias.
                self.adopt_stationary_hold()
            if not self.armed:
                if not allow_rearm:raise RuntimeError('小幅觀察禁止重新啟用扭力或重設保持目標')
                # Seed only on torque enable/rearm. Reseeding partial moves would
                # convert load-related tracking error into cumulative sag.
                for n,v in state.items():
                    if self.stop.is_set():raise SessionCancelled('已停止')
                    self._write(n,42,v)
                for n in JOINTS:
                    if self.stop.is_set():raise SessionCancelled('已停止')
                    self._write(n,40,1,1)
                self.goals=dict(state);self.bias={n:0 for n in JOINTS};self.armed=True

    def glide(self,targets,speed=None,return_origin=False,precision=16,feedback=False,observe=None,guard=None,allow_rearm=True,raw_bounds=None,feedback_observe=None,measured_bounds=None,endpoint_accept=None,endpoint_stability_ticks=None,measured_guard=None,deferred_joints=None,stall_response_wait=False,phase_governor=None,inward_limit_contract=None,settle_timeout_s=.7,settle_telemetry=False,allow_stable_partial_observation=False):
        if (isinstance(settle_timeout_s,bool) or not isinstance(settle_timeout_s,(int,float))
                or not math.isfinite(settle_timeout_s) or not 0<settle_timeout_s<=3.):
            raise ValueError('Endpoint settling timeout must be finite and within 0–3 seconds')
        if type(settle_telemetry) is not bool:raise ValueError('Endpoint settling telemetry must be boolean')
        if phase_governor is not None and (not callable(phase_governor) or feedback or deferred_joints
                or guard is None or measured_guard is None or raw_bounds is None or measured_bounds is None):
            raise ValueError('Phase governor requires a complete screened open-loop segment contract')
        if type(allow_stable_partial_observation) is not bool:
            raise ValueError('Partial survey observation must be boolean')
        if allow_stable_partial_observation:
            from .motion_program import (_validate_contract, SegmentPhaseGovernor,
                                         WORKSPACE_ORIGIN_PROFILE, _partial_observation_progress)
            if (not isinstance(phase_governor, SegmentPhaseGovernor) or feedback or allow_rearm
                    or 'gripper' in targets or self.pwm_owner is not None):
                raise ValueError('Partial observation requires an engineering workspace segment')
            survey = phase_governor.segment
            _validate_contract(survey)
            if (survey.get('origin_limit_profile') != WORKSPACE_ORIGIN_PROFILE
                    or targets != {n: survey['joint_targets'][n] for n in survey['active_joints']}
                    or raw_bounds != survey['raw_bounds'] or measured_bounds != survey['measured_bounds']):
                raise ValueError('Partial observation requires the exact engineering workspace segment')
        inward={}
        if inward_limit_contract is not None:
            from .motion_program import _validate_contract,SegmentPhaseGovernor
            _validate_contract(inward_limit_contract)
            if (return_origin or feedback or allow_rearm or deferred_joints or stall_response_wait
                    or not isinstance(phase_governor,SegmentPhaseGovernor)
                    or phase_governor.segment is not inward_limit_contract
                    or targets!={n:inward_limit_contract['joint_targets'][n] for n in inward_limit_contract['active_joints']}
                    or raw_bounds!=inward_limit_contract['raw_bounds']
                    or measured_bounds!=inward_limit_contract['measured_bounds']):
                raise ValueError('Inward departure requires the exact sealed local segment')
            inward=inward_limit_contract.get('inward_limit_departures',{})
            if (not inward or set(inward)!=set(inward_limit_contract.get('inward_limit_joints',()))
                    or not set(inward)<=set(targets)&BODY_JOINTS):
                raise ValueError('Invalid inward departure axes')
        if type(stall_response_wait) is not bool or (stall_response_wait and
                (not feedback or set(targets)!={'elbow_flex'} or allow_rearm or self.pwm_owner is not None)):
            raise ValueError('Stalled-response observation requires an empty-hand elbow feedback move without rearming')
        if measured_guard is not None and not callable(measured_guard):
            raise ValueError('measured_guard must be callable')
        if deferred_joints is not None and (not feedback or
                not isinstance(deferred_joints,(tuple,list,set,frozenset)) or
                any(not isinstance(n,str) for n in deferred_joints) or
                not set(deferred_joints)<=set(targets)&BODY_JOINTS):
            raise ValueError('Deferred joints require body feedback and must be commanded body joints')
        deferred=frozenset(deferred_joints or ())
        if endpoint_stability_ticks is not None and (not feedback or endpoint_accept is None or
                type(endpoint_stability_ticks) is not int or not 0<=endpoint_stability_ticks<=2):
            raise ValueError('Endpoint stability requires feedback, a predicate, and 0..2 ticks')
        if 'gripper' in targets and (self.gripper_power._record() or {}).get('contact_possible'):
            raise RuntimeError('持有物件時夾爪只能透過明確放置／釋放流程操作')
        self.validate(targets,margin=0 if return_origin else 12)
        if feedback and (not raw_bounds or not set(targets)<=BODY_JOINTS):
            raise ValueError('Body feedback requires explicit screened raw_bounds; gripper is excluded')
        if raw_bounds is not None:check_bounds(targets,raw_bounds,targets)
        if measured_bounds is not None:check_bounds(targets,measured_bounds,targets)
        speed=self.config['max_ticks_per_second'] if speed is None else speed
        if isinstance(speed,bool) or not isinstance(speed,(int,float)) or not math.isfinite(speed) or not 0<speed<=self.config['max_ticks_per_second']:raise ValueError('移動速度超出限制')
        self.moving=True
        try:
            self.begin(allow_rearm=allow_rearm)
            start=self.read();physical_start={n:self.goals[n] for n in targets}
            if inward:
                if dict(self.goals)!=inward_limit_contract['held_goals']:
                    raise RuntimeError('向內退開的保持目標已改變，請重新規劃')
                self.validate(start,margin=0);self.validate(physical_start,margin=0)
            if measured_bounds is not None:check_bounds(start,measured_bounds)
            if measured_guard:measured_guard(start)
            if guard:guard(start)
            physical_target={n:round(v+self.bias[n]) for n,v in targets.items()}
            self.validate(physical_target,margin=0 if return_origin else 12)
            for n,departure in inward.items():
                direction=departure['direction'];lo,hi=departure['hard_bounds']
                c=self.calibration[n]
                if (direction not in (-1,1) or not c['range_min']<=lo<=hi<=c['range_max']
                        or not lo<=start[n]<=hi or not lo<=physical_start[n]<=hi
                        or direction*(physical_target[n]-physical_start[n])<=0
                        or physical_target[n]!=inward_limit_contract['physical_goal_targets'][n]):
                    raise ValueError('向內退開的起點、方向或保持補償已改變')
            if raw_bounds is not None:
                check_bounds(start,raw_bounds);check_bounds(physical_start,raw_bounds,targets)
                check_bounds({**start,**physical_target},raw_bounds)
            if guard:guard({**start,**physical_target})
            # Deferred axes keep their acknowledged physical hold, with no goal
            # resends. Their logical endpoint remains part of worst-axis feedback.
            deferred_bias={n:tick(physical_start[n])-tick(targets[n]) for n in deferred}
            if any(abs(v)>EndpointFeedback.MAX_BIAS for v in deferred_bias.values()):
                raise ValueError('Deferred endpoint exceeds the 60-tick compensation budget')
            trajectory_target={n:v for n,v in physical_target.items() if n not in deferred}
            duration=max(.3,max((abs(v-physical_start[n]) for n,v in trajectory_target.items()),default=0)/speed)
            t0=time.monotonic();step=0;commanded_fraction=0.;phase_clock=t0
            last_goal_time={n:t0 for n in trajectory_target}
            while True:
                with self.io:
                    if self.stop.is_set():raise SessionCancelled('動作已停止')
                    if not self.camera.fresh():raise RuntimeError('腕部畫面中斷，已停止移動')
                    if guard or measured_guard:
                        actual=self.read()
                        if measured_guard:measured_guard(actual)
                        if guard:guard(actual)
                    phase_now=time.monotonic()
                    nominal=min(1.,(phase_now-t0)/duration)
                    requested=nominal
                    t=requested;waiting_for_path=False
                    if phase_governor is not None:
                        # Discard time spent waiting on an earlier cycle;
                        # releasing a wait must not catch up in one jump.
                        requested=min(nominal,commanded_fraction+max(0.,phase_now-phase_clock)/duration)
                        phase_clock=phase_now
                        # Serial writes occur at different instants. Bound
                        # each axis from its own last acknowledged goal too;
                        # one late write must not borrow another axis's time.
                        for n,end in trajectory_target.items():
                            delta=end-physical_start[n]
                            if delta:
                                current=(self.goals[n]-physical_start[n])/delta
                                elapsed=max(0.,phase_now-last_goal_time[n])
                                requested=min(requested,current+elapsed/duration+.499999/abs(delta))
                        t=requested
                        governed=phase_governor(actual,requested,commanded_fraction)
                        if governed is None:
                            t=commanded_fraction;waiting_for_path=True
                        elif (isinstance(governed,bool) or not isinstance(governed,(int,float))
                                or not math.isfinite(governed)
                                or not commanded_fraction-1e-12<=governed<=requested+1e-12):
                            raise ValueError('Invalid governed path fraction')
                        else:
                            t=governed
                            if t<=commanded_fraction+1e-12 and nominal>t+1e-12:
                                waiting_for_path=True
                    # Linear velocity bounded by configured ticks/s, 25 Hz, no long open-loop motion.
                    desired={} if waiting_for_path else {n:round(physical_start[n]+(v-physical_start[n])*t) for n,v in trajectory_target.items()}
                    for n,v in desired.items():
                        if self.stop.is_set():raise SessionCancelled('已停止')
                        if phase_governor is not None and self.goals[n]==v:continue
                        write_margin=0 if return_origin else 12
                        if n in inward:
                            departure=inward[n];direction=departure['direction']
                            lo,hi=departure['hard_bounds']
                            if (not lo<=v<=hi or direction*(v-physical_start[n])<0
                                    or direction*(v-self.goals[n])<0):
                                raise RuntimeError('退開端點時不允許往外送出馬達目標')
                            # The sealed endpoint retains the usual reserve;
                            # only intermediate one-way goals use hard limits.
                            write_margin=0
                        if raw_bounds is not None or measured_bounds is not None or guard or measured_guard:
                            try:self._screen_local_goal({n:v},raw_bounds,guard,write_margin,measured_bounds,measured_guard)
                            except RuntimeError as exc:
                                from .motion_program import MotionPathViolation
                                # A fresh read already passed every measured
                                # guard. An unsent leading goal can wait; an
                                # actual deviation or any other fault cannot.
                                if (phase_governor is not None and isinstance(exc,MotionPathViolation)
                                        and getattr(exc,'guard_context',{}).get('checked_pose_kind')=='prospective_goal'):
                                    waiting_for_path=True;break
                                raise
                        self._write(n,42,v);self.goals[n]=v
                        if phase_governor is not None:
                            commanded_fraction=max(commanded_fraction,t)
                            last_goal_time[n]=time.monotonic()
                    if step%3==0:
                        actual=self.read()
                        if measured_bounds is not None:check_bounds(actual,measured_bounds)
                        if measured_guard:measured_guard(actual)
                        for n,v in self.goals.items():
                            if abs(actual[n]-v)>self.config['max_tracking_error_ticks']:raise RuntimeError(f'{n} 跟隨誤差過大，停止並保持姿勢')
                            if self._read(n,63,1)>60:raise RuntimeError(f'{n} 溫度過高，停止移動')
                    if observe is not None:
                        observe({'elapsed':time.monotonic()-t0,'duration':duration,
                                 'fraction':t,'speed':speed,'goals':dict(desired),
                                 **({'nominal_fraction':nominal,'speed_limited_fraction':requested,
                                     'commanded_fraction':commanded_fraction,
                                     'waiting_for_path':waiting_for_path} if phase_governor is not None else {})})
                    step+=1
                if t>=1 and not waiting_for_path:break
                if self.stop.wait(.04):raise SessionCancelled('動作已停止')
            if feedback:
                if deferred:
                    with self.io:
                        self._screen_local_goal({},raw_bounds,guard,0 if return_origin else 12,measured_bounds,measured_guard)
                        rebased={n:tick(self.goals[n])-tick(targets[n]) for n in deferred}
                        if any(abs(v)>EndpointFeedback.MAX_BIAS for v in rebased.values()):
                            raise ValueError('Deferred endpoint exceeds the 60-tick compensation budget')
                        # RAM ledger only: target+bias must describe the motor
                        # goal already in force, never a fabricated write.
                        self.bias.update(rebased)
                actual=self.finish_feedback(targets,return_origin,raw_bounds=raw_bounds,guard=guard,observe=feedback_observe,measured_bounds=measured_bounds,endpoint_accept=endpoint_accept,endpoint_stability_ticks=endpoint_stability_ticks,measured_guard=measured_guard,stall_response_wait=stall_response_wait)
                self.revision+=1
                return actual
            # Wait for the already acknowledged final goal. This loop never
            # adds a corrective write or changes force, bias or tolerance.
            settle_started=time.monotonic();deadline=settle_started+settle_timeout_s;settle_samples=0
            stable_window=[]
            while True:
                if self.stop.wait(.08):raise SessionCancelled('動作已停止')
                if not self.camera.fresh():raise RuntimeError('腕部畫面中斷，已停止移動')
                actual=self.read()
                if raw_bounds is not None:check_bounds(actual,raw_bounds)
                if measured_bounds is not None:check_bounds(actual,measured_bounds)
                if measured_guard:measured_guard(actual)
                if guard:guard(actual)
                for n,v in self.goals.items():
                    if abs(actual[n]-v)>self.config['max_tracking_error_ticks']:raise RuntimeError(f'{n} 跟隨誤差過大，停止並保持姿勢')
                    if self._read(n,63,1)>60:raise RuntimeError(f'{n} 溫度過高，停止移動')
                if allow_stable_partial_observation:
                    stable_window.append((time.monotonic(),dict(actual)))
                    if any(max(p[n] for _,p in stable_window)-min(p[n] for _,p in stable_window)>2 for n in JOINTS):
                        stable_window=stable_window[-1:]
                if settle_telemetry and observe is not None and settle_samples<32:
                    settle_samples+=1
                    observe({'kind':'endpoint_settling','elapsed_s':time.monotonic()-settle_started,
                        'actual':dict(actual),'goals':dict(self.goals),'logical_targets':dict(targets),
                        'endpoint_error_ticks':{n:actual[n]-v for n,v in targets.items()},
                        'timeout_s':settle_timeout_s,'sample_index':settle_samples})
                if all(abs(actual[n]-v)<=precision for n,v in targets.items()):
                    if any(abs(v-start[n])>=50 and abs(actual[n]-start[n])<.4*abs(v-start[n]) for n,v in targets.items()):raise RuntimeError('關節沒有足夠的實際位移，停止觀察')
                    break
                if time.monotonic()>deadline:
                    if (allow_stable_partial_observation and stable_window[-1][0]-stable_window[0][0]>=.4
                            and _partial_observation_progress(actual,start,targets)):
                        if self.stop.is_set():raise SessionCancelled('動作已停止')
                        self.revision+=1
                        return {'completion':'stable_partial_observation','actual':dict(actual),
                            'goals':dict(self.goals),'logical_targets':dict(targets),
                            'reached_nominal_endpoint':False,
                            'stable_duration_s':stable_window[-1][0]-stable_window[0][0],
                            'stable_joint_span_ticks':{n:max(p[n] for _,p in stable_window)-min(p[n] for _,p in stable_window) for n in JOINTS},
                            'endpoint_error_ticks':{n:actual[n]-v for n,v in targets.items()}}
                    raise RuntimeError('未到達目標位置，已保持姿勢：'+str({n:round(actual[n]-v) for n,v in targets.items()}))
            self.revision+=1
        except MaintenanceRejected:
            raise
        except Exception as e:
            self.error=str(e);self.stop.set()
            failure={'error':self.error,'time':time.time(),'actual':dict(self.position),'goals':dict(self.goals)}
            for key in ('path_diagnostic','guard_context'):
                if hasattr(e,key):failure[key]=getattr(e,key)
            self.hold()
            failure['error']=self.error
            # Preserve the reason for a latched stop across an idle restart.
            # Disk failure must never prevent the immediate hold or mask e.
            try:
                pending=self.failure_path.with_suffix('.tmp')
                pending.write_text(json.dumps(failure,ensure_ascii=False));pending.replace(self.failure_path)
            except OSError:pass
            raise
        finally:self.moving=False
    def _screen_local_goal(self,desired,raw_bounds,guard,margin,measured_bounds=None,measured_guard=None):
        """Measured constraints see fresh encoders, never hypothetical motor lead."""
        if measured_guard is not None and not callable(measured_guard):
            raise ValueError('measured_guard must be callable')
        if self.stop.is_set():raise SessionCancelled('位置修正已停止')
        if not self.camera.fresh():raise RuntimeError('腕部畫面中斷，停止位置修正')
        t0=time.monotonic();actual=self.read();t1=time.monotonic()
        checked_kind='measured';candidate=None
        try:
            self.validate(actual,margin=0)
            if raw_bounds is not None:check_bounds(actual,raw_bounds)
            if measured_bounds is not None:check_bounds(actual,measured_bounds)
            if measured_guard:measured_guard(actual)
            if guard:guard(actual)
            candidate={**actual,**desired}
            checked_kind='prospective_goal' if desired else 'measured'
            if desired:self.validate(desired,margin=margin)
            if raw_bounds is not None:check_bounds(candidate,raw_bounds)
            if guard:guard(candidate)
        except Exception as exc:
            exc.guard_context={'kind':'local_goal_screen','checked_pose_kind':checked_kind,
                'actual':dict(actual),'prospective_pose':candidate,'desired':dict(desired),
                'physical_goals':dict(self.goals),'read_start_monotonic':t0,
                'read_end_monotonic':t1,'raised_monotonic':time.monotonic()}
            raise
        if self.stop.is_set():raise SessionCancelled('位置修正已停止')
        if not self.camera.fresh():raise RuntimeError('腕部畫面中斷，停止位置修正')
        return actual,t0,t1
    def finish_feedback(self,targets,return_origin=False,*,raw_bounds,guard=None,observe=None,measured_bounds=None,endpoint_accept=None,endpoint_stability_ticks=None,measured_guard=None,stall_response_wait=False):
        """Bounded local body correction. Caller owns the enclosing abort/hold.

        When supplied, endpoint_accept is the exclusive completion criterion;
        scalar convergence cannot override it. Acceptance needs three samples
        within the same three-second budget, including telemetry/log callbacks.
        """
        if not raw_bounds:raise ValueError('Explicit screened raw_bounds are required')
        if type(stall_response_wait) is not bool or (stall_response_wait and
                (set(targets)!={'elbow_flex'} or self.pwm_owner is not None)):
            raise ValueError('Stalled-response observation requires an empty-hand elbow endpoint')
        if measured_guard is not None and not callable(measured_guard):
            raise ValueError('measured_guard must be callable')
        if endpoint_stability_ticks is not None and (type(endpoint_stability_ticks) is not int or
                not 0<=endpoint_stability_ticks<=2 or endpoint_accept is None):
            raise ValueError('Endpoint stability requires an acceptance predicate and 0..2 ticks')
        margin=0 if return_origin else 12
        bounds={n:(max(raw_bounds[n][0],self.calibration[n]['range_min']+margin),
                   min(raw_bounds[n][1],self.calibration[n]['range_max']-margin)) for n in targets}
        started=time.monotonic();accepted_samples=0;accepted_window=[]
        # Cartesian completion can require one more encoder tick even when the
        # old 3-tick scalar tolerance is met. Continue bounded correction toward
        # the measured IK target instead of declaring a contradictory failure.
        control=LocalEndpoint(targets,self.bias,self.goals,bounds,precision=1 if endpoint_accept else 3,
                              started_at=started,stall_response_wait=stall_response_wait)
        while True:
            if self.stop.wait(control.poll_delay(time.monotonic())):raise SessionCancelled('動作已停止')
            with self.io:
                actual,t0,t1=self._screen_local_goal({},raw_bounds,guard,margin,measured_bounds,measured_guard)
                for n,goal in self.goals.items():
                    if abs(actual[n]-goal)>self.config['max_tracking_error_ticks']:raise RuntimeError(f'{n} 跟隨誤差過大')
                    if self._read(n,63,1)>60:raise RuntimeError(f'{n} 溫度過高')
                if endpoint_accept is not None and endpoint_accept(actual):
                    # Cartesian acceptance is a separate consecutive streak;
                    # interrupted scalar samples cannot later complete a hold.
                    control.stable=0
                    control.waiting_response=False
                    accepted_window.append(dict(actual))
                    if endpoint_stability_ticks is not None and any(
                            max(p[n] for p in accepted_window)-min(p[n] for p in accepted_window)>endpoint_stability_ticks
                            for n in actual):
                        accepted_window=[dict(actual)]
                    accepted_samples=len(accepted_window)
                    if time.monotonic()-started>=3:raise RuntimeError('末端量測等待逾時')
                    event={'phase':'cartesian_endpoint','status':'accepted' if accepted_samples>=3 else 'settling',
                           'actual':actual,'accepted_samples':accepted_samples,
                           'read_start_monotonic':t0,'read_end_monotonic':t1,
                           'logical_targets':dict(targets),'goals':dict(self.goals),'bias':dict(self.bias)}
                    if observe is not None:observe(event)
                    # Telemetry/log callbacks share the same global budget.
                    if time.monotonic()-started>=3:raise RuntimeError('末端量測等待逾時')
                    if accepted_samples>=3:return actual
                    continue
                accepted_samples=0;accepted_window=[]
                decision=control.observe(actual,time.monotonic())
                if decision.status=='abort':raise RuntimeError('位置修正中止：'+decision.reason)
                event={'phase':'endpoint','status':decision.status,'actual':actual,
                       'read_start_monotonic':t0,'read_end_monotonic':t1,
                       'logical_targets':dict(targets),'goals':dict(self.goals),'bias':dict(self.bias)}
                if decision.reason:event['reason']=decision.reason
                if decision.status=='done' and endpoint_accept is not None:
                    event.update(status='rejected',reason='custom endpoint criterion is not satisfied')
                    if observe is not None:observe(event)
                    raise RuntimeError('關節已穩定，但末端位置仍未達到接受條件')
                if decision.status=='propose':
                    correction=decision.correction;n=correction.joint
                    measured,t0,t1=self._screen_local_goal({n:correction.goal},raw_bounds,guard,margin,measured_bounds,measured_guard)
                    # The guard/read latency counts toward the same global budget.
                    control.validate_before_write(decision,time.monotonic())
                    write_started=time.monotonic();self._write(n,42,correction.goal)
                    written_at=time.monotonic();control.commit(decision,written_at)
                    self.goals[n]=correction.goal;self.bias[n]=correction.bias
                    event.update(status='committed',joint=n,goal=correction.goal,
                                 measured_before_write=measured,write_read_start_monotonic=t0,
                                 write_read_end_monotonic=t1,write_start_monotonic=write_started,
                                 write_completed_monotonic=written_at,goals=dict(self.goals),bias=dict(self.bias))
                if observe is not None:observe(event)
                if decision.status=='done':return actual
    def hold(self):
        self.stop.set();failures={};self.armed=False
        with self.io:
            # A simultaneous resume may have run while hold waited for I/O.
            self.stop.set();self.armed=False
            if not self.port:failures['connection']='尚未連接馬達，無法確認保持姿勢'
            if self.port:
                for n in JOINTS:
                    try:
                        # A failed cap restoration may already have applied 500.
                        # Only the power-session protocol may neutralize a jaw
                        # with active/pending/unknown cap state, including stop.
                        if n=='gripper':self.gripper_power.assert_motion_allowed()
                        actual=self._read(n,56)
                        self.validate({n:actual},margin=0)
                        self._write(n,42,actual)
                        self.position[n]=actual;self.goals[n]=actual
                    except Exception as e:failures[n]=str(e)
            if failures:self.error='部分關節無法確認停止：'+', '.join(failures)
        return {'held':not failures,'failures':failures}

    def _require_not_maintenance(self):
        if getattr(self,'maintenance_latched',False):
            raise MaintenanceRejected('控制器正在準備重新啟動，請等重新啟動完成後再恢復')

    def maintenance_quiesce(self):
        """Latch a verified stationary hold without changing any motor register.

        The API owns job_lock before entering this I/O lock. This process then
        refuses new begin/resume/recovery calls until exit; emergency hold stays
        available. Existing compensation goals, torque and PWM remain untouched.
        """
        with self.io:
            if self.moving or self.pwm_owner is not None:
                raise RuntimeError('請先完成目前動作與夾爪工作，再重新啟動控制器')
            if not self.connected or not self.port:
                raise RuntimeError('馬達尚未連接，無法確認原本的保持目標')
            self.gripper_power.assert_motion_allowed()
            samples=[];goals=None
            for index in range(4):
                started=time.monotonic();actual=self.read()
                observed={n:self._read(n,42) for n in JOINTS}
                self.validate(actual,margin=0);self.validate(observed,margin=0)
                if goals is not None and goals!=observed:
                    raise RuntimeError('保持目標還在改變，尚未進入維護狀態')
                goals=observed
                health={n:{'torque':self._read(n,40,1),'status':self._read(n,65,1),
                           'moving':self._read(n,66,1)} for n in JOINTS}
                if any(v['torque']!=1 or v['status']!=0 or v['moving'] not in (0,1) for v in health.values()):
                    raise RuntimeError('馬達保持狀態不正常，尚未進入維護狀態')
                if any(abs(actual[n]-goals[n])>(16 if n=='gripper' else 60) for n in JOINTS):
                    raise RuntimeError('馬達偏離原保持目標，尚未進入維護狀態')
                samples.append(dict(actual=dict(actual),physical_goals=dict(goals),servo_health=health,
                    read_start_monotonic=started,read_end_monotonic=time.monotonic()))
                if any(max(s['actual'][n] for s in samples)-min(s['actual'][n] for s in samples)>2 for n in JOINTS):
                    raise RuntimeError('手臂尚未停穩，尚未進入維護狀態')
                if index<3:time.sleep(.08)
            if self.moving or self.pwm_owner is not None:
                raise RuntimeError('維護檢查期間出現新動作，請重新確認')
            self.gripper_power.assert_motion_allowed()
            if (getattr(self,'maintenance_receipt',None) is not None
                    and self.maintenance_receipt['preserved_goals']!=goals):
                raise RuntimeError('進入維護後保持目標已改變，請重新檢查')
            self.maintenance_latched=True;self.stop.set();self.armed=False
            if getattr(self.stop,'persistence_error',''):
                raise RuntimeError('維護停止狀態未保存，暫時不要重新啟動控制器')
            self.goals=dict(goals)
            receipt=dict(maintenance_latched=True,stopped=True,preserved_goals=dict(goals),
                actual=dict(actual),samples=samples,zero_motor_writes=True,
                restart_required=True,automatic_resume=False)
            self.maintenance_receipt=receipt
            return receipt

    def resume(self):
        with self.io:
            self._require_not_maintenance()
            self.gripper_power.assert_motion_allowed()
            if self.connected:
                for n in JOINTS:
                    if self._read(n,65,1):raise RuntimeError(f'{n} 馬達保護尚未解除')
            self.stop.clear()
            try:self.failure_path.unlink(missing_ok=True)
            except OSError as exc:
                self.stop.set()
                raise RuntimeError('故障紀錄無法清除，保持停止') from exc
            self.error=''
    def recover_gripper_power(self):
        with self.io:
            self._require_not_maintenance()
            self.connect()
            result=self.gripper_power.recover()
            if 'neutral_goal' in result:
                self.goals['gripper']=result['neutral_goal'];self.bias['gripper']=0
            self.pwm_owner=None
            return result
    def demonstrate_gripper(self,report):
        """Explicit empty-gripper demonstration, with no arm trajectory."""
        try:
            self.begin();start=self.read()['gripper'];minimum=self.calibration['gripper']['range_min']
            closed=minimum+16;opened=minimum+404
            self.validate({'gripper':closed});self.validate({'gripper':opened})
            for cycle in range(2):
                for label,target in [('張開',opened),('收合',closed)]:
                    report(f'夾爪{label} · {cycle+1} / 2')
                    self.glide({'gripper':target},speed=70,precision=16)
                    until=time.monotonic()+3
                    while time.monotonic()<until:
                        if self.stop.wait(.1):raise RuntimeError('夾爪示範已停止')
                        if not self.camera.fresh():raise RuntimeError('腕部畫面中斷，停止示範')
            self.glide({'gripper':start},speed=70,return_origin=True,precision=16)
            report('夾爪收放示範完成')
        except MaintenanceRejected:
            raise
        except Exception as e:
            self.error=str(e);self.stop.set();self.hold();raise
    def gesture(self,kind):
        self._require_not_maintenance()
        self.connect();origin=self.read();d=self.config['scan_delta_ticks']
        if kind=='look':
            # A small nod and elbow movement, relative to actual pose. Never replay historic raw poses.
            goal=self.config['gaze_wrist_flex']
            if abs(goal-origin['wrist_flex'])>500:raise RuntimeError('目前姿势與抬頭基準相差過大，請先確認姿勢')
            count=max(1,math.ceil(abs(goal-origin['wrist_flex'])/90))
            pan_goal=self.config.get('gaze_pan',origin['shoulder_pan'])
            if abs(pan_goal-origin['shoulder_pan'])>400:raise RuntimeError('目前方向離觀察範圍太遠，保持原位')
            plan=[{'wrist_flex':round(origin['wrist_flex']+(goal-origin['wrist_flex'])*i/count),'shoulder_pan':round(origin['shoulder_pan']+(pan_goal-origin['shoulder_pan'])*i/count)} for i in range(1,count+1)]
        elif kind in ('wave','shrug'):
            plan=[{'wrist_roll':origin['wrist_roll']+d},{'wrist_roll':origin['wrist_roll']-d},{'wrist_roll':origin['wrist_roll']+d},{'wrist_roll':origin['wrist_roll']}]
        else:raise ValueError('不支援的姿勢')
        for i,pose in enumerate(plan):self.validate(pose,margin=0 if i==len(plan)-1 else 12)
        for i,pose in enumerate(plan):self.glide(pose,return_origin=i==len(plan)-1,precision=45)
    def scan(self,snap,report):
        try:
            self._require_not_maintenance()
            return self._scan(snap,report)
        except MaintenanceRejected:
            raise
        except Exception as e:
            self.error=str(e);self.stop.set();self.hold();raise
    def _scan(self,snap,report):
        self.gesture('look')
        self.connect();origin=self.read();origin_goals=dict(self.goals);d=self.config['scan_delta_ticks']
        # 6 degrees of pan per side, a modest elbow/head nod. Collect actual observations.
        plan=[{},
              {'shoulder_pan':origin['shoulder_pan']-90,'wrist_flex':origin['wrist_flex']-60},
              {'shoulder_pan':origin['shoulder_pan']+90,'elbow_flex':origin['elbow_flex']-90,'wrist_roll':origin['wrist_roll']+100},
              {'shoulder_pan':origin['shoulder_pan'],'wrist_flex':origin['wrist_flex']+70,'wrist_roll':origin['wrist_roll']-100},
              {'shoulder_pan':origin['shoulder_pan']-60,'elbow_flex':origin['elbow_flex']-140,'wrist_flex':origin['wrist_flex']-50,'wrist_roll':origin['wrist_roll']},
              {'shoulder_pan':origin['shoulder_pan']+40,'wrist_flex':origin['wrist_flex']+90},
              {n:origin_goals[n]-self.bias[n] for n in ['shoulder_pan','wrist_flex','elbow_flex','wrist_roll']}]

        for i,p in enumerate(plan):
            if p:self.validate(p,margin=0 if i==len(plan)-1 else 12)
        frames=[]
        for i,p in enumerate(plan):
            if p:self.glide(p,return_origin=i==len(plan)-1,precision=45)
            if self.stop.is_set():raise RuntimeError('環顧已停止')
            report(f'環顧四周 {i+1} / {len(plan)}')
            frame=self.settled_snapshot(snap);frames.append(frame)
            with open(self.runtime/'observations.jsonl','a') as log:log.write(json.dumps(frame)+'\n')
        return frames
    def settled_snapshot(self,snap,guard=None):
        try:
            self._require_not_maintenance()
            history=[];times=[];window=[];deadline=time.monotonic()+3
            while True:
                if self.stop.wait(.12):
                    self._require_not_maintenance()
                    raise RuntimeError('環顧已停止')
                t0=time.monotonic();actual=self.read();t1=time.monotonic()
                if guard:guard(actual)
                history.append(actual);history=history[-6:];times.append(t1);times=times[-6:]
                window.append({'read_start_monotonic':t0,'read_end_monotonic':t1,'joints':actual});window=window[-6:]
                if len(history)==6 and all(max(p[n] for p in history)-min(p[n] for p in history)<=2 for n in JOINTS):break
                if time.monotonic()>deadline:raise RuntimeError('手臂持續漂移，無法配對影像與角度')
            before_start=time.monotonic();before=self.read();before_time=time.monotonic()
            if guard:guard(before)
            # Poll metadata, not snap(): serializing the same old 1080p image
            # repeatedly holds the camera lock and competes with new arrivals.
            # No motion occurs here. A temporarily unavailable exposure may
            # recover within this finite wait, but only a fresh new image can
            # be returned, with continuously stable measured joints around it.
            generation=self.camera.generation
            if not isinstance(generation,str) or not generation:raise RuntimeError('尚未取得腕部相機身分')
            lower={n:min(p[n] for p in history+[before]) for n in JOINTS}
            upper={n:max(p[n] for p in history+[before]) for n in JOINTS}
            if any(upper[n]-lower[n]>2 for n in JOINTS):raise RuntimeError('取像期間手臂仍在移動')
            brackets=[{'read_start_monotonic':before_start,'read_end_monotonic':before_time,'joints':before}]
            deadline=before_time+SETTLED_IMAGE_WAIT_SECONDS;polls=0
            while True:
                if self.stop.wait(.06):
                    self._require_not_maintenance()
                    raise RuntimeError('環顧已停止')
                if time.monotonic()>deadline:raise RuntimeError('沒有與關節角度同步的新影像')
                t0=time.monotonic();actual=self.read();t1=time.monotonic()
                if guard:guard(actual)
                for n in JOINTS:lower[n]=min(lower[n],actual[n]);upper[n]=max(upper[n],actual[n])
                if any(upper[n]-lower[n]>2 for n in JOINTS):raise RuntimeError('取像期間手臂仍在移動')
                brackets.append({'read_start_monotonic':t0,'read_end_monotonic':t1,'joints':actual})
                if self.camera.generation!=generation:raise RuntimeError('腕部相機在取像期間重新連接')
                polls+=1
                try:stamp=self.camera.latest_stamp()
                except RuntimeError:
                    # latest_stamp explicitly rejects stale/dark/missing input.
                    # Wait for a valid arrival, never serialize the old image.
                    continue
                captured=stamp.get('frame_capture_monotonic')
                if (stamp.get('camera_source')!='wrist' or type(captured) not in (int,float)
                        or not math.isfinite(captured) or captured>time.monotonic()):
                    raise RuntimeError('腕部影像時刻無效')
                if captured<before_time:continue
                if time.monotonic()>deadline:raise RuntimeError('沒有與關節角度同步的新影像')
                frame=snap()
                if self.stop.wait(0):
                    self._require_not_maintenance()
                    raise RuntimeError('環顧已停止')
                captured=frame.get('capture_monotonic')
                if (frame.get('source')!='wrist' or frame.get('camera_generation')!=generation
                        or self.camera.generation!=generation):
                    raise RuntimeError('腕部相機在取像期間重新連接')
                if (type(captured) not in (int,float) or not math.isfinite(captured)
                        or captured<before_time or captured<stamp['frame_capture_monotonic']):
                    raise RuntimeError('沒有與關節角度同步的新影像')
                break
            after_start=time.monotonic();after=self.read();after_time=time.monotonic()
            if guard:guard(after)
            if self.stop.wait(0):
                self._require_not_maintenance()
                raise RuntimeError('環顧已停止')
            if self.camera.generation!=generation:raise RuntimeError('腕部相機在取像期間重新連接')
            if frame['capture_monotonic']>after_start:raise RuntimeError('腕部影像的單調時刻不在關節觀察期間')
            if not 0<=after_time-frame['capture_monotonic']<1.:raise RuntimeError('腕部相機畫面已過期')
            for n in JOINTS:lower[n]=min(lower[n],after[n]);upper[n]=max(upper[n],after[n])
            if any(upper[n]-lower[n]>2 for n in JOINTS):raise RuntimeError('取像期間手臂仍在移動')
            # Use the nearest completed joint read before this exact exposure;
            # the first stable read remains the strict new-image threshold.
            bracket=max((s for s in brackets if s['read_end_monotonic']<=frame['capture_monotonic']),
                        key=lambda s:s['read_end_monotonic'])
            before_start=bracket['read_start_monotonic'];before=bracket['joints']
            frame['joints']={n:round((before[n]+after[n])/2) for n in JOINTS}
            frame['timestamp_basis']='jpeg-arrival after settled joint window'
            frame['joint_capture_span_ticks']=max(abs(after[n]-before[n]) for n in JOINTS)
            frame['settled_duration_s']=times[-1]-times[0]
            frame['settled_joint_window']=window
            frame['joint_observation_bracket_monotonic']=[before_start,after_time]
            frame['image_wait_polls']=polls
            frame['image_wait_seconds']=after_time-before_time
            frame['image_wait_joint_samples']=len(brackets)+1
            frame['image_wait_joint_span_ticks']=max(upper[n]-lower[n] for n in JOINTS)
            if 'capture_monotonic' in frame:
                frame['image_joint_skew_s']=max(abs(frame['capture_monotonic']-before_start),abs(after_time-frame['capture_monotonic']))
            return frame
        except MaintenanceRejected:
            raise
        except Exception as e:
            self.error=str(e);self.stop.set();self.hold();raise
    def refinement_scan(self,snap,report,origin,pilot=False,framing_ticks=0):
        try:
            self._require_not_maintenance()
            return self._refinement_scan(snap,report,origin,pilot,framing_ticks)
        except MaintenanceRejected:
            raise
        except Exception as e:
            self.error=str(e);self.stop.set();self.hold();raise
    def wide_observation(self,snap,report):
        """Engineering-qualified observation region; no object contact or grasp."""
        from .observation_plan import build_plan,validate_pose,orientation_views
        import uuid
        try:
            self.begin();origin=self.read()
            screened={'shoulder_pan':2161,'shoulder_lift':1103,'elbow_flex':2915,
                      'wrist_flex':2318,'wrist_roll':998,'gripper':1524}
            if any(abs(origin[n]-screened[n])>(70 if n=='wrist_flex' else 12) for n in JOINTS):
                raise RuntimeError('目前姿態不在已檢查的加寬觀察範圍內')
            self.gesture('look');origin=self.read()
            if any(abs(origin[n]-screened[n])>12 for n in JOINTS):
                raise RuntimeError('目前姿態不在已檢查的加寬觀察範圍內')
            finish={n:self.goals[n]-self.bias[n] for n in JOINTS}
            if any(abs(finish[n]-origin[n])>20 for n in JOINTS):
                raise RuntimeError('保持目標偏移過大，無法執行加寬觀察')
            plan=build_plan(origin,self.calibration,framing_ticks=70,return_target=finish)
            framed=plan['framed_origin']
            # Rotation constraints supplement the larger translation baseline.
            orientations=orientation_views(framed,self.calibration,bias=self.bias)
            stages=plan['stages'][:2]+orientations+plan['stages'][2:]
            for stage in stages:validate_pose(stage['targets'],self.calibration,bias=self.bias)
            frames=[];group=uuid.uuid4().hex
            for index,stage in enumerate(stages):
                report(f'拉開視角，核對深度 {index+1} / {len(stages)}')
                if stage['motion']:
                    for target in stage.get('waypoints',[stage['targets']]):self.glide(target,precision=45)
                frame=self.settled_snapshot(snap)
                frame.update(observation_stage=stage['label'],validation_group=group+':'+stage['validation_group'])
                frames.append(frame)
                with open(self.runtime/'observations.jsonl','a') as log:log.write(json.dumps(frame)+'\n')
                (self.runtime/'wide-observations-last.json').write_text(json.dumps(frames,indent=2))
            return frames
        except MaintenanceRejected:
            raise
        except Exception as e:
            self.error=str(e);self.stop.set();self.hold();raise
    def local_gaze_observation(self,snap,report,on_capture=None):
        """Small, separately screened image-response experiment; no approach."""
        from .observation_plan import build_local_gaze_plan,require_local_gaze_region,GAZE_AXES
        import uuid
        frames=[];group=uuid.uuid4().hex
        try:
            self._require_not_maintenance()
            self.connect();require_local_gaze_region(self.read())
            self.adopt_stationary_hold();self.begin(allow_rearm=False)
            reference=self.settled_snapshot(snap,guard=require_local_gaze_region);origin=reference['joints']
            require_local_gaze_region(origin)
            saved={n:self.goals[n]-self.bias[n] for n in GAZE_AXES}
            stages=build_local_gaze_plan(origin,self.calibration,return_target=saved,bias=self.bias)
            for i,stage in enumerate(stages):
                report(f'核對視線的小幅移動 {i+1} / {len(stages)}')
                region=None
                def guard(actual):
                    require_local_gaze_region(actual,moving=region is not None,active_axis=region)
                    if self.camera.generation!=reference['camera_generation']:raise RuntimeError('腕部相機在採樣途中重新連接')
                for target,region in zip(stage['waypoints'],stage['waypoint_regions']):
                    self.glide(target,speed=35,precision=8,guard=guard,allow_rearm=False)
                frame=reference if i==0 else self.settled_snapshot(snap,guard=guard)
                guard(frame['joints'])
                if frame['camera_generation']!=reference['camera_generation']:raise RuntimeError('腕部取像不是同一個相機連線')
                labels=('center_0','pan_plus','center_1','pan_minus','center_2','wrist_plus','center_3','wrist_minus','center_4')
                frame.update(observation_stage=stage['label'],stage=labels[i],protocol_id=group,
                             validation_group=group+':'+stage['validation_group'])
                frames.append(frame)
                with open(self.runtime/'observations.jsonl','a') as log:log.write(json.dumps(frame)+'\n')
                (self.runtime/'local-gaze-observations-last.json').write_text(json.dumps(frames,indent=2))
                if on_capture:on_capture(frame)
            return frames
        except MaintenanceRejected:
            raise
        except Exception as e:
            self.error=str(e);self.stop.set();self.hold();raise
    def _refinement_scan(self,snap,report,origin,pilot=False,framing_ticks=0):
        from .geometry import Kinematics
        from .selfcal import inv
        import numpy as np
        self.connect()
        names=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll']
        current=self.read()
        if any(abs(current[n]-origin[n])>60 for n in names):raise RuntimeError('補看起點已改變，請重新搜尋')
        # One fresh origin for the entire bounded sequence; never rebase the
        # second step onto the raised first step. Restore the saved motor goals.
        reference=self.settled_snapshot(snap);origin=reference['joints']
        saved_goals={n:self.goals[n] for n in names}
        fk=Kinematics(Path(__file__).resolve().parents[1]/'assets/so101_follower_cam.urdf')
        frames=[]
        def record(frame):
            frames.append(frame)
            with open(self.runtime/'observations.jsonl','a') as log:log.write(json.dumps(frame)+'\n')
            (self.runtime/'refinement-last.json').write_text(json.dumps(frames,indent=2))
            return frame
        record(reference)
        if isinstance(framing_ticks,bool) or not isinstance(framing_ticks,int) or not 0<=framing_ticks<=70:raise ValueError('觀察取景角度超出限制')
        if framing_ticks:
            report('調整視線，讓物件表面的細節更清楚')
            self.glide({'wrist_flex':origin['wrist_flex']+framing_ticks},precision=16)
            framed=record(self.settled_snapshot(snap));origin=framed['joints']
        start_camera=fk.camera(origin)
        if not pilot:
            # Small overlapping orientation changes preserve textured material
            # tracks while providing more than a single rotation axis.
            views=[{'wrist_roll':origin['wrist_roll']+45},
                   {'wrist_roll':origin['wrist_roll']-45},
                   {'wrist_roll':origin['wrist_roll'],'shoulder_pan':origin['shoulder_pan']-35},
                   {'shoulder_pan':origin['shoulder_pan']+35},
                   {'shoulder_pan':origin['shoulder_pan']}]
            for target in views:self.validate(target)
            for i,target in enumerate(views):
                report(f'沿著物件細節換角度 {i+1} / {len(views)}')
                self.glide(target,precision=16);record(self.settled_snapshot(snap))
        for lift_delta,elbow_delta in ([(30,-60)] if pilot else [(30,-60),(60,-120),(120,-60)]):
            report('平移腕部，再用實際角度對齊視線')
            body_target={'shoulder_lift':origin['shoulder_lift']+lift_delta,'elbow_flex':origin['elbow_flex']+elbow_delta,'wrist_flex':origin['wrist_flex']-lift_delta-elbow_delta}
            self.glide(body_target,precision=45)
            body=record(self.settled_snapshot(snap))['joints']
            wrist_target=sum(origin[n] for n in ['shoulder_lift','elbow_flex','wrist_flex'])-body['shoulder_lift']-body['elbow_flex']
            if abs(wrist_target-body['wrist_flex'])>90:raise RuntimeError('所需視線修正超過觀察幅度')
            # An observation needs measured geometry, not a forced exact goal.
            # A stalled precision controller previously stopped this experiment.
            # Use the normal bounded move and qualify the ACTUAL rotation below;
            # never retry micro-corrections to force a translation qualification.
            self.glide({'wrist_flex':wrist_target,'shoulder_pan':origin['shoulder_pan'],'wrist_roll':origin['wrist_roll']},precision=16)
            frame=self.settled_snapshot(snap)
            relative=inv(start_camera)@fk.camera(frame['joints'])
            angle=float(np.degrees(np.arccos(np.clip((np.trace(relative[:3,:3])-1)/2,-1,1))))
            baseline=float(np.linalg.norm(relative[:3,3]))
            pitch_error=abs(sum(frame['joints'][n]-origin[n] for n in ['shoulder_lift','elbow_flex','wrist_flex']))
            frame['measured_translation_check']={'rotation_deg':angle,'baseline_m':baseline,'pitch_error_ticks':pitch_error,'qualified':pitch_error<=2 and angle<.4 and baseline>=.010}
            record(frame)
        report('觀察完成，回到原本的保持目標')
        self.glide({n:saved_goals[n]-self.bias[n] for n in names},return_origin=True,precision=45)
        record(self.settled_snapshot(snap))
        return frames
    def status(self):return {'connected':self.connected,'port':self.config['follower_port'],'position':dict(self.position),'moving':self.moving,'stopped':self.stop.is_set(),'error':self.error,'revision':self.revision,'maintenance_latched':getattr(self,'maintenance_latched',False)}
    def close(self):
        # A settled loaded servo already has a holding goal. Re-seeding that goal
        # on every idle service restart would accumulate gravity sag.
        preserved=False
        with self.io:
            if getattr(self,'maintenance_latched',False):
                # The maintenance API already verified the hold under the
                # admission/I/O locks. Never reseed actual during its shutdown.
                self.stop.set();self.armed=False;preserved=True
            elif self.port and not self.moving:
                try:
                    self.gripper_power.assert_motion_allowed()
                    samples=[];goals=None
                    for _ in range(4):
                        actual=self.read();observed={n:self._read(n,42) for n in JOINTS}
                        if goals is not None and goals!=observed:raise RuntimeError('Hold goals changed')
                        goals=observed
                        if any(self._read(n,40,1)!=1 or self._read(n,65,1)!=0 for n in JOINTS):raise RuntimeError('Hold not healthy')
                        if any(abs(actual[n]-goals[n])>(16 if n=='gripper' else 60) for n in JOINTS):raise RuntimeError('Hold error outside retained range')
                        samples.append(actual);time.sleep(.08)
                    if any(max(x[n] for x in samples)-min(x[n] for x in samples)>2 for n in JOINTS):raise RuntimeError('Hold not stationary')
                    preserved=True;self.stop.set();self.armed=False
                except Exception:pass
        if not preserved:self.hold()
        with self.io:
            if self.port:self.port.closePort()
            if self.lockfile:self.lockfile.close()
