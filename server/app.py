from __future__ import annotations
import asyncio, base64, copy, random, subprocess, concurrent.futures, contextlib, json, os, re, secrets, stat, threading, time, uuid
from pathlib import Path
from typing import Literal
import cv2, numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse, FileResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, ConfigDict, Field
from .camera import WristCamera
from .robot import Robot
from .brain import Brain, BrainTransportError
from .geometry import SpatialEstimator
from .local_secrets import load_provider_credentials
from .jev import JevPolicy
from .world_state import WorldStateMemory
from .er2_brain import GeminiRoboticsER2Brain
from .motiongpt_arm import MOTION_PRESETS, plan_single_arm_motion, execute_retargeted_motion, is_natural_motion_query

ROOT=Path(__file__).resolve().parents[1];RUNTIME=ROOT/'.runtime';RUNTIME.mkdir(exist_ok=True)
load_provider_credentials(ROOT)
tactician=JevPolicy()
CONFIG=json.loads((ROOT/'config/robot.json').read_text())
er2_brain=GeminiRoboticsER2Brain(default_thinking_level=CONFIG.get('thinking_level') or CONFIG.get('reasoning_effort','low'))
camera=WristCamera(CONFIG,RUNTIME/'frames')
robot=Robot(CONFIG,json.loads((ROOT/'config/follower-calibration.json').read_text()),RUNTIME,camera)
brain=Brain(CONFIG,RUNTIME)
try:
    if (RUNTIME/'brain-config.json').is_file():
        _saved_brain=json.loads((RUNTIME/'brain-config.json').read_text())
        if _saved_brain.get('engine') in ('astra','argon'):
            brain.engine_override=_saved_brain['engine']
        if _saved_brain.get('thinking_level') in ('none','low','medium','high'):
            brain.thinking_level_override=_saved_brain['thinking_level']
            er2_brain.default_thinking_level=_saved_brain['thinking_level']
except Exception:
    pass
spatial=SpatialEstimator(ROOT)
from .kinematic_offsets import runtime_offsets,OffsetKinematics
if runtime_offsets(ROOT):spatial.fk=OffsetKinematics(ROOT/'assets/so101_follower_cam.urdf',runtime_offsets(ROOT))
world_memory=WorldStateMemory(spatial.fk)
executor=concurrent.futures.ThreadPoolExecutor(max_workers=1,thread_name_prefix='robot-owner')
jobs={};scenes={};frames={};uploads={};job_lock=threading.RLock();frame_lock=threading.RLock();active=None
snapshot_owner=threading.local()
TOKEN=secrets.token_urlsafe(32)
engineering_capture_lock=threading.Lock()

@contextlib.asynccontextmanager
async def lifespan(app):
    camera.start()
    yield
    robot.stop.set()
    for j in jobs.values():j['cancel'].set()
    await asyncio.to_thread(robot.close);camera.close();executor.shutdown(wait=False,cancel_futures=True)
app=FastAPI(title='LeRobot Argon Local',lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware,allowed_hosts=['localhost','127.0.0.1','testserver'])

@app.get('/api/engineering/scene/{artifact}')
def engineering_scene(artifact:str,frame_id:str|None=None,snapshot_id:str|None=None):
    """Read-only exported geometry; never opens a device or starts motion."""
    media = {'scene.json': 'application/json', 'scene.glb': 'model/gltf-binary',
             'scene.xml': 'application/xml', 'verification.json': 'application/json',
             'depth.json': 'application/json', 'wrist.jpg': 'image/jpeg'}
    if artifact not in media:
        raise HTTPException(404, '沒有這個工程檔案')
    if snapshot_id is not None and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}',snapshot_id):
        raise HTTPException(404,'沒有這份工程快照')
    root = (RUNTIME / 'sim-scene').resolve()
    directory = (root / (snapshot_id or 'current')).resolve()
    if directory.parent != root:
        raise HTTPException(404,'沒有這份工程快照')
    directory_fd = None
    try:
        # Keep one directory inode even if /current is renamed or its symlink
        # is swapped while this response is being prepared. Do not defer a
        # FileResponse path reopen until after the manifest has been checked.
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        def read_artifact(name,limit):
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
            with os.fdopen(fd, 'rb') as handle:
                info = os.fstat(handle.fileno())
                if not stat.S_ISREG(info.st_mode):
                    raise HTTPException(404, '沒有這個工程檔案')
                if info.st_size > limit:
                    raise HTTPException(413, '工程檔案超過讀取上限')
                content = handle.read(limit + 1)
                if len(content) > limit:
                    raise HTTPException(413, '工程檔案超過讀取上限')
                return content
        manifest = read_artifact('scene.json', 1024 * 1024)
        if artifact != 'scene.json' and not frame_id:
            raise HTTPException(400, '請先載入場景資料，再以該份快照 ID 讀取檔案')
        if frame_id is not None:
            try:
                expected = json.loads(manifest)['frame']['id']
            except (ValueError, KeyError, TypeError):
                raise HTTPException(409, '場景資料已改變，請重新載入快照')
            if not isinstance(expected, str) or not expected or expected != frame_id:
                raise HTTPException(409, '場景資料已改變，請重新載入快照')
        limit = 64 * 1024 * 1024 if artifact == 'scene.glb' else 16 * 1024 * 1024 if artifact == 'wrist.jpg' else 1024 * 1024
        content = manifest if artifact == 'scene.json' else read_artifact(artifact, limit)
        return Response(content, media_type=media[artifact])
    except OSError:
        raise HTTPException(404, '尚未匯出這份場景，請先執行工程場景匯出')
    finally:
        if directory_fd is not None:
            os.close(directory_fd)

@app.get('/api/engineering/scenes')
def engineering_scenes():
    root=RUNTIME/'sim-scene';entries=[]
    if not root.exists():return {'snapshots':[]}
    for directory in root.iterdir():
        if (directory.name=='current' or directory.is_symlink() or not directory.is_dir()
                or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}',directory.name)):
            continue
        try:
            path=directory/'scene.json'
            if path.is_symlink() or path.stat().st_size>1024*1024:continue
            state=json.loads(path.read_text());frame=state['frame']
            entries.append({'snapshot_id':directory.name,'frame_id':frame['id'],
                            'captured_at':frame.get('captured_at'),
                            'has_depth':(directory/'depth.json').is_file()})
        except (OSError,ValueError,KeyError,TypeError):continue
    return {'snapshots':sorted(entries,key=lambda row:row['captured_at'] or 0,reverse=True)[:100]}

@app.post('/api/engineering/scene-capture')
def engineering_scene_capture():
    from .engineering_snapshot import capture_snapshot,publish_snapshot
    if not engineering_capture_lock.acquire(blocking=False):
        raise HTTPException(409,'正在擷取工程快照，請稍候')
    try:
        # No submit/begin/hold calls: failures must not issue motor commands.
        with job_lock:
            if ((active and jobs[active]['status']=='running') or robot.moving
                    or robot.pwm_owner is not None or grasp.state not in ('idle','approaching')):
                raise HTTPException(409,'請等待目前工作結束，再擷取工程快照')
            frame=capture_snapshot(robot,camera)
        return publish_snapshot(RUNTIME,frame)
    except (RuntimeError,ValueError,OSError) as exc:
        raise HTTPException(503,str(exc)) from exc
    finally:
        engineering_capture_lock.release()

@app.middleware('http')
async def local_only(request:Request,call_next):
    origin=request.headers.get('origin')
    if origin and origin not in ['http://localhost:3100','http://127.0.0.1:3100','http://localhost:8102','http://127.0.0.1:8102']:
        return Response('Origin rejected',status_code=403)
    if request.method in ['POST','PUT','DELETE','PATCH'] and not secrets.compare_digest(request.headers.get('x-lerobot-token',''),TOKEN):
        return Response('Local session required',status_code=403)
    response=await call_next(request);response.headers['Cache-Control']='no-store';return response

class SearchRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    query:str=Field(default='',max_length=500)
    mode:Literal['search','lucky','point','inspect']='search'
    reference_id:str|None=None;frame_id:str|None=None
    x:float|None=Field(default=None,ge=0,le=1);y:float|None=Field(default=None,ge=0,le=1)
class ActionRequest(BaseModel):
    scene_id:str;object_index:int=Field(ge=0);action_index:int=Field(ge=0)
class ObjectActionRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    kind:Literal['inspect','pick','wave','place','stack']
    query:str=Field(default='',max_length=500)
class CommandRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    instruction:str=Field(min_length=1,max_length=500)
class ManualWaveRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
class UploadRequest(BaseModel):
    image:str=Field(max_length=12_000_000)
class EmptyGripperRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    frame_id:str
    empty_verified:Literal[True]
    validation_only:bool=False
class JawCaptureRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    frame_id:str
    empty_verified:Literal[True]
    protocol:Literal['standard','wide']='standard'

def public_job(j):
    with job_lock:
        result=copy.deepcopy({k:v for k,v in j.items() if k not in ('cancel','frame_pins','manual_wave_signal')})
    geometry=(result.get('result') or {}).get('spatial')
    if geometry:
        # Keep per-pixel engineering evidence on disk; polling the search page
        # needs the conclusion, not thousands of repeated correspondence rows.
        for segment in geometry.get('tracking',[]):
            segment.pop('pairs',None);segment.pop('frames',None);segment.pop('visibility',None)
        for obj in geometry.get('objects',[]):
            for fold in obj.get('calibration_stability',{}).get('folds',[]):fold.pop('pairs',None)
    return result
def publish_scene(j,scene):
    with job_lock:
        j['result']=copy.deepcopy(scene);scenes[scene['id']]=copy.deepcopy(scene)
def report(j,message,stage='observing'):
    with job_lock:
        j.update(message=message,stage=stage,updated_at=time.time());j['events'].append({'message':message,'stage':stage,'time':time.time()})
        j['events']=j['events'][-20:]

def execute_object_program(j,query,task='grasp'):
    """Publish exact executor phases and retain partial evidence on failure."""
    stage='planning'
    def change_stage(value):
        nonlocal stage
        stage=value
    previous=grasp.latest
    try:
        value=grasp.run_program(query,lambda message:report(j,message,stage),j['cancel'],
                                task=task,report_stage=change_stage)
    except Exception:
        if grasp.latest is not previous:
            with job_lock:j['engineering_result']=copy.deepcopy(grasp.latest)
        raise
    with job_lock:j['engineering_result']=value
    key={'grasp':'grasp_confirmed','place':'place_confirmed','stack':'stack_confirmed'}[task]
    if value.get('completed') is not True or value.get(key) is not True:
        raise RuntimeError(value.get('reason') or '這次物件操作尚未確認完成')
    return value
def snap(owner=None):
    if owner is None:owner=getattr(snapshot_owner,'job',None)
    f=camera.snapshot()
    with frame_lock:
        frames[f['id']]=f
        if owner is not None:owner['frame_pins'].add(f['id'])
        with job_lock:
            protected={fid for job in jobs.values() if job['status']=='running' for fid in job.get('frame_pins',set())}
            protected.update(f['id'] for scene in scenes.values() for f in scene['frames'])
            # Keep the evidence the result UI actually renders, even after the
            # job stops running and a camera preview continues taking snapshots.
            receipts=[scene.get('grasp') for scene in scenes.values()]
            receipts.extend(job.get('engineering_result') for job in jobs.values()
                            if job.get('kind') in ('grasp-attempt','object-program'))
            receipts.extend(step.get('receipt') for job in jobs.values() if job.get('kind')=='command'
                            for step in (job.get('command_result') or {}).get('steps',[]))
            protected.update(receipt[key]['id'] for receipt in receipts if isinstance(receipt,dict)
                             for key in ('pre_lift_frame','held_frame') if isinstance(receipt.get(key),dict) and 'id' in receipt[key])
            protected.update(observation['after']['id'] for receipt in receipts if isinstance(receipt,dict)
                             for observation in receipt.get('observations',[])
                             if observation.get('completed') is True and isinstance(observation.get('after'),dict)
                             and observation['after'].get('source')=='wrist' and 'id' in observation['after'])
        for key in list(frames):
            if len(frames)<=200:break
            if key in protected:continue
            old=frames.pop(key);Path(old['path']).unlink(missing_ok=True)
    return f

def capture_segment(observations):
    return {'keyframes':[f['id'] for f in observations],
            'frames':camera.tracking_history(observations[0]['sequence'],observations[-1]['sequence'])}

def framing_correction(scene, observations, current):
    """Small gaze correction from a recent box, without guessing object size."""
    index=scene.get('selected_index',-1)
    if index<0:return 0
    obj=scene['objects'][index];view=observations[obj['frame_index']]
    names=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll']
    if any(abs(view['joints'][n]-current['joints'][n])>20 for n in names):return 0
    b=obj['bbox'];image=cv2.imread(view['path'])
    if image is None:return 0
    height,width=image.shape[:2]
    # Bring texture above the jaw exclusion region, keeping top-edge headroom.
    # This is a bounded framing heuristic; subsequent images verify geometry.
    shift=max(0.,min((b['y']+b['h']/2-.31)*height,(b['y']-.10)*height))
    return min(70,round(shift/(width*.75)*4096/(2*np.pi)))

def submit(kind,fn,*,allow_stopped=False,allow_holding=False,initial_stage='starting',initial_message='準備看看眼前的世界'):
    global active
    with job_lock:
        if getattr(robot,'maintenance_latched',False):raise HTTPException(409,'控制器正在準備重新啟動，暫時不接受新工作')
        if active and jobs[active]['status']=='running':raise HTTPException(409,'LeRobot 正在處理上一個動作')
        if robot.stop.is_set() and not allow_stopped:raise HTTPException(409,'機械臂已停止；請按恢復')
        if robot.pwm_owner is not None and not allow_holding and kind not in ('inspect','grasp-step','grasp-open','grasp-attempt','grasp-release','command'):
            raise HTTPException(409,'夾爪正由抓取流程保持，請先完成或釋放目前物件')
        jid=uuid.uuid4().hex;j={'id':jid,'kind':kind,'status':'running','stage':initial_stage,'message':initial_message,'events':[],'created_at':time.time(),'cancel':threading.Event(),'frame_pins':set(),'result':None}
        if kind=='wave':
            j.update(manual_wave_signal=threading.Event(),manual_wave_requested=False,manual_wave_phase='assessing')
        jobs[jid]=j;active=jid
        while len(jobs)>50:jobs.pop(next(iter(jobs)))
    def run():
        snapshot_owner.job=j
        try:
            fn(j)
            if j['cancel'].is_set():raise RuntimeError('已停止')
            with job_lock:j['status']='completed'
        except Exception as e:
            with job_lock:j['status']='cancelled' if j['cancel'].is_set() else 'failed'
            report(j,str(e),'error')
        finally:
            del snapshot_owner.job
    executor.submit(run)
    return public_job(j)

def remember_scene(scene,observations):
    # Enrich the original evidence; inference/geometry completion is not a
    # new capture and must never refresh its observation time or scene ID.
    paired=[f for f in observations if f.get('joints') and f.get('capture_monotonic') is not None]
    for frame in paired:
        world_memory.remember(frame,objects=scene['objects'],object_frames=observations,spatial=scene.get('spatial'))

def enrich(scene,observations):
    scene['id']=uuid.uuid4().hex;scene['created_at']=min(f['captured_at'] for f in observations);scene['completed_at']=time.time();scene['robot_revision']=robot.revision;scene['model']=CONFIG['model'];scene['elapsed']=brain.last_latency
    scene['frames']=[{k:v for k,v in f.items() if k!='path'} for f in observations]
    remember_scene(scene,observations)
    for obj in scene['objects']:
        if CONFIG.get('experimental_pick_enabled') and obj['safe_to_touch'] and not any(a['kind']=='pick' for a in obj['actions']):
            obj['actions'].append({'kind':'pick','label':'試著抓起來','reason':'一次規劃接近與抓取，輕合後核對是否拿起','target':''})
        for a in obj['actions']:
            a['available']=a['kind'] in ('inspect','wave') or (a['kind'] in ('pick','place','stack') and obj['safe_to_touch'] and bool(CONFIG.get('experimental_pick_enabled')))
            a['unavailable_reason']='' if a['available'] else '這個物件的實驗操作目前無法啟用'
        if not any(a['kind']=='inspect' for a in obj['actions']):obj['actions'].insert(0,{'kind':'inspect','label':'再看仔細一點','reason':'換個角度確認細節','target':'','available':True,'unavailable_reason':''})
    with job_lock:
        scenes[scene['id']]=copy.deepcopy(scene)
        while len(scenes)>40:scenes.pop(next(iter(scenes)))
    return scene

def text_intent_search(req):
    return (req.mode=='search' and bool(req.query.strip()) and not req.reference_id
            and not req.frame_id and req.x is None and req.y is None)

def run_search(req,j):
    if text_intent_search(req):
        from .search_intent import classify_search_intent,action_suggestion
        report(j,'先理解你想找什麼，或想讓手臂做什麼','understanding')
        intent=classify_search_intent(brain,req.query,j['cancel'],tactician=tactician)
        suggestion=action_suggestion(req.query,intent)
        if suggestion is not None:
            with job_lock:
                j['action_suggestion']=suggestion
                if is_natural_motion_query(req.query) and hasattr(grasp,'model'):
                    try:
                        j['motiongpt_plan']=plan_single_arm_motion(req.query,model=grasp.model)
                    except Exception:
                        pass
            report(j,'已整理成動作建議，點選後才會執行' if suggestion['availability'] in ('available','experimental')
                   else suggestion['reason'],'done')
            return
        # Text classification is read-only even while stopped/holding. A noun
        # search still follows the original camera/motion ownership checks.
        if robot.stop.is_set():raise RuntimeError('機械臂已停止；請按恢復後搜尋物件')
        if robot.pwm_owner is not None:raise RuntimeError('夾爪正由抓取流程保持，請先完成或放開目前物件')
    if j['cancel'].is_set():raise RuntimeError('已停止')
    if not camera.fresh():raise RuntimeError(camera.error or '沒有即時腕部畫面')
    reference=None;observation_receipt=None;observation_notice=''
    if req.reference_id:
        reference=uploads.get(req.reference_id)
        if reference is None:raise ValueError('參考照片已失效，請重新上傳')
    if req.mode=='point':
        with frame_lock:
            f=frames.get(req.frame_id)
            if not f or time.time()-f['captured_at']>120:raise ValueError('預覽照片已過期，請重新開啟相機')
            j['frame_pins'].add(f['id'])
        if req.x is None or req.y is None:raise ValueError('缺少點選位置')
        observations=[f];point={'x':req.x,'y':req.y,'frame_index':0}
    elif req.mode=='inspect':observations=[snap(j)];point=None
    else:
        report(j,'動動手腕，看看左右','scanning')
        revision=robot.revision
        try:
            observation_receipt=grasp.observe_views(lambda m:report(j,m,'scanning'),j['cancel'])
            observations=observation_receipt['frames']
        except Exception as exc:
            # A declined sweep before any motion still permits fresh visual
            # search. Never hide cancellation or a partially executed movement.
            # A model/network failure says nothing about clearance. Starting
            # another inference here would repeat the same cloud wait.
            if isinstance(exc,BrainTransportError):raise
            if j['cancel'].is_set() or robot.stop.is_set() or robot.revision!=revision:raise
            observation_notice='目前先原地觀察：'+str(exc)
            report(j,'目前空間不適合轉腕，先看眼前的物件','observing')
            observations=[snap(j)]
        point=None
    if j['cancel'].is_set():raise RuntimeError('已停止')
    report(j,'Argon ER 2 正在辨識眼前物件與可以做的事','thinking')
    all_observations=observations
    if len(observations)>4:observations=[observations[i] for i in [0,2,5,6]]
    result=brain.perceive(req.query,observations,reference,point,j['cancel'])
    brain_elapsed=brain.last_latency
    scene=enrich(result,observations);publish_scene(j,scene)
    if observation_receipt:
        # Single-axis roll improves coverage, but cannot identify all camera
        # extrinsics/focal length. Its CAD baseline is not calibrated depth.
        scene['observation']={k:observation_receipt[k] for k in ('measured_roll_range_deg','return_error_deg','camera_baseline_model_mm','not_external_metrology')}
        scene['observation'].update(mode='wrist_roll_arc',view_count=len(all_observations),metric_calibrated=False)
        scene['notice']='已從左右確認物件；抓取時會再核對夾爪位置。'
    elif observation_notice:scene['notice']=observation_notice
    scene['brain_elapsed']=brain_elapsed;scene['elapsed']=round(time.time()-j['created_at'],1)
    publish_scene(j,scene)
    if req.mode=='lucky':
        # Model identification confidence cannot unlock a not-yet-calibrated grasp.
        if scene['empty']:
            report(j,'眼前沒有合適的小物件，向你揮揮手','moving');scene['wave']=grasp.planned_wave(lambda m:report(j,m,'moving'),j['cancel']);scene['performed']='向你揮手'
        elif scene['selected_index']>=0:
            obj=scene['objects'][scene['selected_index']]
            kind=random.choice(['look','wave']);report(j,f'好手氣選中「{obj["name"]}」','moving')
            if kind=='look':robot.gesture(kind)
            else:scene['wave']=grasp.planned_wave(lambda m:report(j,m,'moving'),j['cancel'])
            scene['performed']=f'換個角度觀察{obj["name"]}' if kind=='look' else f'找到{obj["name"]}，向你揮手示意'
        else:scene['performed']='沒有足夠把握，保持原位'
    if req.mode=='point' and scene['selected_index']>=0:
        obj=scene['objects'][scene['selected_index']];scene['query']=obj['name'];obj['matches_query']=True
        scene['notice_kind']='recognition_only'
        scene['notice']='已認出物件，尚未嘗試抓取。可選擇下方的「嘗試輕夾」。'
    scene['robot_revision']=robot.revision;scene['completed_at']=time.time();scene['elapsed']=round(time.time()-j['created_at'],1)
    publish_scene(j,scene)
    report(j,'搜尋完成','done')

@app.get('/api/session')
def session():return {'token':TOKEN}
@app.post('/api/gripper-demo')
def gripper_demo():
    return submit('gripper-demo',lambda j:robot.demonstrate_gripper(lambda m:report(j,m,'moving')))
@app.post('/api/gripper-baseline')
def gripper_baseline(req:EmptyGripperRequest):
    # Engineering-only entry point: the caller must inspect the fresh wrist
    # frame. Ordinary product actions never assume an empty gripper.
    with frame_lock:
        f=frames.get(req.frame_id)
        if not f or f['source']!='wrist' or time.time()-f['captured_at']>30:
            raise HTTPException(409,'請先檢查新的腕部影像，確認夾爪空置')
        evidence={'frame_id':f['id'],'captured_at':f['captured_at'],'source':'wrist','method':'engineering visual inspection'}
    from .active_touch import collect_empty_jaw
    return submit('gripper-baseline',lambda j:collect_empty_jaw(robot,lambda m:report(j,m,'moving'),j['cancel'],evidence,req.validation_only))
@app.post('/api/gripper-power/recover')
def recover_gripper_power():
    with job_lock:
        if getattr(robot,'maintenance_latched',False):raise HTTPException(409,'控制器正在準備重新啟動，暫時無法復原夾爪')
        if active and jobs[active]['status']=='running':raise HTTPException(409,'請先停止目前工作')
        try:return robot.recover_gripper_power()
        except RuntimeError as e:raise HTTPException(409,str(e))

@app.post('/api/engineering/jaw-calibration/capture')
def jaw_calibration_capture(req:JawCaptureRequest):
    from .jaw_acquisition import collect_jaw_images, validate_empty_evidence
    from .spatial_runtime import geometry_revision, _read_profile
    with frame_lock:
        evidence=copy.deepcopy(frames.get(req.frame_id))
    try:validate_empty_evidence(evidence,camera.generation,empty_verified=req.empty_verified)
    except ValueError as exc:raise HTTPException(409,str(exc))
    with job_lock:
        if robot.moving or grasp.state not in ('idle','approaching'):
            raise HTTPException(409,'請先完成目前動作或持物流程，再拍攝空夾爪')
        try:robot.gripper_power.assert_motion_allowed()
        except RuntimeError as exc:raise HTTPException(409,str(exc))
        revision=geometry_revision(grasp.model)
        profile=_read_profile(CONFIG,'spatial_calibration_profile')
        def run(j):
            result=collect_jaw_images(robot,snap,lambda m:report(j,m,'observing'),j['cancel'],evidence,
                empty_verified=req.empty_verified,camera_id=CONFIG['wrist_camera_id'],
                mount_revision=CONFIG.get('camera_mount_revision','existing-mount-unverified'),
                geometry_revision=revision,camera_profile=profile,protocol=req.protocol)
            with job_lock:j['jaw_calibration_result']=result
            report(j,'空夾爪五開口影像已保存，尚未形成幾何校準','done')
        return submit('jaw-calibration-capture',run,initial_stage='observing',
                      initial_message='準備拍攝五個空夾爪開口；手臂身體關節保持不動')
@app.post('/api/observe-wide')
def observe_wide(req:SearchRequest):
    def run(j):
        all_views=robot.wide_observation(lambda:snap(j),lambda m:report(j,m,'scanning'))
        histories=[capture_segment(all_views)]
        views=[all_views[i] for i in [1,3,8,9]]
        # Persist the camera segment before cloud work, including failed calls.
        spatial.record_inputs(all_views,[],views,histories)
        report(j,'Argon ER 2 正在核對加寬視角中的物件','thinking')
        scene=enrich(brain.perceive(req.query,views,None,None,j['cancel']),views)
        publish_scene(j,scene)
        report(j,'比對不同距離與返回視角中的位置','measuring')
        scene['spatial']=spatial.estimate(all_views,scene['objects'],views,j['cancel'],histories=histories)
        remember_scene(scene,views)
        scene['notice']='已完成加寬觀察；抓取仍須通過獨立精度驗證。'
        scene['brain_elapsed']=brain.last_latency;scene['elapsed']=round(time.time()-j['created_at'],1)
        publish_scene(j,scene);report(j,'加寬觀察完成','done')
    return submit('wide-observation',run)
@app.get('/api/status')
def status():return {'robot':robot.status(),'camera':camera.status(),'brain':brain.status(),'er2':er2_brain.status_dict(),'motiongpt':{'available':True,'presets':list(MOTION_PRESETS.keys())},'tactics':tactician.status(),'job':public_job(jobs[active]) if active and active in jobs else None,'grasp':{'state':grasp.state,'low_power_owner':robot.pwm_owner is not None},'capabilities':{'search':True,'look':True,'wave':True,'command':True,'motiongpt':True,'er2':er2_brain.available(),'pick':bool(CONFIG.get('experimental_pick_enabled')),'place':bool(CONFIG.get('experimental_pick_enabled')),'stack':bool(CONFIG.get('experimental_pick_enabled')),'observer_in_production':False}}
@app.post('/api/observe-local')
def observe_local(req:SearchRequest):
    def run(j):
        views=[];histories=[]
        try:robot.local_gaze_observation(lambda:snap(j),lambda m:report(j,m,'scanning'),on_capture=views.append)
        finally:
            # Preserve unsuccessful observations before the live ring advances.
            if views:
                histories=[capture_segment(views)]
                spatial.record_inputs(views,[],[views[0]],histories,archive_name='wrist-local-gaze-input.zip')
        scene_views=[views[0]]
        report(j,'Argon ER 2 正在辨識這次小幅觀察的目標','thinking')
        scene=enrich(brain.perceive(req.query,scene_views,None,None,j['cancel']),scene_views)
        spatial.record_inputs(views,scene['objects'],scene_views,histories,archive_name='wrist-local-gaze-input.zip')
        scene['notice']='小幅視線採樣完成；置中控制仍須通過獨立影像驗證。'
        scene['elapsed']=round(time.time()-j['created_at'],1)
        publish_scene(j,scene);report(j,'小幅觀察完成','done')
    return submit('local-gaze-observation',run)
# Small engineering steps share the production single-owner executor.
class GraspStepRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    direction:Literal['left','right','up','down','forward','backward']
    distance_mm:float=Field(gt=0,le=5)
    frame_id:str
    clear_path_verified:Literal[True]

from .grasp import WristGrasp
grasp=WristGrasp(robot,brain,snap,RUNTIME)
from .kinematic_grasp import KinematicGraspProgram
kinematic_grasp=KinematicGraspProgram(robot,brain,grasp,RUNTIME)
grasp.tactician=tactician
grasp.state_memory=world_memory

# Calibration uses the same single owner as robot jobs, but captures never
# issue a motion command. D435/observer frames have no entry point here.
from .calibration_store import CalibrationSessionStore
calibration_sessions=CalibrationSessionStore(ROOT)
calibration_view_batches={}
calibration_view_origins={}
from .calibration_workspace_journal import WorkspaceJournal
calibration_workspace_journal=WorkspaceJournal(ROOT)
calibration_workspace_candidates={}

class CalibrationStartRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    board_static:Literal[True]
    print_scale_checked:Literal[True]

class CalibrationCaptureRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    board_static:bool

class SpatialModeRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    mode:Literal['legacy','shadow','v2']

class CalibrationActivateRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    solve_id:str=Field(pattern=r'^[0-9a-f]{32}$')

class JawProfileActivateRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    profile_sha256:str=Field(pattern=r'^[0-9a-f]{64}$')

class CalibrationViewPlanRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    session_id:str=Field(pattern=r'^[0-9a-f]{32}$')
    rotations_only:bool=False

class CalibrationViewExecuteRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    batch_id:str=Field(pattern=r'^[0-9a-f]{32}$')
    candidate_id:str=Field(min_length=1,max_length=80)
    approval:dict

class WorkspacePlanRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    session_id:str=Field(pattern=r'^[0-9a-f]{32}$')

class WorkspacePrepareRequest(WorkspacePlanRequest):
    station_index:int=Field(ge=0,strict=True)

class WorkspaceExecuteRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    candidate_id:str=Field(pattern=r'^[0-9a-f]{32}$')
    approval:dict

def _calibration_view_session(session_id):
    from .spatial_runtime import geometry_revision
    try:
        session=calibration_sessions.status(session_id)
        if session['status'] in ('invalidated','solving') or not session['board_static_declared']:
            raise ValueError('棋盤工作階段已失效或正在計算，請先固定棋盤並建立新的工作階段')
        if (session['geometry_revision']!=geometry_revision(grasp.model)
                or session['camera_id']!=CONFIG['wrist_camera_id']
                or session['mount_revision']!=CONFIG.get('camera_mount_revision','existing-mount-unverified')):
            raise ValueError('這批校準與目前的相機或手臂設定不同')
        return session
    except (ValueError,KeyError) as exc:raise HTTPException(409,str(exc))

def _workspace_record(session_id):
    from .calibration_workspace_motion import _profile
    from .spatial_runtime import _read_profile
    _calibration_view_session(session_id)
    record=calibration_workspace_journal.load(session_id)
    if record is None:raise ValueError('請先建立這批校準的完整取景計畫')
    protocol=record['protocol']
    selected=_read_profile(CONFIG,'spatial_calibration_profile')
    fingerprint=_profile(selected,grasp.model,CONFIG['wrist_camera_id'],
        CONFIG.get('camera_mount_revision','existing-mount-unverified'))
    if fingerprint!=protocol['camera_profile_sha256']:
        raise ValueError('相機校準版本已改變，不能沿用舊工程取景路徑')
    if record['inflight'] is not None:
        raise ValueError('上一站的實際移動尚未核對完成；保持目前姿勢，不重播舊指令')
    return record

@app.post('/api/engineering/calibration/workspace/plan')
def calibration_workspace_plan(req:WorkspacePlanRequest):
    if robot.moving:raise HTTPException(409,'手臂還在移動，請等停穩再規劃校準')
    _calibration_view_session(req.session_id)
    def run(j):
        if j['cancel'].is_set():raise RuntimeError('已停止')
        from .calibration_workspace_motion import build_workspace_protocol
        from .spatial_runtime import _read_profile
        existing=calibration_workspace_journal.load(req.session_id)
        if existing is not None:
            record=_workspace_record(req.session_id)
        else:
            robot.connect();robot.begin(allow_rearm=False)
            frame=grasp.capture()
            session=_calibration_view_session(req.session_id)
            anchor=calibration_sessions.motion_origin(req.session_id,
                {**frame,'camera_id':CONFIG['wrist_camera_id']})
            if (anchor['generation']!=frame['camera_generation']
                    or session['camera_generation'] not in (None,frame['camera_generation'])
                    or any(abs(frame['joints'][n]-anchor['joints'][n])>3 for n in frame['joints'])):
                raise ValueError('目前姿勢不在這批校準的原始起點，不能重設起點以延長範圍')
            protocol=build_workspace_protocol(grasp,frame,session_id=req.session_id,
                camera_profile=_read_profile(CONFIG,'spatial_calibration_profile'),
                camera_id=CONFIG['wrist_camera_id'],
                mount_revision=CONFIG.get('camera_mount_revision','existing-mount-unverified'))
            if j['cancel'].is_set():raise RuntimeError('已停止')
            record=calibration_workspace_journal.create(req.session_id,protocol)
        with job_lock:j['workspace_protocol']=record['protocol']
        report(j,'完整肩肘取景計畫已保存；尚未移動','done')
    return submit('calibration-workspace-plan',run,initial_stage='planning',
        initial_message='計算涵蓋肩、肘與手腕的校準姿勢')

@app.post('/api/engineering/calibration/workspace/prepare')
def calibration_workspace_prepare(req:WorkspacePrepareRequest):
    if robot.moving:raise HTTPException(409,'手臂還在移動，請等停穩再規劃下一站')
    _calibration_view_session(req.session_id)
    def run(j):
        if j['cancel'].is_set():raise RuntimeError('已停止')
        from .calibration_workspace_motion import (prepare_workspace_station,
            board_reference_from_capture,describe_workspace_station)
        record=_workspace_record(req.session_id)
        robot.connect();robot.begin(allow_rearm=False)
        frame=grasp.capture()
        protocol=record['protocol']
        board_reference=None
        try:
            board_reference=board_reference_from_capture(grasp,protocol['camera_profile'],frame,
                camera_id=CONFIG['wrist_camera_id'],
                mount_revision=CONFIG.get('camera_mount_revision','existing-mount-unverified'))
        except ValueError:
            pass  # Missing corners are not a fabricated board measurement.
        candidate=prepare_workspace_station(grasp,protocol,frame,station_index=req.station_index,
            completed_station_ids=record['completed_station_ids'],board_reference=board_reference,
            previous_receipt=record['receipts'][-1] if record['receipts'] else None,
            camera_rebind=record.get('camera_rebind'))
        if j['cancel'].is_set():raise RuntimeError('已停止')
        identity=uuid.uuid4().hex
        with job_lock:
            # Re-preparing replaces old review candidates for this session.
            for key,old in list(calibration_workspace_candidates.items()):
                if old['session_id']==req.session_id:calibration_workspace_candidates.pop(key)
            calibration_workspace_candidates[identity]=candidate
            while len(calibration_workspace_candidates)>12:
                calibration_workspace_candidates.pop(next(iter(calibration_workspace_candidates)))
            j['workspace_candidate']={**describe_workspace_station(candidate),'candidate_id':identity,
                'frame':{k:v for k,v in frame.items() if k!='path'}}
        report(j,'下一站已用目前位置重算；等待核對影像與整段路徑','done')
    return submit('calibration-workspace-prepare',run,initial_stage='planning',
        initial_message='核對下一個肩肘姿勢與棋盤取景')

@app.post('/api/engineering/calibration/workspace/execute')
def calibration_workspace_execute(req:WorkspaceExecuteRequest):
    with job_lock:
        if robot.moving:raise HTTPException(409,'手臂還在移動，請等停穩再執行校準')
        candidate=calibration_workspace_candidates.get(req.candidate_id)
        if candidate is None:raise HTTPException(409,'這個校準候選已用過或已被更新')
        _calibration_view_session(candidate['session_id'])
        def run(j):
            from .calibration_workspace_motion import execute_workspace_station,validate_workspace_station
            record=_workspace_record(candidate['session_id'])
            if j['cancel'].is_set():raise RuntimeError('已停止')
            reference=grasp.capture()
            validate_workspace_station(grasp,record['protocol'],candidate,reference,
                session_id=candidate['session_id'],completed_station_ids=record['completed_station_ids'],
                approval=req.approval,previous_receipt=record['receipts'][-1] if record['receipts'] else None,
                camera_rebind=record.get('camera_rebind'))
            # Persist before any possible dispatch. A crash is not permission
            # to replay the segment after restarting the controller.
            calibration_workspace_journal.begin(candidate['session_id'],candidate['station_id'],req.candidate_id)
            movement=execute_workspace_station(grasp,record['protocol'],candidate,reference,j['cancel'],
                lambda message:report(j,message,'moving'),session_id=candidate['session_id'],
                completed_station_ids=record['completed_station_ids'],approval=req.approval,
                previous_receipt=record['receipts'][-1] if record['receipts'] else None,
                camera_rebind=record.get('camera_rebind'))
            with job_lock:j['calibration_motion_result']=movement
            calibration_workspace_journal.moved(candidate['session_id'],candidate['station_id'],movement)
            if j['cancel'].is_set():raise RuntimeError('已停止')
            if candidate['capture_required']:
                result=calibration_sessions.capture(candidate['session_id'],
                    {**movement['after'],'camera_id':CONFIG['wrist_camera_id']},board_static=True)
                calibration_workspace_journal.captured(candidate['session_id'],candidate['station_id'],result)
                with job_lock:j['calibration_result']=result
                message=result.get('message') or '已保存本次棋盤取景'
            elif (movement.get('completion') == 'stable_partial_observation'
                    or movement.get('reached_nominal_endpoint') is False):
                message='已停穩並保存實際姿勢；未到名目位置，下一站會按實測位置規劃'
            else:message='已到達中途姿勢；下一站會重新核對腕部影像'
            report(j,message,'done')
        result=submit('calibration-workspace-execute',run,initial_stage='moving',
            initial_message='移到已核對的肩肘姿勢，停穩後取像')
        calibration_workspace_candidates.pop(req.candidate_id)
        return result

@app.post('/api/engineering/calibration/views/plan')
def calibration_view_plan(req:CalibrationViewPlanRequest):
    if robot.moving:raise HTTPException(409,'手臂還在移動，請等停穩再計算校準視角')
    _calibration_view_session(req.session_id)
    def run(j):
        from .calibration_motion import build_calibration_candidates,describe_calibration_candidates
        if j['cancel'].is_set():raise RuntimeError('已停止')
        robot.connect();robot.begin(allow_rearm=False)
        frame=grasp.capture()
        session=_calibration_view_session(req.session_id)
        if session['camera_generation'] not in (None,frame['camera_generation']):
            raise ValueError('相機已重新連接，請建立新的棋盤工作階段')
        anchor=calibration_sessions.motion_origin(req.session_id,
            {**frame,'camera_id':CONFIG['wrist_camera_id']})
        with job_lock:calibration_view_origins[req.session_id]=anchor
        if anchor['generation']!=frame['camera_generation']:
            raise ValueError('相機已重新連接，請建立新的棋盤工作階段')
        batch=build_calibration_candidates(grasp,frame,origin=anchor['joints'],
            session_id=req.session_id,cancel=j['cancel'],rotations_only=req.rotations_only)
        batch_id=uuid.uuid4().hex
        with job_lock:
            calibration_view_batches[batch_id]=batch
            while len(calibration_view_batches)>12:calibration_view_batches.pop(next(iter(calibration_view_batches)))
            j['calibration_motion']={**describe_calibration_candidates(batch),'batch_id':batch_id,
                'source':'wrist-only','frame':{k:v for k,v in frame.items() if k!='path'}}
        report(j,'取景路徑已計算；尚未移動，等待核對這一張腕部影像','done')
    return submit('calibration-view-plan',run,initial_stage='planning',initial_message='計算退後、抬高與不同角度的校準取景')

@app.post('/api/engineering/calibration/views/execute')
def calibration_view_execute(req:CalibrationViewExecuteRequest):
    with job_lock:
        if robot.moving:raise HTTPException(409,'手臂還在移動，請等停穩再切換校準視角')
        batch=calibration_view_batches.get(req.batch_id)
        if batch is None:raise HTTPException(409,'取景計畫已用過或過期，請重新計算')
        _calibration_view_session(batch['session_id'])
        def run(j):
            from .calibration_motion import execute_calibration_candidate
            if j['cancel'].is_set():raise RuntimeError('已停止')
            _calibration_view_session(batch['session_id'])
            reference=grasp.capture()
            movement=execute_calibration_candidate(grasp,batch,req.candidate_id,reference,
                j['cancel'],lambda message:report(j,message,'moving'),
                session_id=batch['session_id'],approval=req.approval)
            with job_lock:j['calibration_motion_result']=movement
            if j['cancel'].is_set():raise RuntimeError('已停止')
            frame={**movement['after'],'camera_id':CONFIG['wrist_camera_id']}
            result=calibration_sessions.capture(batch['session_id'],frame,board_static=True)
            with job_lock:j['calibration_result']=result
            report(j,result.get('message') or '取景與棋盤拍攝完成','done')
        job_result=submit('calibration-view-execute',run,initial_stage='moving',initial_message='依已核對路徑換角度，停穩後拍棋盤')
        calibration_view_batches.pop(req.batch_id)
        return job_result

def _require_spatial_change_idle():
    if ((active and jobs[active]['status']=='running') or robot.pwm_owner is not None
            or robot.moving or not robot.stop.is_set()):
        raise HTTPException(409,'請先停止目前工作，並確認夾爪沒有持物，再切換空間設定')
    if grasp.state not in ('idle','approaching'):
        raise HTTPException(409,'夾爪還有待確認的接觸或持物，請先完成放下流程')
    try:robot.gripper_power.assert_motion_allowed()
    except RuntimeError as exc:raise HTTPException(409,'還有未完成的夾爪保持紀錄，先完成恢復或放下流程') from exc

def _save_spatial_config(changes):
    """Caller holds job_lock and checked the stop latch. No hardware writes."""
    from .spatial_runtime import settings, public_state
    updated={**CONFIG,**changes}
    settings(updated)
    path=ROOT/'config/robot.json'
    previous=path.read_bytes()
    if json.loads(previous)!=CONFIG:
        raise HTTPException(409,'磁碟設定已被其他程式修改，請重新載入服務後再切換')
    releases=ROOT/'.runtime/spatial-config-history';releases.mkdir(exist_ok=True,parents=True)
    identity=uuid.uuid4().hex
    (releases/(identity+'.json')).write_bytes(previous)
    temporary=path.with_name('robot.'+identity+'.tmp')
    try:
        temporary.write_text(json.dumps(updated,ensure_ascii=False,indent=2)+'\n')
        if path.read_bytes()!=previous:
            raise HTTPException(409,'磁碟設定在切換期間被修改，已保留外部修改，請重新載入服務')
        temporary.replace(path)
    finally:temporary.unlink(missing_ok=True)
    CONFIG.update(changes)
    robot.config=CONFIG
    # Cached object locations cannot survive a change of coordinate model.
    scenes.clear();grasp.latest=None
    global world_memory
    world_memory=WorldStateMemory(spatial.fk);grasp.state_memory=world_memory
    robot.revision+=1
    return {'changed':True,'restart_required':False,'motors_commanded':False,
            'previous_config_id':identity,'spatial':public_state(CONFIG)}

@app.post('/api/engineering/calibration/mode')
def spatial_mode(req:SpatialModeRequest):
    with job_lock:
        _require_spatial_change_idle()
        return _save_spatial_config({'spatial_backend':req.mode,
            'motion_backend':'task_space_v2' if req.mode=='v2' else 'legacy',
            'object_model_provider':'geometry'})

@app.post('/api/engineering/calibration/sessions/{session_id}/activate')
def calibration_activate(session_id:str,req:CalibrationActivateRequest):
    from .spatial_runtime import geometry_revision
    from .spatial_contract import WristCalibration
    with job_lock:
        _require_spatial_change_idle()
        try:
            if not re.fullmatch(r'[0-9a-f]{32}',session_id):
                raise ValueError('無效的校準工作階段')
            session=calibration_sessions.status(session_id)
            for key,expected in (('camera_id',CONFIG['wrist_camera_id']),
                    ('mount_revision',CONFIG.get('camera_mount_revision','existing-mount-unverified')),
                    ('geometry_revision',geometry_revision(grasp.model))):
                if session[key]!=expected:raise ValueError('校準工作階段與目前相機或手臂設定不一致')
            solved=session['last_solve']
            if (session['status']!='solved' or not solved or solved['solve_id']!=req.solve_id
                    or solved['status']!='validated'):
                raise ValueError('只能套用目前工作階段最新且通過驗證的校準')
            directory=calibration_sessions.root/session_id/'solves'/req.solve_id
            candidate_path=directory/'candidate-profile.json'
            candidate_bytes=candidate_path.read_bytes()
            report_bytes=(directory/'report.json').read_bytes()
            profile=json.loads(candidate_bytes)
            report_data=json.loads(report_bytes)
            if (report_data['status']!='validated' or profile!=report_data['calibration_profile']
                    or profile.get('source')!='wrist-only' or report_data.get('source')!='wrist-only'
                    or profile.get('board_session_id')!=session_id
                    or report_data.get('board_session_id')!=session_id):
                raise ValueError('校準候選與獨立驗證報告不同')
            calibrated=WristCalibration.from_mapping(profile,
                native_image_size=session['image_size'],camera_id=CONFIG['wrist_camera_id'],
                geometry_revision=geometry_revision(grasp.model),cad_handeye=np.eye(4),
                mount_revision=CONFIG.get('camera_mount_revision','existing-mount-unverified'))
            if not calibrated.metric_calibrated:raise ValueError('校準尚未通過獨立驗證')
            if session['image_size'][0]!=CONFIG.get('wrist_capture_width',1920):
                raise ValueError('目前相機解析度與校準照片不同')
            with camera.lock:
                if camera.frame is not None and list(camera.frame.shape[1::-1])!=session['image_size']:
                    raise ValueError('即時腕部相機尺寸與校準照片不同')
            # Publish exactly the verified candidate/report bytes. Keeping
            # these under config makes the selected profile survive a fresh
            # checkout without copying photographs or transient robot state.
            published=ROOT/'config/calibration'
            if published.is_symlink():raise ValueError('校準設定目錄不能是符號連結')
            published.mkdir(parents=True,exist_ok=True)
            prefix=f'{session_id}-{req.solve_id}'
            artifacts=((published/(prefix+'-profile.json'),candidate_bytes),
                       (published/(prefix+'-report.json'),report_bytes))
            for path,data in artifacts:
                if path.is_symlink() or (path.exists() and path.read_bytes()!=data):
                    raise ValueError('同一校準版本已有不同內容，已保留原設定；請重新求解另存版本')
            for path,data in artifacts:
                if path.exists():continue
                temporary=published/(prefix+'-'+uuid.uuid4().hex+'.tmp')
                try:
                    with temporary.open('xb') as handle:
                        handle.write(data);handle.flush();os.fsync(handle.fileno())
                    try:os.link(temporary,path)
                    except FileExistsError:
                        if path.is_symlink() or path.read_bytes()!=data:
                            raise ValueError('校準版本在保存期間被修改，已保留原設定')
                finally:temporary.unlink(missing_ok=True)
            # Recheck both complete files before the final config mutation.
            if any(path.is_symlink() or path.read_bytes()!=data for path,data in artifacts):
                raise ValueError('校準檔案保存核對失敗，尚未切換設定')
            selected=str(artifacts[0][0].relative_to(ROOT))
        except (ValueError,OSError,KeyError,TypeError) as exc:raise HTTPException(409,str(exc))
        return _save_spatial_config({'spatial_calibration_profile':selected})

@app.get('/api/engineering/calibration')
def calibration_status():
    from .spatial_runtime import public_state
    return {'spatial':public_state(CONFIG), 'sessions':calibration_sessions.list_sessions(),
            'board_url':'/calibration/lerobot-a4-calibration-board.pdf'}

@app.post('/api/engineering/jaw-calibration/activate')
def jaw_profile_activate(req:JawProfileActivateRequest):
    from .jaw_profile_selection import reviewed_jaw_profile
    with job_lock:
        _require_spatial_change_idle()
        try:
            selected,jaw=reviewed_jaw_profile(ROOT,req.profile_sha256,CONFIG,grasp.model)
            # Readback only. Selection does not connect, rearm or move to a
            # sampled opening, and an out-of-range opening is not extrapolated.
            raw=robot.read()['gripper'] if robot.connected else jaw.samples[0].raw
            geometry=jaw.at(raw)
        except (ValueError,OSError,KeyError,TypeError,RuntimeError) as exc:
            raise HTTPException(409,str(exc))
        result=_save_spatial_config({'jaw_calibration_profile':selected})
        result['jaw_geometry']=geometry.to_dict()
        result['current_opening_checked']=bool(robot.connected)
        return result

@app.post('/api/engineering/calibration/sessions')
def calibration_start(req:CalibrationStartRequest):
    from .spatial_runtime import geometry_revision
    with job_lock:
        if active and jobs[active]['status']=='running':
            raise HTTPException(409,'請等目前工作結束，再開始校準')
        try:
            return calibration_sessions.start(camera_id=CONFIG['wrist_camera_id'],
                mount_revision=CONFIG.get('camera_mount_revision','existing-mount-unverified'),
                geometry_revision=geometry_revision(grasp.model),board_static=req.board_static)
        except (ValueError,OSError) as exc:raise HTTPException(409,str(exc))

@app.post('/api/engineering/calibration/sessions/{session_id}/capture')
def calibration_capture(session_id:str,req:CalibrationCaptureRequest):
    try:session=calibration_sessions.status(session_id)
    except ValueError as exc:raise HTTPException(404,str(exc))
    from .spatial_runtime import geometry_revision
    if session['geometry_revision']!=geometry_revision(grasp.model):
        raise HTTPException(409,'手臂幾何設定已改變，請建立新的校準工作階段')
    def run(j):
        if j['cancel'].is_set():raise RuntimeError('已停止')
        robot.connect()
        frame=grasp.capture()
        if j['cancel'].is_set():raise RuntimeError('已停止')
        result=calibration_sessions.capture(session_id,
            {**frame,'camera_id':CONFIG['wrist_camera_id']},board_static=req.board_static)
        with job_lock:j['calibration_result']=result
        report(j,result.get('message') or '已保存校準影像','done')
    return submit('calibration-capture',run,initial_stage='observing',
                  initial_message='等手臂停穩，拍下棋盤格與馬達角度')

@app.post('/api/engineering/calibration/sessions/{session_id}/solve')
def calibration_solve(session_id:str):
    try:session=calibration_sessions.status(session_id)
    except ValueError as exc:raise HTTPException(404,str(exc))
    from .spatial_runtime import geometry_revision
    if session['geometry_revision']!=geometry_revision(grasp.model):
        raise HTTPException(409,'手臂幾何設定已改變，這組照片只能保留作比較')
    def run(j):
        result=calibration_sessions.solve(session_id,grasp.model,cancel=j['cancel'])
        with job_lock:j['calibration_result']=result
        report(j,result.get('message') or '校準計算完成，結果尚未啟用','done')
    return submit('calibration-solve',run,allow_stopped=True,
                  initial_stage='measuring',initial_message='用照片計算座標，再以保留視角驗證')

@app.get('/api/world-state')
def world_state():
    # Cached evidence only: opening this endpoint never captures or moves.
    return world_memory.latest(camera_generation=camera.generation,camera_fresh=camera.fresh())

@app.post('/api/engineering/calibration/sessions/{session_id}/extend')
def calibration_extend(session_id:str):
    # Reuse the immutable capture journal and explicit activation boundary.
    # This is an offline solve: no connect, capture, motor goals or cloud call.
    from .spatial_runtime import geometry_revision, _read_profile
    try:
        session=calibration_sessions.status(session_id)
        if session['geometry_revision']!=geometry_revision(grasp.model):
            raise ValueError('手臂模型已改變，這批照片只能保留作比較')
        if not CONFIG.get('spatial_calibration_profile'):
            raise ValueError('請先完成一次相機校準，再驗證其他手臂姿勢')
        profile=_read_profile(CONFIG,'spatial_calibration_profile')
        for key,expected in (('camera_id',CONFIG['wrist_camera_id']),
                ('mount_revision',CONFIG.get('camera_mount_revision','existing-mount-unverified')),
                ('geometry_revision',geometry_revision(grasp.model))):
            if session[key]!=expected or profile.get(key)!=expected:
                raise ValueError('校準照片、相機設定與目前手臂模型不一致')
    except (ValueError,KeyError,OSError,TypeError) as exc:raise HTTPException(409,str(exc))
    def run(j):
        result=calibration_sessions.solve(session_id,grasp.model,cancel=j['cancel'],frozen_profile=profile)
        with job_lock:j['calibration_result']=result
        report(j,result.get('message') or '工作範圍驗證完成，尚未啟用','done')
    return submit('calibration-workspace-validation',run,allow_stopped=True,
        initial_stage='measuring',initial_message='保留相機參數，檢查新姿勢的實測誤差')

@app.get('/api/skills')
def skills():
    from .skill_plan import skill_catalogue
    from .motion_policy import minimum_visual_confidence
    return {**skill_catalogue(),'tactics':tactician.status(),
            'confidence_policy':{k:minimum_visual_confidence(k) for k in ('local_move','wave','close','grasp_verify')}}

@app.post('/api/command')
def command(req:CommandRequest):
    if not req.instruction.strip():raise HTTPException(422,'請先輸入想做的動作')
    def run(j):
        if getattr(brain,'_is_gemini',lambda:False)() and is_natural_motion_query(req.instruction):
            if j['cancel'].is_set():raise RuntimeError('已停止')
            plan=plan_single_arm_motion(req.instruction,model=grasp.model)
            with job_lock:
                j['motiongpt_plan']=plan
            robot.connect()
            robot.begin()
            receipt=execute_retargeted_motion(robot,plan,progress=lambda stage,msg:report(j,msg,stage))
            cmd_res={
                'plan':{'status':'ready','summary':plan['summary_zh'],'unsupported_reason':'','steps':[{'skill':'wave','direction':'none','distance_mm':0.0,'target':'','label':plan['title_zh']}]},
                'completed':True,
                'status':'completed',
                'steps':[{'index':0,'skill':'wave','label':plan['title_zh'],'completed':True,'status':'completed','receipt':receipt}],
            }
            with job_lock:
                j['motiongpt_result']=receipt
                j['engineering_result']=receipt
                j['command_result']=cmd_res
            report(j,f"已完成單手動作：{plan['title_zh']}",'done')
            return
        from .skill_plan import plan_skills
        from .skill_runtime import SkillRunner
        # An unavailable image can reject a new command before touching the
        # stationary robot capture/hold path.
        if j['cancel'].is_set():raise RuntimeError('已停止')
        if not camera.fresh():raise RuntimeError(camera.error or '沒有即時腕部畫面')
        report(j,'Argon ER 2 正在把你的指令拆成動作','planning')
        # A fresh backend has no serial protocol until connect reads the
        # device/calibration. Planning must not rearm or rewrite motor goals.
        # Each later skill still owns its begin/low-power-release checks.
        robot.connect()
        if j['cancel'].is_set():raise RuntimeError('已停止')
        frame=grasp.capture()
        state=world_memory.remember(frame)
        plan=plan_skills(brain,req.instruction,frame,j['cancel'],world_state=state)
        def publish(result):
            with job_lock:j['command_result']=copy.deepcopy(result)
        runner=SkillRunner(grasp,brain,pick_enabled=bool(CONFIG.get('experimental_pick_enabled')),tactician=tactician)
        result=runner.execute(plan,lambda m,s='moving':report(j,m,s),j['cancel'],publish)
        report(j,'指定動作已完成' if result['completed'] else result.get('reason','這個動作目前還不能執行'),'done')
    return submit('command',run)

class GraspReleaseRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    frame_id:str
    support_verified:bool=False
    empty_verified:bool=False

@app.post('/api/grasp-release')
def grasp_release(req:GraspReleaseRequest):
    with frame_lock:
        f=frames.get(req.frame_id)
        if not f or time.time()-f['captured_at']>30 or 'joints' not in f:
            raise HTTPException(409,'需要新的腕部影像與實測姿勢')
    def run(j):
        j['engineering_result']=grasp.release(f,lambda m:report(j,m,'moving'),j['cancel'],support_verified=req.support_verified,empty_verified=req.empty_verified)
        report(j,'已完成釋放並核對空爪','done')
    return submit('grasp-release',run,allow_stopped=True)

@app.post('/api/grasp-open')
def grasp_open(req:EmptyGripperRequest):
    with frame_lock:
        f=frames.get(req.frame_id)
        if not f or time.time()-f['captured_at']>30 or 'joints' not in f:
            raise HTTPException(409,'需要新的腕部影像與實測姿勢')
    def run(j):
        j['engineering_result']=grasp.open_empty(f,lambda m:report(j,m,'moving'))
        report(j,'夾爪已張開','done')
    return submit('grasp-open',run)

@app.post('/api/grasp-attempt')
def grasp_attempt(req:SearchRequest):
    if not CONFIG.get('experimental_pick_enabled',False):raise HTTPException(409,'實驗抓取目前未開啟')
    if CONFIG.get('grasp_backend')=='kinematic':
        # Fitted kinematic model: home, look around, plan locally, then pick up, put back, release, retreat.
        def run(j):
            planned=kinematic_grasp.plan(req.query,lambda m:report(j,m,'planning'),j['cancel'])
            j['engineering_result']=planned
            if not planned['executable']:raise RuntimeError('抓取路徑有問題，未執行：'+'；'.join(planned['problems'])[:300])
            value=kinematic_grasp.execute(planned['plan_id'],lambda m:report(j,m,'moving'),j['cancel'])
            j['engineering_result']={**value,'task':'grasp','plan':planned}
            report(j,'已拿起物件並放回桌面' if value.get('grasp_confirmed') else '動作完成，但未確認拿起','done')
        return submit('grasp-attempt',run)
    def run(j):
        execute_object_program(j,req.query)
        report(j,'已拿起物件','done')
    return submit('grasp-attempt',run)

class KinematicGraspRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    query:str=Field(min_length=1,max_length=200)

class KinematicGraspExecute(BaseModel):
    model_config=ConfigDict(extra='forbid')
    plan_id:str=Field(min_length=1,max_length=80)
    return_object:bool=True

@app.post('/api/kinematic-grasp/plan')
def kinematic_grasp_plan(req:KinematicGraspRequest):
    """Observation motions only (home, look around, one closer view); returns a grasp plan for review."""
    if not CONFIG.get('kinematic_model'):raise HTTPException(409,'尚未設定運動學模型')
    def run(j):
        j['engineering_result']=kinematic_grasp.plan(req.query,lambda m:report(j,m,'planning'),j['cancel'])
        report(j,'抓取計畫完成，等待確認','done')
    return submit('grasp-attempt',run)

@app.post('/api/kinematic-grasp/execute')
def kinematic_grasp_execute(req:KinematicGraspExecute):
    if not CONFIG.get('experimental_pick_enabled',False):raise HTTPException(409,'實驗抓取目前未開啟')
    def run(j):
        value=kinematic_grasp.execute(req.plan_id,lambda m:report(j,m,'moving'),j['cancel'],return_object=req.return_object)
        j['engineering_result']=value
        report(j,'已拿起物件並放回' if value.get('grasp_confirmed') else '動作完成，但未確認拿起','done')
    return submit('grasp-attempt',run)

@app.post('/api/grasp-step')
def grasp_step(req:GraspStepRequest):
    with frame_lock:
        reference=frames.get(req.frame_id)
        if not reference or time.time()-reference['captured_at']>30 or 'joints' not in reference:
            raise HTTPException(409,'請先檢查新的腕部畫面與實測姿勢')
    def run(j):
        result=grasp.move(req.direction,req.distance_mm,reference,lambda m:report(j,m,'moving'))
        j['engineering_result']=result
        report(j,'小步移動完成，已更新影像與實際角度','done')
    return submit('grasp-step',run)

@app.post('/api/grasp-observation')
def grasp_observation():
    with job_lock:
        if active and jobs[active]['status']=='running':raise HTTPException(409,'目前正在執行動作')
        robot.connect()
        if robot.stop.is_set():
            before=robot.read();f=snap();f['joints']=robot.read()
            if max(abs(before[n]-f['joints'][n]) for n in before)>2:raise HTTPException(409,'手臂仍在移動')
        else:
            robot.begin(allow_rearm=False);f=grasp.capture()
        return {k:v for k,v in f.items() if k!='path'}

@app.get('/api/gripper/telemetry')
def gripper_telemetry():
    try:return robot.gripper_telemetry()
    except RuntimeError as e:raise HTTPException(503,str(e))
@app.get('/api/robot/diagnostics')
def robot_diagnostics():
    with job_lock:
        if active and jobs[active]['status']=='running':raise HTTPException(409,'請在目前工作結束後讀取馬達診斷')
        try:return robot.joint_diagnostics()
        except RuntimeError as e:raise HTTPException(503,str(e))

@app.get('/api/robot/onboard-calibration')
def robot_onboard_calibration():
    with job_lock:
        if active and jobs[active]['status']=='running':
            raise HTTPException(409,'請等目前工作結束，再讀取機內馬達校準')
        try:return robot.read_onboard_calibration()
        except RuntimeError as e:raise HTTPException(503,str(e))
@app.get('/api/camera/frame')
def camera_frame():
    try:return Response(camera.jpeg(),media_type='image/jpeg')
    except RuntimeError as e:raise HTTPException(503,str(e))
@app.get('/api/camera/stream')
async def stream():
    async def generate():
        while True:
            try:data=camera.jpeg()
            except RuntimeError:await asyncio.sleep(.3);continue
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+data+b'\r\n';await asyncio.sleep(.12)
    return StreamingResponse(generate(),media_type='multipart/x-mixed-replace; boundary=frame')
@app.post('/api/camera/snapshot')
def snapshot():
    try:return {k:v for k,v in snap().items() if k!='path'}
    except RuntimeError as e:raise HTTPException(503,str(e))
@app.get('/api/frames/{fid}')
def frame(fid:str):
    f=frames.get(fid)
    if not f:raise HTTPException(404,'照片已失效')
    return FileResponse(f['path'],media_type='image/jpeg')
@app.post('/api/search')
def search(req:SearchRequest):
    text_only=text_intent_search(req)
    return submit(req.mode,lambda j:run_search(req,j),allow_stopped=text_only,allow_holding=text_only,
                  initial_stage='understanding' if text_only else 'starting',
                  initial_message='正在理解你的搜尋' if text_only else '準備看看眼前的世界')
@app.get('/api/jobs/{jid}')
def job(jid:str):
    if jid not in jobs:raise HTTPException(404,'工作不存在')
    return public_job(jobs[jid])
@app.post('/api/stop')
def stop():
    robot.stop.set()
    if active and active in jobs:jobs[active]['cancel'].set()
    contact_error=None
    try:
        if robot.pwm_owner is not None and robot.gripper_power.state in ('active','holding','releasing'):
            record=robot.gripper_power._record()
            if record and record.get('contact_possible'):robot.gripper_power.stop_contact(robot.pwm_owner)
    except Exception as exc:
        robot.gripper_power.leave_pending();contact_error=str(exc)
    finally:result=robot.hold()
    return {'stopped':True,**result,'contact_stop_error':contact_error}
@app.post('/api/engineering/maintenance-quiesce')
def maintenance_quiesce():
    """Maintenance-only admission latch; ordinary emergency stop is unchanged."""
    with job_lock:
        if (any(j.get('status')=='running' for j in jobs.values())
                or robot.moving or robot.pwm_owner is not None
                or grasp.state not in ('idle','approaching')):
            raise HTTPException(409,'請先完成目前工作與持物流程，再重新啟動控制器')
        try:return robot.maintenance_quiesce()
        except (RuntimeError,ValueError) as exc:raise HTTPException(409,str(exc)) from exc
@app.post('/api/resume')
def resume():
    with job_lock:
        if getattr(robot,'maintenance_latched',False):raise HTTPException(409,'控制器正在準備重新啟動，請等重新啟動完成後再恢復')
        if active and active in jobs and jobs[active]['status']=='running':raise HTTPException(409,'正在停止，請稍候')
        try:robot.resume()
        except RuntimeError as e:raise HTTPException(409,str(e))
    return {'stopped':False}
@app.post('/api/look')
def look():return submit('look',lambda j:(report(j,'抬起手腕看看你','moving'),robot.gesture('look'),report(j,'可以點選眼前的物件','done')))
@app.post('/api/robot/home')
def robot_home():
    with job_lock:
        if active and active in jobs and jobs[active]['status']=='running' and not robot.moving:
            jobs[active]['cancel'].set()
            jobs[active]['status']='cancelled'
            jobs[active]['message']='已切換至歸回原位'
    def run(j):
        receipt=robot.go_home(report=lambda m:report(j,m,'moving'))
        j['engineering_result']=receipt
        report(j,'已歸回正前方預設原位 (shoulder_pan=2030)','done')
    return submit('home',run,initial_stage='moving',initial_message='正在將手臂歸回正前方預設原位…')

def run_inspection(query,j):
    report(j,'重新拍一張，看看物件的細節','observing')
    frame=snap(j)
    report(j,'Argon ER 2 正在讀取外觀、文字與可確認的細節','thinking')
    details=brain.inspect_details(query,[frame],j['cancel'])
    objects=[]
    if details['visible'] and details['bbox']:
        objects=[{'name':details['object_name'],'description':details['summary'],'confidence':details['confidence'],
                  'bbox':details['bbox'],'frame_index':0,'matches_query':True,
                  'safe_to_touch':details['safe_to_touch'],'actions':details['actions']}]
    scene=enrich({'query':query,'summary':details['summary'],'objects':objects,
                  'selected_index':0 if objects else -1,'empty':not objects,
                  'uncertainty':'；'.join(details['unknowns'])},[frame])
    scene['details']={**details,'frame':{k:v for k,v in frame.items() if k!='path'}}
    scene['elapsed']=round(time.time()-j['created_at'],1)
    publish_scene(j,scene);report(j,'細節已更新','done')

@app.post('/api/object-action')
def object_action(req:ObjectActionRequest):
    # Names are fresh semantic requests, never cached pixel/motor coordinates.
    if req.kind!='wave' and not req.query.strip():raise HTTPException(422,'請選擇要互動的物件')
    if req.kind=='inspect':return submit('inspect',lambda j:run_inspection(req.query,j),allow_stopped=True)
    if req.kind=='pick':return grasp_attempt(SearchRequest(query=req.query))
    if req.kind in ('place','stack'):
        if not CONFIG.get('experimental_pick_enabled'):raise HTTPException(409,'實驗物件操作目前未啟用')
        def manipulate(j):
            execute_object_program(j,req.query,req.kind)
            report(j,'已確認堆疊完成' if req.kind=='stack' else '已確認物件放到目的地','done')
        return submit('object-program',manipulate)
    def run(j):
        def wave_report(message):
            with job_lock:
                if message.startswith('向你揮揮手'):j['manual_wave_phase']='waving'
                report(j,message,'moving')
        receipt=grasp.planned_wave(wave_report,j['cancel'],manual_override=j['manual_wave_signal'])
        j['engineering_result']=receipt;report(j,'揮手完成','done')
    return submit('wave',run)

@app.post('/api/jobs/{jid}/force-wave')
def force_wave(jid:str,req:ManualWaveRequest):
    """Human-only, one-job override; the existing worker remains sole motor owner."""
    with job_lock:
        j=jobs.get(jid)
        if not j:raise HTTPException(404,'找不到這次揮手，請重新開始')
        if jid!=active or j['kind']!='wave' or j['status']!='running':
            raise HTTPException(409,'只能接手目前正在準備的揮手')
        if robot.stop.is_set() or j['cancel'].is_set():
            raise HTTPException(409,'機械臂已停止；請按恢復後重新開始')
        if robot.pwm_owner is not None:
            raise HTTPException(409,'夾爪正保持物件，請先完成目前工作')
        if j['manual_wave_requested']:return public_job(j)
        if j['manual_wave_phase']=='waving':raise HTTPException(409,'已經在揮手，不需要強制執行')
        j.update(manual_wave_requested=True,manual_wave_requested_at=time.time())
        j['manual_wave_signal'].set()
        report(j,'依你的指示，略過障礙物判斷；完成目前這段動作後原地揮手。','moving')
        return public_job(j)

@app.post('/api/actions')
def action(req:ActionRequest):
    s=scenes.get(req.scene_id)
    if not s or time.time()-s['created_at']>180:raise HTTPException(409,'場景已過期，請重新搜尋')
    try:o=s['objects'][req.object_index];a=o['actions'][req.action_index]
    except IndexError:raise HTTPException(422,'無效的動作')
    if not a['available']:raise HTTPException(409,a['unavailable_reason'])
    if a['kind'] in ('pick','place','stack') and not o.get('safe_to_touch',False):
        raise HTTPException(409,'這個物件目前不適合抓取或搬放')
    if a['kind']=='pick':
        if not CONFIG.get('experimental_pick_enabled',False):raise HTTPException(409,'實驗抓取目前未開啟')
        def run(j):
            publish_scene(j,copy.deepcopy(s))
            result=execute_object_program(j,o['name'])
            updated=copy.deepcopy(s);updated['grasp']=result
            updated['performed']=f"已拿起{o['name']}"
            updated['robot_revision']=robot.revision;updated['completed_at']=time.time()
            publish_scene(j,updated);report(j,updated['performed'],'done')
        return submit('grasp-attempt',run)
    if a['kind']=='inspect':return submit('inspect',lambda j:run_inspection(o['name'],j),allow_stopped=True)
    if a['kind']=='wave':return object_action(ObjectActionRequest(kind='wave'))
    if a['kind'] in ('place','stack'):
        destination=a.get('target','').strip()
        if not destination:raise HTTPException(422,'請在搜尋框說明物件要放到哪裡')
        return object_action(ObjectActionRequest(kind=a['kind'],query=f"把{o['name']}放到{destination}"))
    raise HTTPException(422,'不支援的動作')
@app.post('/api/upload')
def upload(req:UploadRequest):
    try:
        data=base64.b64decode(req.image.split(',')[-1],validate=True)
        if len(data)>8_000_000:raise ValueError('照片太大，請選擇 8 MB 以下的照片')
        f=cv2.imdecode(np.frombuffer(data,dtype=np.uint8),cv2.IMREAD_COLOR)
        if f is None and b'ftyp' in data[:32]:
            temporary=RUNTIME/'uploads';temporary.mkdir(exist_ok=True);stem=uuid.uuid4().hex
            heic=temporary/f'{stem}.heic';jpg=temporary/f'{stem}.jpg';heic.write_bytes(data)
            try:
                subprocess.run(['sips','-s','format','jpeg',str(heic),'--out',str(jpg)],check=True,timeout=15,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                f=cv2.imread(str(jpg))
            finally:heic.unlink(missing_ok=True);jpg.unlink(missing_ok=True)
        if f is None:raise ValueError('請使用 JPEG、PNG、WebP 或 HEIC 照片')
        if max(f.shape[:2])>1600:f=cv2.resize(f,(int(f.shape[1]*1600/max(f.shape[:2])),int(f.shape[0]*1600/max(f.shape[:2]))))
        uid=uuid.uuid4().hex;path=RUNTIME/'uploads'/f'{uid}.jpg';path.parent.mkdir(exist_ok=True);cv2.imwrite(str(path),f);uploads[uid]=path
        while len(uploads)>20:uploads.pop(next(iter(uploads))).unlink(missing_ok=True)
        return {'id':uid,'url':f'/api/uploads/{uid}'}
    except Exception as e:raise HTTPException(422,str(e))

@app.get('/api/uploads/{uid}')
def reference_image(uid:str):
    path=uploads.get(uid)
    if path is None:raise HTTPException(404,'參考照片已過期')
    return FileResponse(path,media_type='image/jpeg')

class MotionGPTPlanRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    query:str=Field(default='揮揮手',min_length=1,max_length=300)
    preferred_arm:Literal['auto','right','left']='auto'

class ER2SpatialRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    query:str=Field(default='',max_length=300)
    frame_id:str|None=None
    thinking_level:Literal['none','low','medium','high']|None=None

class BrainConfigRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    engine:Literal['astra','argon']|None=None
    thinking_level:Literal['none','low','medium','high']|None=None
    gemini_api_key:str|None=Field(default=None,max_length=500)

def _update_env_local_gemini_key(api_key:str)->None:
    env_path=ROOT/'.env.local'
    lines=[]
    if env_path.is_file():
        lines=[ln for ln in env_path.read_text().splitlines() if not ln.strip().startswith('GEMINI_API_KEY=')]
    if api_key:
        lines.append(f'GEMINI_API_KEY={api_key}')
        os.environ['GEMINI_API_KEY']=api_key
    else:
        os.environ.pop('GEMINI_API_KEY',None)
    env_path.write_text('\n'.join(lines)+('\n' if lines else ''))

@app.get('/api/brain/config')
def get_brain_config():
    return {'brain':brain.status(),'er2':er2_brain.status_dict()}

@app.post('/api/brain/config')
def update_brain_config(req:BrainConfigRequest):
    if req.gemini_api_key is not None:
        _update_env_local_gemini_key(req.gemini_api_key.strip())
    if req.engine is not None:
        brain.engine_override=req.engine
    if req.thinking_level is not None:
        brain.thinking_level_override=req.thinking_level
        er2_brain.default_thinking_level=req.thinking_level
    try:
        (RUNTIME/'brain-config.json').write_text(json.dumps({
            'engine':brain.engine(),
            'thinking_level':brain.thinking_level(),
        },ensure_ascii=False,indent=2)+'\n')
    except OSError:
        pass
    brain.last_error='';brain.error_kind=''
    return {'brain':brain.status(),'er2':er2_brain.status_dict()}

@app.get('/api/motiongpt/presets')
def motiongpt_presets():
    return {'presets':list(MOTION_PRESETS.values())}

@app.post('/api/motiongpt/plan')
def motiongpt_plan(req:MotionGPTPlanRequest):
    """Generate a HumanML3D 22-joint clip and retarget the active single arm to SO-101 without motor I/O."""
    try:
        return plan_single_arm_motion(req.query,preferred_arm=req.preferred_arm,model=grasp.model)
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc

@app.post('/api/motiongpt/execute')
def motiongpt_execute(req:MotionGPTPlanRequest):
    """Plan and execute a single-arm retargeted MotionGPT trajectory on the SO-101 arm."""
    try:
        plan=plan_single_arm_motion(req.query,preferred_arm=req.preferred_arm,model=grasp.model)
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc
    def run(j):
        with job_lock:
            j['motiongpt_plan']=plan
        if j['cancel'].is_set():raise RuntimeError('已停止')
        robot.connect()
        robot.begin()
        receipt=execute_retargeted_motion(robot,plan,progress=lambda stage,msg:report(j,msg,stage))
        with job_lock:
            j['motiongpt_result']=receipt
            j['engineering_result']=receipt
        report(j,f"已完成單手動作：{plan['title_zh']}",'done')
    return submit('motiongpt',run,initial_stage='planning',initial_message=f"正在重映射單手動作：{plan['title_zh']}")

@app.post('/api/er2/spatial')
def er2_spatial(req:ER2SpatialRequest):
    """Run Gemini Robotics ER 2 (`gemini-robotics-er-2-preview`) 0-1000 pointing, boxes, and 3D table projection."""
    if req.frame_id:
        with frame_lock:
            f=frames.get(req.frame_id)
        if not f:raise HTTPException(404,'指定的腕部照片已過期')
    else:
        try:
            f=snap()
        except RuntimeError as exc:
            # If the live wrist camera is temporarily asleep/disconnected, fall back to the latest captured wrist frame
            with frame_lock:
                f=list(frames.values())[-1] if frames else None
            if not f:
                saved_jpgs=sorted((RUNTIME/'frames').glob('*.jpg'),key=lambda p:p.stat().st_mtime)
                if saved_jpgs:
                    latest_p=saved_jpgs[-1]
                    fid=latest_p.stem
                    f={'id':fid,'path':str(latest_p),'url':f'/api/frames/{fid}','captured_at':latest_p.stat().st_mtime,'source':'wrist'}
                    with frame_lock:frames[fid]=f
                else:
                    raise HTTPException(503,str(exc)) from exc
    try:
        effective_tl=req.thinking_level or brain.thinking_level()
        result=er2_brain.detect_spatial_targets(f['path'],req.query,thinking_level=effective_tl)
        # If calibrated 3D table-ray model and measured joints are available, enrich with metric 3D coordinates
        joints=f.get('joints')
        if not joints:
            try:joints=robot.pose() if not robot.moving else dict(robot.position)
            except Exception:joints=dict(robot.position) if robot.position else None
        if not joints:
            from .robot import HOME_POSE
            joints=dict(HOME_POSE)
        if joints:
            try:
                from .kinematic_grasp import Geometry, cylinder_from_feet
                geo=Geometry(ROOT,CONFIG)
                for item in result.get('items',[]):
                    try:
                        feet=item['feet_px']
                        loc=cylinder_from_feet(
                            geo,
                            joints,
                            feet['bottom_left_px'],
                            feet['bottom_right_px'],
                            feet.get('top_center_px'),
                        )
                        item['metric_3d']={
                            'center_m':[round(float(v),4) for v in loc['centre']],
                            'radius_mm':round(float(loc['radius'])*1000.0,1),
                            'height_mm':round(float(loc['height'] or 0.0)*1000.0,1),
                        }
                    except Exception:
                        pass
            except Exception:
                pass
        result['frame']={k:v for k,v in f.items() if k!='path'}
        return result
    except Exception as exc:
        raise HTTPException(502,str(exc)) from exc

