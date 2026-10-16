"""Production vision input: one explicitly identified wrist camera, never an observer."""
from __future__ import annotations
import threading, time, uuid
from collections import deque
from pathlib import Path
import cv2, numpy as np


def frame_quality(frame):
    """Reject globally near-black input, not a dark object in a visible scene.

    The 95th percentile ignores isolated bright/hot pixels. Both low mean and
    low p95 are required, so a dark object with brighter context remains usable.
    This is an exposure gate, not proof of object visibility or grasp clearance.
    """
    if (not isinstance(frame,np.ndarray) or frame.dtype!=np.uint8 or
            frame.ndim!=3 or frame.shape[2]!=3 or not frame.size):
        return {'usable':False,'status':'decode_failed','reason':'腕部相機影像無法解碼'}
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    histogram=cv2.calcHist([gray],[0],None,[256],[0,256]).ravel()
    count=gray.size;cumulative=histogram.cumsum()
    mean=float(histogram.astype(np.float64)@np.arange(256)/count)
    p95=int(np.searchsorted(cumulative,count*.95))
    p99=int(np.searchsorted(cumulative,count*.99))
    usable=not(mean<24 and p95<32)
    return {'usable':usable,'status':'usable' if usable else 'too_dark',
            'reason':'' if usable else '腕部相機畫面太暗，請確認照明與鏡頭是否被遮住',
            'mean_luma':round(mean,2),'p95_luma':p95,'p99_luma':p99,
            'bright_fraction':round(float(histogram[32:].sum()/count),6)}

class WristCamera:
    def __init__(self, config, directory):
        self.config=config; self.directory=Path(directory); self.directory.mkdir(parents=True,exist_ok=True)
        self.lock=threading.RLock(); self.quit=threading.Event(); self.frame=None; self.timestamp=0.; self.error='相機連接中'; self.name=''
        self.sequence=0;self.history=deque(maxlen=500);self.generation=None
        self.image_quality={'usable':False,'status':'waiting','reason':'尚未取得腕部影像'}
    def start(self):
        self.thread=threading.Thread(target=self._capture,daemon=True,name='wrist-camera');self.thread.start()
    def _capture(self):
        import subprocess
        root=Path(__file__).resolve().parents[1]
        latest=self.directory.parent/'wrist-live.jpg'
        helper=root/'.runtime'/'wrist-capture'
        while not self.quit.is_set():
            process=None
            try:
                latest.unlink(missing_ok=True)
                process=subprocess.Popen([str(helper),self.config['wrist_camera_name'],self.config['wrist_camera_id'],str(latest),str(self.config.get('wrist_capture_width',960))],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
                generation=uuid.uuid4().hex
                self.name=self.config['wrist_camera_name'];last=0;last_arrival=time.monotonic()
                while not self.quit.wait(.06):
                    if process.poll() is not None:
                        detail=process.stderr.read().decode(errors='replace')[-500:]
                        raise RuntimeError('腕部相機無法開啟：'+detail)
                    if time.monotonic()-last_arrival>5:raise RuntimeError('腕部相機沒有新畫面，重新連接')
                    if not latest.exists():continue
                    stamp=latest.stat().st_mtime_ns
                    if stamp==last:continue
                    last=stamp;last_arrival=time.monotonic()
                    self._accept_frame(cv2.imread(str(latest)),generation,last_arrival)
            except Exception as e:
                with self.lock:self._invalidate(str(e),'capture_error')
            finally:
                if process and process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=2)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
            self.quit.wait(2)
    def _invalidate(self,reason,status):
        # Call only under self.lock. Never leave the last good image eligible
        # for a snapshot after observing an unusable arrival or capture failure.
        self.frame=None;self.timestamp=0.;self.error=reason
        self.image_quality={'usable':False,'status':status,'reason':reason}
    def _accept_frame(self,frame,generation,captured_monotonic=None):
        with self.lock:
            arrival=time.monotonic() if captured_monotonic is None else captured_monotonic
            self.sequence+=1;self.generation=generation
            quality=frame_quality(frame)
            if not quality['usable']:
                self._invalidate(quality['reason'],quality['status']);self.image_quality=quality
                return False
            try:ok,encoded=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,94])
            except cv2.error:
                self._invalidate('相機影像編碼失敗','encode_failed');return False
            if not ok:
                self._invalidate('相機影像編碼失敗','encode_failed');return False
            self.frame=frame;self.timestamp=arrival;self.error='';self.image_quality=quality
            self.history.append({'sequence':self.sequence,'jpeg':encoded.tobytes(),'source':'wrist',
                                 'capture_monotonic':arrival,'camera_generation':generation})
            return True
    def status(self):
        with self.lock:
            return {'connected':self.fresh(),'name':self.name,'error':self.error,'source':'wrist',
                    'age_ms':round((time.monotonic()-self.timestamp)*1000) if self.timestamp else None,
                    'image_quality':dict(self.image_quality)}
    def fresh(self):
        with self.lock:
            return (self.frame is not None and self.image_quality.get('usable') is True and
                    0<=time.monotonic()-self.timestamp<1.0 and not self.error)
    def jpeg(self):
        with self.lock:
            if not self.fresh(): raise RuntimeError(self.error or '腕部相機畫面已過期')
            ok,b=cv2.imencode('.jpg',self.frame,[cv2.IMWRITE_JPEG_QUALITY,94])
            if not ok: raise RuntimeError('相機影像編碼失敗')
            return b.tobytes()
    def snapshot(self):
        with self.lock:
            data=self.jpeg();sequence=self.sequence;captured_at=time.time()-(time.monotonic()-self.timestamp)
            capture_monotonic=self.timestamp;generation=self.generation;size=[self.frame.shape[1],self.frame.shape[0]]
        sid=uuid.uuid4().hex
        path=self.directory/f'{sid}.jpg';path.write_bytes(data)
        return {'id':sid,'path':str(path),'url':f'/api/frames/{sid}','source':'wrist','captured_at':captured_at,'sequence':sequence,
                'capture_monotonic':capture_monotonic,'camera_generation':generation,'image_size':size}
    def tracking_history(self,start_sequence,end_sequence):
        with self.lock:return [dict(f) for f in self.history if start_sequence<=f['sequence']<=end_sequence]
    def latest_stamp(self):
        with self.lock:
            if not self.fresh():raise RuntimeError('腕部相機畫面已過期')
            return {'frame_sequence':self.sequence,'frame_capture_monotonic':self.timestamp,'camera_source':'wrist'}
    def close(self):
        self.quit.set()
        if hasattr(self,'thread'):self.thread.join(timeout=4)
