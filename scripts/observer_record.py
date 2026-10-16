"""Archive an existing D435 observer stream for human/VLM engineering review.

This process only reads the standalone observer JPEG. It has no robot API,
model client, motor access, or connection to the production perception path.
Capture times below are host file-arrival times, not sensor exposure times.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time

import cv2
import numpy as np


def record(latest, output, seconds, fps=5.):
    output.mkdir(parents=True, exist_ok=False)
    started=time.monotonic();wall_ns=time.time_ns();wall=wall_ns/1e9;stop=False;last=None;count=0
    def finish(*_):
        nonlocal stop
        stop=True
    old={sig:signal.signal(sig,finish) for sig in (signal.SIGINT,signal.SIGTERM)}
    metadata={'version':1,'source':'engineering_observer','production_eligible':False,
              'purpose':'external visual verification, not robot control',
              'timestamp_reference':'host_file_arrival_not_sensor_exposure',
              'started_wall_time':wall,'started_wall_time_ns':wall_ns,'started_monotonic':started,
              'maximum_duration_s':seconds,'maximum_fps':fps}
    (output/'session.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(str(output),flush=True)
    try:
        with (output/'frames.jsonl').open('w') as log:
            while not stop and time.monotonic()-started<seconds:
                before=time.monotonic()
                try:
                    stamp=latest.stat().st_mtime_ns
                    # Never archive a cached pre-session image as a live frame.
                    if stamp!=last and stamp>=wall_ns and 0<=time.time_ns()-stamp<=1_000_000_000:
                        data=latest.read_bytes()
                        if latest.stat().st_mtime_ns!=stamp:continue
                        frame=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
                        if frame is not None:
                            name=f'frame-{count:06d}.jpg'
                            (output/name).write_bytes(data)
                            entry={'file':name,'index':count,'source':'engineering_observer',
                                   'host_wall_time':time.time(),'host_monotonic':time.monotonic(),
                                   'file_mtime_ns':stamp,'sha256':hashlib.sha256(data).hexdigest(),
                                   'image_size':[int(frame.shape[1]),int(frame.shape[0])]}
                            log.write(json.dumps(entry)+'\n');log.flush()
                            last=stamp;count+=1
                except FileNotFoundError:pass
                time.sleep(max(.005,1/fps-(time.monotonic()-before)))
    finally:
        for sig,handler in old.items():signal.signal(sig,handler)
        metadata.update(frame_count=count,finished_wall_time=time.time(),
                        elapsed_s=time.monotonic()-started,interrupted=stop,
                        physical_success_verified=False)
        (output/'session.json').write_text(json.dumps(metadata,indent=2)+'\n')
    if not count:raise RuntimeError('No fresh observer images were received')
    # Relative files only; ffmpeg can preserve the actual sampled timing.
    entries=[json.loads(line) for line in (output/'frames.jsonl').read_text().splitlines()]
    timeline=[]
    for index,entry in enumerate(entries):
        duration=(entries[index+1]['host_monotonic']-entry['host_monotonic']) if index+1<len(entries) else 1/fps
        timeline.extend([f"file '{entry['file']}'",f'duration {duration:.6f}'])
    timeline.append(f"file '{entries[-1]['file']}'")
    (output/'video.ffconcat').write_text('ffconcat version 1.0\n'+'\n'.join(timeline)+'\n')
    print(json.dumps({'frames':count,'elapsed_s':round(metadata['elapsed_s'],2),
                      'source':'engineering_observer','physical_success_verified':False}),flush=True)


def encode_video(output):
    """Optional local export, preserving the archived host frame intervals."""
    import imageio_ffmpeg
    video=output/'observer.mp4'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-n','-safe','1',
                    '-i',str(output/'video.ffconcat'),'-c:v','libx264',
                    '-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
    return video


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--latest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=300)
    parser.add_argument('--fps',type=float,default=5)
    parser.add_argument('--encode-video',action='store_true',help='Export H.264 with the optional imageio-ffmpeg package')
    args=parser.parse_args()
    if not .1<=args.seconds<=1800 or not 1<=args.fps<=10:parser.error('Use 0.1–1800 seconds and 1–10 fps')
    record(args.latest,args.output,args.seconds,args.fps)
    if args.encode_video:print(str(encode_video(args.output)),flush=True)


if __name__=='__main__':main()
