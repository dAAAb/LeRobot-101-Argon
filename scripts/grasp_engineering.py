"""Local engineering client. No serial access; server retains sole ownership."""
import json,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE='http://127.0.0.1:8102'
def request(path,body=None):
    headers={}
    if body is not None:
        token=json.load(urllib.request.urlopen(BASE+'/api/session',timeout=5))['token']
        headers={'content-type':'application/json','x-lerobot-token':token}
    r=urllib.request.Request(BASE+path,data=None if body is None else json.dumps(body).encode(),headers=headers)
    with urllib.request.urlopen(r,timeout=15) as response:return json.load(response)
if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='observe':
        f=request('/api/grasp-observation',{});(ROOT/'.runtime/grasp-observation.json').write_text(json.dumps(f))
        data=urllib.request.urlopen(BASE+f['url'],timeout=5).read();(ROOT/'.runtime/grasp-observation.jpg').write_bytes(data)
        print(json.dumps(f,ensure_ascii=False))
    elif mode=='step':
        f=json.loads((ROOT/'.runtime/grasp-observation.json').read_text())
        print(json.dumps(request('/api/grasp-step',{'direction':sys.argv[2],'distance_mm':float(sys.argv[3]),'frame_id':f['id'],'clear_path_verified':True}),ensure_ascii=False))
    elif mode=='attempt':print(json.dumps(request('/api/grasp-attempt',{'query':sys.argv[2] if len(sys.argv)>2 else '白黃圓筒'}),ensure_ascii=False))
    elif mode=='open':
        f=json.loads((ROOT/'.runtime/grasp-observation.json').read_text())
        print(json.dumps(request('/api/grasp-open',{'frame_id':f['id'],'empty_verified':True}),ensure_ascii=False))
    elif mode=='release-empty':
        f=json.loads((ROOT/'.runtime/grasp-observation.json').read_text())
        print(json.dumps(request('/api/grasp-release',{'frame_id':f['id'],'empty_verified':True}),ensure_ascii=False))
    elif mode=='live':
        path=ROOT/'.runtime/grasp-live.jpg'
        path.write_bytes(urllib.request.urlopen(BASE+'/api/camera/frame',timeout=5).read())
        print(path)
    elif mode=='resume':print(json.dumps(request('/api/resume',{}),ensure_ascii=False))
    elif mode=='status':
        s=request('/api/status');(ROOT/'.runtime/grasp-status.json').write_text(json.dumps(s,ensure_ascii=False))
        print(json.dumps({'robot':s['robot'],'job':{k:v for k,v in (s.get('job') or {}).items() if k not in ('result','engineering_result','events')}},ensure_ascii=False))
