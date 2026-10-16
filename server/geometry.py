"""Metric multi-view reconstruction from wrist pixels and measured robot joint states.
An approximate camera mounting prior is NEVER represented as measured hand-eye calibration.
"""
from __future__ import annotations
import math, xml.etree.ElementTree as ET
from pathlib import Path
import cv2, numpy as np

def observation_families(observations):
    """Keep return visits and near-identical measured poses in one holdout fold."""
    parent=list(range(len(observations)))
    def root(i):
        while parent[i]!=i:i=parent[i]
        return i
    names=('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll')
    for i,a in enumerate(observations):
        for j,b in enumerate(observations[:i]):
            explicit=a.get('validation_group') and a.get('validation_group')==b.get('validation_group')
            near=all(abs(a['joints'][n]-b['joints'][n])<=4 for n in names)
            if explicit or near:parent[root(i)]=root(j)
    groups={}
    for i in range(len(observations)):groups.setdefault(root(i),[]).append(i)
    return list(groups.values())

class Kinematics:
    def __init__(self,urdf):
        self.joints=[]
        for j in ET.parse(urdf).getroot().findall('joint'):
            o=j.find('origin');p=j.find('parent');c=j.find('child');a=j.find('axis')
            if p is None or c is None:continue
            xyz=np.fromstring(o.get('xyz','0 0 0'),sep=' ') if o is not None else np.zeros(3)
            rpy=np.fromstring(o.get('rpy','0 0 0'),sep=' ') if o is not None else np.zeros(3)
            Rz=cv2.Rodrigues(np.array([0.,0.,rpy[2]]))[0];Ry=cv2.Rodrigues(np.array([0.,rpy[1],0.]))[0];Rx=cv2.Rodrigues(np.array([rpy[0],0.,0.]))[0]
            T=np.eye(4);T[:3,:3]=Rz@Ry@Rx;T[:3,3]=xyz
            self.joints.append((j.get('name'),j.get('type'),p.get('link'),c.get('link'),T,np.fromstring(a.get('xyz','0 0 1'),sep=' ') if a is not None else np.array([0.,0.,1.])))
    def transforms(self,raw):
        required=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll']
        if any(n not in raw or isinstance(raw[n],bool) or not isinstance(raw[n],(int,float)) or not math.isfinite(raw[n]) for n in required):raise ValueError('Missing or nonfinite measured joint state')
        parents={j[2] for j in self.joints};children={j[3] for j in self.joints};poses={next(iter(parents-children)):np.eye(4)}
        remaining=list(self.joints);unresolved=set()
        while remaining:
            progress=False
            for j in list(remaining):
                name,kind,parent,child,T,axis=j
                # Motor travel calibration does not measure the jaw's CAD zero.
                # Omit this unresolved branch; never expose a fictitious jaw
                # position or use the fixed finger frame as the two-jaw center.
                if name=='gripper' or parent in unresolved:
                    unresolved.add(child);remaining.remove(j);progress=True;continue
                if parent not in poses:continue
                rotation=np.eye(4)
                if kind!='fixed':
                    # OS visualization convention. Its geometric accuracy is an explicit prior, not proven by motor calibration.
                    angle=(raw.get(name,2047)-2047)*2*math.pi/4096
                    rotation[:3,:3]=cv2.Rodrigues(axis*angle)[0]
                poses[child]=poses[parent]@T@rotation;remaining.remove(j);progress=True
            if not progress:raise ValueError('URDF contains a disconnected joint tree')
        return poses
    def camera(self,raw):return self.transforms(raw)['wrist_camera_optical']

class SpatialEstimator:
    """No object dimensions, reference-photo size, D435, or human measurements as input."""
    def __init__(self,root):
        self.root=Path(root);self.fk=Kinematics(self.root/'assets/so101_follower_cam.urdf')
    def record_inputs(self, observations, objects, scene_frames, histories, *, archive_name='wrist-spatial-input.zip'):
        """One bounded local engineering record; never an extra perception input."""
        import json,zipfile
        if any(o.get('source')!='wrist' for o in observations) or any(f.get('source')!='wrist' for h in histories for f in h['frames']):
            raise ValueError('Engineering archive accepts wrist observations only')
        if archive_name not in ('wrist-spatial-input.zip','wrist-local-gaze-input.zip'):raise ValueError('Unknown wrist engineering archive')
        path=self.root/'.runtime'/archive_name
        temporary=path.with_suffix('.tmp')
        manifest={'observations':observations,'objects':objects,'scene_frames':scene_frames,'histories':[],'keyframe_groups':[],'keyframe_images':{}}
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_STORED) as bundle:
            for i,o in enumerate(observations):
                name=f'keyframes/{i}.jpg';bundle.write(o['path'],name);manifest['keyframe_images'][o['id']]=name
            for i,segment in enumerate(histories):
                history=segment['frames']
                manifest['keyframe_groups'].append(segment['keyframes'])
                entries=[]
                for f in history:
                    name=f'{i}/{f["sequence"]}.jpg'
                    bundle.writestr(name,f['jpeg'])
                    entries.append({**{k:v for k,v in f.items() if k!='jpeg'},'archive_path':name})
                manifest['histories'].append(entries)
            bundle.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False))
        temporary.replace(path)
    def correspondences(self, observations, cancel=None, histories=None):
        if any(o.get('source')!='wrist' for o in observations):raise ValueError('Only wrist observations are permitted')
        if histories is not None:
            from .video_tracks import track_wrist_video
            matches, metadata, size = [], [], None
            for segment in histories:
                history=segment['frames']
                ids=segment['keyframes']
                indexes=[i for i,o in enumerate(observations) if o['id'] in ids]
                if len(indexes)!=len(ids):raise ValueError('觀察視角不完整')
                group=[observations[i] for i in indexes]
                pairs, image_size, diagnostics=track_wrist_video(group,history,cancel,require_timestamps=True,max_step_reference_width_px=960)
                metadata.append(diagnostics)
                if diagnostics['status']!='ok':continue
                if size is not None and size!=image_size:raise ValueError('腕部相機影像尺寸改變')
                size=image_size
                matches.extend((indexes[i],indexes[j],p1,p2) for i,j,p1,p2 in pairs)
            return matches,size or (0,0),metadata
        from .features import wrist_tracks
        return wrist_tracks(observations, cancel)
    def estimate(self,observations,objects,scene_frames,cancel=None,histories=None):
        from .selfcal import inv,fit_focal_and_handeye,sampson_errors,reconstruct_pair
        summary={'method':'robot-motion metric multi-view','source':'wrist-only','status':'insufficient','grasp_ready':False,'objects':[]}
        try:
            if any(o.get('source')!='wrist' for o in observations):raise ValueError('Only wrist observations are permitted')
            if len({o.get('camera_generation') for o in observations})>1:raise ValueError('Wrist camera session changed during reconstruction')
            summary['evidence']={'observations':[{k:o.get(k) for k in ('id','camera_generation','capture_monotonic')} for o in observations]}
            if histories is not None:self.record_inputs(observations,objects,scene_frames,histories)
            if len(observations)<5:raise ValueError('需要更多不同角度與平移的腕部影像')
            # More pictures or a longer baseline around the same axis cannot
            # recover the missing camera-mount translation. Reject that data
            # before fitting features, rather than return a prior-driven scale.
            from .motion_observability import analyze_motion_observability
            G=[self.fk.transforms(o['joints'])['gripper_link'] for o in observations]
            motion=analyze_motion_observability(G)
            summary['motion_observability']=motion
            if not motion['translation_identifiable_from_rotations']:
                summary['next_observation_requirement']='nonparallel_rotation_axes'
                raise ValueError('這批影像缺少不同方向的轉動；沿同一方向多拍幾張仍無法確定相機位置')
            correspondence=self.correspondences(observations,cancel,histories)
            matches,size=correspondence[:2]
            if histories is not None:summary['tracking']=correspondence[2]
            if len(matches)<6:raise ValueError('可追蹤的物件特徵不足，需要換角度再看')
            if cancel and cancel.is_set():raise ValueError('已停止空間估算')
            X0=inv(G[0])@self.fk.camera(observations[0]['joints'])
            matches=[m for m in matches if np.linalg.norm((G[m[0]]@X0)[:3,3]-(G[m[1]]@X0)[:3,3])>=.01]
            if len(matches)<6:raise ValueError('平移基線不足，需要增加觀察角度')
            # Withhold complete view pairs, not random points from a fitted pair.
            holdout=matches[::4];train=[m for i,m in enumerate(matches) if i%4]
            X,K,diag=fit_focal_and_handeye(G,train,size,X0,size[0]*.75,max_nfev=160,cancel=cancel)
            if cancel and cancel.is_set():raise ValueError('已停止空間估算')
            held=np.abs(sampson_errors(G,X,K,holdout))
            diag['holdout_p95_px']=float(np.percentile(held,95));diag['matched_pairs']=len(matches)
            diag['camera_mount_shift_mm']=float(np.linalg.norm(X[:3,3]-X0[:3,3])*1000)
            summary['diagnostics']=diag
            if diag['holdout_p95_px']>3:raise ValueError('新視角無法重現估算，需要更多觀察')
            if np.linalg.norm(X[:3,3]-X0[:3,3])>.07:raise ValueError('相機位置估算與手臂尺寸不一致，暫不採用')
            if not diag['success']:raise ValueError('空間估算尚未收斂')
            # Leave all visits to a pose out together. A near-duplicate return
            # must not make a held-out view appear independently predictable.
            folds=[];fold_calibrations=[]
            for family in observation_families(observations):
                pose=family[0];members=set(family)
                training=[m for m in matches if not members.intersection(m[:2])]
                testing=[m for m in matches if members.intersection(m[:2])]
                if len(training)<4 or not testing:continue
                try:
                    xf,kf,df=fit_focal_and_handeye(G,training,size,X0,size[0]*.75,max_nfev=120,cancel=cancel)
                    error=float(np.percentile(np.abs(sampson_errors(G,xf,kf,testing)),95))
                    fold_calibrations.append({'pose':pose,'passed':bool(error<=3 and df['success']),'X':xf,'K':kf})
                    folds.append({'pose':pose,'poses':family,'p95_px':error,'passed':error<=3,'focal_px':float(kf[0,0]),'mount_delta_mm':float(np.linalg.norm(xf[:3,3]-X[:3,3])*1000)})
                except ValueError as e:
                    folds.append({'pose':pose,'poses':family,'passed':False,'reason':str(e)})
                    fold_calibrations.append({'pose':pose,'passed':False})
            diag['withheld_poses']=folds
            summary['needs_more_views']=(any(t['status']!='ok' for t in summary.get('tracking',[])) or len(folds)<3 or not all(f['passed'] and f['mount_delta_mm']<10 for f in folds))
            summary['stability']='uncertain' if summary['needs_more_views'] else 'repeatable'

            clouds=[]
            for i,j,p1,p2 in matches:
                pts,valid,pdiag=reconstruct_pair(G,X,K,i,j,p1,p2)
                clouds.append((i,j,p1,p2,pts,valid,pdiag))
            for object_index,obj in enumerate(objects):
                ref=scene_frames[obj['frame_index']]['id'];idx=next((i for i,o in enumerate(observations) if o['id']==ref),None)
                if idx is None:continue
                b=obj['bbox'];x,y,bw,bh=[b[k] for k in ['x','y','w','h']];collected=[];pairstats=[];tracks={};target_matches=[]
                for i,j,p1,p2,pts,valid,pdiag in clouds:
                    if idx not in (i,j):continue
                    pixels=p1 if i==idx else p2
                    inside=(pixels[:,0]>=x*size[0])&(pixels[:,0]<=(x+bw)*size[0])&(pixels[:,1]>=y*size[1])&(pixels[:,1]<=(y+bh)*size[1])
                    if inside.sum()>=5:target_matches.append((i,j,p1[inside],p2[inside]))
                    keep=inside&valid&(np.linalg.norm(pts,axis=1)<1.0)&(pts[:,2]>-.05)
                    if keep.sum()>=5:
                        collected.extend(pts[keep]);pairstats.append(pdiag)
                        for pixel,point in zip(pixels[keep],pts[keep]):tracks.setdefault(tuple(np.round(pixel,2)),[]).append(point)
                if len(collected)<15 or len(pairstats)<2:
                    summary['objects'].append({'name':obj['name'],'status':'insufficient','reason':'同一物件的有效視差不足'});continue
                from .fold_stability import target_fold_stability
                metric_stability=target_fold_stability(G,X,K,target_matches,fold_calibrations,reconstruct_pair=reconstruct_pair,cancel=cancel)
                if metric_stability['needs_more_views']:summary['needs_more_views']=True;summary['stability']='uncertain'
                disagreements=[max(np.linalg.norm(a-b) for a in values for b in values) for values in tracks.values() if len(values)>1]
                consistency=float(np.percentile(disagreements,95)) if disagreements else None
                if consistency is None or consistency>.008:summary['needs_more_views']=True;summary['stability']='uncertain'
                pts=np.array(collected);center=np.median(pts,axis=0);dist=np.linalg.norm(pts-center,axis=1);pts=pts[dist<=np.percentile(dist,85)]
                center=np.median(pts,axis=0)
                cam=G[idx]@X
                camera_points=np.einsum('ij,kj->ki',inv(cam),np.c_[pts,np.ones(len(pts))])
                if not np.isfinite(camera_points).all():raise ValueError('相機座標計算包含非有限數值')
                depth=float(np.median(camera_points[:,2]));distance=float(np.linalg.norm(center-cam[:3,3]))
                # Perspective extent near the reconstructed target depth. Vertical
                # extent is a projection estimate, not an inferred hidden back side.
                width=bw*size[0]*depth/K[0,0];height=bh*size[1]*depth/K[1,1]
                spread=float(np.median(np.linalg.norm(pts-center,axis=1)))
                summary['objects'].append({'name':obj['name'],'status':'estimated','position_m':center.round(4).tolist(),'position_reference':'median-visible-surface-not-object-center','distance_m':round(distance,3),'projected_width_m':round(width,3),'projected_height_m':round(height,3),'tracked_points':len(pts),'independent_pairs':len(pairstats),'surface_spread_m':round(spread,3),'same_point_p95_disagreement_m':round(consistency,4) if consistency is not None else None,'calibration_stability':metric_stability,'precision':'experimental','grasp_ready':False})
                summary['objects'][-1].update(object_index=object_index,scene_frame_id=ref)
            summary.update(status='estimated' if any(o['status']=='estimated' for o in summary['objects']) else 'insufficient',diagnostics=diag,camera_matrix=K.tolist(),handeye=X.tolist(),note='使用手臂連桿尺寸與實際關節角度恢復尺度；尺寸是可見投影範圍，尚未通過夾取精度驗證。')
            (self.root/'.runtime'/'spatial-latest.json').write_text(__import__('json').dumps(summary,ensure_ascii=False,indent=2))
        except (ValueError,RuntimeError,cv2.error,np.linalg.LinAlgError) as e:
            summary['diagnostic_reason']=str(e)
            summary['needs_more_views']=True
            summary['reason']=('目前照片的觀察方向太相似，還算不準物件距離；需要從不同方向觀察。'
                               if summary.get('next_observation_requirement')=='nonparallel_rotation_axes'
                               else '不同視角尚未得到一致的立體位置，會保留辨識结果並繼續尋找更好的觀察角度。')
        return summary
