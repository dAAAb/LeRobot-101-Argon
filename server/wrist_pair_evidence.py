"""Pure wrist-image displacement evidence; no motor/camera/network operations.
Input frames are wrist records with source/generation/monotonic timestamp and
path, jpeg bytes, or an in-memory uint8 image. Bboxes/ROIs are explicit normalized
{x,y,w,h} by default. Target, fixed finger and background have disjoint support.

No known dimensions, metric depth, force or true-grasp inference. Even positive
co-motion evidence cannot independently establish contact, lift or a true grasp.
"""
from __future__ import annotations
import math,time
import cv2
import numpy as np

def _rect(box,shape,normalized):
    values=[box[k] for k in ('x','y','w','h')] if isinstance(box,dict) else list(box)
    if len(values)!=4 or any(isinstance(v,bool) or not isinstance(v,(int,float,np.number)) or not math.isfinite(float(v)) for v in values):raise ValueError('Invalid ROI')
    x,y,w,h=map(float,values);H,W=shape
    if normalized:x*=W;w*=W;y*=H;h*=H
    if w<=0 or h<=0 or x<0 or y<0 or x+w>W+1e-6 or y+h>H+1e-6:raise ValueError('ROI exceeds image')
    return np.array([x,y,w,h])

def _box_mask(shape,box):
    m=np.zeros(shape,np.uint8);x,y,w,h=box
    m[int(math.ceil(y)):int(math.floor(y+h)),int(math.ceil(x)):int(math.floor(x+w))]=255
    return m

def _erode(mask,r):
    return cv2.erode(mask,np.ones((2*r+1,2*r+1),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)

def _gray(frame):
    if 'image' in frame:im=np.asarray(frame['image'])
    elif 'jpeg' in frame:im=cv2.imdecode(np.frombuffer(frame['jpeg'],np.uint8),cv2.IMREAD_GRAYSCALE)
    elif 'path' in frame:im=cv2.imread(str(frame['path']),cv2.IMREAD_GRAYSCALE)
    else:raise ValueError('Missing image source')
    if im is None or im.dtype!=np.uint8:raise ValueError('Invalid uint8 wrist image')
    if im.ndim==3:im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    if im.ndim!=2 or min(im.shape)<64:raise ValueError('Image dimensions unsupported')
    return np.ascontiguousarray(im)

def _warp_points(M,p):return np.einsum('ij,kj->ki',M[:,:2],p)+M[:,2]

def _ncc(a,b,p,q,window):
    aa=cv2.getRectSubPix(a,(window,window),tuple(map(float,p))).astype(np.float32)
    bb=cv2.getRectSubPix(b,(window,window),tuple(map(float,q))).astype(np.float32)
    aa-=aa.mean();bb-=bb.mean();den=float(np.linalg.norm(aa)*np.linalg.norm(bb))
    return float(np.sum(aa*bb)/den) if den>1e-6 else -1.

def _similarity(p,q,threshold):
    if len(p)<3:return None,np.zeros(len(p),bool)
    M,good=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=threshold,maxIters=2000,confidence=.995,refineIters=10)
    if M is None or good is None or not np.isfinite(M).all():return None,np.zeros(len(p),bool)
    return M,good.ravel().astype(bool)

def _scale(p,q):
    i,j=np.triu_indices(len(p),1);d0=np.linalg.norm(p[i]-p[j],axis=1);d1=np.linalg.norm(q[i]-q[j],axis=1)
    keep=d0>=10
    if keep.sum()<10:return None,None
    ratios=d1[keep]/d0[keep];center=float(np.median(ratios))
    return center,float(np.median(np.abs(ratios-center)))

def _summary(src,dst,fb,ncc,seeded,steps,threshold,min_tracks):
    out={'status':'insufficient','seeded_tracks':int(seeded),'track_count':int(len(src)),
         'retained_fraction':float(len(src)/max(1,seeded)),'steps':steps,
         'median_dx_px':None,'median_dy_px':None,'median_scale':None,'scale_mad':None,
         'fb_p95_px':float(np.percentile(fb,95)) if len(fb) else None,
         'ncc_median':float(np.median(ncc)) if len(ncc) else None,
         '_src':src,'_dst':dst,'_M':None}
    if len(src)<min_tracks:return out
    M,good=_similarity(src,dst,threshold)
    if M is None or good.sum()<min_tracks:return out
    p,q=src[good],dst[good];flow=q-p;median=np.median(flow,axis=0)
    residual=np.linalg.norm(q-_warp_points(M,p),axis=1);scale,scale_mad=_scale(p,q)
    out.update(track_count=int(len(p)),inlier_fraction=float(good.mean()),retained_fraction=float(len(p)/max(1,seeded)),
        median_dx_px=float(median[0]),median_dy_px=float(median[1]),
        flow_mad_px=np.median(np.abs(flow-median),axis=0).tolist(),median_scale=scale,scale_mad=scale_mad,
        rotation_deg=float(np.degrees(np.arctan2(M[1,0],M[0,0]))),
        similarity_residual_p95_px=float(np.percentile(residual,95)),
        similarity_matrix=M.tolist(),_src=p,_dst=q,_M=M)
    # These are tracking-consistency labels, never grasp/safety authorization.
    if good.mean()>=.6 and out['retained_fraction']>=.35 and np.percentile(residual,95)<=threshold*1.5:
        out['status']='ok'
    else:out['status']='inconsistent'
    return out

def _track(images,seed_mask,*,exclusions,window,level,fb_max,step_max,min_ncc,min_tracks,max_features,threshold,cancel,deadline):
    radius=(window//2+2)*(1<<level)+2*((1<<level)-1)
    mask=_erode(seed_mask,radius)
    p=cv2.goodFeaturesToTrack(images[0],maxCorners=max_features,qualityLevel=.015,minDistance=7,mask=mask,blockSize=5)
    p=np.empty((0,2),np.float32) if p is None else p[:,0]
    source=p.copy();seeded=len(p);fb_all=np.zeros(len(p));ncc_all=np.ones(len(p));steps=[]
    bounds=[];Mglobal=np.array([[1.,0.,0.],[0.,1.,0.]])
    lk={'winSize':(window,window),'maxLevel':level,'criteria':(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,35,.002)}
    for index,(a,b) in enumerate(zip(images,images[1:])):
        if cancel is not None and cancel.is_set():raise ValueError('Tracking cancelled')
        if time.monotonic()>deadline:raise ValueError('Tracking time budget exceeded')
        before=len(p)
        if before<min_tracks:
            steps.append({'index':index+1,'before':before,'survived':0,'reason':'too_few_tracks'})
            p=np.empty((0,2),np.float32);source=p.copy();fb_all=np.empty(0);ncc_all=np.empty(0);break
        q,st,_=cv2.calcOpticalFlowPyrLK(a,b,p[:,None],None,**lk)
        if q is None or st is None:
            p=np.empty((0,2),np.float32);source=p.copy();fb_all=np.empty(0);ncc_all=np.empty(0);break
        q=q[:,0];finite=np.isfinite(q).all(axis=1);safe=q.copy();safe[~finite]=p[~finite]
        back,bst,_=cv2.calcOpticalFlowPyrLK(b,a,safe[:,None],None,**lk)
        if back is None or bst is None:
            p=np.empty((0,2),np.float32);source=p.copy();fb_all=np.empty(0);ncc_all=np.empty(0);break
        fb=np.linalg.norm(back[:,0]-p,axis=1);H,W=a.shape
        valid=(st[:,0]>0)&(bst[:,0]>0)&finite&(fb<=fb_max)&(np.linalg.norm(q-p,axis=1)<=step_max)
        valid&=(q[:,0]>=radius)&(q[:,0]<W-radius)&(q[:,1]>=radius)&(q[:,1]<H-radius)
        forbidden=exclusions[index+1] if exclusions is not None else None
        if forbidden is not None:
            expanded=cv2.dilate(forbidden,np.ones((2*radius+1,2*radius+1),np.uint8))
            ids=np.where(valid)[0];xy=np.rint(q[ids]).astype(int);valid[ids]&=expanded[xy[:,1],xy[:,0]]==0
        ncc=np.full(len(p),-1.,float)
        for i in np.where(valid)[0]:ncc[i]=_ncc(a,b,p[i],q[i],window)
        valid&=ncc>=min_ncc
        srcstep=p[valid];dststep=q[valid]
        M,inliers=_similarity(srcstep,dststep,threshold)
        ids=np.where(valid)[0];valid[:]=False
        if M is not None:valid[ids[inliers]]=True
        source=source[valid];p=q[valid];fb_all=np.maximum(fb_all[valid],fb[valid]);ncc_all=np.minimum(ncc_all[valid],ncc[valid])
        steps.append({'index':index+1,'before':before,'survived':len(p),
                      'fb_p95_px':float(np.percentile(fb[valid],95)) if valid.any() else None})
        Mglobal,good=_similarity(source,p,threshold) if len(p)>=3 else (None,None)
        bounds.append(Mglobal)
    result=_summary(source,p,fb_all,ncc_all,seeded,steps,threshold,min_tracks)
    result['support_radius_px']=radius;result['_transforms']=bounds
    return result

def analyze_wrist_sequence(frames,target_bbox,*,fixed_finger_roi=None,background_rois=None,
        normalized=True,max_features=180,min_tracks=8,fb_max_px=.8,max_step_reference_px=24,
        max_seconds=3.,cancel=None,include_tracks=False,pyramid_max_level=None):
    """Sequential LK evidence. Use intermediate video frames when available.

    Two endpoint frames are accepted for a local move, but explicitly labeled
    endpoint-only: correspondence alone cannot establish continuity/attachment.
    All public displacements/scales remain in original image pixels, no resizing.
    """
    started=time.monotonic()
    if not 2<=len(frames)<=64:raise ValueError('Expected2..64 wrist frames')
    if type(max_features) is not int or not 16<=max_features<=500 or type(min_tracks) is not int or not 6<=min_tracks<=max_features:raise ValueError('Invalid feature budget')
    if not (0<fb_max_px<=.8 and 0<max_step_reference_px<=48 and 0<max_seconds<=10):raise ValueError('Invalid tracking bounds')
    generation=frames[0].get('camera_generation')
    if not isinstance(generation,str) or not generation:raise ValueError('Camera generation required')
    times=[]
    for frame in frames:
        if frame.get('source')!='wrist' or frame.get('camera_generation')!=generation:raise ValueError('Only one wrist-camera generation is accepted')
        t=frame.get('capture_monotonic')
        if isinstance(t,bool) or not isinstance(t,(int,float)) or not math.isfinite(t):raise ValueError('Finite frame monotonic timestamp required')
        times.append(t)
    if any(b<=a for a,b in zip(times,times[1:])):raise ValueError('Frames must be chronologically increasing')
    images=[_gray(f) for f in frames];shape=images[0].shape
    if any(im.shape!=shape for im in images) or sum(im.size for im in images)>140_000_000:raise ValueError('Image geometry changed or input budget exceeded')
    H,W=shape;target=_rect(target_bbox,shape,normalized);target_mask=_box_mask(shape,target)
    finger=_rect(fixed_finger_roi,shape,normalized) if fixed_finger_roi is not None else None
    finger_mask=_box_mask(shape,finger) if finger is not None else np.zeros(shape,np.uint8)
    if np.any((target_mask>0)&(finger_mask>0)):raise ValueError('Target and fixed-finger source regions overlap')
    if background_rois is None:
        background=np.zeros(shape,np.uint8);background[:int(H*.58)]=255
    else:
        background=np.zeros(shape,np.uint8)
        for box in background_rois:background|=_box_mask(shape,_rect(box,shape,normalized))
    background[(target_mask>0)|(finger_mask>0)]=0
    level=(2 if len(frames)==2 else 1) if pyramid_max_level is None else pyramid_max_level
    if type(level) is not int or not 0<=level<=2:raise ValueError('Pyramid level must be0..2')
    kwargs=dict(window=15,level=level,fb_max=fb_max_px,step_max=max_step_reference_px*W/960,
                min_ncc=.78,min_tracks=min_tracks,max_features=max_features,threshold=1.5,
                cancel=cancel,deadline=started+max_seconds)
    target_e=_track(images,target_mask,exclusions=[finger_mask]*len(images),**kwargs)
    finger_e=_track(images,finger_mask,exclusions=[target_mask]*len(images),**kwargs) if finger is not None else {
        'status':'not_supplied','track_count':0,'reason':'Explicit rigid fixed-finger ROI required; moving jaw is not an anchor'}
    # Exclude the target's full observed swept bbox from background patch support.
    # Include all successful target transforms; failures cannot make target pixels
    # become valid table/background evidence.
    swept=target_mask.copy();x,y,w,h=target;corners=np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]],float)
    for M in target_e.get('_transforms',[]):
        if M is not None:
            polygon=np.rint(_warp_points(M,corners)).astype(np.int32);cv2.fillConvexPoly(swept,polygon,255)
    forbidden=cv2.bitwise_or(swept,finger_mask)
    background[forbidden>0]=0
    background_e=_track(images,background,exclusions=[forbidden]*len(images),**kwargs)
    evidence={'state':'insufficient','object_following_gripper_evidence':False,
        'confirmed_grasp':False,'true_grasp':False,'force_estimate':None,
        'reason':'Need usable target, rigid fixed-finger and independent background tracks'}
    if all(e.get('status')=='ok' for e in [target_e,finger_e,background_e]) and target_e.get('median_scale') is not None and finger_e.get('median_scale') is not None:
        F=finger_e['_M']
        tr=target_e['_dst']-_warp_points(F,target_e['_src'])
        br=background_e['_dst']-_warp_points(F,background_e['_src'])
        tn=np.linalg.norm(tr,axis=1);bn=np.linalg.norm(br,axis=1)
        noise=max(.5,float(target_e['fb_p95_px'] or 0)+float(finger_e['fb_p95_px'] or 0))
        tolerance=max(1.,3*noise);target_p95=float(np.percentile(tn,95));background_motion=float(np.median(bn))
        ratio=(target_e.get('median_scale') or 1)/(finger_e.get('median_scale') or 1)
        static=target_p95<=2*tolerance and abs(math.log(ratio))<=.025
        background_distinct=background_motion>=max(3*tolerance,3.)
        enough_retention=min(target_e['retained_fraction'],finger_e['retained_fraction'])>=.6
        follows=static and background_distinct and target_p95<background_motion*.35 and enough_retention
        evidence.update(state='co_motion_evidence' if follows else ('target_moves_relative_to_fixed_finger' if not static else 'independent_camera_motion_not_established'),
            object_following_gripper_evidence=bool(follows),
            reason='Image co-motion only; contact, clear lift and grasp are not independently established' if follows else 'Images do not establish object following the gripper',
            target_relative_fixed_median_dx_dy_px=np.median(tr,axis=0).tolist(),
            target_relative_fixed_p95_px=target_p95,background_relative_fixed_median_motion_px=background_motion,
            target_relative_fixed_scale=ratio,noise_based_tolerance_px=tolerance,
            independent_background_motion=bool(background_distinct))
    out={'source':'wrist-only','camera_generation':generation,'image_size':[W,H],
         'frame_sequences':[f.get('sequence') for f in frames],
         'frame_capture_monotonic':times,
         'target_bbox_input_px':target.tolist(),
         'frame_count':len(frames),'temporal_basis':'endpoint-only correspondence' if len(frames)==2 else 'sequential-frame correspondence',
         'time_span_s':times[-1]-times[0],'largest_frame_gap_s':max(np.diff(times)),
         'target':target_e,'fixed_finger':finger_e,'background':background_e,'lift_evidence':evidence,
         'metric_dimensions_used':False,'elapsed_seconds':time.monotonic()-started,
         'limitations':['BBox interiors can contain occluders; LK is not semantic segmentation.',
           'Fixed-finger ROI must be rigid material, never the moving jaw.',
           'Background default excludes the lower42 percent, but caller ROIs are preferred.',
           'Co-motion can also arise without grasp; combine with visible containment, actual upward motion and separation from the table.',
           'No image-only output authorizes robot motion or confirms grasp.']}
    for e in [target_e,finger_e,background_e]:
        if include_tracks and '_src' in e:e['tracks_start_px']=e['_src'].tolist();e['tracks_end_px']=e['_dst'].tolist()
        for key in list(e):
            if key.startswith('_'):del e[key]
    return out

def analyze_wrist_pair(before,after,target_bbox,**kwargs):
    return analyze_wrist_sequence([before,after],target_bbox,**kwargs)
