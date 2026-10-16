"""Pure image aperture landmark from two explicitly identified inner-jaw ROIs.

No CAD zero, depth, object dimensions, force, grasp or motor commands. The
returned aperture center is a visible image feature, NOT a calibrated grasp TCP.
ROIs identify left and right jaws; pixels/edge positions are estimated, not given.
"""
import math
import cv2
import numpy as np

def _roi(box,shape):
    H,W=shape
    a=np.asarray([box[k] for k in ('x','y','w','h')],float)
    if not np.isfinite(a).all() or np.any(a<0) or np.any(a[2:]<=0) or a[0]+a[2]>1 or a[1]+a[3]>1:
        raise ValueError('Expected normalized xywh ROI')
    x,y,w,h=a*np.array([W,H,W,H]);return int(math.ceil(x)),int(math.ceil(y)),int(math.floor(x+w)),int(math.floor(y+h))

def _fit_rows(gray,roi,sign):
    x0,y0,x1,y1=_roi(roi,gray.shape)
    if x1-x0<16 or y1-y0<24:raise ValueError('Jaw ROI is too small')
    points=[]
    for y in range(max(4,y0),min(gray.shape[0]-4,y1),2):
        line=np.mean(gray[y-3:y+4],axis=0)
        x=np.arange(max(4,x0),min(gray.shape[1]-4,x1))
        contrast=sign*(line[x+4]-line[x-4]);i=int(np.argmax(contrast))
        if contrast[i]>=8:points.append((float(y),float(x[i]),float(contrast[i])))
    if len(points)<12:return {'status':'insufficient','edge_rows':len(points)}
    p=np.array(points);y,x=p[:,:2].T;best=None
    for i in range(len(p)):
        for j in range(i+1,len(p)):
            if y[j]-y[i]<20:continue
            a=(x[j]-x[i])/(y[j]-y[i]);b=x[i]-a*y[i]
            if abs(a)>.6:continue
            keep=np.abs(x-a*y-b)<=3
            key=(int(keep.sum()),float(p[keep,2].sum()))
            if best is None or key>best[0]:best=(key,keep)
    if best is None:return {'status':'insufficient','edge_rows':len(points)}
    keep=best[1];a,b=np.polyfit(y[keep],x[keep],1)
    residual=np.abs(x-a*y-b);keep=residual<=3
    if keep.sum()<12:return {'status':'insufficient','edge_rows':len(points)}
    inliers=p[keep];predictions=[]
    for parity in (0,1):
        yy,xx=inliers[parity::2,:2].T
        if len(yy)<6:return {'status':'insufficient','edge_rows':len(points)}
        aa,bb=np.polyfit(yy,xx,1);predictions.append(aa*inliers[:,0]+bb)
    split_p95=float(np.percentile(np.abs(predictions[0]-predictions[1]),95))
    spread=float(np.ptp(inliers[:,0]));fraction=float(keep.mean())
    return {'status':'ok' if spread>=20 and fraction>=.55 and split_p95<=3 else 'inconsistent',
            'edge_rows':len(points),'inlier_rows':int(keep.sum()),'inlier_fraction':fraction,
            'row_span_px':spread,'x_equals_a_y_plus_b':[float(a),float(b)],
            'residual_p95_px':float(np.percentile(residual[keep],95)),
            'alternating_rows_fit_disagreement_p95_px':split_p95,
            'median_contrast_gray':float(np.median(inliers[:,2])),
            '_inliers':inliers}

def aperture_landmark(image,*,left_inner_roi,right_inner_roi):
    """White jaws against darker gap; camera/mount semantic ROIs are explicit.

    Fit independently estimated silhouette edges, then cross-check row support.
    Neighboring rows are correlated by blur, so parity validation establishes
    only within-image repeatability, not independent geometric calibration.
    """
    if isinstance(image,(str,bytes)):gray=cv2.imread(image,0)
    else:
        gray=np.asarray(image)
        if gray.ndim==3:gray=cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)
    if gray is None or gray.ndim!=2 or gray.dtype!=np.uint8:raise ValueError('Expected uint8 image')
    smoothed=cv2.GaussianBlur(gray.astype(np.float32),(0,0),2)
    left=_fit_rows(smoothed,left_inner_roi,-1);right=_fit_rows(smoothed,right_inner_roi,1)
    result={'status':'insufficient','source':'wrist-image','left':left,'right':right,
            'calibrated_grasp_tcp':False,'metric_dimensions_used':False,
            'motion_authorized':False,'confirmed_grasp':False,
            'limitations':['Explicit ROIs must identify inner silhouette of separate jaws.',
                'Appearance-only landmark; jaw opening, camera depth and contact location remain unknown.',
                'Alternating rows test repeatability within one blurred image, not independent calibration.']}
    if left['status']=='ok' and right['status']=='ok':
        rows=np.intersect1d(left['_inliers'][:,0],right['_inliers'][:,0])
        if len(rows)>=12 and np.ptp(rows)>=20:
            y=float(np.median(rows));lx=float(np.polyval(left['x_equals_a_y_plus_b'],y));rx=float(np.polyval(right['x_equals_a_y_plus_b'],y))
            gaps=np.polyval(right['x_equals_a_y_plus_b'],rows)-np.polyval(left['x_equals_a_y_plus_b'],rows)
            if np.min(gaps)>8:
                H,W=gray.shape;center=[(lx+rx)/2,y]
                result.update(status='ok',image_size=[W,H],aperture_center_px=center,
                    aperture_center_normalized=[center[0]/W,y/H],
                    left_inner_px=[lx,y],right_inner_px=[rx,y],visible_gap_px=rx-lx,
                    common_edge_rows=len(rows),common_y_range_px=[float(rows.min()),float(rows.max())])
    for group in (left,right):group.pop('_inliers',None)
    return result
