"""Metric hand-eye from FK and visual relative pose directions. No hardware access.

Conventions: G_i = base_T_gripper_i; X = gripper_T_camera.
OpenCV recoverPose(points_i, points_j) returns B_ji = camera_j_T_camera_i.
A_ji = inv(G_j) @ G_i, so A_ji X = X B_ji.
"""
import itertools, json, argparse
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import lsq_linear, least_squares
import cv2


def inv(T):
    out = np.eye(4)
    out[:3,:3] = T[:3,:3].T
    out[:3,3] = -T[:3,:3].T @ T[:3,3]
    return out


def solve_metric_handeye(Gs, pairs, max_condition=1000, max_rotation_rms_deg=3, max_translation_rms_m=0.005):
    """pairs: (i, j, OpenCV R_ji, unit t_ji), no object dimensions needed.

    Non-parallel rotation axes required. All captures must be settled, scene static.
    This initializer assumes K/undistortion and the FK metric geometry are valid.
    A robust production implementation should bootstrap, weight and reject outliers.
    """
    As = [inv(Gs[j]) @ Gs[i] for i,j,R,t in pairs]
    a = np.array([Rotation.from_matrix(A[:3,:3]).as_rotvec() for A in As])
    b = np.array([Rotation.from_matrix(R).as_rotvec() for i,j,R,t in pairs])
    # Rotations below 1 degree are poor calibration constraints.
    valid = (np.linalg.norm(a,axis=1) > np.deg2rad(1)) & (np.linalg.norm(b,axis=1) > np.deg2rad(1))
    if valid.sum() < 3:
        raise ValueError('Too few usable motion pairs')
    _, axes_s, _ = np.linalg.svd(a[valid], full_matrices=False)
    if axes_s[1] < axes_s[0] * 0.05:
        raise ValueError('Single-axis / nearly single-axis rotation degeneracy')
    RX, _ = Rotation.align_vectors(a[valid], b[valid])
    RX = RX.as_matrix()
    m = len(pairs)
    M = np.zeros((3*m, 3+m)); y = np.zeros(3*m)
    for k, (A, (_,_,_,t)) in enumerate(zip(As,pairs)):
        t = np.asarray(t).reshape(3); t = t/np.linalg.norm(t)
        M[3*k:3*k+3, :3] = A[:3,:3] - np.eye(3)
        M[3*k:3*k+3, 3+k] = -RX @ t
        y[3*k:3*k+3] = -A[:3,3]
    s = np.linalg.svd(M, compute_uv=False)
    rank = np.linalg.matrix_rank(M)
    if rank < M.shape[1]:
        raise ValueError(f'Translation/scale system rank deficient: {rank}/{M.shape[1]}')
    if s[0]/s[-1] > max_condition:
        raise ValueError(f'Translation/scale system poorly conditioned: {s[0]/s[-1]:.1f}')
    # Physical per-pair scales must be positive; X translation unrestricted here.
    with np.errstate(all='ignore'):  # macOS Accelerate can leave FP flags after SVD.
        fit = lsq_linear(M, y, bounds=(np.r_[[-np.inf]*3,[1e-8]*m], np.inf))
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError('Non-finite / unsuccessful translation solve')
    X = np.eye(4); X[:3,:3] = RX; X[:3,3] = fit.x[:3]
    angle_err = [Rotation.from_matrix(A[:3,:3]@RX@R.T@RX.T).magnitude() for A,(_,_,R,t) in zip(As,pairs)]
    with np.errstate(all='ignore'):
        linear_residual = np.einsum('ij,j->i',M,fit.x)-y
    diag = {
        'rank':int(rank), 'unknowns':int(M.shape[1]), 'condition':float(s[0]/s[-1]),
        'rotation_rms_deg':float(np.rad2deg(np.sqrt(np.mean(np.square(angle_err))))),
        'translation_rms_m':float(np.sqrt(np.mean(np.square(linear_residual)))),
        'min_baseline_m':float(fit.x[3:].min()),
        'rotation_axis_singular_values':axes_s.tolist(),
    }
    if diag['rotation_rms_deg']>max_rotation_rms_deg:
        raise ValueError(f"Rotation inconsistency: {diag['rotation_rms_deg']:.2f} deg RMS")
    if diag['translation_rms_m']>max_translation_rms_m:
        raise ValueError(f"Translation inconsistency: {diag['translation_rms_m']*1000:.2f} mm RMS")
    if diag['min_baseline_m'] < 0.001:
        raise ValueError('At least one visual baseline collapsed to < 1 mm')
    return X, fit.x[3:], diag


def skew(t):
    x,y,z=np.asarray(t).reshape(3)
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])


def intrinsics(focal, image_size):
    w,h=image_size
    return np.array([[focal,0,(w-1)/2],[0,focal,(h-1)/2],[0,0,1.]])


def sampson_errors(Gs, X, K, matches):
    """Signed pixel Sampson errors for static non-robot features.

    matches: [(i,j, Nx2 points_i, Nx2 points_j), ...].
    Uses metric FK but does not require object sizes/3D point initialization.
    """
    KI=np.linalg.inv(K); residuals=[]
    for i,j,p1,p2 in matches:
        B=inv(Gs[j]@X)@(Gs[i]@X)
        if np.linalg.norm(B[:3,3]) < 1e-6:
            raise ValueError('Zero camera translation provides no epipolar constraint')
        F=KI.T@skew(B[:3,3])@B[:3,:3]@KI
        p1=np.c_[np.asarray(p1),np.ones(len(p1))]
        p2=np.c_[np.asarray(p2),np.ones(len(p2))]
        a=np.einsum('ij,kj->ki',F,p1); b=np.einsum('ij,kj->ki',F.T,p2)
        r=np.einsum('ij,ij->i',p2,a)
        denom=np.sqrt(np.maximum(np.sum(a[:,:2]**2+b[:,:2]**2,axis=1),1e-30))
        residuals.append(r/denom)
    return np.concatenate(residuals)


def fit_focal_and_handeye(Gs, matches, image_size, X0, focal0,
                         min_focal_ratio=.2, max_focal_ratio=3., max_nfev=300, cancel=None,
                         robust_loss='huber', robust_scale_px=1.5, max_sampson_px=3.):
    """Research-only boardless initializer: focal + camera hand-eye from known motion.

    Fixed aspect=1, principal point=image center, distortion=0 are INITIAL MODEL
    assumptions, not automatically valid calibration. Use undistorted captures
    if available. Needs multiple nonparallel motions and scene depth variation.
    Fits epipolar constraints with Huber loss; no object metric dimension input.
    Validate on withheld entire pose pairs, target tracks, and new viewpoints.
    """
    if len(matches)<4: raise ValueError('Need at least four diverse pose pairs')
    if robust_loss not in ('huber', 'cauchy'):
        raise ValueError('Unsupported calibration robust loss')
    if not all(np.isfinite(v) and v > 0 for v in (robust_scale_px, max_sampson_px)):
        raise ValueError('Calibration residual limits must be positive and finite')
    w,h=image_size
    def unpack(x):
        X=np.eye(4)
        X[:3,:3]=Rotation.from_rotvec(x[1:4]).as_matrix()@X0[:3,:3]
        X[:3,3]=x[4:7]
        return X,intrinsics(np.exp(x[0]),image_size)
    def residual(x):
        if cancel is not None and cancel.is_set():raise RuntimeError('空間估算已停止')
        X,K=unpack(x)
        return sampson_errors(Gs,X,K,matches)
    lower=np.r_[np.log(min_focal_ratio*w),[-np.pi]*3,[-.2]*3]
    upper=np.r_[np.log(max_focal_ratio*w),[np.pi]*3,[.2]*3]
    fits=[]
    # Unknown 180-degree image rotation needs an explicit alternative X0 seed;
    # an optimizer should not be expected to cross that ambiguity by itself.
    for factor in [.7,1.,1.4]:
        x0=np.r_[np.log(np.clip(focal0*factor,np.exp(lower[0])*1.01,np.exp(upper[0])*.99)),[0.,0.,0.],X0[:3,3]]
        with np.errstate(all='ignore'):
            fit=least_squares(residual,x0,bounds=(lower,upper),loss=robust_loss,f_scale=robust_scale_px,
                              x_scale=np.r_[.5,[.25]*3,[.05]*3],max_nfev=max_nfev,
                              ftol=1e-10,xtol=1e-10,gtol=1e-10)
        if np.isfinite(fit.x).all() and np.isfinite(fit.cost):fits.append(fit)
    if not fits:raise ValueError('No finite self-calibration fit')
    best=min(fits,key=lambda z:z.cost)
    X,K=unpack(best.x); errors=residual(best.x)
    Jscaled=best.jac*np.r_[.5,[.25]*3,[.05]*3][None,:]
    singular=np.linalg.svd(Jscaled,compute_uv=False)
    rank=int(np.linalg.matrix_rank(Jscaled,tol=singular[0]*1e-5))
    diag={'focal_px':float(K[0,0]),'median_sampson_px':float(np.median(np.abs(errors))),
          'p95_sampson_px':float(np.percentile(np.abs(errors),95)),
          'scaled_jacobian_rank':rank,'scaled_jacobian_condition':float(singular[0]/max(singular[-1],1e-20)),
          'seed_results':[{'focal_px':float(np.exp(z.x[0])),'cost':float(z.cost)} for z in fits],
          'success':bool(best.success), 'robust_loss': robust_loss,
          'robust_scale_px': float(robust_scale_px), 'max_sampson_px': float(max_sampson_px),
          'assumptions':['fixed principal point','square pixels','zero distortion']}
    # Fail closed on critical motion degeneracy / implausible fit, but these gates
    # DO NOT alone establish grasp safety; they only qualify a calibration candidate.
    if not best.success:raise ValueError('Boardless solver did not converge')
    if rank<7:raise ValueError(f'Boardless calibration unobservable, rank {rank}/7')
    if diag['scaled_jacobian_condition']>1e4:raise ValueError('Boardless calibration ill-conditioned')
    if diag['p95_sampson_px']>max_sampson_px:raise ValueError('Boardless calibration large epipolar residual')
    if K[0,0] < min_focal_ratio*w*1.01 or K[0,0] > max_focal_ratio*w*.99:
        raise ValueError('Focal solution hits search boundary')
    # Local covariance is only an observability diagnostic: unmodelled distortion,
    # FK error, rolling shutter, and mismatches can dominate this optimistic value.
    variance=max(float(np.median(np.abs(errors))/.6745)**2,.1**2)
    covariance=np.linalg.pinv(best.jac.T@best.jac)*variance
    diag['local_focal_sigma_px']=float(K[0,0]*np.sqrt(max(covariance[0,0],0)))
    diag['local_translation_sigma_mm']=(np.sqrt(np.maximum(np.diag(covariance)[4:7],0))*1000).tolist()
    return X,K,diag


def reconstruct_pair(Gs,X,K,i,j,p1,p2):
    """Metric sparse points with positive-depth, angle, and reprojection diagnostics."""
    C1=Gs[i]@X;C2=Gs[j]@X
    P1=K@inv(C1)[:3,:];P2=K@inv(C2)[:3,:]
    H=cv2.triangulatePoints(P1,P2,np.asarray(p1,dtype=float).T,np.asarray(p2,dtype=float).T)
    points=(H[:3]/H[3]).T
    uv1=(P1@np.c_[points,np.ones(len(points))].T).T
    uv2=(P2@np.c_[points,np.ones(len(points))].T).T
    z1=uv1[:,2];z2=uv2[:,2]
    err=np.maximum(np.linalg.norm(uv1[:,:2]/z1[:,None]-p1,axis=1),np.linalg.norm(uv2[:,:2]/z2[:,None]-p2,axis=1))
    a=points-C1[:3,3];b=points-C2[:3,3]
    a/=np.linalg.norm(a,axis=1)[:,None];b/=np.linalg.norm(b,axis=1)[:,None]
    angle=np.rad2deg(np.arccos(np.clip(np.einsum('ij,ij->i',a,b),-1,1)))
    good=np.isfinite(points).all(axis=1)&(z1>0)&(z2>0)&(err<2)&(angle>2)
    return points,good,{'baseline_m':float(np.linalg.norm(C1[:3,3]-C2[:3,3])),
                       'positive_depth_fraction':float(np.mean((z1>0)&(z2>0))),
                       'median_parallax_deg':float(np.median(angle)),
                       'p95_reprojection_px':float(np.percentile(err,95)),
                       'retained':int(good.sum()),'total':len(points)}


def synthetic_demo(single_axis=False):
    rng = np.random.default_rng(7)
    X = np.eye(4)
    X[:3,:3] = Rotation.from_euler('xyz',[2.75, 0.03, 0.35]).as_matrix()
    X[:3,3] = [0.005,0.048,-0.043]
    Gs=[]
    for k in range(8):
        G=np.eye(4)
        angles=[0,0,0.07*k] if single_axis else rng.normal(0,0.25,3)
        G[:3,:3]=Rotation.from_euler('xyz',angles).as_matrix()
        G[:3,3]=rng.normal(0,0.025,3)+[0.2,0,0.25]
        Gs.append(G)
    pairs=[]; scales=[]
    for i,j in itertools.combinations(range(len(Gs)),2):
        C_i=Gs[i]@X; C_j=Gs[j]@X
        B=inv(C_j)@C_i
        scales.append(np.linalg.norm(B[:3,3]))
        pairs.append((i,j,B[:3,:3],B[:3,3]/scales[-1]))
    estimate, scale_estimate, diag=solve_metric_handeye(Gs,pairs)
    diag['handeye_translation_error_mm']=float(np.linalg.norm(estimate[:3,3]-X[:3,3])*1000)
    diag['handeye_rotation_error_deg']=float(np.rad2deg(Rotation.from_matrix(estimate[:3,:3]@X[:3,:3].T).magnitude()))
    diag['baseline_max_error_mm']=float(np.max(np.abs(scale_estimate-np.asarray(scales)))*1000)
    return diag


def synthetic_focal_demo():
    rng=np.random.default_rng(9); size=(960,540); K=intrinsics(720,size)
    X=np.eye(4);X[:3,:3]=Rotation.from_euler('xyz',[.08,-.1,.3]).as_matrix();X[:3,3]=[.014,.036,-.028]
    Gs=[]
    for k in range(8):
        G=np.eye(4);G[:3,:3]=Rotation.from_rotvec(rng.normal(0,.16,3)).as_matrix();G[:3,3]=rng.normal(0,.025,3)
        Gs.append(G)
    points=rng.uniform([-.14,-.09,.35],[.14,.09,.7],(250,3))
    pix=[]
    for G in Gs:
        U=(K@inv(G@X)[:3,:]@np.c_[points,np.ones(len(points))].T).T
        pix.append(U[:,:2]/U[:,2,None]+rng.normal(0,.15,(len(points),2)))
    pairs=list(itertools.combinations(range(len(Gs)),2))
    matches=[(i,j,pix[i],pix[j]) for i,j in pairs]
    holdout=matches[-4:];train=matches[:-4]
    X0=X.copy();X0[:3,:3]=Rotation.from_euler('xyz',[.09,-.07,.12]).as_matrix()@X[:3,:3];X0[:3,3]+=[.005,-.005,.004]
    estimate,Ke,diag=fit_focal_and_handeye(Gs,train,size,X0,600)
    diag['focal_error_pct']=float(abs(Ke[0,0]-K[0,0])/K[0,0]*100)
    diag['handeye_translation_error_mm']=float(np.linalg.norm(estimate[:3,3]-X[:3,3])*1000)
    diag['handeye_rotation_error_deg']=float(np.rad2deg(Rotation.from_matrix(estimate[:3,:3]@X[:3,:3].T).magnitude()))
    he=sampson_errors(Gs,estimate,Ke,holdout)
    diag['holdout_p95_sampson_px']=float(np.percentile(np.abs(he),95))
    i,j,p1,p2=holdout[-1]
    pe,valid,rd=reconstruct_pair(Gs,estimate,Ke,i,j,p1,p2)
    rd['median_metric_error_mm']=float(np.median(np.linalg.norm(pe[valid]-points[valid],axis=1))*1000)
    diag['heldout_metric_reconstruction']=rd
    return diag


def urdf_fk(path, joint_angles, tip='gripper_link'):
    """Read-only URDF FK; angles are URDF radians, not LeRobot normalized values."""
    import xml.etree.ElementTree as ET
    root=ET.parse(path).getroot();by_child={}
    for joint in root.findall('joint'):
        if joint.find('parent') is None:continue
        by_child[joint.find('child').get('link')]=joint
    def link_pose(link):
        if link not in by_child:return np.eye(4)
        j=by_child[link];origin=j.find('origin');T=np.eye(4)
        if origin is not None:
            T[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
            T[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
        if j.get('type') in ['revolute','continuous']:
            axis=np.fromstring(j.find('axis').get('xyz'),sep=' ')
            V=np.eye(4);V[:3,:3]=Rotation.from_rotvec(axis*joint_angles.get(j.get('name'),0)).as_matrix();T=T@V
        return link_pose(j.find('parent').get('link'))@T
    return link_pose(tip)


def synthetic_so101_demo():
    path='/Users/neo/LeRobot-101-OS/web/assets/so101/so101_follower_cam.urdf'
    names=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll']
    rng=np.random.default_rng(11);q0=np.array([0.,-.7,.6,.1,0.])
    qs=[q0]+[q0+rng.uniform(-.15,.15,5) for _ in range(9)]
    Gs=[urdf_fk(path,dict(zip(names,q))) for q in qs]
    # Camera mount prior perturbed to emulate unknown installation error.
    X=inv(Gs[0])@urdf_fk(path,dict(zip(names,q0)),tip='wrist_camera_optical')
    X[:3,:3]=Rotation.from_euler('xyz',[.07,-.05,.14]).as_matrix()@X[:3,:3]
    X[:3,3]+=[.003,-.002,.002]
    vision=[];scales=[]
    for i,j in itertools.combinations(range(len(Gs)),2):
        B=inv(Gs[j]@X)@(Gs[i]@X);s=np.linalg.norm(B[:3,3]);scales.append(s)
        vision.append((i,j,B[:3,:3],B[:3,3]/s))
    Xe,se,d=solve_metric_handeye(Gs,vision)
    d['URDF_SO101_metric_translation_error_mm']=float(np.linalg.norm(Xe[:3,3]-X[:3,3])*1000)
    d['known_baselines_mm']=[float(min(scales)*1000),float(max(scales)*1000)]
    size=(960,540);K=intrinsics(680,size)
    # 3D scene defined relative to first camera, then moved into base frame.
    pc=rng.uniform([-.11,-.06,.22],[.11,.06,.43],(350,3))
    points=((Gs[0]@X)@np.c_[pc,np.ones(len(pc))].T).T[:,:3]
    pix=[]
    for G in Gs:
        U=(K@inv(G@X)[:3,:]@np.c_[points,np.ones(len(points))].T).T
        pix.append(U[:,:2]/U[:,2,None]+rng.normal(0,.15,(len(points),2)))
    matches=[]
    for i,j in itertools.combinations(range(len(Gs)),2):
        valid=(pix[i][:,0]>10)&(pix[i][:,0]<950)&(pix[i][:,1]>10)&(pix[i][:,1]<530)&(pix[j][:,0]>10)&(pix[j][:,0]<950)&(pix[j][:,1]>10)&(pix[j][:,1]<530)
        if valid.sum()>=30:matches.append((i,j,pix[i][valid],pix[j][valid]))
    train=matches[:-6];holdout=matches[-6:]
    X0=inv(Gs[0])@urdf_fk(path,dict(zip(names,q0)),tip='wrist_camera_optical')
    est,Ke,fd=fit_focal_and_handeye(Gs,train,size,X0,700)
    fd['focal_truth_px']=680
    fd['handeye_translation_error_mm']=float(np.linalg.norm(est[:3,3]-X[:3,3])*1000)
    fd['holdout_p95_sampson_px']=float(np.percentile(np.abs(sampson_errors(Gs,est,Ke,holdout)),95))
    return {'metric_direction_solver':d,'boardless_focal_solver':fd,'synthetic_only':True}


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--focal',action='store_true');parser.add_argument('--so101',action='store_true');args=parser.parse_args()
    print(json.dumps({'diverse_motion':synthetic_demo()},indent=2))
    try: synthetic_demo(single_axis=True)
    except ValueError as e: print(json.dumps({'single_axis_rejected':str(e)}))
    if args.focal: print(json.dumps({'focal_selfcal_synthetic':synthetic_focal_demo()},indent=2))
    if args.so101: print(json.dumps({'SO101_URDF_synthetic':synthetic_so101_demo()},indent=2))
