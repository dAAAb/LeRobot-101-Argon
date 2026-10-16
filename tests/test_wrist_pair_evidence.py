"""Synthetic offline tests only; no captured image editing, hardware or project writes."""
import importlib.util, unittest, json, time
import cv2,numpy as np
from pathlib import Path
from server import wrist_pair_evidence as e
W,H=640,480
TARGET=(260,110,120,150);FINGER=(55,340,170,110)
def box(r):x,y,w,h=r;return dict(x=x/W,y=y/H,w=w/W,h=h/H)
def texture(rng,shape):
    return cv2.GaussianBlur(rng.integers(25,230,shape,dtype=np.uint8),(3,3),.65)
def synthetic_frames(*,attached=True,occlude=False,background_moving=True,scales=None,count=3):
    rng=np.random.default_rng(34);base=texture(rng,(H,W));target=texture(rng,(TARGET[3],TARGET[2]));finger=texture(rng,(FINGER[3],FINGER[2]));frames=[]
    for i in range(count):
        d=4*i if background_moving else 0
        im=cv2.warpAffine(base,np.float32([[1,0,d],[0,1,0]]),(W,H),borderMode=cv2.BORDER_REFLECT)
        x,y,w,h=TARGET;scale=scales[i] if scales else 1
        patch=cv2.resize(target,(round(w*scale),round(h*scale)))
        x+=0 if attached else d;x-=round((patch.shape[1]-w)/2);y-=round((patch.shape[0]-h)/2)
        im[y:y+patch.shape[0],x:x+patch.shape[1]]=128 if occlude and i==count-1 else patch
        x,y,w,h=FINGER;im[y:y+h,x:x+w]=finger
        frames.append({'image':im,'source':'wrist','camera_generation':'synthetic-one','capture_monotonic':100+i*.1,'sequence':i})
    return frames
class EvidenceTests(unittest.TestCase):
    def analyze(self,frames,**kwargs):return e.analyze_wrist_sequence(frames,box(TARGET),fixed_finger_roi=box(FINGER),**kwargs)
    def test_attached_target_stationary_pad_moving_background_is_evidence_only(self):
        r=self.analyze(synthetic_frames());self.assertEqual(r['target']['status'],'ok')
        self.assertTrue(r['lift_evidence']['object_following_gripper_evidence'],r)
        self.assertFalse(r['lift_evidence']['confirmed_grasp']);self.assertFalse(r['lift_evidence']['true_grasp'])
        self.assertIsNone(r['lift_evidence']['force_estimate'])
    def test_world_target_moves_with_background_not_pad(self):
        r=self.analyze(synthetic_frames(attached=False))
        self.assertAlmostEqual(r['target']['median_dx_px'],8,delta=.3)
        self.assertFalse(r['lift_evidence']['object_following_gripper_evidence'])
        self.assertEqual(r['lift_evidence']['state'],'target_moves_relative_to_fixed_finger')
    def test_no_motion_cannot_be_grasp_evidence(self):
        r=self.analyze(synthetic_frames(background_moving=False))
        self.assertFalse(r['lift_evidence']['object_following_gripper_evidence'])
    def test_occlusion_is_insufficient_never_grasp(self):
        r=self.analyze(synthetic_frames(occlude=True))
        self.assertNotEqual(r['target']['status'],'ok');self.assertFalse(r['lift_evidence']['object_following_gripper_evidence'])
    def test_target_scale_is_recovered_without_dimensions(self):
        r=self.analyze(synthetic_frames(scales=[1,1.02,1.04]))
        self.assertEqual(r['target']['status'],'ok');self.assertAlmostEqual(r['target']['median_scale'],1.04,delta=.015)
    def test_missing_pad_does_not_claim_attachment(self):
        r=e.analyze_wrist_sequence(synthetic_frames(),box(TARGET))
        self.assertEqual(r['fixed_finger']['status'],'not_supplied');self.assertFalse(r['lift_evidence']['object_following_gripper_evidence'])
    def test_generation_reconnect_is_rejected(self):
        f=synthetic_frames();f[-1]['camera_generation']='new'
        with self.assertRaises(ValueError):self.analyze(f)
    def test_nonmonotonic_frame_rejected(self):
        f=synthetic_frames();f[-1]['capture_monotonic']=f[0]['capture_monotonic']
        with self.assertRaises(ValueError):self.analyze(f)
    def test_overlapping_finger_target_rois_rejected(self):
        with self.assertRaises(ValueError):e.analyze_wrist_sequence(synthetic_frames(),box(TARGET),fixed_finger_roi=box(TARGET))
    def test_pair_api_labels_endpoint_limit(self):
        f=synthetic_frames(count=2)
        r=e.analyze_wrist_pair(f[0],f[1],box(TARGET),fixed_finger_roi=box(FINGER))
        self.assertEqual(r['temporal_basis'],'endpoint-only correspondence')
        json.dumps(r)
if __name__=='__main__':
    unittest.main()
