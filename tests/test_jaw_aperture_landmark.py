"""Synthetic image tests; no captured image editing or motor/camera access."""
import importlib.util,unittest
import cv2,numpy as np
from server import jaw_aperture_landmark as m
LEFT={'x':.25,'y':.30,'w':.16,'h':.38}
RIGHT={'x':.48,'y':.30,'w':.18,'h':.38}
def image(missing=False,blur=3,step=False):
    H,W=480,640;im=np.full((H,W),130,np.uint8)
    for y in range(H):
        left=210+int(.04*(y-240));right=350+int(.03*(y-240))
        if step and y>285:right+=30
        im[y,:left]=200
        if not missing:im[y,right:]=200
    return cv2.GaussianBlur(im,(0,0),blur)
class Tests(unittest.TestCase):
    def analyze(self,im):return m.aperture_landmark(im,left_inner_roi=LEFT,right_inner_roi=RIGHT)
    def test_blurred_aperture_is_localized_without_features(self):
        r=self.analyze(image());self.assertEqual(r['status'],'ok')
        self.assertAlmostEqual(r['aperture_center_px'][0],280,delta=2)
        self.assertAlmostEqual(r['visible_gap_px'],140,delta=2)
        self.assertFalse(r['calibrated_grasp_tcp']);self.assertFalse(r['motion_authorized'])
    def test_second_jaw_step_is_rejected_from_line(self):
        r=self.analyze(image(step=True));self.assertEqual(r['status'],'ok')
        self.assertAlmostEqual(r['visible_gap_px'],140,delta=2)
        self.assertLess(r['right']['inlier_fraction'],.9)
    def test_missing_jaw_is_not_aperture(self):
        self.assertEqual(self.analyze(image(missing=True))['status'],'insufficient')
    def test_featureless_image_is_not_aperture(self):
        self.assertEqual(self.analyze(np.full((480,640),128,np.uint8))['status'],'insufficient')
    def test_bad_rois_rejected(self):
        with self.assertRaises(ValueError):m.aperture_landmark(image(),left_inner_roi={**LEFT,'w':1},right_inner_roi=RIGHT)
if __name__=='__main__':unittest.main()
