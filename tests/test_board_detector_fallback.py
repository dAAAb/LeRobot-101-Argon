"""Classic fallback keeps native coordinates and rejects genuinely missing corners."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from server.board_calibration import detect_board


class BoardDetectorFallbackTests(unittest.TestCase):
    def board(self, incomplete=False):
        square = 72
        source = np.full((800, 1000), 255, np.uint8)
        for y in range(7):
            for x in range(10):
                if (x + y) % 2 == 0:
                    source[100+y*square:100+(y+1)*square,
                           100+x*square:100+(x+1)*square] = 0
        if incomplete:
            source[:, 100:100+3*square] = 255
        homography = cv2.getPerspectiveTransform(
            np.array([[0, 0], [999, 0], [999, 799], [0, 799]], np.float32),
            np.array([[360, 160], [1600, 260], [1420, 1120], [270, 960]], np.float32))
        image = cv2.warpPerspective(source, homography, (2003, 1199), borderValue=255)
        corners = np.mgrid[1:10, 1:7].T.reshape(-1, 2) * square + [99.5, 99.5]
        expected = cv2.perspectiveTransform(corners.astype(np.float32)[None], homography)[0]
        return image, expected

    def test_sb_false_negative_falls_back_once_and_refines_real_native_corners(self):
        image, expected = self.board()
        actual_classic = cv2.findChessboardCorners
        with patch('server.board_calibration.cv2.findChessboardCornersSB', return_value=(False, None)), \
                patch('server.board_calibration.cv2.findChessboardCorners', wraps=actual_classic) as classic:
            detection = detect_board(image)
        self.assertTrue(detection['found'], detection)
        self.assertEqual(classic.call_count, 1)
        self.assertLessEqual(max(classic.call_args.args[0].shape), 960)
        self.assertEqual(len(detection['corners_px']), 54)
        errors = np.linalg.norm(np.array(detection['corners_px']) - expected, axis=1)
        self.assertLess(float(np.percentile(errors, 95)), .2)
        self.assertGreater(np.max(detection['corners_px']), 960)
        self.assertEqual(detection['detector']['method'], 'bounded-classic-fallback-and-native-subpixel')

    def test_actual_missing_columns_are_rejected_by_both_detector_paths(self):
        image, _ = self.board(incomplete=True)
        actual_classic = cv2.findChessboardCorners
        with patch('server.board_calibration.cv2.findChessboardCorners', wraps=actual_classic) as classic:
            detection = detect_board(image)
        self.assertEqual(classic.call_count, 1)
        self.assertTrue(detection['detector']['classic_fallback_attempted'])
        self.assertFalse(detection['found'])
        self.assertEqual(detection['corners_px'], [])

    def test_sb_success_does_not_run_fallback(self):
        image, _ = self.board()
        with patch('server.board_calibration.cv2.findChessboardCorners') as classic:
            detection = detect_board(image)
        self.assertTrue(detection['found'])
        classic.assert_not_called()
        self.assertFalse(detection['detector']['classic_fallback_attempted'])


if __name__ == '__main__':
    unittest.main()
