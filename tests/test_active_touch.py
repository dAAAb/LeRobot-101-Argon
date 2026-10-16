"""Acquisition must reject an unmatched pose before any motion/power entry."""
import json,tempfile,threading,time,unittest
from pathlib import Path
from unittest.mock import Mock
from server.active_touch import collect_empty_jaw
from server.empty_jaw_baseline import analyze,synthetic

class AcquisitionContextTests(unittest.TestCase):
    def test_missing_or_different_body_pose_cannot_fit_a_baseline(self):
        for pose in [None,dict.fromkeys(('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll'),2010)]:
            rows=synthetic();rows[50]['body_position']=pose
            self.assertFalse(analyze(rows)['fit_allowed'])

    def test_validation_rejects_changed_start_before_begin_or_low_pwm(self):
        with tempfile.TemporaryDirectory() as directory:
            rows=synthetic();report=analyze(rows)
            runtime=Path(directory)
            (runtime/'gripper-empty-baseline.json').write_text(json.dumps(report))
            (runtime/'gripper-empty-latest.jsonl').write_text(json.dumps({'type':'sample','capture_end_monotonic':time.monotonic()-1})+'\n')
            robot=Mock();robot.runtime=runtime
            robot.read.return_value={**report['body_context']['reference'],'wrist_roll':2005,'gripper':1520}
            with self.assertRaisesRegex(ValueError,'Body pose'):
                collect_empty_jaw(robot,Mock(),threading.Event(),{},validation_only=True)
            robot.begin.assert_not_called();robot.glide.assert_not_called();robot.gripper_power.enter.assert_not_called()
