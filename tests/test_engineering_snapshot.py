"""Current-pose capture never commands motors or attaches historic points."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import importlib.util
from unittest.mock import Mock, patch

from server.engineering_snapshot import capture_snapshot, publish_snapshot
from server.local_cartesian import ALL


class EngineeringSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.clock=10.
        self.pose={name:2000 for name in ALL}
        def read():
            self.clock+=.01
            return dict(self.pose)
        def sleep(seconds):self.clock+=seconds
        self.robot=SimpleNamespace(connected=True,moving=False,pwm_owner=None,read=Mock(side_effect=read),
                                   begin=Mock(),hold=Mock(),connect=Mock(),glide=Mock(),stop=Mock())
        self.camera=SimpleNamespace(generation='wrist-A',
            latest_stamp=Mock(side_effect=lambda:{'frame_capture_monotonic':self.clock-.005,'camera_source':'wrist'}),
            snapshot=Mock(side_effect=lambda:{'id':'fresh','source':'wrist','camera_generation':'wrist-A',
                'capture_monotonic':self.clock-.005,'captured_at':100.,'path':'fresh.jpg'}))
        for name,replacement in [('monotonic',lambda:self.clock),('sleep',sleep)]:
            guard=patch('server.engineering_snapshot.time.'+name,side_effect=replacement)
            guard.start();self.addCleanup(guard.stop)

    def tearDown(self):
        for name in ('begin','hold','connect','glide','stop'):
            self.assertEqual(getattr(self.robot,name).mock_calls,[],name)

    def test_read_only_snapshot_pairs_fresh_exposure_with_stable_six_axis_reads(self):
        frame=capture_snapshot(self.robot,self.camera)
        self.assertEqual(frame['joints'],self.pose)
        self.assertEqual(frame['joint_capture_span_ticks'],0)
        self.assertGreaterEqual(frame['settled_duration_s'],.6)
        self.assertEqual(self.robot.read.call_count,7)
        self.camera.snapshot.assert_called_once()
        a,b=frame['joint_observation_bracket_monotonic']
        self.assertLessEqual(a,frame['capture_monotonic']);self.assertGreaterEqual(b,frame['capture_monotonic'])

    def test_drift_reconnection_and_old_frame_reject_without_hold_or_rearm(self):
        original=self.robot.read.side_effect
        def drift():
            result=original()
            if self.robot.read.call_count==7:result['elbow_flex']+=3
            return result
        self.robot.read.side_effect=drift
        with self.assertRaisesRegex(RuntimeError,'停穩'):capture_snapshot(self.robot,self.camera)
        self.robot.read.side_effect=original
        self.camera.snapshot.side_effect=lambda:{'source':'wrist','camera_generation':'other','capture_monotonic':self.clock-.005}
        with self.assertRaisesRegex(RuntimeError,'配對'):capture_snapshot(self.robot,self.camera)
        self.camera.snapshot.side_effect=lambda:{'source':'wrist','camera_generation':'wrist-A','capture_monotonic':1.}
        with self.assertRaisesRegex(RuntimeError,'配對'):capture_snapshot(self.robot,self.camera)

    def test_disconnected_read_fails_without_connecting_or_rearming(self):
        self.robot.connected=False
        with self.assertRaises(RuntimeError):capture_snapshot(self.robot,self.camera)
        self.robot.read.assert_not_called()

    def test_malformed_final_read_is_rejected_without_changing_motor_state(self):
        for bad in ({}, {**self.pose,'elbow_flex':True}, {**self.pose,'elbow_flex':float('nan')},
                    {**self.pose,'elbow_flex':4096}):
            self.robot.read.side_effect=[dict(self.pose)]*6+[bad]
            with self.subTest(bad=bad), self.assertRaisesRegex(RuntimeError,'完整'):
                capture_snapshot(self.robot,self.camera)

    def test_publish_preserves_historic_bundle_and_new_pose_has_no_old_depth(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'sim-scene';old=root/'current';old.mkdir(parents=True)
            (old/'depth.json').write_text('{"old_points":true}')
            (old/'scene.json').write_text('{"old_pose":true}')
            def export(destination,frame):
                destination.mkdir()
                for name in ('scene.glb','scene.xml','verification.json','wrist.jpg'):
                    (destination/name).write_text('new')
                return {'frame':frame,'artifacts':{'glb':'scene.glb','wrist_image':'wrist.jpg'}}
            with patch('server.engineering_snapshot._export_snapshot_bundle',side_effect=export) as exporter:
                result=publish_snapshot(folder,{'id':'new','captured_at':200})
            exporter.assert_called_once()
            self.assertEqual(exporter.call_args.args[1]['id'],'new')
            self.assertEqual(exporter.call_args.kwargs,{})
            self.assertTrue(old.is_symlink());self.assertFalse((old/'depth.json').exists())
            self.assertFalse(result['depth_included'])
            self.assertEqual(json.loads((old/'scene.json').read_text())['frame']['id'],'new')
            archived=list(root.glob('archive-*'));self.assertEqual(len(archived),1)
            self.assertTrue((archived[0]/'depth.json').is_file())
            with patch('server.engineering_snapshot._export_snapshot_bundle',side_effect=RuntimeError('export failed')):
                with self.assertRaises(RuntimeError):publish_snapshot(folder,{'id':'failed'})
            self.assertEqual(old.resolve().name,result['snapshot_id'])

    def test_missing_isolated_runtime_does_not_publish_or_replace_old_bundle(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);current=root/'sim-scene/current';current.mkdir(parents=True)
            (current/'scene.json').write_text('old snapshot')
            with patch('server.engineering_snapshot.ROOT',root), patch('server.engineering_snapshot.importlib.util.find_spec',return_value=None):
                with self.assertRaisesRegex(RuntimeError,'尚未備妥'):
                    publish_snapshot(root,{'id':'new'})
            self.assertEqual((current/'scene.json').read_text(),'old snapshot')

    def test_existing_isolated_worker_exports_a_real_complete_depth_free_bundle(self):
        from server.engineering_snapshot import ROOT
        worker=ROOT/'.runtime/mujoco-venv/bin/python'
        if not worker.is_file():self.skipTest('Existing isolated engineering runtime is unavailable')
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);image=root/'wrist.jpg'
            cv2.imwrite(str(image),np.full((80,120,3),128,dtype=np.uint8))
            frame={'id':'offline-worker-test','source':'wrist','camera_generation':'fixture',
                   'capture_monotonic':100.,'captured_at':1000.,'path':str(image),'image_size':[120,80],
                   'joints':dict(zip(ALL,[2314,1016,3042,2140,765,1859]))}
            # Force only the interpreter-selection branch; the existing worker,
            # MuJoCo, trimesh, meshes and export verification all run for real.
            with patch('server.engineering_snapshot.importlib.util.find_spec',return_value=None):
                result=publish_snapshot(root,frame)
            current=root/'sim-scene/current'
            state=json.loads((current/'scene.json').read_text())
            self.assertEqual(state['frame']['id'],frame['id'])
            self.assertEqual(state['frame']['joints'],frame['joints'])
            self.assertTrue(state['verification']['passed'])
            self.assertFalse((current/'depth.json').exists())
            self.assertFalse(result['depth_included'])
            for name in ('scene.glb','scene.xml','verification.json','wrist.jpg'):
                self.assertGreater((current/name).stat().st_size,0)


if __name__=='__main__':unittest.main()
