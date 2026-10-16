"""Offline engineering export. SDK is never imported and no device is opened."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scripts.observer_depth import capture,export_observation,select_pair
from types import SimpleNamespace
from unittest.mock import Mock,patch
from server.world_state import build_world_state,policy_state


class ObserverDepthTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.destination=Path(self.temp.name)/'artifact'
        self.depth=np.array([[1000,0],[1200,1500]],dtype=np.uint16)
        self.points=np.array([[0.,0.,1.],[0.,0.,0.],[0.,.1,1.2],[.1,.1,1.5]],dtype=np.float32)
        self.metadata={'depth_scale_m':.001,'depth_intrinsics':{'width':2,'height':2},
            'color_intrinsics':{},'depth_to_color_extrinsics':{},'device_serial':'synthetic',
            'device_name':'synthetic D435 fixture','depth_timestamp':{},'color_timestamp':{},'host_receipt_monotonic':100.}
        self.color=np.zeros((2,2,3),dtype=np.uint8)

    def export(self,**changes):
        values={'depth_raw':self.depth,'color_rgb':self.color,'points_m':self.points,'metadata':self.metadata}
        values.update(changes)
        return export_observation(self.destination,**values)

    def test_artifact_has_optical_coordinates_unknown_registration_and_invalid_depth_mask(self):
        result=self.export()
        self.assertEqual(result['source'],'engineering_observer')
        self.assertFalse(result['production_eligible']);self.assertIsNone(result['robot_base_transform'])
        self.assertIsNone(result['object_dimensions_m']);self.assertEqual(result['valid_depth_fraction'],.75)
        self.assertEqual(result['geometry']['sample_count'],3)
        with np.load(self.destination/'measurements.npz') as data:
            self.assertFalse(data['native_valid_mask'][0,1]);self.assertEqual(data['sample_points_m'].shape,(3,3))
        self.assertTrue((self.destination/'state.json').exists())

    def test_unit_mismatch_is_rejected_before_writing_artifact(self):
        with self.assertRaisesRegex(ValueError,'scale'):self.export(points_m=self.points*1000)
        self.assertFalse(self.destination.exists())

    def test_geometry_is_not_invented_when_all_depth_is_invalid(self):
        result=self.export(depth_raw=np.zeros((2,2),dtype=np.uint16),points_m=np.zeros((4,3)))
        self.assertIsNone(result['geometry']);self.assertEqual(result['valid_depth_fraction'],0)

    def test_invalid_shapes_scales_and_intrinsics_are_rejected(self):
        for updates in ({'depth_raw':self.depth.astype(float)},
                        {'points_m':self.points[:2]},
                        {'metadata':{**self.metadata,'depth_scale_m':float('nan')}},
                        {'metadata':{**self.metadata,'depth_intrinsics':{'width':3,'height':2}}},
                        {'aligned_depth_raw':np.zeros((4,4),dtype=np.uint16)}):
            with self.assertRaises(ValueError):self.export(**updates)

    def test_short_bag_returns_last_valid_pair_at_normal_eof(self):
        first=SimpleNamespace(get_timestamp=lambda:100.)
        pair=SimpleNamespace(get_depth_frame=lambda:first,get_color_frame=lambda:True)
        pipeline=SimpleNamespace(wait_for_frames=Mock(side_effect=[pair,RuntimeError('EOF')]))
        playback=SimpleNamespace(current_status=lambda:'stopped')
        self.assertIs(select_pair(pipeline,2.,playback=playback,stopped_status='stopped'),pair)

    def test_bag_uses_sensor_time_not_wall_clock_and_io_failure_is_not_eof(self):
        def pair(stamp):return SimpleNamespace(get_depth_frame=lambda:SimpleNamespace(get_timestamp=lambda:stamp),get_color_frame=lambda:True)
        first,last=pair(100.),pair(2200.)
        pipeline=SimpleNamespace(wait_for_frames=Mock(side_effect=[first,last]))
        with patch('scripts.observer_depth.time.monotonic',return_value=5.):
            self.assertIs(select_pair(pipeline,2.,playback=SimpleNamespace(current_status=lambda:'playing'),stopped_status='stopped'),last)
        pipeline.wait_for_frames=Mock(side_effect=[first,RuntimeError('USB error')])
        with self.assertRaisesRegex(RuntimeError,'USB error'):
            select_pair(pipeline,2.,playback=SimpleNamespace(current_status=lambda:'playing'),stopped_status='stopped')

    def test_no_overwrite_and_no_production_state_conversion(self):
        value=self.export()
        with self.assertRaises(FileExistsError):self.export()
        with self.assertRaises(ValueError):build_world_state(value,None)
        with self.assertRaises(ValueError):policy_state(value)

    def fake_sdk(self,*,name='RealSense D435I',serial='exact-serial',started_device=None):
        """Identity/stream lifecycle fake; never imports the installed SDK."""
        events=[]
        device=SimpleNamespace(get_info=lambda key:{'name':name,'serial':serial}[key])
        resolved=SimpleNamespace(get_device=lambda:device)
        started=SimpleNamespace(get_device=lambda:started_device or device)
        context=object();wrapper=object()
        pipeline=SimpleNamespace(
            set_device=Mock(side_effect=lambda value:events.append(('set_device',value))),
            start=Mock(side_effect=lambda value:(events.append(('start',value)),started)[1]),
            stop=Mock(side_effect=lambda:events.append(('stop',None))))
        config=SimpleNamespace(enable_device=Mock(),enable_stream=Mock(),
            resolve=Mock(side_effect=lambda value:(events.append(('resolve',value)),resolved)[1]))
        sdk=SimpleNamespace(context=Mock(return_value=context),pipeline=Mock(return_value=pipeline),
            pipeline_wrapper=Mock(return_value=wrapper),config=Mock(return_value=config),
            product_line=SimpleNamespace(D400=2),stream=SimpleNamespace(depth='depth',color='color'),
            format=SimpleNamespace(z16='z16',rgb8='rgb8'),
            camera_info=SimpleNamespace(name='name',serial_number='serial'),
            playback_status=SimpleNamespace(stopped='stopped'))
        return sdk,context,wrapper,pipeline,config,device,events

    def test_live_uses_identity_checked_same_device_and_context_before_start(self):
        sdk,context,wrapper,pipeline,config,device,events=self.fake_sdk()
        with patch.dict('sys.modules',{'pyrealsense2':sdk}), \
             patch('scripts.observer_depth.select_pair',side_effect=RuntimeError('fixture end of capture')):
            with self.assertRaisesRegex(RuntimeError,'fixture end'):
                capture(serial='exact-serial',seconds=.2)
        sdk.context.assert_called_once_with({'inherit':False,'device-mask':2,'dds':{'enabled':False}})
        sdk.pipeline.assert_called_once_with(context)
        sdk.pipeline_wrapper.assert_called_once_with(pipeline)
        config.resolve.assert_called_once_with(wrapper)
        config.enable_device.assert_called_once_with('exact-serial')
        self.assertEqual(events,[('resolve',wrapper),('set_device',device),('start',config),('stop',None)])
        pipeline.stop.assert_called_once()

    def test_live_wrong_model_or_serial_is_rejected_without_starting_stream(self):
        for changes in ({'name':'RealSense D455'},{'serial':'different-serial'}):
            with self.subTest(changes=changes):
                sdk,_,_,pipeline,_,_,_=self.fake_sdk(**changes)
                with patch.dict('sys.modules',{'pyrealsense2':sdk}), \
                     patch('scripts.observer_depth.select_pair') as frames:
                    with self.assertRaisesRegex(RuntimeError,'explicitly requested D435'):
                        capture(serial='exact-serial')
                pipeline.set_device.assert_not_called();pipeline.start.assert_not_called()
                pipeline.stop.assert_not_called();frames.assert_not_called()

    def test_live_rechecks_actual_device_and_stops_if_identity_changed(self):
        other=SimpleNamespace(get_info=lambda key:{'name':'RealSense D435I','serial':'unexpected'}[key])
        sdk,_,_,pipeline,_,_,_=self.fake_sdk(started_device=other)
        with patch.dict('sys.modules',{'pyrealsense2':sdk}), \
             patch('scripts.observer_depth.select_pair') as frames:
            with self.assertRaisesRegex(RuntimeError,'explicitly requested D435'):
                capture(serial='exact-serial')
        pipeline.start.assert_called_once();pipeline.stop.assert_called_once();frames.assert_not_called()


if __name__=='__main__':unittest.main()
