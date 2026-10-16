"""Profile target selection and real glide on synthetic register buses only."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from server.jaw_opening import empty_opening_contract, require_opening_contract
from server.gripper_power import SessionCancelled
from server.spatial_contract import JawCalibration, JawSample
import test_active_open as open_fixtures


def profile(revision='synthetic-opening'):
    return JawCalibration(revision,'test-geometry',(
        JawSample(1554,[-.01,0.,.1],[.01,0.,.1]),
        JawSample(2046,[-.01,0.,.1],[.055,0.,.1])),[0.,0.,1.],
        .002,'validated','synthetic-heldout','independent-heldout')


def selection(jaw=None):
    return ('test-geometry',None,jaw or profile(),{})


class OpeningContractTests(unittest.TestCase):
    def setUp(self):
        self.config={'spatial_backend':'v2','jaw_calibration_profile':'test-only.json'}
        self.model=SimpleNamespace(calibration={'gripper':{'range_min':1456,'range_max':2983}})
    def test_profile_target_leaves_encoder_reserve_and_discloses_aperture(self):
        with patch('server.jaw_opening._profile_snapshot',return_value=selection()):
            contract=empty_opening_contract(self.model,self.config)
            self.assertEqual(contract['target_raw'],2038)
            self.assertEqual(contract['measured_raw_range'],[1554,2046])
            self.assertGreater(contract['target_geometry']['aperture_m'],.064)
            self.assertEqual(require_opening_contract(contract,self.model,self.config),contract)
            altered=deepcopy(contract);altered['target_raw']+=1
            with self.assertRaisesRegex(ValueError,'重新審核'):
                require_opening_contract(altered,self.model,self.config)
        with patch('server.jaw_opening._profile_snapshot',return_value=selection(profile('changed'))):
            with self.assertRaisesRegex(ValueError,'重新審核'):
                require_opening_contract(contract,self.model,self.config)
    def test_missing_v2_selection_keeps_legacy_and_decreasing_aperture_rejected(self):
        with patch('server.jaw_opening._profile_snapshot',side_effect=AssertionError('no profile read')):
            self.assertEqual(empty_opening_contract(self.model,{})['target_raw'],1860)
            self.assertEqual(empty_opening_contract(self.model,{'spatial_backend':'v2'})['target_raw'],1860)
        jaw=JawCalibration('reversed','test-geometry',(
            JawSample(1554,[-.01,0.,.1],[.055,0.,.1]),JawSample(2046,[-.01,0.,.1],[.01,0.,.1])),
            [0.,0.,1.],.002,'validated','synthetic-heldout','independent-heldout')
        with patch('server.jaw_opening._profile_snapshot',return_value=selection(jaw)):
            with self.assertRaisesRegex(ValueError,'全程只會張開'):
                empty_opening_contract(self.model,self.config)


class WiderActiveOpenTests(unittest.TestCase):
    """Use the actual controller with a synthetic bus; no inherited test copies."""
    model=open_fixtures.ActiveOpenTests.model
    capture=open_fixtures.ActiveOpenTests.capture
    setUp=open_fixtures.ActiveOpenTests.setUp
    open=open_fixtures.ActiveOpenTests.open
    def wide_setup(self):
        for table in (self.robot.goals,self.initial,self.initial_holds):table['gripper']=1856
        self.bus.reg[6].update({42:1856,56:1856})
        self.robot.config.update(spatial_backend='v2',jaw_calibration_profile='test-only.json')
        selected=patch('server.jaw_opening._profile_snapshot',return_value=selection())
        selected.start();self.addCleanup(selected.stop)
        return empty_opening_contract(self.model,self.robot.config)
    def test_explicit_wide_contract_moves_only_jaw_at_low_pwm_and_empty_finish_keeps_opening(self):
        contract=self.wide_setup();opened=self.open(opening_contract=contract,cancel=self.cancel)
        self.assertEqual(opened['joints']['gripper'],2038)
        self.assertEqual(self.bus.reg[6][48],100)
        self.assertEqual({e[:2] for e in self.bus.events},{(6,48),(6,42)})
        self.assertEqual(self.robot.read()['gripper'],2038)
        receipt=self.robot.gripper_power.finish()
        self.assertEqual(receipt['neutral_goal'],2038)
        self.assertEqual(self.bus.reg[6][42],2038)
        self.assertEqual(self.bus.reg[6][48],500)
        self.assertEqual(self.robot.read(),{**self.initial,'gripper':2038})
    def test_existing_caller_never_silently_receives_the_wider_profile_target(self):
        self.wide_setup();opened=self.open(cancel=self.cancel)
        self.assertEqual(opened['joints']['gripper'],1860)
    def test_changed_target_is_rejected_before_pwm_or_goal_writes(self):
        contract=self.wide_setup();contract['target_raw']=2040
        with self.assertRaisesRegex(ValueError,'重新審核'):
            self.open(opening_contract=contract,cancel=self.cancel)
        self.assertEqual(self.bus.events,[])
    def test_profile_change_during_fresh_capture_is_rejected_before_pwm(self):
        contract=self.wide_setup()
        with patch('server.jaw_opening._profile_snapshot',side_effect=[selection(),selection(profile('changed'))]):
            with self.assertRaisesRegex(ValueError,'重新審核'):
                self.open(opening_contract=contract,cancel=self.cancel)
        self.assertEqual(self.bus.events,[])
    def test_wide_open_cancel_retains_low_cap_without_reaching_target(self):
        contract=self.wide_setup();write=self.robot._write;goals=[]
        def checked(name,address,value,*args):
            result=write(name,address,value,*args)
            if name=='gripper' and address==42:
                goals.append(value)
                if len(goals)==5:self.cancel.set()
            return result
        self.robot._write=checked
        with self.assertRaises(SessionCancelled):self.open(opening_contract=contract,cancel=self.cancel)
        self.assertLess(self.bus.reg[6][56],2038)
        self.assertEqual(self.bus.reg[6][48],100)
        self.assertNotIn((6,48,500),self.bus.events)


if __name__=='__main__':unittest.main()
