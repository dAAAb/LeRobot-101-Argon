"""Unit tests for LeRobot 101 Argon: single-arm MotionGPT retargeting & Gemini Robotics ER 2."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from server import app as app_module
from server.er2_brain import GeminiRoboticsER2Brain, _clamp_1000, _extract_json
from server.gripper_power import SessionCancelled
from server.motiongpt_arm import (
    BODY_JOINTS,
    GESTURE_ANCHOR,
    detect_preferred_arm,
    execute_retargeted_motion,
    is_natural_motion_query,
    plan_single_arm_motion,
    translate_motion_query,
)


class SingleArmMotionGPTRetargetTests(unittest.TestCase):
    def test_detect_preferred_arm_and_query_translation(self):
        self.assertEqual(detect_preferred_arm('請用左手跟我揮揮手'), 'left')
        self.assertEqual(detect_preferred_arm('右手打一套拳'), 'right')
        self.assertEqual(detect_preferred_arm('伸個懶腰'), 'auto')

        spec_wave = translate_motion_query('左手揮揮手')
        self.assertEqual(spec_wave['id'], 'right-wave')
        self.assertEqual(spec_wave['preferred_arm'], 'left')
        self.assertIn('left', spec_wave['english_prompt'].lower())

        spec_box = translate_motion_query('打一套拳擊')
        self.assertEqual(spec_box['id'], 'boxing')
        self.assertEqual(spec_box['preferred_arm'], 'right')

        self.assertTrue(is_natural_motion_query('揮揮手'))
        self.assertTrue(is_natural_motion_query('伸懶腰'))
        self.assertFalse(is_natural_motion_query('把桌上的白色圓罐抓起來'))

    def test_retarget_all_presets_enforces_table_clearance_and_joint_bounds(self):
        cal = json.loads((Path(__file__).resolve().parent.parent / 'config' / 'follower-calibration.json').read_text())
        queries = [
            ('揮揮手', 'auto', 'right'),
            ('左手揮揮手', 'left', 'left'),
            ('伸懶腰', 'auto', 'right'),
            ('打一套拳擊', 'auto', 'right'),
            ('開心跳舞', 'auto', 'right'),
            ('鞠躬敬禮', 'auto', 'right'),
            ('舉手歡呼', 'auto', 'right'),
            ('點頭說好', 'auto', 'right'),
            ('招手過來', 'auto', 'right'),
        ]
        for query, pref, expected_arm in queries:
            with self.subTest(query=query, pref=pref):
                plan = plan_single_arm_motion(query, preferred_arm=pref)
                self.assertEqual(plan['arm_selection']['selected_arm'], expected_arm)
                self.assertTrue(plan['table_clearance_passed'])
                self.assertGreaterEqual(plan['min_table_clearance_mm'], 54.5)
                self.assertGreater(len(plan['waypoints']), 5)
                self.assertEqual(len(plan['skeleton_frames']), len(plan['so101_frames']))
                self.assertEqual(len(plan['skeleton_frames'][0]), 22)

                for wp in plan['waypoints']:
                    for joint in BODY_JOINTS:
                        val = wp['joints'][joint]
                        self.assertGreaterEqual(val, cal[joint]['range_min'] + 24)
                        self.assertLessEqual(val, cal[joint]['range_max'] - 24)

    def test_execute_retargeted_motion_glides_and_returns_receipt(self):
        plan = plan_single_arm_motion('揮揮手')
        fake_robot = MagicMock()
        fake_robot.stopped = False
        fake_robot.pose.return_value = {**GESTURE_ANCHOR, 'gripper': 2048}

        stages: list[tuple[str, str]] = []
        receipt = execute_retargeted_motion(
            fake_robot,
            plan,
            progress=lambda stage, msg: stages.append((stage, msg)),
        )
        self.assertTrue(receipt['completed'])
        self.assertEqual(receipt['motion_id'], 'right-wave')
        self.assertEqual(receipt['executed_waypoints'], len(plan['waypoints']))
        self.assertGreater(fake_robot.glide.call_count, len(plan['waypoints']))
        self.assertTrue(any(s == 'moving' for s, _ in stages))

    def test_execute_retargeted_motion_honors_stop_latch(self):
        plan = plan_single_arm_motion('伸懶腰')
        fake_robot = MagicMock()
        fake_robot.stopped = True
        with self.assertRaises(SessionCancelled):
            execute_retargeted_motion(fake_robot, plan)


class ER2BrainAndAPITests(unittest.TestCase):
    def test_er2_helpers_and_status(self):
        self.assertEqual(_clamp_1000(1250), 1000)
        self.assertEqual(_clamp_1000(-20), 0)
        self.assertEqual(_clamp_1000(432.6), 433)
        parsed = _extract_json('```json\n{"summary": "桌面有白罐", "items": []}\n```')
        self.assertEqual(parsed['summary'], '桌面有白罐')

        er2 = GeminiRoboticsER2Brain()
        st = er2.status_dict()
        self.assertIn('model', st)
        self.assertEqual(st['model'], 'gemini-robotics-er-2-preview')

    def test_motiongpt_and_status_api_endpoints(self):
        client = TestClient(app_module.app)
        headers = {'x-lerobot-token': app_module.TOKEN}

        status_res = client.get('/api/status')
        self.assertEqual(status_res.status_code, 200)
        status_json = status_res.json()
        self.assertIn('er2', status_json)
        self.assertIn('motiongpt', status_json)
        self.assertTrue(status_json['capabilities']['motiongpt'])

        presets_res = client.get('/api/motiongpt/presets')
        self.assertEqual(presets_res.status_code, 200)
        self.assertGreaterEqual(len(presets_res.json()['presets']), 8)

        plan_res = client.post(
            '/api/motiongpt/plan',
            headers=headers,
            json={'query': '右手揮揮手', 'preferred_arm': 'right'},
        )
        self.assertEqual(plan_res.status_code, 200)
        plan_data = plan_res.json()
        self.assertEqual(plan_data['id'], 'right-wave')
        self.assertTrue(plan_data['table_clearance_passed'])

        cfg_astra = client.post(
            '/api/brain/config',
            headers=headers,
            json={'engine': 'astra', 'thinking_level': 'medium'},
        )
        self.assertEqual(cfg_astra.status_code, 200)
        self.assertEqual(cfg_astra.json()['brain']['engine'], 'astra')
        self.assertEqual(cfg_astra.json()['brain']['model'], 'gpt-6-astra')
        self.assertEqual(cfg_astra.json()['brain']['thinking_level'], 'medium')

        cfg_argon = client.post(
            '/api/brain/config',
            headers=headers,
            json={'engine': 'argon', 'thinking_level': 'low'},
        )
        self.assertEqual(cfg_argon.status_code, 200)
        self.assertEqual(cfg_argon.json()['brain']['engine'], 'argon')
        self.assertEqual(cfg_argon.json()['brain']['model'], 'gemini-robotics-er-2-preview')
        self.assertEqual(cfg_argon.json()['brain']['thinking_level'], 'low')


if __name__ == '__main__':
    unittest.main()
