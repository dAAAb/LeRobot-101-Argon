"""Offline checks for the perception schemas passed to Codex Structured Outputs.

These inspect the full wire schema, including referenced definitions. They do not
invoke Astra or assert that schema compatibility proves a successful inference.
"""
from copy import deepcopy
import unittest

from pydantic import ValidationError

from server.brain import (
    Scene, Inspection, GraspAdvice, GraspVerification, JawView, WaveClearance,
)


class PerceptionSchemaTests(unittest.TestCase):
    def test_every_perception_output_has_closed_required_objects(self):
        # WaveControllerGeometry is controller input, not an Astra output; its
        # locally optional fields must not be confused with the wire contract.
        for model in (Scene, Inspection, GraspAdvice, GraspVerification, JawView,
                      WaveClearance):
            with self.subTest(model=model.__name__):
                schema = model.model_json_schema()
                objects = []

                def inspect(node, path='$'):
                    if isinstance(node, dict):
                        if node.get('type') == 'object':
                            objects.append(path)
                            self.assertIs(node.get('additionalProperties'), False, path)
                            self.assertEqual(set(node.get('required', [])),
                                             set(node.get('properties', {})), path)
                        self.assertNotIn('default', node, path)
                        for key, value in node.items():
                            inspect(value, f'{path}.{key}')
                    elif isinstance(node, list):
                        for index, value in enumerate(node):
                            inspect(value, f'{path}[{index}]')

                inspect(schema)
                self.assertTrue(objects)

    def test_scene_uses_closed_box_and_retains_identification_and_action_fields(self):
        scene = {
            'query': '白罐', 'summary': '可見一個白色圓筒。',
            'objects': [{
                'name': '白色圓筒', 'description': '白色筒身，頂端有蓋。',
                'confidence': .85, 'bbox': {'x': .4, 'y': .2, 'w': .1, 'h': .6},
                'frame_index': 3, 'matches_query': True, 'safe_to_touch': True,
                'actions': [{'kind': 'pick', 'label': '嘗試夾取',
                             'reason': '先確認夾取位置。', 'target': '白色圓筒'}],
            }],
            'selected_index': 0, 'empty': False, 'uncertainty': '距離尚未測量。',
        }
        self.assertEqual(Scene.model_validate(scene).model_dump(), scene)
        for member in ('confidence', 'matches_query', 'safe_to_touch', 'actions'):
            missing = deepcopy(scene)
            del missing['objects'][0][member]
            with self.subTest(member=member), self.assertRaises(ValidationError):
                Scene.model_validate(missing)
        for bbox in ({'x': .4, 'y': .2, 'w': .1},
                     {'x': .4, 'y': .2, 'w': .1, 'h': .6, 'depth': .2}):
            malformed = deepcopy(scene)
            malformed['objects'][0]['bbox'] = bbox
            with self.subTest(bbox=bbox), self.assertRaises(ValidationError):
                Scene.model_validate(malformed)

    def test_unobserved_inspection_box_is_explicit_null_not_an_omitted_field(self):
        invisible = {
            'visible': False, 'confidence': 0., 'object_name': '',
            'summary': '目前看不到指定物件。', 'attributes': [], 'visible_text': [],
            'unknowns': ['物件是否仍在桌上'], 'bbox': None,
            'safe_to_touch': False, 'actions': [],
        }
        self.assertEqual(Inspection.model_validate(invisible).model_dump(), invisible)
        del invisible['bbox']
        with self.assertRaises(ValidationError):
            Inspection.model_validate(invisible)
        for model in (Inspection, GraspAdvice):
            with self.subTest(model=model.__name__):
                schema = model.model_json_schema()
                self.assertIn('bbox', schema['required'])
                self.assertIn({'type': 'null'}, schema['properties']['bbox']['anyOf'])


if __name__ == '__main__':
    unittest.main()
