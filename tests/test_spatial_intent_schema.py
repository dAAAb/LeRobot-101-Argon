"""Check the actual wire schema against Structured Outputs object rules.

This does not simulate model success: it inspects the exact generated schema
Brain writes to --output-schema, including referenced nested definitions.
"""
from copy import deepcopy
import unittest

from pydantic import ValidationError

from server.object_program import TargetBBox
from server.spatial_object_program import SpatialObjectIntent


class SpatialIntentSchemaTests(unittest.TestCase):
    def payload(self):
        return dict(source_object='圓筒',target_visible=True,
            target_bbox=dict(x=.3,y=.2,w=.2,h=.6),shape='upright_cylinder',
            initial_jaw_empty=True,human_clear=True,observation=None,reason='可見筒身',
            cylinder_contours=[dict(frame_id='wrist-1',
                body_sides_normalized=[[[.3,.2],[.3,.8]],[[.5,.2],[.5,.8]]],
                support_boundary_normalized=[])],support_surface='other_or_unknown')

    def test_wire_schema_recursively_requires_and_closes_every_object(self):
        schema=SpatialObjectIntent.model_json_schema()
        objects=[]
        def check(node,path='$'):
            if isinstance(node,dict):
                if node.get('type')=='object':
                    objects.append(path)
                    self.assertIs(node.get('additionalProperties'),False,path)
                    self.assertEqual(set(node.get('required',[])),set(node.get('properties',{})),path)
                self.assertNotIn('default',node,path)
                for key,value in node.items():check(value,path+'.'+key)
            elif isinstance(node,list):
                for i,value in enumerate(node):check(value,f'{path}[{i}]')
        check(schema)
        self.assertEqual(len(objects),4)  # Intent, contour, bbox, observation.
        self.assertEqual(schema['$defs']['TargetBBox']['required'],['x','y','w','h'])
        for key in ('target_bbox','observation'):
            self.assertIn({'type':'null'},schema['properties'][key]['anyOf'])

    def test_required_unobserved_fields_have_explicit_empty_or_unknown_values(self):
        payload=self.payload()
        payload.update(target_bbox=None,cylinder_contours=[],support_surface='other_or_unknown')
        self.assertEqual(SpatialObjectIntent.model_validate(payload).model_dump(),payload)
        for key in ('cylinder_contours','support_surface'):
            missing=deepcopy(payload);del missing[key]
            with self.subTest(key=key),self.assertRaises(ValidationError):
                SpatialObjectIntent.model_validate(missing)
        missing=self.payload();del missing['cylinder_contours'][0]['support_boundary_normalized']
        with self.assertRaises(ValidationError):SpatialObjectIntent.model_validate(missing)

    def test_closed_bbox_preserves_validated_object_program_contract(self):
        payload=self.payload();intent=SpatialObjectIntent.model_validate(payload)
        self.assertIsInstance(intent.target_bbox,TargetBBox)
        self.assertEqual(intent.model_dump(),payload)
        for bbox in (dict(x=.3,y=.2,w=.2,h=.6,z=.1),dict(x=.9,y=.2,w=.2,h=.6)):
            with self.subTest(bbox=bbox),self.assertRaises(ValidationError):
                SpatialObjectIntent.model_validate({**payload,'target_bbox':bbox})


if __name__=='__main__':unittest.main()
