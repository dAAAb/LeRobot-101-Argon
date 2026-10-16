import json,unittest
from pathlib import Path
from server.observation_plan import build_plan,validate_pose,JOINTS
ROOT=Path(__file__).resolve().parents[1]
class ObservationPlanTests(unittest.TestCase):
 def setUp(self):
  self.cal=json.loads((ROOT/'config/follower-calibration.json').read_text())
  self.origin=dict(zip(JOINTS,[2161,1103,2915,2318,998,1524]))
 def test_absolute_stages_and_staged_return(self):
  plan=build_plan(self.origin,self.cal);stages=plan['stages']
  self.assertEqual(stages[4]['targets'],dict(self.origin,shoulder_lift=1253,elbow_flex=2615,wrist_flex=2538))
  self.assertEqual([s['targets']['elbow_flex'] for s in stages[4:8]],[2615,2715,2815,2915])
  for before,after in zip(stages[4:7],stages[5:8]):
   self.assertLessEqual(max(abs(before['targets'][n]-after['targets'][n]) for n in JOINTS),100)
  self.assertEqual(stages[2]['validation_group'],stages[6]['validation_group'])
  self.assertEqual(stages[3]['validation_group'],stages[5]['validation_group'])
 def test_nominal_orientation_and_fixed_joints(self):
  plan=build_plan(self.origin,self.cal)
  for s in plan['stages'][1:-1]:
   self.assertEqual(sum(s['targets'][n] for n in ('shoulder_lift','elbow_flex','wrist_flex')),6406)
   for n in ('shoulder_pan','wrist_roll','gripper'):self.assertEqual(s['targets'][n],self.origin[n])
 def test_refuses_invalid_or_physically_biased_targets(self):
  for bad in [dict(self.origin,wrist_flex=True),dict(self.origin,wrist_flex=float('nan')),dict(self.origin,wrist_flex=2780)]:
   with self.assertRaises(ValueError):build_plan(bad,self.cal)
  with self.assertRaises(ValueError):validate_pose(self.origin,self.cal,bias={'elbow_flex':200})
 def test_restore_keeps_saved_hold_goal(self):
  finish=dict(self.origin,elbow_flex=2922)
  stages=build_plan(self.origin,self.cal,return_target=finish)['stages']
  self.assertEqual(stages[-1]['targets'],finish)
  self.assertEqual(stages[-1]['validation_group'],stages[0]['validation_group'])

class PoseFamilyTests(unittest.TestCase):
 def test_return_visits_are_not_independent_holdouts(self):
  from server.geometry import observation_families
  def obs(x,group=None):
   d={'joints':dict(zip(JOINTS,[2100+x,1100,2910,2320,1000,1520]))}
   if group:d['validation_group']=group
   return d
  groups=observation_families([obs(0),obs(30,'a'),obs(70),obs(39,'a'),obs(2)])
  self.assertEqual(sorted(sorted(g) for g in groups),[[0,4],[1,3],[2]])

class OrientationTests(unittest.TestCase):
 def test_excursions_cross_center_and_keep_step_bounds(self):
  from server.observation_plan import orientation_views
  cal=json.loads((ROOT/'config/follower-calibration.json').read_text())
  center=dict(zip(JOINTS,[2161,1103,2915,2388,998,1524]));views=orientation_views(center,cal)
  previous=center
  for i,v in enumerate(views):
   if i>0:self.assertIn(center,v['waypoints'])
   for point in v['waypoints']:
    self.assertLessEqual(abs(point['wrist_roll']-previous['wrist_roll']),50)
    self.assertLessEqual(abs(point['shoulder_pan']-previous['shoulder_pan']),35)
    for n in ('shoulder_lift','elbow_flex','wrist_flex','gripper'):self.assertEqual(point[n],center[n])
    previous=point
   self.assertEqual(previous,v['targets'])

class LocalGazeTests(unittest.TestCase):
 def test_small_excursions_return_and_never_command_body_or_gripper(self):
  from server.observation_plan import build_local_gaze_plan,LOCAL_CENTER,GAZE_AXES
  cal=json.loads((ROOT/'config/follower-calibration.json').read_text())
  origin=dict(LOCAL_CENTER);saved={n:origin[n]+4 for n in GAZE_AXES}
  stages=build_local_gaze_plan(origin,cal,return_target=saved)
  self.assertEqual(len(stages),9)
  self.assertEqual([s['validation_group'] for s in stages[::2]],['center']*5)
  previous={n:origin[n] for n in GAZE_AXES}
  for stage in stages:
   for p in stage['waypoints']:
    self.assertEqual(set(p),set(GAZE_AXES))
    self.assertLessEqual(max(abs(p[n]-previous[n]) for n in GAZE_AXES),20)
    self.assertTrue(all(abs(p[n]-origin[n])<=20 for n in GAZE_AXES));previous=p
  self.assertEqual(previous,saved)
  # Intermediate centers use measured origin; only the final central segment
  # restores saved goals, after passing through the measured center first.
  for stage in stages[2:8:2]:self.assertEqual(stage['targets'],origin)
  self.assertEqual(stages[-1]['waypoints'][-2],{n:origin[n] for n in GAZE_AXES})
  self.assertEqual(stages[-1]['waypoint_regions'][-1],'saved')
 def test_out_of_region_and_biased_physical_goals_are_rejected(self):
  from server.observation_plan import build_local_gaze_plan,LOCAL_CENTER,GAZE_AXES
  cal=json.loads((ROOT/'config/follower-calibration.json').read_text())
  for origin,bias in [(dict(LOCAL_CENTER,elbow_flex=2937),None),(dict(LOCAL_CENTER),{'wrist_flex':5})]:
   with self.assertRaises(ValueError):build_local_gaze_plan(origin,cal,return_target={n:origin[n] for n in GAZE_AXES},bias=bias)
 def test_diagonal_corner_is_not_part_of_either_excursion_corridor(self):
  from server.observation_plan import require_local_gaze_region,LOCAL_CENTER
  pose=dict(LOCAL_CENTER,shoulder_pan=2185,wrist_flex=2354)
  for region in ('shoulder_pan','wrist_flex','saved'):
   with self.assertRaises(ValueError):require_local_gaze_region(pose,moving=True,active_axis=region)
 def test_saved_physical_start_cannot_begin_a_diagonal_excursion(self):
  from server.observation_plan import build_local_gaze_plan,LOCAL_CENTER,GAZE_AXES
  cal=json.loads((ROOT/'config/follower-calibration.json').read_text())
  origin={**LOCAL_CENTER,**{n:LOCAL_CENTER[n]+4 for n in GAZE_AXES}}
  with self.assertRaises(ValueError):build_local_gaze_plan(origin,cal,return_target={n:origin[n]+4 for n in GAZE_AXES})
