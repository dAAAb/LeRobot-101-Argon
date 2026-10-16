"""Synthetic camera geometry, no devices or known product dimensions."""
from copy import deepcopy
import unittest
import cv2
import numpy as np

from server.active_depth import _project
from server.cylinder_silhouette import fit_cylinder_silhouettes


def look_at(origin, target):
    z = np.asarray(target) - origin; z /= np.linalg.norm(z)
    x = np.cross(z, [0., 0., 1.]); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    t = np.eye(4); t[:3, :3] = np.stack([x, y, z], axis=1); t[:3, 3] = origin
    return t


class CylinderSilhouetteTests(unittest.TestCase):
    def setUp(self):
        self.center = np.array([.30, .04, 0.]); self.radius = .018
        self.k = np.array([[1000., 0., 960.], [0., 1000., 540.], [0., 0., 1.]])
        self.records = []
        for i, origin in enumerate(([.03, -.02, .16], [.05, .01, .18], [.01, .05, .17], [.04, .09, .18])):
            origin = np.array(origin); t = look_at(origin, self.center + [0., 0., .07])
            to_camera = origin - self.center; to_camera[2] = 0.; distance = np.linalg.norm(to_camera)
            u = to_camera / distance; v = np.cross([0., 0., 1.], u)
            tangent = [self.center + self.radius ** 2 / distance * u + sign * self.radius *
                       np.sqrt(1 - (self.radius / distance) ** 2) * v for sign in (-1, 1)]
            sides = [[_project(p + [0., 0., height], t, self.k, np.zeros(5)).tolist()
                      for height in (.03, .10)] for p in tangent]
            front = self.center + self.radius * u
            self.records.append(dict(source='wrist', frame_id=f'actual-{i}', camera_generation='one-session',
                calibration_fingerprint='synthetic-camera', camera_transform_base_m=t.tolist(),
                camera_matrix=self.k.tolist(), distortion=[0.] * 5, image_size=[1920, 1080],
                body_sides_px=sides, support_boundary_px=[_project(front, t, self.k, np.zeros(5)).tolist()]))
        self.uncertainty = dict(pixel_edge_mislocalization_px=0., relative_translation_m=0.,
            relative_rotation_deg=0., focal_relative=0., common_translation_m=0., common_rotation_deg=0.,
            support_plane_translation_m=0., support_plane_rotation_deg=0.)
        self.plane = dict(point_base_m=[0., 0., 0.], normal_base=[0., 0., 1.],
                          assumption='Synthetic cylinder footprint and known support plane coincide exactly.')

    def fit(self, records=None, **kw):
        return fit_cylinder_silhouettes(records or self.records, shape_intent='upright_cylinder',
            axis_base=kw.pop('axis_base', [0., 0., 1.]), uncertainty=kw.pop('uncertainty', self.uncertainty), **kw)

    def test_sides_recover_center_and_radius_without_texture_or_object_size(self):
        result = self.fit()
        self.assertAlmostEqual(result['radius_m'], self.radius, places=9)
        np.testing.assert_allclose(result['centerline_point_base_m'], self.center + [0., 0., .065], atol=1e-8)
        self.assertLess(result['diagnostics']['tangent_plane_rms_m'], 1e-10)
        self.assertFalse(result['object_dimensions_used'])
        self.assertFalse(result['metric_accuracy_independently_validated'])
        self.assertFalse(result['planning_eligible'])
        self.assertFalse(result['execution_authorized'])
        self.assertIsNone(result['full_height_m'])

    def test_ground_boundary_constrains_circle_not_its_center(self):
        result = self.fit(support_plane=self.plane)
        self.assertAlmostEqual(result['radius_m'], self.radius, places=8)
        np.testing.assert_allclose(result['centerline_point_base_m'][:2], self.center[:2], atol=1e-8)
        points = np.asarray(result['diagnostics']['support_boundary_points_base_m'])
        np.testing.assert_allclose(np.linalg.norm(points[:, :2] - self.center[:2], axis=1), self.radius, atol=1e-8)
        self.assertGreater(np.linalg.norm(points[0, :2] - result['centerline_point_base_m'][:2]), .017)
        self.assertGreater(min(result['diagnostics']['support_boundary_front_distance_m']), 0.)

    def test_support_outline_cannot_select_the_hidden_back_circle_branch(self):
        records = deepcopy(self.records)
        # Side-plane localization has a coherent range disagreement. The
        # unconstrained optimum would put observed bottom pixels behind the
        # circle center, even though those pixels are a visible front outline.
        near_center = self.center - [.025, 0., 0.]
        for row in records:
            t = np.array(row['camera_transform_base_m']); origin = t[:3, 3]
            direction = origin - near_center; direction[2] = 0.; distance = np.linalg.norm(direction)
            u = direction / distance; v = np.cross([0., 0., 1.], u)
            tangent = [near_center + self.radius ** 2 / distance * u + sign * self.radius *
                       np.sqrt(1 - (self.radius / distance) ** 2) * v for sign in (-1, 1)]
            row['body_sides_px'] = [[_project(p + [0., 0., h], t, self.k, np.zeros(5)).tolist()
                                     for h in (.03, .10)] for p in tangent]
        result = self.fit(records, support_plane=self.plane,
                          uncertainty={**self.uncertainty, 'relative_translation_m': .03})
        self.assertGreaterEqual(min(result['diagnostics']['support_boundary_front_distance_m']), -1e-7)

    def test_noisy_near_tangent_outline_converges_on_visibility_boundary(self):
        records = deepcopy(self.records)
        row = records[0]; t = np.asarray(row['camera_transform_base_m'])
        toward = t[:3, 3] - self.center; toward[2] = 0.
        distance = np.linalg.norm(toward); toward /= distance
        sideways = np.cross([0., 0., 1.], toward)
        tangent = self.center + self.radius ** 2 / distance * toward + self.radius * np.sqrt(
            1 - (self.radius / distance) ** 2) * sideways
        # A <3 px localization error moves this near-tangent bottom pixel
        # 0.5 mm behind the nominal center's front half-plane. The solution
        # must fit ON the boundary, not accept that hidden point or reject
        # an otherwise feasible noisy observation after a soft penalty fit.
        noisy = self.center + self.radius * sideways - .0005 * toward
        pixel = _project(noisy, t, self.k, np.zeros(5))
        self.assertLess(np.linalg.norm(pixel - _project(tangent, t, self.k, np.zeros(5))), 3.)
        row['support_boundary_px'] = [pixel.tolist()]
        result = self.fit(records, support_plane=self.plane, uncertainty={**self.uncertainty,
            'pixel_edge_mislocalization_px': 3., 'relative_translation_m': .001})
        distances = result['diagnostics']['support_boundary_front_distance_m']
        self.assertGreaterEqual(min(distances), -1e-7)
        self.assertLess(abs(distances[0]), 1e-7)
        self.assertLess(np.linalg.norm(np.asarray(result['centerline_point_base_m'])[:2] - self.center[:2]), .001)
        self.assertLess(abs(result['radius_m'] - self.radius), .0003)
        self.assertTrue(np.isfinite(result['center_position_error_m']))
        self.assertGreater(result['center_position_error_m'], .001)
        self.assertEqual(set(result['uncertainty_modes']), {'independent_views', 'shared_camera_local_mount'})
        self.assertFalse(result['execution_authorized'])

    def test_hidden_back_outline_is_rejected_when_sides_do_not_support_front_fit(self):
        records = deepcopy(self.records)
        for row in records:
            t = np.asarray(row['camera_transform_base_m'])
            toward = t[:3, 3] - self.center; toward[2] = 0.; toward /= np.linalg.norm(toward)
            hidden = self.center - self.radius * toward
            row['support_boundary_px'] = [_project(hidden, t, self.k, np.zeros(5)).tolist()]
        # A circle may pass through the mislabeled back pixels, but moving
        # its center far enough to make them front-visible contradicts the
        # accurately localized sides. The constraint cannot override that.
        with self.assertRaisesRegex(ValueError, 'disagree beyond'):
            self.fit(records, support_plane=self.plane, uncertainty={**self.uncertainty,
                'pixel_edge_mislocalization_px': 2., 'relative_translation_m': .0005})

    def test_independent_plane_error_is_retained_when_it_is_actually_declared(self):
        nominal = self.fit(support_plane=self.plane)
        changed = self.fit(support_plane=self.plane, uncertainty={**self.uncertainty,
            'support_plane_translation_m': .002, 'support_plane_rotation_deg': .2})
        self.assertGreater(changed['radius_sensitivity_m'], nominal['radius_sensitivity_m'])
        self.assertGreater(changed['center_position_error_m'], nominal['center_position_error_m'])
        self.assertIn('relative_support_plane_translation', changed['covariance_contributions'])

    def test_common_rigid_coordinates_preserve_radius_and_transform_center(self):
        original = self.fit(support_plane=self.plane)
        r = cv2.Rodrigues(np.array([.3, -.2, .4]))[0]; translation = np.array([.2, -.1, .03])
        records = deepcopy(self.records)
        for row in records:
            t = np.array(row['camera_transform_base_m']); t[:3, :3] = r @ t[:3, :3]; t[:3, 3] = r @ t[:3, 3] + translation
            row['camera_transform_base_m'] = t.tolist()
        plane = dict(self.plane, point_base_m=translation.tolist(), normal_base=(r @ [0., 0., 1.]).tolist())
        changed = self.fit(records, axis_base=(r @ [0., 0., 1.]).tolist(), support_plane=plane)
        self.assertAlmostEqual(changed['radius_m'], original['radius_m'], places=8)
        np.testing.assert_allclose(changed['centerline_point_base_m'],
            r @ original['centerline_point_base_m'] + translation, atol=1e-8)

    def test_common_translation_uncertainty_moves_center_without_inflating_radius(self):
        before = self.fit()
        after = self.fit(uncertainty={**self.uncertainty, 'common_translation_m': .02})
        self.assertAlmostEqual(after['radius_sensitivity_m'], before['radius_sensitivity_m'], places=10)
        self.assertGreaterEqual(after['center_position_error_m'], .02)
        self.assertEqual(after['covariance_contributions']['common_translation'][3][3], 0.)

    def test_common_rotation_with_axis_does_not_inflate_radius(self):
        result = self.fit(uncertainty={**self.uncertainty, 'common_rotation_deg': 1.})
        self.assertGreater(result['center_position_error_m'], .001)
        self.assertLess(result['radius_sensitivity_m'], 1e-8)

    def test_relative_camera_error_retains_center_radius_correlation(self):
        result = self.fit(uncertainty={**self.uncertainty, 'relative_translation_m': .002,
                                      'relative_rotation_deg': .2, 'pixel_edge_mislocalization_px': 2.})
        self.assertGreater(result['radius_sensitivity_m'], .0001)
        cov = np.asarray(result['conditional_center_radius_covariance'])
        self.assertGreater(np.linalg.norm(cov[:3, 3]), 1e-9)
        self.assertGreater(np.linalg.eigvalsh(cov)[0], -1e-12)

    def test_correlated_mount_is_an_alternative_mode_and_not_averaged_away(self):
        result = self.fit(support_plane=self.plane, uncertainty={**self.uncertainty,
            'relative_translation_m': .013, 'relative_rotation_deg': 2.16,
            'common_translation_m': .010, 'pixel_edge_mislocalization_px': 6.})
        modes = result['uncertainty_modes']
        self.assertGreater(modes['shared_camera_local_mount']['center_position_sensitivity_m'],
                           modes['independent_views']['center_position_sensitivity_m'])
        self.assertEqual(result['uncertainty_budget_sources']['center'], 'shared_camera_local_mount')
        self.assertAlmostEqual(result['center_position_error_m'],
            max(row['center_position_sensitivity_m'] for row in modes.values()), places=9)
        self.assertAlmostEqual(result['radius_sensitivity_m'],
            max(row['radius_sensitivity_m'] for row in modes.values()), places=9)
        shared = modes['shared_camera_local_mount']['covariance_contributions']
        self.assertIn('shared_camera_local_translation', shared)
        self.assertNotIn('view_0_translation', shared)
        np.testing.assert_allclose(result['conditional_center_radius_covariance'],
            modes[result['conditional_covariance_mode']]['conditional_center_radius_covariance'])
        # The same relative-pose scale is not charged twice as independent
        # and common errors. Other image/base terms occur once per mode.
        for mode in modes.values():
            np.testing.assert_allclose(mode['conditional_center_radius_covariance'],
                sum((np.asarray(c) for c in mode['covariance_contributions'].values()), np.zeros((4,4))))

    def test_correlated_mode_covariance_transforms_with_a_common_base_rotation(self):
        scales = {**self.uncertainty, 'relative_translation_m': .003,
                  'relative_rotation_deg': .5, 'pixel_edge_mislocalization_px': 2.}
        original = self.fit(support_plane=self.plane, uncertainty=scales)
        r = cv2.Rodrigues(np.array([.2, -.3, .1]))[0]
        records = deepcopy(self.records)
        for row in records:
            t = np.asarray(row['camera_transform_base_m']); t[:3] = r @ t[:3]
            row['camera_transform_base_m'] = t.tolist()
        plane = dict(self.plane, normal_base=(r @ [0., 0., 1.]).tolist())
        changed = self.fit(records, axis_base=plane['normal_base'], support_plane=plane, uncertainty=scales)
        transform = np.eye(4); transform[:3,:3] = r
        # The shared local mount perturbation is the SAME physical change
        # after relabeling the base frame; verify its complete covariance.
        name = 'shared_camera_local_mount'
        old = np.asarray(original['uncertainty_modes'][name]['conditional_center_radius_covariance'])
        new = np.asarray(changed['uncertainty_modes'][name]['conditional_center_radius_covariance'])
        np.testing.assert_allclose(new, transform @ old @ transform.T, atol=1e-10, rtol=1e-5)
        self.assertAlmostEqual(original['radius_sensitivity_m'], changed['radius_sensitivity_m'], places=6)
        self.assertAlmostEqual(original['center_position_error_m'], changed['center_position_error_m'], places=6)

    def test_actual_mismatched_view_is_not_explained_by_an_unbounded_fit(self):
        records = deepcopy(self.records)
        records[1]['body_sides_px'] = (np.asarray(records[1]['body_sides_px']) + [80., 0.]).tolist()
        with self.assertRaisesRegex(ValueError, 'disagree'):
            self.fit(records, uncertainty={**self.uncertainty, 'pixel_edge_mislocalization_px': 2.})

    def test_tilted_sides_cannot_hide_behind_agreeing_fitted_tangent_planes(self):
        for displacement in (10., 30., 60.):
            records = deepcopy(self.records)
            for row in records:
                sides = np.asarray(row['body_sides_px'])
                for side in sides:
                    direction = side[-1] - side[0]; direction /= np.linalg.norm(direction)
                    normal = np.array([-direction[1], direction[0]])
                    side += displacement * np.array([-1., 1.])[:, None] * normal
                row['body_sides_px'] = sides.tolist()
            # These nearest axis-containing planes agree to <0.1 mm even
            # though the actual endpoints deviate by 10–60 pixels. A large
            # translation budget must not excuse an angular inconsistency.
            with self.subTest(displacement=displacement), self.assertRaisesRegex(ValueError, 'Body-edge rays'):
                self.fit(records, uncertainty={**self.uncertainty,
                    'pixel_edge_mislocalization_px': 2., 'relative_translation_m': .03})

    def test_bent_side_with_correct_endpoints_is_rejected_from_interior_rays(self):
        records = deepcopy(self.records)
        for row in records:
            sides = []
            for side in np.asarray(row['body_sides_px']):
                direction = side[-1] - side[0]; direction /= np.linalg.norm(direction)
                middle = np.mean(side, axis=0) + 30. * np.array([-direction[1], direction[0]])
                sides.append([side[0].tolist(), middle.tolist(), side[-1].tolist()])
            row['body_sides_px'] = sides
        with self.assertRaisesRegex(ValueError, 'Body-edge rays'):
            self.fit(records, uncertainty={**self.uncertainty,
                'pixel_edge_mislocalization_px': 2., 'relative_translation_m': .03})

    def test_frame_identity_camera_session_and_shape_are_required(self):
        for mutation in (lambda r:r[0].update(source='D435'),
                         lambda r:r[1].update(frame_id=r[0]['frame_id']),
                         lambda r:r[1].update(camera_generation='another'),
                         lambda r:r[1].update(calibration_fingerprint='another')):
            records = deepcopy(self.records); mutation(records)
            with self.assertRaises(ValueError): self.fit(records)
        with self.assertRaisesRegex(ValueError, 'prior'):
            fit_cylinder_silhouettes(self.records, shape_intent='unknown', axis_base=[0.,0.,1.], uncertainty=self.uncertainty)


if __name__ == '__main__': unittest.main()
