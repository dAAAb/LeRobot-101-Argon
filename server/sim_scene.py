"""Offline engineering scene export, with no camera, serial or controller API.

The measured motor pose drives the same URDF convention as geometry.Kinematics.
MuJoCo independently verifies that conversion; it does not calibrate the robot,
infer object depth, authorize motion, or certify a grasp/contact dynamics model.
Optional MuJoCo/trimesh imports occur only when an engineer requests an export.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .local_cartesian import ALL, NAMES, RAD_PER_TICK, default_model, xyz_transform

ROOT = Path(__file__).resolve().parents[1]
TARGET_SOURCES = {'wrist_multiview_estimate', 'manual_measurement', 'engineering_observer'}
# Appearance of this user's white printed SO101. These overrides belong only
# to the engineering artifact, never to the URDF or collision model.
DISPLAY_MATERIALS = {'3d_printed': [.92, .92, .92, 1.],
                     'sts3215': [.075, .08, .085, 1.]}


def _text(value, maximum=200):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError('Expected a bounded evidence label')
    return value.strip()


def _number(value, minimum=-2., maximum=2.):
    if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError('Expected a bounded finite number')
    return float(value)


def _raw(value):
    if (not isinstance(value, Mapping) or set(value) != set(ALL)
            or any(type(v) is not int or not 0 <= v <= 4095 for v in value.values())):
        raise ValueError('Scene requires six measured integer encoder values')
    return {n: value[n] for n in ALL}


def _pose(matrix):
    return {'position_m': np.asarray(matrix[:3, 3]).tolist(),
            'quaternion_xyzw': Rotation.from_matrix(matrix[:3, :3]).as_quat().tolist()}


def _display_materials(model):
    materials = {x.get('name'): np.fromstring(x.find('color').get('rgba'), sep=' ').tolist()
                 for x in model.tree.findall('material') if x.find('color') is not None}
    return {**materials, **DISPLAY_MATERIALS}


def _jaw_display(model, raw):
    """A visible CAD jaw, explicitly separate from calibrated physical FK.

    Onboard min/max describe motor travel, not the jaw's CAD zero or aperture.
    Map that measured travel fraction to the CAD range only as a display prior.
    Raw already includes the onboard homing offset; do not apply it again.
    The controller continues using its full-360-degree jaw collision envelope.
    """
    joint = model.tree.find("joint[@name='gripper']")
    calibration = model.calibration['gripper']
    lo, hi = calibration['range_min'], calibration['range_max']
    if not lo < hi:
        raise ValueError('Jaw display needs a nonempty onboard travel range')
    limit = joint.find('limit')
    lower, upper = float(limit.get('lower')), float(limit.get('upper'))
    if not math.isfinite(lower + upper) or not lower < upper:
        raise ValueError('Jaw display needs a finite CAD angle range')
    fraction = (raw['gripper'] - lo) / (hi - lo)
    if not 0 <= fraction <= 1:
        raise ValueError('Jaw encoder is outside its onboard travel range')
    angle = lower + fraction * (upper - lower)
    rotation = np.eye(4)
    axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
    rotation[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
    local = xyz_transform(joint.find('origin')) @ rotation
    world = model.fk.transforms(raw)[joint.find('parent').get('link')] @ local
    description = {
        'mapping': 'onboard_travel_fraction_to_URDF_range_prior',
        'angle_rad': angle, 'raw_travel_fraction': fraction,
        'raw_range': [lo, hi], 'urdf_angle_range_rad': [lower, upper],
        'mechanical_zero_calibrated': False, 'aperture_calibrated': False,
        'homing_offset_applied_again': False, 'visual_only': True,
        'link_name': joint.find('child').get('link'), 'link_pose': _pose(world),
        'not_used_by': ['motion_planning', 'collision_screening', 'depth_estimation'],
    }
    return description, local, world


def _targets(values):
    if values is None:
        return []
    if not isinstance(values, list) or len(values) > 16:
        raise ValueError('At most 16 explicit target records may be exported')
    result = []
    for item in values:
        if not isinstance(item, dict) or item.get('source') not in TARGET_SOURCES:
            raise ValueError('Targets need an explicit measurement or estimate source')
        target = {'id': _text(item.get('id'), 80), 'label': _text(item.get('label'), 120),
                  'source': item['source'], 'evidence_id': _text(item.get('evidence_id')),
                  'coordinate_frame': 'robot_base_model',
                  'production_eligible': False, 'collision_volume_known': False,
                  'position_m': None, 'uncertainty_radius_m': None,
                  'reference': item.get('reference', 'visible_surface')}
        if target['reference'] not in ('visible_surface', 'object_center', 'unknown'):
            raise ValueError('Unknown target position reference')
        if item.get('position_m') is not None:
            if item.get('coordinate_frame') != 'robot_base_model':
                raise ValueError('Target must already be registered to the robot base model')
            position = item['position_m']
            if not isinstance(position, list) or len(position) != 3:
                raise ValueError('Target position needs three coordinates')
            target['position_m'] = [_number(v) for v in position]
            target['uncertainty_radius_m'] = _number(item.get('uncertainty_radius_m'), 0., 1.)
            if item['source'] == 'engineering_observer':
                target['registration_evidence_id'] = _text(item.get('registration_evidence_id'))
        result.append(target)
    if len({t['id'] for t in result}) != len(result):
        raise ValueError('Target identifiers must be unique')
    return result


def build_scene(frame, *, targets=None, candidates=None, project_root=None):
    """Create allowlisted JSON from an explicit saved observation; never sample IO.

    Archived observations are useful, but are explicitly marked as snapshots.
    Targets are point estimates with uncertainty, not fabricated solid objects.
    Candidate poses are supplied proposals, never issued motor commands.
    """
    if not isinstance(frame, dict) or frame.get('source') != 'wrist':
        raise ValueError('Robot pose must come from a wrist observation record')
    raw = _raw(frame.get('joints'))
    stamp = _number(frame.get('capture_monotonic'), 0., 1e12)
    generation = _text(frame.get('camera_generation'))
    root = Path(project_root or ROOT)
    model = default_model(root)
    model.validate_raw(raw, margin_ticks=0)
    transforms = model.fk.transforms(raw)
    proposed = []
    if candidates is not None:
        if not isinstance(candidates, list) or len(candidates) > 32:
            raise ValueError('At most 32 candidate poses may be displayed')
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError('Invalid candidate pose')
            segment = candidate.get('segment') or candidate
            target = segment.get('joint_targets')
            if target is None:  # Jaw-only observation has no camera translation.
                continue
            target = _raw(target)
            model.validate_raw(target, margin_ticks=0)
            T = model.fk.camera(target)
            proposed.append({'id': _text(candidate.get('id'), 80), 'joints': target,
                             'camera_pose': _pose(T), 'executed': False,
                             'source': 'CAD_proposal', 'motion_authorized': False})
    jaw = model.jaw
    jaw_display, _, _ = _jaw_display(model, raw)
    radius = float(np.max(np.linalg.norm(jaw[:, :2], axis=1)))
    low, high = float(jaw[:, 2].min()), float(jaw[:, 2].max())
    return {
        'version': 1, 'engineering_only': True, 'production_eligible': False,
        'motion_authorized': False, 'coordinate_frame': 'robot_base_model',
        'units': 'm', 'up_axis': 'Z', 'snapshot_only': True,
        'exported_at': time.time(),
        'frame': {'source': 'wrist', 'id': str(frame.get('id', 'saved_observation'))[:200],
                  'capture_monotonic': stamp, 'camera_generation': generation,
                  'captured_at': frame.get('captured_at'), 'joints': raw,
                  'image_path': str(frame.get('path', '')),
                  'timestamp_basis': str(frame.get('timestamp_basis', 'saved_observation'))[:200]},
        'robot': {'urdf_sha256': hashlib.sha256(model.urdf_path.read_bytes()).hexdigest(),
                  'pose_source': 'measured_encoders_with_CAD_zero_prior',
                  'raw_to_angle': '(raw - 2047) * 2*pi/4096',
                  'raw_to_angle_scope': 'five_body_joints_only',
                  'metric_accuracy_verified': False,
                  'appearance': {'source': 'user_reported_white_printed_robot',
                                 'display_only': True, 'materials': DISPLAY_MATERIALS},
                  'links': [{'name': name, **_pose(T)} for name, T in transforms.items()],
                  'jaw': {'raw': raw['gripper'], 'angle_calibrated': False,
                          'rendering': 'mesh_with_visual_only_travel_prior',
                          'visual_only': True, 'display_pose': jaw_display,
                          'mesh_node': 'moving_jaw_so101_v1_link_0',
                          'envelope_node': 'jaw_unknown_full_sweep',
                          'envelope_visible_by_default': False,
                          'collision_guard': 'moving_jaw_full360_envelope_unchanged',
                          'envelope_radius_m': radius, 'envelope_z_bounds_m': [low, high],
                          'is_actual_jaw_pose': False}},
        'table': {'z_m': 0., 'source': 'controller_model_prior',
                  'display_extent_m': [.8, .8], 'extent_is_measured': False},
        'camera': {'optical_pose': _pose(transforms['wrist_camera_optical']),
                   'mount_source': 'CAD_prior', 'intrinsics_calibrated': False,
                   'preview_fovy_degrees': 45., 'fovy_source': 'display_prior',
                   'optical_forward_axis': '+Z', 'mujoco_forward_axis': '-Z',
                   'image_is_not_simulated_sensor_evidence': True},
        'targets': _targets(targets), 'candidates': proposed,
        'limitations': ['CAD agreement does not measure real robot accuracy',
                       'Unknown object depth remains absent',
                       'Jaw envelope is not an aperture or contact measurement',
                       'Solid moving jaw uses a visual-only travel prior, not a measured CAD angle',
                       'Contact dynamics and mesh convexification are not validated',
                       'D435 evidence is engineering-only and requires explicit registration'],
    }


def _fmt(values):
    return ' '.join(format(float(v), '.17g') for v in values)


def _transform_attributes(matrix):
    q = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return {'pos': _fmt(matrix[:3, 3]), 'quat': _fmt(q[[3, 0, 1, 2]])}


def _mesh_path(model, filename):
    path = (model.mesh_root / filename).resolve(strict=True)
    if not path.is_relative_to(model.mesh_root.resolve()) or path.suffix.lower() != '.stl':
        raise ValueError('Only bundled STL robot assets are permitted')
    return path


def _mjcf(state, model, destination):
    """Body FK plus a fixed illustrative jaw snapshot, without a jaw actuator."""
    xml = ET.Element('mujoco', model='SO101 engineering snapshot')
    ET.SubElement(xml, 'compiler', angle='radian', fusestatic='false', autolimits='true')
    ET.SubElement(xml, 'option', gravity='0 0 0', timestep='0.004')
    ET.SubElement(xml, 'statistic', center='0.15 0 0.15', extent='0.7')
    asset = ET.SubElement(xml, 'asset')
    world = ET.SubElement(xml, 'worldbody')
    ET.SubElement(world, 'light', pos='0 0 1', dir='0 0 -1', diffuse='.8 .8 .8')
    ET.SubElement(world, 'geom', name='modeled_table', type='box',
                  pos='0 0 -.025', size='.4 .4 .025', rgba='.76 .79 .82 1')
    mesh_names, joint_order = {}, []
    materials = _display_materials(model)
    links = {x.get('name'): x for x in model.tree.findall('link')}
    joints = {}
    for joint in model.tree.findall('joint'):
        joints.setdefault(joint.find('parent').get('link'), []).append(joint)

    def geometry(body, item, name, collision):
        geometry = item.find('geometry')
        attributes = {'name': name, **_transform_attributes(xyz_transform(item.find('origin'))),
                      'mass': '0', 'group': '3' if collision else '1',
                      'contype': '1' if collision else '0',
                      'conaffinity': '1' if collision else '0'}
        material = item.find('material')
        attributes['rgba'] = '0 0 0 0' if collision else _fmt(materials.get(
            material.get('name') if material is not None else '', [.55, .58, .62, 1]))
        mesh = geometry.find('mesh')
        if mesh is not None:
            path = _mesh_path(model, mesh.get('filename'))
            scale = mesh.get('scale', '1 1 1')
            key = (str(path), scale)
            if key not in mesh_names:
                mesh_names[key] = f'robot_mesh_{len(mesh_names)}'
                (destination / 'meshes').mkdir(exist_ok=True)
                shutil.copy2(path, destination / 'meshes' / path.name)
                ET.SubElement(asset, 'mesh', name=mesh_names[key],
                              file='meshes/' + path.name, scale=scale)
            attributes.update(type='mesh', mesh=mesh_names[key])
        elif geometry.find('box') is not None:
            attributes.update(type='box', size=_fmt(np.fromstring(
                geometry.find('box').get('size'), sep=' ') / 2))
        elif geometry.find('cylinder') is not None:
            cylinder = geometry.find('cylinder')
            attributes.update(type='cylinder', size=_fmt([float(cylinder.get('radius')),
                                                         float(cylinder.get('length')) / 2]))
        else:
            raise ValueError('Unsupported URDF geometry')
        ET.SubElement(body, 'geom', **attributes)

    def add_link(parent, name, joint=None):
        transform = np.eye(4) if joint is None else xyz_transform(joint.find('origin'))
        illustrative_jaw = joint is not None and joint.get('name') == 'gripper'
        if illustrative_jaw:
            _, transform, _ = _jaw_display(model, state['frame']['joints'])
        body = ET.SubElement(parent, 'body', name=name, **_transform_attributes(transform))
        if joint is not None and joint.get('type') != 'fixed' and not illustrative_jaw:
            limit = joint.find('limit')
            ET.SubElement(body, 'joint', name=joint.get('name'), type='hinge',
                          axis=joint.find('axis').get('xyz'),
                          range=limit.get('lower') + ' ' + limit.get('upper'))
            joint_order.append(joint.get('name'))
        link = links[name]
        inertial = link.find('inertial')
        if inertial is not None:
            inertia = inertial.find('inertia')
            values = [float(inertia.get(k)) for k in ('ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz')]
            if min(values[:3]) > 0:
                T = xyz_transform(inertial.find('origin'))
                # Rotate the full inertia into the body frame. fullinertia and
                # inertial quaternion cannot both be specified in MJCF.
                a, b, c, d, e, f = values
                I = T[:3, :3] @ np.array([[a, d, e], [d, b, f], [e, f, c]]) @ T[:3, :3].T
                ET.SubElement(body, 'inertial', pos=_fmt(T[:3, 3]),
                              mass=inertial.find('mass').get('value'),
                              fullinertia=_fmt([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]))
        for kind in ('visual', 'collision'):
            if illustrative_jaw and kind == 'collision':
                continue  # A display prior must not create false contact geometry.
            for i, item in enumerate(link.findall(kind)):
                geometry(body, item, f'{name}_{kind}_{i}', kind == 'collision')
        if name == 'wrist_camera_optical':
            # URDF optical +Z/down converts to MuJoCo -Z/up by pi around X.
            ET.SubElement(body, 'camera', name='wrist_preview_prior', quat='0 1 0 0', fovy='45')
        if name == 'gripper_link':
            jaw = state['robot']['jaw']; low, high = jaw['envelope_z_bounds_m']
            offset = np.eye(4); offset[2, 3] = (low + high) / 2
            ET.SubElement(body, 'geom', name='uncalibrated_jaw_sweep_envelope',
                          type='cylinder', size=_fmt([jaw['envelope_radius_m'], (high - low) / 2]),
                          **_transform_attributes(model.jaw_origin @ offset),
                          rgba='1 .45 .05 .16', group='2', contype='0', conaffinity='0', mass='0')
        for child in joints.get(name, []):
            add_link(body, child.find('child').get('link'), child)

    add_link(world, 'base_link')
    for index, target in enumerate(state['targets']):
        if target['position_m'] is None:
            continue
        ET.SubElement(world, 'geom', name=f'target_point_{index}', type='sphere', size='.004',
                      pos=_fmt(target['position_m']), rgba='.95 .35 .15 1', contype='0', conaffinity='0')
        if target['uncertainty_radius_m'] > 0:
            ET.SubElement(world, 'geom', name=f'target_uncertainty_{index}', type='sphere',
                          size=_fmt([target['uncertainty_radius_m']]), pos=_fmt(target['position_m']),
                          rgba='.95 .35 .15 .12', contype='0', conaffinity='0')
    keyframe = ET.SubElement(xml, 'keyframe')
    ET.SubElement(keyframe, 'key', name='measured_pose',
                  qpos=_fmt([(state['frame']['joints'][n] - 2047) * RAD_PER_TICK for n in joint_order]))
    ET.indent(xml)
    ET.ElementTree(xml).write(destination / 'scene.xml', encoding='utf-8', xml_declaration=True)


def verify_mujoco(scene_path, raw, *, project_root=None):
    """Compare independent MuJoCo FK to every resolved existing URDF link."""
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError('Install the optional engineering MuJoCo dependency to verify the scene') from exc
    model = default_model(project_root or ROOT)
    raw = _raw(raw)
    simulation = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(simulation)
    for name in NAMES:
        joint = simulation.joint(name)
        data.qpos[joint.qposadr[0]] = (raw[name] - 2047) * RAD_PER_TICK
    mujoco.mj_forward(simulation, data)
    rows = []
    for name, T in model.fk.transforms(raw).items():
        body = data.body(name)
        rotation = np.asarray(body.xmat).reshape(3, 3)
        rows.append({'name': name,
                     'position_error_m': float(np.linalg.norm(body.xpos - T[:3, 3])),
                     'rotation_error_rad': float(Rotation.from_matrix(rotation @ T[:3, :3].T).magnitude())})
    camera = data.camera('wrist_preview_prior')
    camera_forward = -np.asarray(camera.xmat).reshape(3, 3)[:, 2]
    expected_forward = model.fk.camera(raw)[:3, 2]
    max_position = max(row['position_error_m'] for row in rows)
    max_rotation = max(row['rotation_error_rad'] for row in rows)
    camera_error = float(np.linalg.norm(camera_forward - expected_forward))
    passed = max_position < 1e-9 and max_rotation < 1e-9 and camera_error < 1e-9
    return {'passed': passed, 'method': 'independent_MuJoCo_FK_vs_existing_URDF_FK',
            'mujoco_version': mujoco.__version__, 'links': rows,
            'maximum_position_error_m': max_position, 'maximum_rotation_error_rad': max_rotation,
            'camera_forward_vector_error': camera_error, 'actuator_count': int(simulation.nu),
            'real_world_accuracy_verified': False, 'contact_dynamics_verified': False,
            'jaw_angle_verified': False, 'mj_forward_only': True}


def _glb(state, model, path):
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError('Install optional engineering trimesh dependency to export GLB') from exc
    scene = trimesh.Scene()
    transforms = model.fk.transforms(state['frame']['joints'])
    display, _, jaw_world = _jaw_display(model, state['frame']['joints'])
    transforms[display['link_name']] = jaw_world  # Export-only, not Kinematics.
    materials = _display_materials(model)

    def add(mesh, name, transform, rgba, *, role=None, material_name=None):
        if role:
            mesh.metadata['engineering_role'] = role
        if material_name:
            mesh.metadata['source_material'] = material_name
        mesh.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
            name=material_name or name,
            baseColorFactor=np.rint(np.asarray(rgba) * 255).astype(np.uint8),
            roughnessFactor=.75, metallicFactor=.2 if material_name == 'sts3215' else 0.,
            alphaMode='BLEND' if rgba[3] < 1 else 'OPAQUE'))
        scene.add_geometry(mesh, node_name=name, geom_name=name, transform=transform)

    for link in model.tree.findall('link'):
        name = link.get('name')
        if name not in transforms:
            continue
        for index, visual in enumerate(link.findall('visual')):
            geometry = visual.find('geometry'); mesh_element = geometry.find('mesh')
            if mesh_element is not None:
                mesh = trimesh.load_mesh(_mesh_path(model, mesh_element.get('filename')), process=False)
                mesh.apply_scale(np.fromstring(mesh_element.get('scale', '1 1 1'), sep=' '))
            elif geometry.find('box') is not None:
                mesh = trimesh.creation.box(np.fromstring(geometry.find('box').get('size'), sep=' '))
            elif geometry.find('cylinder') is not None:
                cylinder = geometry.find('cylinder')
                mesh = trimesh.creation.cylinder(float(cylinder.get('radius')), float(cylinder.get('length')))
            else:
                raise ValueError('Unsupported visual primitive')
            material = visual.find('material')
            material_name = material.get('name') if material is not None else None
            rgba = materials.get(material_name, [.55, .58, .62, 1])
            add(mesh, f'{name}_{index}', transforms[name] @ xyz_transform(visual.find('origin')), rgba,
                material_name=material_name,
                role='illustrative_moving_jaw' if name == display['link_name'] else 'robot_visual')
    jaw = state['robot']['jaw']; low, high = jaw['envelope_z_bounds_m']
    offset = np.eye(4); offset[2, 3] = (low + high) / 2
    add(trimesh.creation.cylinder(jaw['envelope_radius_m'], high - low), 'jaw_unknown_full_sweep',
        transforms['gripper_link'] @ model.jaw_origin @ offset, [1, .45, .05, .16],
        role='jaw_uncertainty_envelope')
    table = np.eye(4); table[2, 3] = -.025
    add(trimesh.creation.box([.8, .8, .05]), 'table_model_prior', table, [.76, .79, .82, 1])
    for i, target in enumerate(state['targets']):
        if target['position_m'] is None:
            continue
        T = np.eye(4); T[:3, 3] = target['position_m']
        add(trimesh.creation.icosphere(subdivisions=2, radius=.004), f'target_point_{i}', T, [.95, .35, .15, 1])
        if target['uncertainty_radius_m'] > 0:
            add(trimesh.creation.icosphere(subdivisions=2, radius=target['uncertainty_radius_m']),
                f'target_uncertainty_{i}', T, [.95, .35, .15, .12])
    for i, candidate in enumerate(state['candidates']):
        T = np.eye(4); T[:3, 3] = candidate['camera_pose']['position_m']
        add(trimesh.creation.icosphere(subdivisions=1, radius=.004), f'proposed_camera_{i}', T, [.2, .55, 1, .7])
    scene.metadata.update(engineering_only=True, up_axis='Z', coordinate_frame='robot_base_model',
                          units='m', motion_authorized=False, jaw_angle_calibrated=False)
    path.write_bytes(scene.export(file_type='glb'))


def export_scene(destination, frame, *, targets=None, candidates=None, project_root=None):
    """Export a portable, verified artifact directory, without altering hardware.

    Require a new output directory so a failed export cannot replace a known
    good scene or mingle assets from an older snapshot. Rename it at the API
    layer if a stable /current URL is needed.
    """
    root = Path(project_root or ROOT)
    state = build_scene(frame, targets=targets, candidates=candidates, project_root=root)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Use a new scene directory')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.scene-', dir=destination.parent) as temporary:
        folder = Path(temporary)
        model = default_model(root)
        _mjcf(state, model, folder)
        verification = verify_mujoco(folder / 'scene.xml', state['frame']['joints'], project_root=root)
        if not verification['passed']:
            raise ValueError('MuJoCo and existing FK disagree; scene export rejected')
        _glb(state, model, folder / 'scene.glb')
        state['artifacts'] = {'glb': 'scene.glb', 'mjcf': 'scene.xml', 'verification': 'verification.json'}
        state['verification'] = {k: v for k, v in verification.items() if k != 'links'}
        (folder / 'scene.json').write_text(json.dumps(state, ensure_ascii=False, allow_nan=False, indent=2))
        (folder / 'verification.json').write_text(json.dumps(verification, allow_nan=False, indent=2))
        # Preserve asset attribution with each portable engineering export.
        (folder / 'ASSET_NOTICE.md').write_text((root / 'assets/NOTICE.md').read_text())
        shutil.copy2(root / 'assets/so101/LICENSE.SO-ARM100', folder / 'LICENSE.SO-ARM100')
        folder.rename(destination)
    return state
