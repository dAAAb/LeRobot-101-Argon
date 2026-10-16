/** Real Three.js scene objects; no browser, camera, or motor access. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { isJawEnvelope, showJawEnvelope, snapshotEvidencePresentation, visibleExtentPresentation } from '../lib/engineering-display.ts';

test('saved evidence never displays a cloud by default and ages from capture time', () => {
  const old = snapshotEvidencePresentation(100, false, 500000);
  assert.equal(old.historical, true);
  assert.equal(old.calibrated, false);
  assert.equal(old.cloudVisibleByDefault, false);
  assert.equal(old.ageSeconds, 400);
  assert.equal(snapshotEvidencePresentation(490, true, 500000).historical, false);
  assert.equal(snapshotEvidencePresentation(490, true, 500000).cloudVisibleByDefault, false);
  assert.equal(snapshotEvidencePresentation(undefined, false, 500000).historical, true);
});

function fixture() {
  const root = new THREE.Group();
  const material = () => new THREE.MeshStandardMaterial({ color: 0xf7f7f4 });
  const fixed = new THREE.Mesh(new THREE.BoxGeometry(), material()); fixed.name = 'gripper_link_0';
  const moving = new THREE.Mesh(new THREE.BoxGeometry(), material()); moving.name = 'moving_jaw_so101_0';
  const envelope = new THREE.Mesh(new THREE.CylinderGeometry(), new THREE.MeshStandardMaterial({
    color: 0xff730d, transparent: true, opacity: .16,
  })); envelope.name = 'jaw_unknown_full_sweep';
  root.add(fixed, moving, envelope);
  return { root, fixed, moving, envelope };
}

test('hiding the uncertain sweep preserves both solid jaws and their white materials', () => {
  const { root, fixed, moving, envelope } = fixture();
  showJawEnvelope(root, false);
  assert.equal(envelope.visible, false);
  for (const jaw of [fixed, moving]) {
    assert.equal(jaw.visible, true);
    assert.equal(jaw.material.color.getHex(), 0xf7f7f4);
    assert.equal(jaw.material.opacity, 1);
    assert.equal(jaw.material.transparent, false);
    assert.equal(jaw.material.depthWrite, true);
    assert.equal(jaw.material.wireframe, false);
  }
});

test('explicitly showing the sweep uses lines that cannot cover the solid jaw', () => {
  const { root, envelope } = fixture();
  showJawEnvelope(root, false);
  showJawEnvelope(root, true);
  assert.equal(envelope.visible, true);
  assert.equal(envelope.material.wireframe, true);
  assert.equal(envelope.material.transparent, true);
  assert.equal(envelope.material.opacity, .18);
  assert.equal(envelope.material.depthWrite, false);
  assert.equal(envelope.material.depthTest, true);
  showJawEnvelope(root, false);
  assert.equal(envelope.visible, false);
});

test('a tagged envelope group applies to all of its mesh materials without hiding the jaw branch', () => {
  const { root, moving } = fixture();
  const overlay = new THREE.Group();
  overlay.userData.engineering_role = 'jaw_uncertainty_envelope';
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(), [new THREE.MeshStandardMaterial(), new THREE.MeshStandardMaterial()]);
  overlay.add(mesh); root.add(overlay);
  assert.equal(isJawEnvelope(overlay), true);
  assert.equal(isJawEnvelope(moving), false);
  showJawEnvelope(root, true);
  assert.equal(overlay.visible, true);
  assert.ok(mesh.material.every(material => material.wireframe && !material.depthWrite));
  showJawEnvelope(root, false);
  assert.equal(overlay.visible, false);
  assert.equal(moving.visible, true);
});

function extentEvidence() {
  return { status: 'estimated', current_frame_id: 'frame-1', roi_frame_id: 'frame-1', frame_ids: ['frame-1'], visible_extent: {
    width_m: .035, height_m: .11, width_interval_m: [.03, .04], height_interval_m: [.09, .13],
    touches_image_edges: [], full_object_dimensions_verified: false,
    scope: 'visible_bbox_projection_at_tracked_surface_depth', reference_frame_id: 'frame-1',
  } };
}

test('visible projection dimensions are shown only for the matching estimated frame', () => {
  const evidence = extentEvidence();
  assert.deepEqual(visibleExtentPresentation(evidence, 'frame-1'), {widthM: .035, heightM: .11, cropped: false, earlierReference: false});
  evidence.visible_extent.touches_image_edges = ['top'];
  assert.equal(visibleExtentPresentation(evidence, 'frame-1').cropped, true);
  assert.equal(visibleExtentPresentation(evidence, 'other-frame'), null);
  evidence.visible_extent.reference_frame_id = 'other-frame';
  assert.equal(visibleExtentPresentation(evidence, 'frame-1'), null);
});

test('an earlier ROI remains bound to its own evidence frame instead of the current image', () => {
  const evidence = extentEvidence();
  evidence.roi_frame_id = 'earlier-roi';
  evidence.frame_ids = ['earlier-roi', 'frame-1'];
  evidence.visible_extent.reference_frame_id = 'earlier-roi';
  assert.deepEqual(visibleExtentPresentation(evidence, 'frame-1'), {
    widthM: .035, heightM: .11, cropped: false, earlierReference: true,
  });
  assert.equal(visibleExtentPresentation({...evidence, roi_frame_id: 'unrelated'}, 'frame-1'), null);
  assert.equal(visibleExtentPresentation({...evidence, frame_ids: ['frame-1']}, 'frame-1'), null);
  assert.equal(visibleExtentPresentation({...evidence, frame_ids: undefined}, 'frame-1'), null);
});

test('insufficient or malformed extent data never becomes a numeric size claim', () => {
  const evidence = extentEvidence();
  assert.equal(visibleExtentPresentation(null, 'frame-1'), null);
  assert.equal(visibleExtentPresentation({...evidence, status: 'insufficient'}, 'frame-1'), null);
  assert.equal(visibleExtentPresentation({...evidence, visible_extent: null}, 'frame-1'), null);
  for (const change of [{width_m: NaN}, {height_m: Infinity}, {width_m: 0}, {height_m: -1},
    {width_m: '.035'}, {full_object_dimensions_verified: true}, {scope: 'assumed_object_size'},
    {touches_image_edges: null}]) {
    assert.equal(visibleExtentPresentation({...evidence, visible_extent: {...evidence.visible_extent, ...change}}, 'frame-1'), null);
  }
});
