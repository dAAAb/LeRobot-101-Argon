import * as THREE from 'three';

export function snapshotEvidencePresentation(capturedAt: number | undefined, metricCalibrated: boolean | undefined, nowMs = Date.now()) {
  const known = typeof capturedAt === 'number' && Number.isFinite(capturedAt) && capturedAt > 0;
  const ageSeconds = known ? Math.max(0, nowMs / 1000 - capturedAt) : null;
  return { historical: ageSeconds === null || ageSeconds > 60, ageSeconds,
    capturedLabel: known ? new Date(capturedAt * 1000).toLocaleString('zh-TW', { hour12: false }) : '拍攝時間未記錄',
    calibrated: metricCalibrated === true,
    cloudVisibleByDefault: false };
}

export type VisibleExtent = {
  width_m: number;
  height_m: number;
  width_interval_m: number[];
  height_interval_m: number[];
  touches_image_edges: string[];
  full_object_dimensions_verified: false;
  scope: 'visible_bbox_projection_at_tracked_surface_depth';
  reference_frame_id: string;
};

export function visibleExtentPresentation(depth: {
  status: string; current_frame_id?: string; roi_frame_id?: string; frame_ids?: string[];
  visible_extent?: VisibleExtent | null;
} | null, frameId: string): { widthM: number; heightM: number; cropped: boolean; earlierReference: boolean } | null {
  const extent = depth?.visible_extent;
  if (depth?.status !== 'estimated' || depth.current_frame_id !== frameId || !extent
      || extent.reference_frame_id !== depth.roi_frame_id
      || !Array.isArray(depth.frame_ids) || !depth.frame_ids.includes(extent.reference_frame_id)
      || extent.scope !== 'visible_bbox_projection_at_tracked_surface_depth'
      || extent.full_object_dimensions_verified !== false
      || ![extent.width_m, extent.height_m].every(value => typeof value === 'number' && Number.isFinite(value) && value > 0)
      || !Array.isArray(extent.touches_image_edges)) return null;
  return { widthM: extent.width_m, heightM: extent.height_m, cropped: extent.touches_image_edges.length > 0,
    earlierReference: extent.reference_frame_id !== frameId };
}

/** The optional uncertainty envelope is an annotation, never the movable jaw. */
export function isJawEnvelope(node: THREE.Object3D): boolean {
  return node.name === 'jaw_unknown_full_sweep'
    || node.name === 'uncalibrated_jaw_sweep_envelope'
    || node.userData.engineering_role === 'jaw_uncertainty_envelope';
}

export function showJawEnvelope(root: THREE.Object3D, visible: boolean): void {
  root.traverse(node => {
    if (!isJawEnvelope(node)) return;
    node.visible = visible;
    node.traverse(child => {
      if (!(child instanceof THREE.Mesh)) return;
      const materials = Array.isArray(child.material) ? child.material : [child.material];
      for (const material of materials) {
        // Keep the solid robot in front. A translucent depth-writing cylinder
        // can otherwise hide the inner jaw, depending on render order.
        material.transparent = true;
        material.opacity = .18;
        material.depthWrite = false;
        material.depthTest = true;
        if ('wireframe' in material) material.wireframe = true;
        material.needsUpdate = true;
      }
    });
  });
}
