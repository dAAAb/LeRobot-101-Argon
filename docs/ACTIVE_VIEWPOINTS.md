# Camera baselines for active observation

The observation planner first tries one coordinated camera translation: retreat,
left, right, or a higher view. It solves all joints with available moving reserve
and keeps near-limit axes stationary. A full segment is screened before Astra
selects it. A higher view is not automatically an overhead view.

Candidates disclose the camera's predicted baseline, optical-axis translation,
transverse baseline, before/after transforms, and nominal forward travel.
Execution receipts recompute these quantities from the measured encoder poses.
Both use the CAD camera prior; they are not calibrated object depth. No object
size, depth, or exact centering is assumed. Each entire path still requires
visual clearance, and actual motion stays within the predeclared tracking tube.

An object box anchored to the current wrist image also guides framing. The
planner projects its visible corners through explicit, uncalibrated focal and
depth hypotheses along each nominal camera path. It filters views predicted to
lose more than 15% of the visible box. These projections are a framing heuristic;
they do not establish object depth, collision clearance, or tracking accuracy.
When the image was refreshed while stationary, metadata retains both the actual
planning image and the original candidate image/pose. Refresh drift must stay
within three encoder ticks, and the caller separately checks image consistency.

If a target is near the image edge, a screened 68-tick wrist-pitch option can be
offered alongside translations. It is labeled `recenter_for_translation` only
when every sampled focal/depth hypothesis predicts at least 3% of image height
in vertical centering improvement and the visibility check passes. This step
aims the camera for a subsequent translation; it does not by itself resolve
metric depth. Ordinary roll/pitch options remain fallbacks when no larger
translation fits.

Candidate endpoints are compared with previous measured wrist poses from the
same attempt and camera generation. A predicted revisit lies within both 5 mm
and 2 degrees of a prior camera pose. Such candidates are removed when another
novel body path exists. D435, different camera generations, malformed records
and future timestamps contribute no evidence. Pose novelty does not guarantee
new visual information or a successful depth estimate.

A translation direction may continue using explicit `_continue_2` and
`_continue_3` IDs, only with eligible wrist history and a predicted novel
endpoint. Without that history, each direction retains its one-use behavior.
Continuations still obey the original attempt's 240-tick/80-mm envelope, the
per-segment geometry and motor guards, and the caller's five-observation and
30-second physical-motion budget. Reframing or a new direction ID never resets
those limits.

To export geometry without connecting to hardware:

```sh
python scripts/export_observation_candidates.py \
  --frame .runtime/active-depth-current.json \
  --goals .runtime/active-depth-diagnostics.json \
  --output .runtime/sim-scene/observation-geometry.json \
  --list-output .runtime/sim-scene/offline-candidates.json
```

`--goals` accepts explicit six-axis goals or saved diagnostics containing
`joints[name].position` and `.goal`. Diagnostic positions must match the saved
frame. The exporter preserves the acknowledged goal offsets, records generation
time, and removes execution seals. Its files are for simulation; live operation
must regenerate candidates from fresh wrist evidence.

Pass `offline-candidates.json` to `scripts/export_sim_scene.py --candidates` to
include candidate camera poses in the MuJoCo/GLB engineering scene. This export
uses no D435 data and issues no motor command.
