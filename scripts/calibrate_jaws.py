#!/usr/bin/env python3
"""Validate measured jaw landmarks offline; never activate a calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.jaw_calibration import fit_jaw_calibration
from server.local_cartesian import default_model
from server.spatial_runtime import geometry_revision
from server.spatial_contract import WristCalibration


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('measurements', type=Path, help='JSON dataset; evidence paths are relative to its directory')
    parser.add_argument('--output', type=Path, required=True, help='New output directory (must not exist)')
    parser.add_argument('--wrist-profile', type=Path, help='Required for calibrated-wrist-3d measurements')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('output directory already exists; use a new run directory')
    try:
        data = json.loads(args.measurements.read_text())
        revision = geometry_revision(default_model())
        wrist_fingerprint = None
        if args.wrist_profile is not None:
            profile = json.loads(args.wrist_profile.read_text())
            try:
                wrist = WristCalibration.from_mapping(profile, native_image_size=profile['native_image_size'],
                    camera_id=profile['camera_id'], geometry_revision=revision,
                    mount_revision=profile['mount_revision'], cad_handeye=profile['handeye'])
            except KeyError as exc:
                raise ValueError('Wrist profile is missing validated camera/mount/handeye binding') from exc
            if not wrist.metric_calibrated:
                raise ValueError('Wrist 3D measurements cannot use an unvalidated camera prior')
            wrist_fingerprint = wrist.fingerprint
        report = fit_jaw_calibration(data, evidence_root=args.measurements.parent,
                                    expected_geometry_revision=revision,
                                    expected_wrist_calibration_fingerprint=wrist_fingerprint)
    except (ValueError, OSError) as exc:
        report = {'passed': False, 'blockers': [str(exc)], 'calibration_profile': None,
                  'automatically_activated': False}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    if report['passed']:
        (args.output / 'profile.json').write_text(json.dumps(report['calibration_profile'], ensure_ascii=False,
                                                           indent=2, allow_nan=False) + '\n')
        print('數值驗證通過；已產生 profile.json 與 report.json。尚未啟用，也不代表已成功抓取。')
        return 0
    print('校準未通過；只保存 report.json，沒有產生可啟用的 profile。', file=sys.stderr)
    for reason in report['blockers']:
        print(reason, file=sys.stderr)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
