"""Revalidate a CAD-assisted jaw after a strictly frozen camera scope extension.

Offline only. Produces new inactive artifacts; no motors, cameras, HTTP calls,
configuration changes or overwrite of the original acquisition are performed.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server.jaw_camera_revalidation import revalidate_saved_jaw_camera


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('jaw-profile', 'jaw-report', 'old-camera', 'new-camera',
                 'camera-extension-report', 'manifest', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    args = parser.parse_args(argv)
    result = revalidate_saved_jaw_camera(jaw_profile_path=args.jaw_profile,
        jaw_report_path=args.jaw_report, old_camera_path=args.old_camera,
        new_camera_path=args.new_camera, camera_extension_report_path=args.camera_extension_report,
        manifest_path=args.manifest, output=args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
