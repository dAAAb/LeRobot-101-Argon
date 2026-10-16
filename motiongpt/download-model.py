"""Download only the eight pinned official MotionGPT assets; verify before use."""
import argparse
import hashlib
import json
import shutil
import ssl
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import certifi

MIN_FREE_BYTES = 8 * 1024 ** 3


def matches(path, row):
    if not path.is_file() or path.stat().st_size != row['bytes']:
        return False
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest() == row['sha256']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / '.models')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(__file__).parent / 'model-manifest.json').read_text())
    for row in manifest['files']:
        relative = PurePosixPath(row['path'])
        url = urlparse(row['url'])
        if relative.is_absolute() or '..' in relative.parts or url.scheme != 'https' or url.hostname not in ('huggingface.co', 'raw.githubusercontent.com'):
            raise RuntimeError('Invalid pinned manifest entry')
        target = root / relative
        if matches(target, row):
            print(f'Verified {row["path"]}', flush=True)
            continue
        if args.verify_only:
            raise RuntimeError(f'Missing or invalid asset: {row["path"]}')
        if shutil.disk_usage(root).free < row['bytes'] + MIN_FREE_BYTES:
            raise RuntimeError('Download would leave less than 8 GiB free')
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f'Downloading official pinned asset: {row["path"]}', flush=True)
        context = ssl.create_default_context(cafile=certifi.where())
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.motiongpt-', delete=False) as destination:
                temporary = Path(destination.name)
                with urllib.request.urlopen(row['url'], context=context, timeout=60) as response:
                    downloaded = 0
                    while chunk := response.read(4 * 1024 * 1024):
                        downloaded += len(chunk)
                        if downloaded > row['bytes']:
                            raise RuntimeError('Remote asset exceeds pinned size')
                        destination.write(chunk)
            if not matches(temporary, row):
                raise RuntimeError(f'Size or SHA-256 mismatch: {row["path"]}')
            temporary.replace(target)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    print('All eight pinned assets are verified. Inference can run offline.', flush=True)


if __name__ == '__main__':
    main()
