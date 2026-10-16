"""Local checkpoint conversion only; PyTorch is never used for MLX inference."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

FORMAT_VERSION = 1


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verified_assets(model_dir):
    root = Path(model_dir).resolve()
    manifest = json.loads((Path(__file__).parent / 'model-manifest.json').read_text())
    for row in manifest['files']:
        path = root / row['path']
        if not path.is_file() or path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
            raise RuntimeError(f'Missing or invalid pinned MotionGPT asset: {row["path"]}')
    return manifest


def ensure_mlx_weights(model_dir):
    """Convert trusted tensors to a local safetensors cache; never download weights."""
    root = Path(model_dir).resolve()
    manifest = verified_assets(root)
    checkpoint_sha = manifest['files'][0]['sha256']
    cache = root / '.mlx-cache'
    weights_path = cache / 'motiongpt-float32.safetensors'
    record_path = cache / 'conversion.json'
    if weights_path.is_file() and record_path.is_file():
        record = json.loads(record_path.read_text())
        if record.get('format_version') == FORMAT_VERSION and record.get('source_sha256') == checkpoint_sha and record.get('sha256') == sha256(weights_path):
            return weights_path
    # This import is restricted to a first-time format conversion. The inference
    # module imports no Torch and all model operations run in MLX on Metal.
    import torch
    from safetensors.numpy import save_file
    if shutil.disk_usage(root).free < 9 * 1024 ** 3:
        raise RuntimeError('MLX conversion needs about 1 GiB while retaining 8 GiB free')
    source = torch.load(root / 'motiongpt_s3_h3d.tar', map_location='cpu', weights_only=True)['state_dict']
    weights = {}
    for key, value in source.items():
        if key.startswith('lm.language_model.'):
            name = key.removeprefix('lm.language_model.')
            if name in ('encoder.embed_tokens.weight', 'decoder.embed_tokens.weight'):
                continue  # Both are exact aliases of shared.weight.
            weights['t5.' + name] = value.float().numpy().copy()
        elif key.startswith('vae.decoder.') or key == 'vae.quantizer.codebook':
            array = value.float().numpy()
            if array.ndim == 3:
                array = array.transpose(0, 2, 1)  # Torch O,I,K -> MLX O,K,I.
            weights[key] = array.copy()
    del source
    cache.mkdir(exist_ok=True)
    temporary = cache / 'motiongpt-float32.safetensors.partial'
    try:
        save_file(weights, str(temporary), metadata={'source_sha256': checkpoint_sha, 'format_version': str(FORMAT_VERSION)})
        temporary.replace(weights_path)
    finally:
        temporary.unlink(missing_ok=True)
    record = {'format_version': FORMAT_VERSION, 'source_sha256': checkpoint_sha,
              'sha256': sha256(weights_path), 'bytes': weights_path.stat().st_size,
              'dtype': 'float32', 'tensors': len(weights)}
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    return weights_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=Path(__file__).parent / '.models')
    args = parser.parse_args()
    print(ensure_mlx_weights(args.model_dir))
