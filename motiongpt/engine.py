"""Minimal local MotionGPT-base inference: FLAN-T5, VQ decoder, HumanML3D joints.

The VQ implementation and joint recovery follow OpenMotionLab/MotionGPT
001aaca8d0ee218fc17f8265d11ac124044fe42f (MIT, see vendor/LICENSE.MotionGPT).
No SMPL assets, training data, evaluation networks, or cloud APIs are used.
"""
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration

from vendor.mgpt_vq import VQVae

MODEL_ID = 'OpenMotionLab/MotionGPT-base'
FPS = 20
MAX_FRAMES = 196
JOINT_NAMES = (
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
)


def verify_files(root):
    manifest = json.loads((Path(__file__).parent / 'model-manifest.json').read_text())
    for row in manifest['files']:
        path = root / row['path']
        if not path.is_file() or path.stat().st_size != row['bytes']:
            raise RuntimeError(f'MotionGPT file missing or incorrect size: {row["path"]}')
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != row['sha256']:
            raise RuntimeError(f'MotionGPT checksum failed: {row["path"]}')


def inverse_rotate(quaternion, vector):
    """Inverse unit-quaternion rotation, using HumanML3D's w,x,y,z convention."""
    axis = -quaternion[..., 1:]
    uv = torch.cross(axis, vector, dim=-1)
    uuv = torch.cross(axis, uv, dim=-1)
    return vector + 2 * (quaternion[..., :1] * uv + uuv)


def recover_joints(features):
    """Recover official rotation-invariant 263D features to global 22x3 joints."""
    angle = torch.zeros_like(features[..., 0])
    angle[..., 1:] = features[..., :-1, 0]
    angle = torch.cumsum(angle, dim=-1)
    quaternion = torch.zeros(features.shape[:-1] + (4,), device=features.device)
    quaternion[..., 0] = torch.cos(angle)
    quaternion[..., 2] = torch.sin(angle)
    root = torch.zeros(features.shape[:-1] + (3,), device=features.device)
    root[..., 1:, [0, 2]] = features[..., :-1, 1:3]
    root = torch.cumsum(inverse_rotate(quaternion, root), dim=-2)
    root[..., 1] = features[..., 3]
    positions = features[..., 4:67].reshape(features.shape[:-1] + (21, 3))
    positions = inverse_rotate(quaternion.unsqueeze(-2).expand(positions.shape[:-1] + (4,)), positions)
    positions[..., 0] += root[..., 0:1]
    positions[..., 2] += root[..., 2:3]
    return torch.cat((root.unsqueeze(-2), positions), dim=-2)


class MotionEngine:
    def __init__(self, model_dir, device='auto'):
        started = time.perf_counter()
        root = Path(model_dir).resolve()
        verify_files(root)
        if device not in ('auto', 'cpu', 'mps'):
            raise ValueError('Choose device auto, cpu, or mps')
        # Batch-one autoregressive decoding was faster on this Mac's CPU than MPS.
        # Keep Metal opt-in; do not equate GPU availability with lower latency.
        self.device = 'cpu' if device == 'auto' else device
        if self.device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('Apple Metal is unavailable in this runtime')
        torch.set_num_threads(6)
        # Only tensor dictionaries from the pinned official checkpoint can be loaded.
        state = torch.load(root / 'motiongpt_s3_h3d.tar', map_location='cpu', weights_only=True)['state_dict']
        self.tokenizer = AutoTokenizer.from_pretrained(root / 'tokenizer', local_files_only=True,
                                                       legacy=True, clean_up_tokenization_spaces=True)
        self.tokenizer.add_tokens([f'<motion_id_{i}>' for i in range(515)])
        config = T5Config.from_pretrained(root / 'tokenizer', local_files_only=True)
        config.vocab_size = len(self.tokenizer)
        self.lm = T5ForConditionalGeneration(config)
        self.lm.load_state_dict({key.removeprefix('lm.language_model.'): value for key, value in state.items() if key.startswith('lm.language_model.')}, strict=True)
        self.vae = VQVae(nfeats=263, quantizer='ema_reset', code_num=512,
                         code_dim=512, output_emb_width=512, down_t=2, stride_t=2,
                         width=512, depth=3, dilation_growth_rate=3, norm=None)
        self.vae.load_state_dict({key.removeprefix('vae.'): value for key, value in state.items() if key.startswith('vae.')}, strict=True)
        del state
        self.lm.to(self.device).eval()
        self.vae.to(self.device).eval()
        self.mean = torch.from_numpy(np.load(root / 'mean.npy', allow_pickle=False)).float().to(self.device)
        self.std = torch.from_numpy(np.load(root / 'std.npy', allow_pickle=False)).float().to(self.device)
        if self.mean.shape != (263,) or self.std.shape != (263,):
            raise RuntimeError('Unexpected HumanML3D normalization shape')
        self.load_seconds = time.perf_counter() - started

    @torch.inference_mode()
    def generate(self, prompt, seed=42):
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 512:
            raise ValueError('Prompt must contain 1 to 512 characters')
        if type(seed) is not int or not 0 <= seed <= 2147483647:
            raise ValueError('Seed must be an integer from 0 to 2147483647')
        started = time.perf_counter()
        torch.manual_seed(seed)
        encoded = self.tokenizer(['Generate motion: ' + prompt.strip()], padding='max_length',
                                 max_length=256, truncation=True, return_tensors='pt')
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        output = self.lm.generate(**encoded, max_new_tokens=128, num_beams=1,
                                  do_sample=True, max_time=45.0)
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        match = re.search(r'<motion_id_512>(.*?)(?:<motion_id_513>|$)', text, re.DOTALL)
        if not match:
            raise RuntimeError('The model did not produce motion tokens. Try a simple English movement description.')
        codes = [int(value) for value in re.findall(r'<motion_id_(\d+)>', match.group(1))]
        if not codes or any(value < 0 or value >= 512 for value in codes):
            raise RuntimeError('The model produced an invalid motion sequence')
        tokens = torch.tensor(codes, dtype=torch.long, device=self.device)
        decoded = self.vae.decode(tokens)
        joints = recover_joints(decoded * self.std + self.mean)[0].cpu().numpy()
        frames = joints.shape[0]
        joints = joints[:MAX_FRAMES]
        if joints.shape[0] < 2 or joints.shape[1:] != (22, 3) or not np.isfinite(joints).all() or np.abs(joints).max() > 100:
            raise RuntimeError('The model produced invalid joint coordinates')
        return {
            'format': 'humanml3d-22', 'fps': FPS,
            'coordinate_system': 'right-handed-y-up',
            'initial_forward': '+Z', 'anatomical_left': '+X',
            'joint_names': list(JOINT_NAMES), 'joints': np.round(joints.astype(np.float64), 6).tolist(),
            'prompt': prompt.strip(), 'model': MODEL_ID, 'seed': seed,
            'device': self.device, 'generated_frames': frames,
            'truncated': frames > MAX_FRAMES, 'motion_tokens': len(codes),
            'generation_seconds': round(time.perf_counter() - started, 4),
        }
