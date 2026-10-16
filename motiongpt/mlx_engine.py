"""MotionGPT with native MLX/Metal inference; no Torch execution in generate()."""
import json
import re
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

from mlx_convert import ensure_mlx_weights
from mlx_model import MlxT5, MlxVQDecoder, recover_joints

JOINT_NAMES = ('pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
               'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
               'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
               'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist')


class MlxMotionEngine:
    def __init__(self, model_dir, device='mlx'):
        started = time.perf_counter()
        if device != 'mlx':
            raise ValueError('MlxMotionEngine requires device mlx')
        if not mx.metal.is_available():
            raise RuntimeError('MLX Metal GPU is unavailable; choose CPU or MPS explicitly')
        mx.set_default_device(mx.gpu)
        self.device = 'mlx'
        self.backend = 'mlx-metal'
        root = Path(model_dir).resolve()
        path = ensure_mlx_weights(root)
        self.weights = mx.load(str(path))
        self.tokenizer = AutoTokenizer.from_pretrained(root / 'tokenizer', local_files_only=True,
                                                       legacy=True, clean_up_tokenization_spaces=True)
        self.tokenizer.add_tokens([f'<motion_id_{i}>' for i in range(515)])
        config = json.loads((root / 'tokenizer/config.json').read_text())
        if (config['d_model'], config['d_ff'], config['num_layers'], config['num_heads'], len(self.tokenizer)) != (768, 2048, 12, 12, 32615):
            raise RuntimeError('Unexpected MotionGPT FLAN-T5-base architecture')
        self.lm = MlxT5(self.weights)
        self.vae = MlxVQDecoder(self.weights)
        self.mean = mx.array(np.load(root / 'mean.npy', allow_pickle=False).astype(np.float32))
        self.std = mx.array(np.load(root / 'std.npy', allow_pickle=False).astype(np.float32))
        if self.mean.shape != (263,) or self.std.shape != (263,):
            raise RuntimeError('Unexpected HumanML3D normalization shape')
        mx.eval(self.weights, self.mean, self.std)
        self.load_seconds = time.perf_counter() - started

    def generate(self, prompt, seed=42):
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 512:
            raise ValueError('Prompt must contain 1 to 512 characters')
        if type(seed) is not int or not 0 <= seed <= 2147483647:
            raise ValueError('Seed must be an integer from 0 to 2147483647')
        started = time.perf_counter()
        # Explicit GPU device stream also applies when called by a service worker.
        with mx.stream(mx.gpu):
            encoded = self.tokenizer(['Generate motion: ' + prompt.strip()], padding='max_length',
                                     max_length=256, truncation=True, return_tensors='np')
            inputs = mx.array(encoded['input_ids'].astype(np.int32))
            mask = mx.array(encoded['attention_mask'].astype(np.int32))
            memory = self.lm.encode(inputs, mask)
            cross = self.lm.cross_cache(memory)
            mx.eval(cross)
            token = mx.array([[0]], dtype=mx.int32)
            key = mx.random.key(seed)
            cache = None
            generated = []
            for _ in range(128):
                logits, cache = self.lm.decode(token, cross, mask, cache)
                scores = logits[:, -1, :]
                # Match HF generation's default do_sample=True, top_k=50.
                cutoff = mx.min(mx.topk(scores, 50, axis=-1), axis=-1, keepdims=True)
                scores = mx.where(scores < cutoff, -mx.inf, scores)
                key, sample_key = mx.random.split(key)
                token = mx.random.categorical(scores, key=sample_key).astype(mx.int32)[:, None]
                mx.eval(token, cache)
                value = token.item()
                generated.append(value)
                if value == 1:
                    break
                if time.perf_counter() - started > 45:
                    raise RuntimeError('MLX generation exceeded the 45-second limit')
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            match = re.search(r'<motion_id_512>(.*?)(?:<motion_id_513>|$)', text, re.DOTALL)
            codes = [] if not match else [int(value) for value in re.findall(r'<motion_id_(\d+)>', match.group(1))]
            if not codes or any(value < 0 or value >= 512 for value in codes):
                raise RuntimeError('The model did not produce valid motion tokens. Try a simple English movement description.')
            features = self.vae.decode(mx.array(codes, dtype=mx.int32))
            joints_gpu = recover_joints(features * self.std + self.mean)
            mx.eval(joints_gpu)
            joints = np.array(joints_gpu[0])
        frames = len(joints)
        joints = joints[:196]
        if joints.shape[0] < 2 or joints.shape[1:] != (22, 3) or not np.isfinite(joints).all() or np.abs(joints).max() > 100:
            raise RuntimeError('The model produced invalid joint coordinates')
        return {'format': 'humanml3d-22', 'fps': 20, 'coordinate_system': 'right-handed-y-up',
                'initial_forward': '+Z', 'anatomical_left': '+X', 'joint_names': list(JOINT_NAMES),
                'joints': np.round(joints.astype(np.float64), 6).tolist(),
                'prompt': prompt.strip(), 'model': 'OpenMotionLab/MotionGPT-base', 'seed': seed,
                'device': self.device, 'backend': self.backend, 'precision': 'float32',
                'generated_frames': frames, 'truncated': frames > 196,
                'motion_tokens': len(codes), 'generation_seconds': round(time.perf_counter() - started, 4)}
