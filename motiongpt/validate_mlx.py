"""Numerical parity with identical inputs: Torch CPU vs native MLX GPU.

Run --reference with the Torch environment, then --compare with MLX.
The reference NPZ remains local; the small metrics JSON can be published.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np


def reference(args):
    import torch
    from engine import MotionEngine, recover_joints
    engine = MotionEngine(args.model_dir, 'cpu')
    encoded = engine.tokenizer(['Generate motion: A person waves their right hand to greet someone.'],
                               padding='max_length', max_length=256, truncation=True, return_tensors='pt')
    with torch.inference_mode():
        memory = engine.lm.encoder(**encoded).last_hidden_state
        torch.manual_seed(42)
        tokens = engine.lm.generate(**encoded, max_new_tokens=128, do_sample=True)
        text = engine.tokenizer.decode(tokens[0], skip_special_tokens=True)
        codes = [int(v) for v in re.findall(r'<motion_id_(\d+)>', text) if int(v) < 512]
        cache = None
        logits = []
        for position in range(min(16, tokens.shape[1])):
            output = engine.lm(encoder_outputs=(memory,), attention_mask=encoded['attention_mask'],
                               decoder_input_ids=tokens[:, position:position + 1], past_key_values=cache, use_cache=True)
            logits.append(output.logits.numpy())
            cache = output.past_key_values
        features = engine.vae.decode(torch.tensor(codes))
        joints = recover_joints(features * engine.std + engine.mean)
    args.reference_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.reference_file, input_ids=encoded['input_ids'].numpy(), mask=encoded['attention_mask'].numpy(),
             memory=memory.numpy(), decoder_ids=tokens[:, :len(logits)].numpy(),
             logits=np.concatenate(logits, axis=1), codes=np.array(codes),
             features=features.numpy(), joints=joints.numpy())
    print(json.dumps({'reference': str(args.reference_file), 'steps': len(logits), 'codes': len(codes)}))


def compare(args):
    import mlx.core as mx
    from mlx_engine import MlxMotionEngine
    from mlx_model import recover_joints
    engine = MlxMotionEngine(args.model_dir)
    baseline = np.load(args.reference_file)
    ids, mask = mx.array(baseline['input_ids'].astype(np.int32)), mx.array(baseline['mask'].astype(np.int32))
    with mx.stream(mx.gpu):
        memory = engine.lm.encode(ids, mask)
        cross = engine.lm.cross_cache(memory)
        cache, logits = None, []
        for token in baseline['decoder_ids'].T:
            output, cache = engine.lm.decode(mx.array(token[:, None].astype(np.int32)), cross, mask, cache)
            mx.eval(output, cache)
            logits.append(np.array(output))
        features = engine.vae.decode(mx.array(baseline['codes'].astype(np.int32)))
        joints = recover_joints(features * engine.std + engine.mean)
        mx.eval(memory, features, joints)
    report = {'backend': engine.backend, 'precision': 'float32', 'device': str(mx.default_device()),
              'scope': 'Identical inputs and decoder tokens; not random-sample equivalence.', 'metrics': {}}
    arrays = {'memory': np.array(memory), 'logits': np.concatenate(logits, axis=1),
              'features': np.array(features), 'joints': np.array(joints)}
    for name, actual in arrays.items():
        delta = actual.astype(np.float64) - baseline[name].astype(np.float64)
        report['metrics'][name] = {'max_abs': float(np.max(np.abs(delta))), 'rmse': float(np.sqrt(np.mean(delta ** 2))),
                                   'finite': bool(np.isfinite(actual).all())}
    report['metrics']['logits']['argmax_matches_all_steps'] = bool(np.array_equal(arrays['logits'].argmax(-1), baseline['logits'].argmax(-1)))
    report['passed'] = (all(v['finite'] and v['max_abs'] < 0.002 for v in report['metrics'].values())
                        and report['metrics']['logits']['argmax_matches_all_steps'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit('MLX numerical parity failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=Path(__file__).parent / '.models')
    parser.add_argument('--reference-file', type=Path, default=Path(__file__).parent / '.evidence/mlx-reference.npz')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / '../results/motiongpt/mlx-parity.json')
    parser.add_argument('--reference', action='store_true')
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()
    if args.reference == args.compare:
        parser.error('Select exactly one of --reference or --compare')
    reference(args) if args.reference else compare(args)
