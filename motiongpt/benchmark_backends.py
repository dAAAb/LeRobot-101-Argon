"""Benchmark actual CPU/MPS/MLX generation, recording differing token counts."""
import argparse
import json
import platform
import time
from pathlib import Path

PROMPTS = {
    'right-wave': 'A person waves their right hand to greet someone.',
    'bow': 'A person slowly bows forward, then stands upright.',
    'dance': 'A person dances happily, swaying side to side and moving both arms.',
    'stretch': 'A person stretches both arms above their head.',
    'boxing': 'A person performs a short boxing combination with both arms.',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=Path(__file__).parent / '.models')
    parser.add_argument('--device', choices=['cpu', 'mps', 'mlx'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', choices=list(PROMPTS), default=list(PROMPTS))
    parser.add_argument('--repeats', type=int, default=2, choices=range(1, 6))
    args = parser.parse_args()
    if args.device == 'mlx':
        from mlx_engine import MlxMotionEngine
        engine = MlxMotionEngine(args.model_dir)
    else:
        from engine import MotionEngine
        engine = MotionEngine(args.model_dir, args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'platform': platform.platform(), 'backend': getattr(engine, 'backend', 'pytorch-' + engine.device),
              'load_seconds': round(engine.load_seconds, 4), 'seed': 42,
              'scope': 'Same prompts and seed; different frameworks use different RNGs, so token lengths and motions may differ.',
              'cases': []}
    for name in args.cases:
        for repeat in range(args.repeats):
            result = engine.generate(PROMPTS[name], seed=42)
            filename = f'{name}-{engine.device}-{repeat + 1}.json'
            (args.output / filename).write_text(json.dumps(result) + '\n')
            row = {'case': name, 'repeat': repeat + 1, 'output': filename,
                   'generation_seconds': result['generation_seconds'], 'motion_tokens': result['motion_tokens'],
                   'generated_frames': result['generated_frames'], 'returned_frames': len(result['joints']),
                   'truncated': result['truncated']}
            report['cases'].append(row)
            print(json.dumps(row), flush=True)
    (args.output / f'benchmark-{args.device}.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
