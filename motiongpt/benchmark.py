"""Generate real motions and record local CPU/MPS timings without a GUI."""
import argparse
import gc
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from engine import MotionEngine

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
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--devices', nargs='+', choices=['cpu', 'mps', 'auto'], default=['auto'])
    parser.add_argument('--cases', nargs='+', choices=list(PROMPTS), default=list(PROMPTS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'scope': 'Actual pinned MotionGPT-base neural generation; timing is this Mac only, not a general quality benchmark.',
              'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'platform': platform.platform(), 'torch': torch.__version__, 'cases': []}
    for device in args.devices:
        engine = MotionEngine(args.model_dir, device)
        for name in args.cases:
            result = engine.generate(PROMPTS[name], 42)
            output = args.output / f'{name}-{engine.device}.json'
            output.write_text(json.dumps(result, ensure_ascii=False) + '\n')
            joints = np.asarray(result['joints'])
            row = {'case': name, 'device': engine.device, 'load_seconds': round(engine.load_seconds, 4),
                   'generation_seconds': result['generation_seconds'], 'frames': len(joints),
                   'motion_tokens': result['motion_tokens'],
                   'duration_seconds': len(joints) / 20, 'truncated': result['truncated'],
                   'finite': bool(np.isfinite(joints).all()),
                   'left_wrist_y_range_m': float(np.ptp(joints[:, 20, 1])),
                   'right_wrist_y_range_m': float(np.ptp(joints[:, 21, 1])),
                   'head_y_range_m': float(np.ptp(joints[:, 15, 1])),
                   'output': output.name}
            report['cases'].append(row)
            print(json.dumps(row), flush=True)
        del engine
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    (args.output / 'benchmark.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
