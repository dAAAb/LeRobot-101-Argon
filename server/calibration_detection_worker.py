"""CPU-only native board detection isolated from the live motor controller.

OpenCV's macOS OpenCL path can terminate the process rather than raise a
Python exception. Pass exact JPEG bytes to a fresh child, with OpenCL disabled
before OpenCV is imported. A child crash or timeout never retries motion.
"""
from __future__ import annotations

import base64
import json
import math
import os
from pathlib import Path
import subprocess
import sys


MAX_JPEG_BYTES = 32 * 1024 * 1024


def detect_board_isolated(jpeg, *, frame, board, timeout=30.):
    if not isinstance(jpeg, bytes) or not 0 < len(jpeg) <= MAX_JPEG_BYTES:
        raise ValueError('校準照片資料無效或過大')
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 30:
        raise ValueError('棋盤辨識需要 30 秒以內的期限')
    payload = json.dumps({'jpeg': base64.b64encode(jpeg).decode('ascii'),
        'frame': frame, 'board': board.to_dict()}, ensure_ascii=False, allow_nan=False).encode('utf-8')
    environment = dict(os.environ)
    environment['OPENCV_OPENCL_RUNTIME'] = 'disabled'
    try:
        completed = subprocess.run([sys.executable, '-m', 'server.calibration_detection_worker'],
            input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=environment, cwd=str(Path(__file__).resolve().parents[1]),
            timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise ValueError('棋盤辨識逾時；這張照片未加入校準，已完成的手臂動作不會重做。') from exc
    except OSError as exc:
        raise ValueError('無法啟動獨立棋盤辨識；這張照片未加入校準。') from exc
    if completed.returncode != 0:
        raise ValueError('獨立棋盤辨識中斷；這張照片未加入校準，已完成的手臂動作不會重做。')
    try:
        result = json.loads(completed.stdout)
        if not isinstance(result, dict) or type(result.get('found')) is not bool:
            raise ValueError('invalid detector response')
        for key, value in (('frame_id', frame['id']), ('camera_generation', frame['camera_generation']),
                ('sequence', frame['sequence']), ('capture_monotonic', frame['capture_monotonic']),
                ('image_size', frame['image_size']), ('board_fingerprint', board.fingerprint)):
            if result.get(key) != value:
                raise ValueError('detector response belongs to another frame or board')
    except (ValueError, KeyError, TypeError, UnicodeDecodeError) as exc:
        raise ValueError('獨立棋盤辨識回傳資料不完整；這張照片未加入校準。') from exc
    return result


def _main():
    # Also protect direct invocation; the spawning parent sets this BEFORE
    # interpreter startup, then the child explicitly requests CPU execution.
    os.environ['OPENCV_OPENCL_RUNTIME'] = 'disabled'
    import cv2
    import numpy as np
    cv2.ocl.setUseOpenCL(False)
    cv2.setNumThreads(1)
    from .board_calibration import BoardSpec, detect_board
    payload = json.loads(sys.stdin.buffer.read())
    jpeg = base64.b64decode(payload['jpeg'], validate=True)
    if not 0 < len(jpeg) <= MAX_JPEG_BYTES:
        raise ValueError('invalid JPEG size')
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('invalid JPEG')
    values = payload['board']
    board = BoardSpec(**{k: values[k] for k in ('columns', 'rows', 'square_size_m')})
    detection = detect_board(image, frame=payload['frame'], board=board)
    sys.stdout.write(json.dumps(detection, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    try:
        _main()
    except Exception:
        sys.stderr.write('CPU board detector failed\n')
        raise SystemExit(1)
