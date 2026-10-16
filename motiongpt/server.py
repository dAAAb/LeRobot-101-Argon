"""Loopback-only optional MotionGPT service. Never downloads models at inference time."""
import asyncio
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool

ALLOWED_ORIGINS = ('http://127.0.0.1:17900', 'http://127.0.0.1:5174', 'http://localhost:5174')
MAX_BODY_BYTES = 8192
Device = Literal['auto', 'cpu', 'mps', 'mlx']


class RuntimeRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    device: Device


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    prompt: str = Field(min_length=1, max_length=512, strict=True)
    seed: int = Field(default=42, ge=0, le=2147483647, strict=True)

    @field_validator('prompt')
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError('Prompt cannot be blank')
        return value.strip()


class RequestBoundary:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        headers = dict(scope.get('headers', []))
        origin = headers.get(b'origin', b'').decode('latin-1')
        if origin and origin not in ALLOWED_ORIGINS:
            return await JSONResponse({'detail': 'Origin not allowed'}, status_code=403)(scope, receive, send)
        try:
            length = int(headers.get(b'content-length', b'0'))
        except ValueError:
            return await JSONResponse({'detail': 'Invalid Content-Length'}, status_code=400)(scope, receive, send)
        if length < 0 or length > MAX_BODY_BYTES:
            return await JSONResponse({'detail': 'Request exceeds 8 KiB'}, status_code=413)(scope, receive, send)
        total = 0

        async def bounded_receive():
            nonlocal total
            message = await receive()
            if message['type'] == 'http.request':
                total += len(message.get('body', b''))
                if total > MAX_BODY_BYTES:
                    raise HTTPException(413, 'Request exceeds 8 KiB')
            return message

        return await self.app(scope, bounded_receive, send)


def available_devices():
    import torch
    devices = ['cpu']
    if torch.backends.mps.is_available():
        devices.append('mps')
    try:
        import mlx.core as mx
        if mx.metal.is_available() and Path(__file__).with_name('mlx_engine.py').is_file():
            devices.append('mlx')
    except ImportError:
        pass
    return devices


def build_engine(device='auto'):
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY', 'PYTHONDONTWRITEBYTECODE'):
        os.environ[name] = '1'
    directory = os.environ.get('AIRI_MOTIONGPT_MODEL_DIR')
    if not directory:
        raise RuntimeError('Set AIRI_MOTIONGPT_MODEL_DIR to the verified optional model directory')
    fallback_reason = None
    if device == 'mlx' or (device == 'auto' and 'mlx' in available_devices()):
        from mlx_engine import MlxMotionEngine
        try:
            return MlxMotionEngine(Path(directory), 'mlx')
        except Exception as error:
            if device != 'auto':
                raise
            fallback_reason = f'MLX could not load; using CPU: {error}'
    from engine import MotionEngine
    engine = MotionEngine(Path(directory), device)
    engine.auto_fallback_reason = fallback_reason
    return engine


def create_app(engine_factory=build_engine, device_probe=available_devices, initial_device=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.available_devices = await run_in_threadpool(device_probe)
        device = initial_device or os.environ.get('AIRI_MOTIONGPT_DEVICE', 'auto')
        if device != 'auto' and device not in app.state.available_devices:
            raise RuntimeError(f'MotionGPT device is unavailable: {device}')
        started = time.perf_counter()
        app.state.engine = await run_in_threadpool(engine_factory, device)
        app.state.load_seconds = round(time.perf_counter() - started, 4)
        app.state.selected_device = device
        app.state.generation_lock = asyncio.Lock()
        yield
        app.state.engine = None

    app = FastAPI(title='AIRI Local MotionGPT', lifespan=lifespan, docs_url=None, redoc_url=None)
    app.add_middleware(RequestBoundary)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost'])
    app.add_middleware(CORSMiddleware, allow_origins=list(ALLOWED_ORIGINS),
                       allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])

    def runtime_status(state):
        engine = state.engine
        return {'status': 'ok', 'ready': engine is not None, 'model': 'OpenMotionLab/MotionGPT-base',
                'device': engine.device, 'fps': 20, 'max_frames': 196,
                'selected_device': state.selected_device, 'available_devices': state.available_devices,
                'load_seconds': state.load_seconds, 'busy': state.generation_lock.locked(),
                'auto_fallback_reason': getattr(engine, 'auto_fallback_reason', None),
                'scope': 'Optional local text-to-motion generation, no cloud'}

    @app.get('/health')
    async def health(request: Request):
        return runtime_status(request.app.state)

    @app.post('/v1/runtime')
    async def runtime(body: RuntimeRequest, request: Request):
        state = request.app.state
        if state.generation_lock.locked():
            raise HTTPException(409, 'MotionGPT is busy. Wait for the current operation to finish.')
        if body.device != 'auto' and body.device not in state.available_devices:
            raise HTTPException(422, f'MotionGPT device is unavailable: {body.device}')
        async with state.generation_lock:
            if body.device != state.selected_device:
                started = time.perf_counter()
                try:
                    # Do not discard the working engine until the replacement is fully loaded.
                    engine = await run_in_threadpool(engine_factory, body.device)
                except Exception as error:
                    raise HTTPException(422, f'Could not load {body.device}; previous backend preserved: {error}') from error
                state.engine = engine
                state.selected_device = body.device
                state.load_seconds = round(time.perf_counter() - started, 4)
        return JSONResponse(runtime_status(state), headers={'Cache-Control': 'no-store'})

    @app.post('/v1/motions/generate')
    async def generate(body: GenerateRequest, request: Request):
        lock = request.app.state.generation_lock
        if lock.locked():
            raise HTTPException(409, 'MotionGPT is generating another motion. Try again after it finishes.')
        async with lock:
            try:
                result = await run_in_threadpool(request.app.state.engine.generate, body.prompt, body.seed)
            except (ValueError, RuntimeError) as error:
                raise HTTPException(422, str(error)) from error
        return JSONResponse(result, headers={'Cache-Control': 'no-store'})

    return app


app = create_app()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=17905, workers=1, access_log=False)
