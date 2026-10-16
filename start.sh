#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p .runtime
export OPENCV_OPENCL_RUNTIME=disabled

ARGON_PYTHON="${ARGON_PYTHON:-${ASTRA_PYTHON:-$PWD/.venv/bin/python}}"
if [[ ! -x "$ARGON_PYTHON" && -x "/opt/anaconda3/bin/python3" ]]; then
  ARGON_PYTHON="/opt/anaconda3/bin/python3"
elif [[ ! -x "$ARGON_PYTHON" && -x "$HOME/LeRobot-101-OS/.venv/bin/python" ]]; then
  ARGON_PYTHON="$HOME/LeRobot-101-OS/.venv/bin/python"
fi
if [[ ! -x "$ARGON_PYTHON" ]]; then
  uv venv --python 3.11 .venv
  uv pip install --python .venv/bin/python -r requirements.txt
  ARGON_PYTHON="$PWD/.venv/bin/python"
fi
if [[ ! -d node_modules ]]; then npm ci; fi
if [[ ! -x .runtime/wrist-capture || scripts/WristCapture.swift -nt .runtime/wrist-capture ]]; then
  swiftc -module-cache-path .runtime/swift-module-cache scripts/WristCapture.swift -o .runtime/wrist-capture.next -framework AVFoundation -framework CoreImage -framework ImageIO
  mv .runtime/wrist-capture.next .runtime/wrist-capture
fi
if lsof -nP -iTCP:8102 -sTCP:LISTEN >/dev/null 2>&1 || lsof -nP -iTCP:3100 -sTCP:LISTEN >/dev/null 2>&1; then
  echo 'LeRobot Argon 服務或同埠程式已在運作。請先關閉原啟動視窗。'; exit 1
fi
"$ARGON_PYTHON" -m uvicorn server.app:app --host 127.0.0.1 --port 8102 --no-access-log &
ARGON_BACKEND_PID=$!
./node_modules/.bin/vinext dev --host 127.0.0.1 --port 3100 &
ARGON_WEB_PID=$!
cleanup(){ kill -TERM "$ARGON_BACKEND_PID" "$ARGON_WEB_PID" 2>/dev/null || true; wait "$ARGON_BACKEND_PID" "$ARGON_WEB_PID" 2>/dev/null || true; }
trap cleanup EXIT
trap 'exit 0' INT TERM
for i in {1..90}; do
  if curl -fsS http://localhost:3100/ -o /dev/null 2>/dev/null; then
    open http://localhost:3100/
    break
  fi
  sleep 1
done
printf 'LeRobot 101 Argon 已啟動：http://localhost:3100/\n按 Ctrl-C 停止服務，手臂保持目前姿勢。\n'
while kill -0 "$ARGON_BACKEND_PID" 2>/dev/null && kill -0 "$ARGON_WEB_PID" 2>/dev/null; do sleep 1; done
