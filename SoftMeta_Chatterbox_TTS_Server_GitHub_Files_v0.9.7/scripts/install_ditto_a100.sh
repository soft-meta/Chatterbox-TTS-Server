#!/usr/bin/env bash
set -euo pipefail

# Optional Ditto PyTorch runtime for SoftMeta Avatar Talking.
# Only the PyTorch checkpoint files are downloaded. TensorRT/ONNX bundles are
# intentionally excluded to reduce installation time and avoid Colab conflicts.

MM="${1:-/content/bin/micromamba}"
ENV_NAME="${SOFTMETA_AVATAR_ENV:-avatar310}"
DITTO_DIR="${SOFTMETA_DITTO_DIR:-/content/ditto-talkinghead}"
SERVER_DIR="${SOFTMETA_SERVER_DIR:-/content/Chatterbox-TTS-Server}"
FORCE="${SOFTMETA_FORCE_REINSTALL:-0}"
READY_MARKER="$DITTO_DIR/.softmeta_ditto_pytorch_v097"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/content/.cache/pip}"

if [[ ! -x "$MM" ]]; then
  echo "micromamba was not found at: $MM" >&2
  exit 2
fi
if [[ ! -f "$SERVER_DIR/requirements-avatar.txt" ]]; then
  echo "SoftMeta server was not found at: $SERVER_DIR" >&2
  exit 2
fi

has_env() {
  "$MM" env list | awk '{print $1}' | grep -qx "$ENV_NAME"
}

runtime_ok() {
  has_env || return 1
  [[ -f "$DITTO_DIR/inference.py" ]] || return 1
  [[ -f "$DITTO_DIR/checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl" ]] || return 1
  [[ -d "$DITTO_DIR/checkpoints/ditto_pytorch" ]] || return 1
  "$MM" run -n "$ENV_NAME" python - <<PY >/dev/null 2>&1
import os
import sys
import torch
import onnxruntime
import mediapipe
import einops
os.chdir(r"$DITTO_DIR")
sys.path.insert(0, r"$DITTO_DIR")
import inference
assert torch.cuda.is_available()
PY
}

if [[ "$FORCE" != "1" && -f "$READY_MARKER" ]] && runtime_ok; then
  echo "Ditto PyTorch runtime is already verified. Reusing it."
  exit 0
fi

if [[ "$FORCE" == "1" ]]; then
  if has_env; then
    "$MM" env remove -n "$ENV_NAME" -y || true
  fi
  rm -rf "$DITTO_DIR"
fi

if ! has_env; then
  "$MM" create -y -n "$ENV_NAME" -c conda-forge python=3.10 pip
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "Detected GPU:"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

if [[ -d "$DITTO_DIR/.git" ]]; then
  git -C "$DITTO_DIR" fetch --depth 1 origin main
  git -C "$DITTO_DIR" reset --hard origin/main
else
  rm -rf "$DITTO_DIR"
  git clone --depth 1 https://github.com/antgroup/ditto-talkinghead.git "$DITTO_DIR"
fi

"$MM" run -n "$ENV_NAME" python -m pip install -U pip wheel setuptools
"$MM" run -n "$ENV_NAME" python -m pip install \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
"$MM" run -n "$ENV_NAME" python -m pip install \
  -r "$SERVER_DIR/requirements-avatar.txt"

"$MM" run -n "$ENV_NAME" python - <<PYDOWNLOAD
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="digital-avatar/ditto-talkinghead",
    local_dir=r"$DITTO_DIR/checkpoints",
    allow_patterns=[
        "ditto_cfg/v0.4_hubert_cfg_pytorch.pkl",
        "ditto_pytorch/**",
    ],
    max_workers=4,
)
print("Ditto PyTorch checkpoints downloaded.")
PYDOWNLOAD

"$MM" run -n "$ENV_NAME" python -m pip check || \
  echo "pip check reported an optional dependency warning; import verification will continue."

"$MM" run -n "$ENV_NAME" python - <<PYVERIFY
from pathlib import Path
import os
import sys
import torch
import onnxruntime as ort
import mediapipe as mp
import einops

root = Path(r"$DITTO_DIR")
required = [
    root / "inference.py",
    root / "checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl",
    root / "checkpoints/ditto_pytorch",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("Missing Ditto files: " + ", ".join(missing))
os.chdir(root)
sys.path.insert(0, str(root))
import inference

print("PyTorch:", torch.__version__)
print("ONNX Runtime:", ort.__version__)
print("ONNX providers:", ort.get_available_providers())
print("MediaPipe:", mp.__version__)
print("Einops:", einops.__version__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the Avatar environment.")
print("GPU:", torch.cuda.get_device_name(0))
print("GPU VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1))
print("Ditto inference module:", inference.__file__)
print("Ditto PyTorch backend is ready.")
PYVERIFY

touch "$READY_MARKER"
echo "Ditto PyTorch runtime is ready."
