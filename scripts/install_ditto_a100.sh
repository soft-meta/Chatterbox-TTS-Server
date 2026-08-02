#!/usr/bin/env bash
set -euo pipefail

# SoftMeta v0.9.1 A100 40GB Ditto installer.
# Usage:
#   bash scripts/install_ditto_a100.sh /content/bin/micromamba
# Optional environment variables:
#   SOFTMETA_AVATAR_ENV=avatar310
#   SOFTMETA_DITTO_DIR=/content/ditto-talkinghead
#   SOFTMETA_SERVER_DIR=/content/Chatterbox-TTS-Server

MM="${1:-/content/bin/micromamba}"
ENV_NAME="${SOFTMETA_AVATAR_ENV:-avatar310}"
DITTO_DIR="${SOFTMETA_DITTO_DIR:-/content/ditto-talkinghead}"
SERVER_DIR="${SOFTMETA_SERVER_DIR:-/content/Chatterbox-TTS-Server}"

if [[ ! -x "$MM" ]]; then
  echo "micromamba was not found at: $MM" >&2
  exit 2
fi
if [[ ! -f "$SERVER_DIR/requirements-avatar.txt" ]]; then
  echo "SoftMeta server was not found at: $SERVER_DIR" >&2
  exit 2
fi

if "$MM" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  "$MM" env remove -n "$ENV_NAME" -y || true
fi
"$MM" create -y -n "$ENV_NAME" -c conda-forge python=3.10 pip

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "Detected GPU:"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

rm -rf "$DITTO_DIR"
git clone --depth 1 https://github.com/antgroup/ditto-talkinghead.git "$DITTO_DIR"

"$MM" run -n "$ENV_NAME" python -m pip install -U pip wheel setuptools
"$MM" run -n "$ENV_NAME" python -m pip install \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
"$MM" run -n "$ENV_NAME" python -m pip install --no-cache-dir \
  -r "$SERVER_DIR/requirements-avatar.txt"

# TensorRT is preferred on A100. A failed install is not fatal because the
# released Ditto PyTorch backend remains available as a fallback.
set +e
"$MM" run -n "$ENV_NAME" python -m pip install --no-cache-dir \
  --extra-index-url https://pypi.nvidia.com \
  tensorrt==8.6.1
TRT_STATUS=$?
set -e
if [[ $TRT_STATUS -ne 0 ]]; then
  echo "TensorRT 8.6.1 could not be installed. SoftMeta will use Ditto PyTorch."
fi

"$MM" run -n "$ENV_NAME" python - <<PYDOWNLOAD
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="digital-avatar/ditto-talkinghead",
    local_dir=r"$DITTO_DIR/checkpoints",
)
print("Ditto checkpoints downloaded.")
PYDOWNLOAD

AVATAR_PYTHON="$($MM run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)')"
"$MM" run -n "$ENV_NAME" python - <<PYVERIFY
from pathlib import Path
import torch
root = Path(r"$DITTO_DIR")
required = [
    root / "inference.py",
    root / "checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl",
    root / "checkpoints/ditto_pytorch",
]
missing = [str(path) for path in required if not path.exists()]
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
if missing:
    raise SystemExit("Missing Ditto files: " + ", ".join(missing))
print("Ditto PyTorch backend is ready.")
trt_cfg = root / "checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
trt_root = root / "checkpoints/ditto_trt_Ampere_Plus"
print("Ditto TensorRT assets ready:", trt_cfg.is_file() and trt_root.is_dir())
PYVERIFY

cat <<SUMMARY

Avatar installation completed.
Set these variables before starting SoftMeta:
  SOFTMETA_AVATAR_PYTHON=$AVATAR_PYTHON
  SOFTMETA_DITTO_DIR=$DITTO_DIR
  SOFTMETA_DITTO_CHECKPOINTS=$DITTO_DIR/checkpoints

Important: review THIRD_PARTY_NOTICES.md before commercial deployment.
SUMMARY
