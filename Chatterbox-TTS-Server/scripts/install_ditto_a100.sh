\
#!/usr/bin/env bash
set -euo pipefail

# SoftMeta v0.9.2 A100 40GB Ditto installer.
# Stable default: official Ditto PyTorch backend.
# TensorRT 8.6.1 is not installed automatically because current Colab CUDA /
# Python images often cannot build that legacy package. Set
# SOFTMETA_TRY_TENSORRT=1 only when using a matching CUDA/TensorRT image.

MM="${1:-/content/bin/micromamba}"
ENV_NAME="${SOFTMETA_AVATAR_ENV:-avatar310}"
DITTO_DIR="${SOFTMETA_DITTO_DIR:-/content/ditto-talkinghead}"
SERVER_DIR="${SOFTMETA_SERVER_DIR:-/content/Chatterbox-TTS-Server}"
TRY_TENSORRT="${SOFTMETA_TRY_TENSORRT:-0}"

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

if [[ "$TRY_TENSORRT" == "1" ]]; then
  echo "Optional TensorRT installation requested."
  set +e
  "$MM" run -n "$ENV_NAME" python -m pip install --no-cache-dir \
    --extra-index-url https://pypi.nvidia.com \
    tensorrt==8.6.1
  TRT_STATUS=$?
  set -e
  if [[ $TRT_STATUS -ne 0 ]]; then
    echo "Optional TensorRT installation failed. Ditto PyTorch remains available."
  fi
else
  echo "Skipping legacy TensorRT 8.6.1 on Colab. Ditto PyTorch is the stable A100 40GB backend."
fi

"$MM" run -n "$ENV_NAME" python - <<PYDOWNLOAD
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="digital-avatar/ditto-talkinghead",
    local_dir=r"$DITTO_DIR/checkpoints",
)
print("Ditto checkpoints downloaded.")
PYDOWNLOAD

"$MM" run -n "$ENV_NAME" python -m pip check

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
    print("GPU VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1))
if missing:
    raise SystemExit("Missing Ditto files: " + ", ".join(missing))
print("Ditto PyTorch backend is ready.")
try:
    import tensorrt
except Exception:
    print("TensorRT: not installed, PyTorch mode selected.")
else:
    print("TensorRT:", tensorrt.__version__)
PYVERIFY

cat <<SUMMARY

Avatar installation completed.
Stable backend: Ditto PyTorch
Set these variables before starting SoftMeta:
  SOFTMETA_AVATAR_PYTHON=$AVATAR_PYTHON
  SOFTMETA_DITTO_DIR=$DITTO_DIR
  SOFTMETA_DITTO_CHECKPOINTS=$DITTO_DIR/checkpoints
  SOFTMETA_ENABLE_TENSORRT=0

Important: review THIRD_PARTY_NOTICES.md before commercial deployment.
SUMMARY
