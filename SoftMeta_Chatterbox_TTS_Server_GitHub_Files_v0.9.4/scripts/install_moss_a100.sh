#!/usr/bin/env bash
set -euo pipefail

# Optional MOSS VoiceGenerator runtime for SoftMeta.
# This script is idempotent: a verified environment is reused on reruns.

MM="${1:-/content/bin/micromamba}"
ENV_NAME="${SOFTMETA_MOSS_ENV:-moss312}"
MOSS_DIR="${SOFTMETA_MOSS_DIR:-/content/MOSS-TTS}"
SERVER_DIR="${SOFTMETA_SERVER_DIR:-/content/Chatterbox-TTS-Server}"
FORCE="${SOFTMETA_FORCE_REINSTALL:-0}"
READY_MARKER="$MOSS_DIR/.softmeta_moss_runtime_v094"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/content/.cache/pip}"

if [[ ! -x "$MM" ]]; then
  echo "micromamba was not found at: $MM" >&2
  exit 2
fi
if [[ ! -f "$SERVER_DIR/requirements-voice.txt" ]]; then
  echo "SoftMeta server was not found at: $SERVER_DIR" >&2
  exit 2
fi

has_env() {
  "$MM" env list | awk '{print $1}' | grep -qx "$ENV_NAME"
}

runtime_ok() {
  has_env || return 1
  "$MM" run -n "$ENV_NAME" python - <<'PY' >/dev/null 2>&1
import torch
import torchaudio
import soundfile
from transformers import AutoModel, AutoProcessor
from speechbrain.inference.speaker import EncoderClassifier
assert torch.cuda.is_available()
PY
}

if [[ "$FORCE" != "1" && -f "$READY_MARKER" ]] && runtime_ok; then
  echo "MOSS VoiceGenerator runtime is already verified. Reusing it."
  exit 0
fi

if [[ "$FORCE" == "1" ]]; then
  if has_env; then
    "$MM" env remove -n "$ENV_NAME" -y || true
  fi
  rm -rf "$MOSS_DIR"
fi

if ! has_env; then
  "$MM" create -y -n "$ENV_NAME" -c conda-forge python=3.12 pip
fi

if [[ -d "$MOSS_DIR/.git" ]]; then
  git -C "$MOSS_DIR" fetch --depth 1 origin main
  git -C "$MOSS_DIR" reset --hard origin/main
else
  rm -rf "$MOSS_DIR"
  git clone --depth 1 https://github.com/OpenMOSS/MOSS-TTS.git "$MOSS_DIR"
fi

"$MM" run -n "$ENV_NAME" python -m pip install -U pip wheel setuptools
"$MM" run -n "$ENV_NAME" python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -e "$MOSS_DIR[torch-runtime]"
"$MM" run -n "$ENV_NAME" python -m pip install \
  -r "$SERVER_DIR/requirements-voice.txt"

# Import verification is the source of truth. pip check is informative because
# upstream ML projects can declare optional package ranges that overlap.
"$MM" run -n "$ENV_NAME" python -m pip check || \
  echo "pip check reported an optional dependency warning; import verification will continue."

"$MM" run -n "$ENV_NAME" python - <<'PYVERIFY'
import sys
from importlib.metadata import version
import torch
import torchaudio
import soundfile
from transformers import AutoModel, AutoProcessor
from speechbrain.inference.speaker import EncoderClassifier

print("MOSS VoiceGenerator environment")
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("TorchAudio:", torchaudio.__version__)
print("Transformers:", version("transformers"))
print("SpeechBrain:", version("speechbrain"))
print("SoundFile:", soundfile.__version__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the MOSS environment.")
print("GPU:", torch.cuda.get_device_name(0))
print("AutoModel:", AutoModel.__name__)
print("AutoProcessor:", AutoProcessor.__name__)
print("Speaker checker:", EncoderClassifier.__name__)
print("MOSS environment verification passed.")
PYVERIFY

touch "$READY_MARKER"
echo "MOSS VoiceGenerator runtime is ready."
