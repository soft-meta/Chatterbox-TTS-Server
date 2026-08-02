\
#!/usr/bin/env bash
set -euo pipefail

# SoftMeta v0.9.2 MOSS VoiceGenerator installer.
# Uses the upstream MOSS-TTS dependency set instead of overriding torch,
# torchaudio, transformers, or scipy with conflicting versions.

MM="${1:-/content/bin/micromamba}"
ENV_NAME="${SOFTMETA_MOSS_ENV:-moss312}"
MOSS_DIR="${SOFTMETA_MOSS_DIR:-/content/MOSS-TTS}"
SERVER_DIR="${SOFTMETA_SERVER_DIR:-/content/Chatterbox-TTS-Server}"

if [[ ! -x "$MM" ]]; then
  echo "micromamba was not found at: $MM" >&2
  exit 2
fi
if [[ ! -f "$SERVER_DIR/requirements-voice.txt" ]]; then
  echo "SoftMeta server was not found at: $SERVER_DIR" >&2
  exit 2
fi

if "$MM" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  "$MM" env remove -n "$ENV_NAME" -y || true
fi
"$MM" create -y -n "$ENV_NAME" -c conda-forge python=3.12 pip

rm -rf "$MOSS_DIR"
git clone --depth 1 https://github.com/OpenMOSS/MOSS-TTS.git "$MOSS_DIR"

"$MM" run -n "$ENV_NAME" python -m pip install -U pip wheel setuptools
"$MM" run -n "$ENV_NAME" python -m pip install --no-cache-dir \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -e "$MOSS_DIR[torch-runtime]"
"$MM" run -n "$ENV_NAME" python -m pip install --no-cache-dir \
  -r "$SERVER_DIR/requirements-voice.txt"

"$MM" run -n "$ENV_NAME" python -m pip check

"$MM" run -n "$ENV_NAME" python - <<'PYVERIFY'
import sys
from importlib.metadata import version
import torch
import torchaudio
import soundfile
from transformers import AutoModel, AutoProcessor
from speechbrain.inference.speaker import EncoderClassifier

print("MOSS VoiceGenerator environment")
print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("TorchAudio:", torchaudio.__version__)
print("Transformers:", version("transformers"))
print("SpeechBrain:", version("speechbrain"))
print("SoundFile:", soundfile.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the MOSS environment.")
print("GPU:", torch.cuda.get_device_name(0))
print("AutoModel:", AutoModel.__name__)
print("AutoProcessor:", AutoProcessor.__name__)
print("Speaker checker:", EncoderClassifier.__name__)
print("MOSS environment verification passed.")
PYVERIFY

echo "MOSS VoiceGenerator installation completed without dependency conflicts."
