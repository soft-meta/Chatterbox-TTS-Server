#!/usr/bin/env bash
set -euo pipefail

MM="${1:-/content/bin/micromamba}"
SERVER_DIR="${SOFTMETA_SERVER_DIR:?SOFTMETA_SERVER_DIR is required}"
ENV_NAME="${SOFTMETA_AVATAR_ENV:-echo310}"
ECHO_DIR="${SOFTMETA_ECHOMIMIC_DIR:-/content/EchoMimicV3}"
MODEL_ROOT="${SOFTMETA_ECHOMIMIC_MODELS:-/content/echomimic_v3_models}"
FLASH_DIR="$MODEL_ROOT/flash"
BASE_DIR="$FLASH_DIR/Wan2.1-Fun-V1.1-1.3B-InP"
AUDIO_DIR="$FLASH_DIR/chinese-wav2vec2-base"
TRANSFORMER_DIR="$FLASH_DIR/transformer"
FORCE="${SOFTMETA_FORCE_REINSTALL:-0}"

CODE_REVISION="7e89489ca51c0d008fc1963ec6c03fc5bd0b9397"
BASE_REVISION="fc913c34361f4ec879e2f9c78b4f11ae50a937d1"
FLASH_REVISION="311e176905a8c4c24b240b530488fe636ce4d249"
AUDIO_REVISION="3991242c806928916fff4a8c0e4f76acf661b743"
READY_MARKER="$MODEL_ROOT/.softmeta_echomimic_v3_flash_v110"
PATCH_MARKER="$ECHO_DIR/.softmeta_echomimic_v3_flash_v110"

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export HF_HOME="${HF_HOME:-/content/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

if [[ "$ECHO_DIR" != /content/* || "$MODEL_ROOT" != /content/* || "$ECHO_DIR" == "$MODEL_ROOT" ]]; then
  echo "EchoMimic code and model paths must be different explicit children of /content." >&2
  exit 2
fi
if [[ ! -x "$MM" ]]; then
  echo "micromamba was not found at: $MM" >&2
  exit 2
fi
if [[ ! -f "$SERVER_DIR/scripts/patch_echomimic_v3_flash.py" ]]; then
  echo "SoftMeta EchoMimicV3 patcher is missing from: $SERVER_DIR" >&2
  exit 2
fi

has_env() {
  "$MM" env list | awk '{print $1}' | grep -qx "$ENV_NAME"
}

runtime_ok() {
  has_env || return 1
  [[ -f "$ECHO_DIR/infer_flash.py" && -f "$PATCH_MARKER" ]] || return 1
  [[ -f "$BASE_DIR/diffusion_pytorch_model.safetensors" ]] || return 1
  [[ -f "$BASE_DIR/Wan2.1_VAE.pth" ]] || return 1
  [[ -f "$BASE_DIR/models_t5_umt5-xxl-enc-bf16.pth" ]] || return 1
  [[ -f "$BASE_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" ]] || return 1
  [[ -f "$AUDIO_DIR/config.json" ]] || return 1
  [[ -f "$TRANSFORMER_DIR/diffusion_pytorch_model.safetensors" ]] || return 1
  "$MM" run -n "$ENV_NAME" python - <<'PY' >/dev/null 2>&1
import torch, diffusers, transformers, librosa, moviepy, pyloudnorm
assert torch.cuda.is_available()
PY
}

if [[ "$FORCE" != "1" ]] && [[ -f "$READY_MARKER" ]] && runtime_ok; then
  echo "EchoMimicV3 Flash is already verified. Reusing it."
  exit 0
fi

if ! has_env; then
  "$MM" create -y -n "$ENV_NAME" python=3.10 pip
fi

if [[ ! -d "$ECHO_DIR/.git" ]]; then
  git clone https://github.com/antgroup/echomimic_v3.git "$ECHO_DIR"
fi
git -C "$ECHO_DIR" fetch --depth 1 origin "$CODE_REVISION"
git -C "$ECHO_DIR" checkout --detach "$CODE_REVISION"
python3 "$SERVER_DIR/scripts/patch_echomimic_v3_flash.py" "$ECHO_DIR"

"$MM" run -n "$ENV_NAME" python -m pip install -U pip wheel "setuptools==80.9.0"
"$MM" run -n "$ENV_NAME" python -m pip install \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

requirements=$(mktemp)
grep -Ev '^(torch|tensorflow|retina-face|gradio|mmgp)([<=>!~ ]|$)' \
  "$ECHO_DIR/requirements.txt" > "$requirements"
"$MM" run -n "$ENV_NAME" python -m pip install -r "$requirements"
rm -f -- "$requirements"
"$MM" run -n "$ENV_NAME" python -m pip install \
  "numpy==1.26.4" "diffusers==0.31.0" "transformers==4.46.3" \
  "accelerate==1.2.1" "moviepy==2.2.1" \
  "huggingface_hub[hf_xet]>=0.34.0" "pyloudnorm==0.1.1" "imageio-ffmpeg>=0.5"

mkdir -p "$BASE_DIR" "$AUDIO_DIR" "$TRANSFORMER_DIR"
free_gb=$(df -Pk /content | awk 'NR==2 {print int($4/1024/1024)}')
if (( free_gb < 28 )); then
  echo "EchoMimicV3 Flash needs at least 28 GB of free disk; found ${free_gb} GB." >&2
  exit 3
fi

SOFTMETA_BASE_DIR="$BASE_DIR" SOFTMETA_AUDIO_DIR="$AUDIO_DIR" \
SOFTMETA_TRANSFORMER_DIR="$TRANSFORMER_DIR" \
SOFTMETA_BASE_REVISION="$BASE_REVISION" SOFTMETA_AUDIO_REVISION="$AUDIO_REVISION" \
SOFTMETA_FLASH_REVISION="$FLASH_REVISION" \
"$MM" run -n "$ENV_NAME" python - <<'PYDOWNLOAD'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP",
    revision=os.environ["SOFTMETA_BASE_REVISION"],
    local_dir=os.environ["SOFTMETA_BASE_DIR"],
    max_workers=8,
)
snapshot_download(
    repo_id="TencentGameMate/chinese-wav2vec2-base",
    revision=os.environ["SOFTMETA_AUDIO_REVISION"],
    local_dir=os.environ["SOFTMETA_AUDIO_DIR"],
    max_workers=8,
)
snapshot_download(
    repo_id="BadToBest/EchoMimicV3",
    revision=os.environ["SOFTMETA_FLASH_REVISION"],
    local_dir=os.environ["SOFTMETA_TRANSFORMER_DIR"],
    allow_patterns=["echomimicv3-flash-pro/diffusion_pytorch_model.safetensors"],
    max_workers=4,
)
PYDOWNLOAD

# Flatten only the selected Flash weight into the official runtime layout.
flash_source="$TRANSFORMER_DIR/echomimicv3-flash-pro/diffusion_pytorch_model.safetensors"
flash_target="$TRANSFORMER_DIR/diffusion_pytorch_model.safetensors"
if [[ -f "$flash_source" && ! -f "$flash_target" ]]; then
  ln "$flash_source" "$flash_target" 2>/dev/null || cp "$flash_source" "$flash_target"
fi

SOFTMETA_ECHO_DIR="$ECHO_DIR" SOFTMETA_BASE_DIR="$BASE_DIR" \
SOFTMETA_AUDIO_DIR="$AUDIO_DIR" SOFTMETA_TRANSFORMER_DIR="$TRANSFORMER_DIR" \
"$MM" run -n "$ENV_NAME" python - <<'PYVERIFY'
import os, sys
from pathlib import Path
import torch

repo = Path(os.environ["SOFTMETA_ECHO_DIR"])
base = Path(os.environ["SOFTMETA_BASE_DIR"])
audio = Path(os.environ["SOFTMETA_AUDIO_DIR"])
transformer = Path(os.environ["SOFTMETA_TRANSFORMER_DIR"])
required = [
    repo / "infer_flash.py",
    repo / ".softmeta_echomimic_v3_flash_v110",
    base / "diffusion_pytorch_model.safetensors",
    base / "Wan2.1_VAE.pth",
    base / "models_t5_umt5-xxl-enc-bf16.pth",
    base / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    audio / "config.json",
    transformer / "diffusion_pytorch_model.safetensors",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit("Missing EchoMimicV3 files: " + ", ".join(missing))
sys.path.insert(0, str(repo))
from src.wan_transformer3d_audio_2512 import WanTransformerAudioMask3DModel
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the EchoMimicV3 environment.")
print("EchoMimicV3 Flash runtime imports and model manifest verified.")
print("GPU:", torch.cuda.get_device_name(0))
print("Flash model:", WanTransformerAudioMask3DModel.__name__)
PYVERIFY

touch "$READY_MARKER"
echo "EchoMimicV3 Flash is ready."
