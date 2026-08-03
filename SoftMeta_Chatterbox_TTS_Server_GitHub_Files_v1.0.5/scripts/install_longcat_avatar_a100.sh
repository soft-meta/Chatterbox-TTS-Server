#!/usr/bin/env bash
set -euo pipefail

# Optional high-realism avatar runtime for SoftMeta. LongCat-Video-Avatar 1.5
# is isolated from the TTS Python environment and uses its official INT8 DiT,
# Whisper-Large audio encoder and distilled eight-step inference path.

MM="${1:-/content/bin/micromamba}"
ENV_NAME="${SOFTMETA_AVATAR_ENV:-longcat310}"
LONGCAT_DIR="${SOFTMETA_LONGCAT_DIR:-/content/LongCat-Video}"
MODEL_ROOT="${SOFTMETA_LONGCAT_MODELS:-/content/longcat_models}"
BASE_MODEL_DIR="$MODEL_ROOT/LongCat-Video"
AVATAR_MODEL_DIR="$MODEL_ROOT/LongCat-Video-Avatar-1.5"
SERVER_DIR="${SOFTMETA_SERVER_DIR:-/content/Chatterbox-TTS-Server}"
FORCE="${SOFTMETA_FORCE_REINSTALL:-0}"
UPSTREAM_COMMIT="6b3f4b8582a8bc3f20f795735f5383716c4ba794"
# This complete revision contains all four INT8 safetensors shards. The older
# d3cfdd revision was an incomplete upload containing only shard 1 of 4.
AVATAR_MODEL_REVISION="92016c71d5d318d0f5d84e4db30015a571484ab6"
READY_MARKER="$MODEL_ROOT/.softmeta_longcat_avatar_v15_v105"

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/content/.cache/pip}"
export HF_HOME="${HF_HOME:-/content/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/softmeta-matplotlib}"
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$MODEL_ROOT" "$MPLCONFIGDIR"

safe_content_child() {
  local target
  target="$(realpath -m -- "$1")"
  [[ "$target" == /content/* && "$target" != /content && "$target" != /content/ ]]
}

if ! safe_content_child "$LONGCAT_DIR" || ! safe_content_child "$MODEL_ROOT"; then
  echo "LongCat code and model directories must be explicit child directories of /content." >&2
  exit 2
fi
if [[ "$(realpath -m -- "$LONGCAT_DIR")" == "$(realpath -m -- "$MODEL_ROOT")" ]]; then
  echo "LongCat code and model directories must be different." >&2
  exit 2
fi

if [[ ! -x "$MM" ]]; then
  echo "micromamba was not found at: $MM" >&2
  exit 2
fi
if [[ ! -f "$SERVER_DIR/scripts/patch_longcat_runtime.py" ]]; then
  echo "SoftMeta LongCat runtime patcher was not found in: $SERVER_DIR" >&2
  exit 2
fi

has_env() {
  "$MM" env list | awk '{print $1}' | grep -qx "$ENV_NAME"
}

model_shards_ok() {
  local index="$AVATAR_MODEL_DIR/base_model_int8/quantized_model.safetensors.index.json"
  [[ -f "$index" ]] || return 1
  "$MM" run -n "$ENV_NAME" python - "$index" <<'PY' >/dev/null 2>&1
import json
import sys
from pathlib import Path

index = Path(sys.argv[1])
payload = json.loads(index.read_text())
shards = sorted(set(payload.get("weight_map", {}).values()))
if not shards or any(not (index.parent / shard).is_file() for shard in shards):
    raise SystemExit(1)
PY
}

runtime_ok() {
  has_env || return 1
  [[ -f "$LONGCAT_DIR/run_demo_avatar_single_audio_to_video.py" ]] || return 1
  [[ -f "$LONGCAT_DIR/.softmeta_longcat_runtime_patch_v105" ]] || return 1
  [[ -f "$BASE_MODEL_DIR/text_encoder/model.safetensors.index.json" ]] || return 1
  [[ -f "$BASE_MODEL_DIR/vae/config.json" ]] || return 1
  [[ -f "$AVATAR_MODEL_DIR/base_model_int8/quantized_model.safetensors.index.json" ]] || return 1
  model_shards_ok || return 1
  [[ -f "$AVATAR_MODEL_DIR/whisper-large-v3/model.safetensors" ]] || return 1
  SOFTMETA_LONGCAT_DIR="$LONGCAT_DIR" "$MM" run -n "$ENV_NAME" python - <<'PY' >/dev/null 2>&1
import os
import sys
import torch
import transformers
import diffusers
import audio_separator

sys.path.insert(0, os.environ["SOFTMETA_LONGCAT_DIR"])
from longcat_video.modules.quantization import load_quantized_dit

assert callable(load_quantized_dit)
assert torch.cuda.is_available()
PY
}

if [[ "$FORCE" != "1" ]] && runtime_ok; then
  touch "$READY_MARKER"
  echo "LongCat Video Avatar 1.5 is already verified. Reusing it."
  exit 0
fi

if [[ "$FORCE" == "1" ]]; then
  if has_env; then
    "$MM" env remove -n "$ENV_NAME" -y || true
  fi
  if [[ -d "$LONGCAT_DIR" ]]; then
    find "$LONGCAT_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
  fi
  if [[ -d "$MODEL_ROOT" ]]; then
    find "$MODEL_ROOT" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
  fi
fi

if ! has_env; then
  "$MM" create -y -n "$ENV_NAME" -c conda-forge python=3.10 pip
fi

echo "Detected GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

if [[ -d "$LONGCAT_DIR/.git" ]]; then
  git -C "$LONGCAT_DIR" fetch --depth 1 origin "$UPSTREAM_COMMIT"
  git -C "$LONGCAT_DIR" reset --hard "$UPSTREAM_COMMIT"
else
  if [[ -e "$LONGCAT_DIR" ]]; then
    find "$LONGCAT_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} + 2>/dev/null || true
  fi
  git clone --filter=blob:none --no-checkout https://github.com/meituan-longcat/LongCat-Video.git "$LONGCAT_DIR"
  git -C "$LONGCAT_DIR" fetch --depth 1 origin "$UPSTREAM_COMMIT"
  git -C "$LONGCAT_DIR" checkout --detach "$UPSTREAM_COMMIT"
fi

python3 "$SERVER_DIR/scripts/patch_longcat_runtime.py" "$LONGCAT_DIR"

"$MM" run -n "$ENV_NAME" python -m pip install -U pip wheel "setuptools==80.9.0"
"$MM" run -n "$ENV_NAME" python -m pip install \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

requirements_main=$(mktemp)
requirements_avatar=$(mktemp)
grep -Ev '^(torch|torchvision|torchaudio|flash-attn)=' "$LONGCAT_DIR/requirements.txt" > "$requirements_main"
# Upstream contains two entries that cannot be installed from PyPI and are not
# Python runtime dependencies of the Avatar pipeline:
# - libsndfile1 is an Ubuntu library already installed by the Colab setup cell.
# - tritonserverclient is not NVIDIA's tritonclient package and has no release.
# Preserve every installable pinned dependency and exclude only those entries.
grep -Ev '^(libsndfile1|tritonserverclient)([<=>!~ ]|$)' \
  "$LONGCAT_DIR/requirements_avatar.txt" > "$requirements_avatar"
"$MM" run -n "$ENV_NAME" python -m pip install -r "$requirements_main"
"$MM" run -n "$ENV_NAME" python -m pip install -r "$requirements_avatar"
rm -f -- "$requirements_main" "$requirements_avatar"
"$MM" run -n "$ENV_NAME" python -m pip install "huggingface_hub[hf_xet]>=0.34.0"
"$MM" run -n "$ENV_NAME" python -m pip install "accelerate>=1.2,<2"

if ! "$MM" run -n "$ENV_NAME" python -c "import flash_attn" >/dev/null 2>&1; then
  export MAX_JOBS="${MAX_JOBS:-4}"
  "$MM" run -n "$ENV_NAME" python -m pip install ninja psutil packaging
  "$MM" run -n "$ENV_NAME" python -m pip install --no-build-isolation flash-attn==2.7.4.post1
fi

if [[ ! -f "$AVATAR_MODEL_DIR/base_model_int8/quantized_model.safetensors.index.json" ]]; then
  free_gb=$(df -Pk /content | awk 'NR==2 {print int($4/1024/1024)}')
  if (( free_gb < 55 )); then
    echo "LongCat requires at least 55 GB of free Colab disk before the first model download; found ${free_gb} GB." >&2
    exit 3
  fi
fi

SOFTMETA_BASE_MODEL_DIR="$BASE_MODEL_DIR" \
SOFTMETA_AVATAR_MODEL_DIR="$AVATAR_MODEL_DIR" \
SOFTMETA_AVATAR_MODEL_REVISION="$AVATAR_MODEL_REVISION" \
"$MM" run -n "$ENV_NAME" python - <<'PYDOWNLOAD'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="meituan-longcat/LongCat-Video",
    local_dir=os.environ["SOFTMETA_BASE_MODEL_DIR"],
    allow_patterns=["tokenizer/**", "text_encoder/**", "vae/**"],
    max_workers=8,
)
snapshot_download(
    repo_id="meituan-longcat/LongCat-Video-Avatar-1.5",
    revision=os.environ["SOFTMETA_AVATAR_MODEL_REVISION"],
    local_dir=os.environ["SOFTMETA_AVATAR_MODEL_DIR"],
    allow_patterns=[
        "base_model_int8/**",
        # The pinned official Avatar 1.5 revision has no lora/ directory.
        # Upstream inference loads a standalone DMD LoRA only when it exists.
        "scheduler/**",
        "vocal_separator/Kim_Vocal_2.onnx",
        "whisper-large-v3/config.json",
        "whisper-large-v3/model.safetensors",
        "whisper-large-v3/preprocessor_config.json",
    ],
    max_workers=8,
)
print("LongCat INT8, Whisper-Large and shared model components downloaded.")
PYDOWNLOAD

"$MM" run -n "$ENV_NAME" python -m pip check

SOFTMETA_LONGCAT_DIR="$LONGCAT_DIR" \
SOFTMETA_BASE_MODEL_DIR="$BASE_MODEL_DIR" \
SOFTMETA_AVATAR_MODEL_DIR="$AVATAR_MODEL_DIR" \
"$MM" run -n "$ENV_NAME" python - <<'PYVERIFY'
import os
import sys
from pathlib import Path
import torch
import transformers
import diffusers

repo = Path(os.environ["SOFTMETA_LONGCAT_DIR"])
base = Path(os.environ["SOFTMETA_BASE_MODEL_DIR"])
avatar = Path(os.environ["SOFTMETA_AVATAR_MODEL_DIR"])
required = [
    repo / "run_demo_avatar_single_audio_to_video.py",
    repo / ".softmeta_longcat_runtime_patch_v105",
    base / "tokenizer/tokenizer.json",
    base / "text_encoder/model.safetensors.index.json",
    base / "vae/config.json",
    avatar / "base_model_int8/quantized_model.safetensors.index.json",
    avatar / "scheduler/scheduler_config.json",
    avatar / "vocal_separator/Kim_Vocal_2.onnx",
    avatar / "whisper-large-v3/model.safetensors",
]
missing = [str(path) for path in required if not path.exists()]
index = avatar / "base_model_int8/quantized_model.safetensors.index.json"
if index.exists():
    import json
    shards = sorted(set(json.loads(index.read_text()).get("weight_map", {}).values()))
    missing.extend(str(index.parent / shard) for shard in shards if not (index.parent / shard).is_file())
if missing:
    raise SystemExit("Missing LongCat files: " + ", ".join(missing))
sys.path.insert(0, str(repo))
from longcat_video.modules.quantization import load_quantized_dit
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the LongCat environment.")
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Diffusers:", diffusers.__version__)
print("GPU:", torch.cuda.get_device_name(0))
print("GPU VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1))
print("LongCat INT8 loader:", load_quantized_dit.__name__)
print("LongCat Video Avatar 1.5 is ready.")
PYVERIFY

touch "$READY_MARKER"
echo "LongCat Video Avatar 1.5 runtime is ready."
