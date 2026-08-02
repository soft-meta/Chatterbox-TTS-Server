from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    "server": {
        "host": "0.0.0.0",
        "port": 8004,
        "auto_load_model": True,
    },
    "tts_engine": {
        "device": "auto",
        "default_model": "chatterbox",
        "predefined_voices_path": "voices",
        "reference_audio_path": "reference_audio",
        "generated_voices_path": "generated_voices",
    },
    "storage": {
        "outputs_path": "outputs",
        "data_path": "data",
        "logs_path": "logs",
        "video_outputs_path": "video_outputs",
    },
    "avatar": {
        "python": "",
        "ditto_dir": "/content/ditto-talkinghead",
        "ditto_checkpoints": "/content/ditto-talkinghead/checkpoints",
        "default_engine": "auto",
        "default_render_mode": "checkpointed",
        "default_segment_seconds": 120,
    },
    "generation_defaults": {
        "preset": "Motivational Speech",
        "language": "en",
        "temperature": 0.8,
        "exaggeration": 0.65,
        "cfg_weight": 0.35,
        "repetition_penalty": 1.2,
        "min_p": 0.05,
        "top_p": 1.0,
        "top_k": 1000,
        "speed_factor": 1.0,
        "seed": 2025,
        "split_text": True,
        "chunk_words": 90,
        "output_format": "wav",
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        save_config(DEFAULT_CONFIG)
        config = deepcopy(DEFAULT_CONFIG)
    else:
        loaded = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        config = deep_merge(DEFAULT_CONFIG, loaded)

    # Environment overrides make Colab and Docker deployment predictable.
    if os.getenv("SOFTMETA_HOST"):
        config["server"]["host"] = os.environ["SOFTMETA_HOST"]
    if os.getenv("SOFTMETA_PORT"):
        config["server"]["port"] = int(os.environ["SOFTMETA_PORT"])
    if os.getenv("SOFTMETA_DEVICE"):
        config["tts_engine"]["device"] = os.environ["SOFTMETA_DEVICE"]
    if os.getenv("SOFTMETA_MODEL"):
        config["tts_engine"]["default_model"] = os.environ["SOFTMETA_MODEL"]
    if os.getenv("SOFTMETA_AVATAR_PYTHON"):
        config["avatar"]["python"] = os.environ["SOFTMETA_AVATAR_PYTHON"]
    if os.getenv("SOFTMETA_DITTO_DIR"):
        config["avatar"]["ditto_dir"] = os.environ["SOFTMETA_DITTO_DIR"]
    if os.getenv("SOFTMETA_DITTO_CHECKPOINTS"):
        config["avatar"]["ditto_checkpoints"] = os.environ["SOFTMETA_DITTO_CHECKPOINTS"]
    return config


def save_config(config: dict[str, Any]) -> None:
    CONFIG_PATH.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
