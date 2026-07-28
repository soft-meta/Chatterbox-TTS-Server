from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    "server": {"host": "0.0.0.0", "port": 8004},
    "tts_engine": {
        "device": "auto",
        "default_model": "chatterbox",
        "predefined_voices_path": "voices",
        "reference_audio_path": "reference_audio",
    },
    "storage": {
        "outputs_path": "outputs",
        "data_path": "data",
        "logs_path": "logs",
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
        return deepcopy(DEFAULT_CONFIG)
    loaded = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    return deep_merge(DEFAULT_CONFIG, loaded)


def save_config(config: dict[str, Any]) -> None:
    CONFIG_PATH.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
