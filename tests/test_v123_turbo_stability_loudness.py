from pathlib import Path
import sys
import types

import numpy as np

# QueueManager imports the engine adapter, which normally imports the separately
# installed softmeta-chatterbox-v2 package. Stub only its import surface here so
# unit tests can exercise server-side audio helpers without model weights.
stub = types.ModuleType("softmeta_chatterbox")
class _GenerationSettings:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
class _SoftMetaChatterboxEngine:
    pass
stub.GenerationSettings = _GenerationSettings
stub.SoftMetaChatterboxEngine = _SoftMetaChatterboxEngine
sys.modules.setdefault("softmeta_chatterbox", stub)

from queue_manager import (
    MOTIVATIONAL_TURBO_PROFILE,
    QueueManager,
    TURBO_CHUNK_PEAK_CEILING_DBFS,
    TURBO_CHUNK_TARGET_RMS_DBFS,
    TURBO_FINAL_TARGET_LUFS,
)

ROOT = Path(__file__).resolve().parents[1]


def test_turbo_profile_is_more_conservative() -> None:
    assert MOTIVATIONAL_TURBO_PROFILE["temperature"] == 0.60
    assert MOTIVATIONAL_TURBO_PROFILE["top_p"] == 0.85
    assert MOTIVATIONAL_TURBO_PROFILE["top_k"] == 600
    assert MOTIVATIONAL_TURBO_PROFILE["chunk_words"] == 85


def test_per_chunk_leveling_raises_quiet_audio_and_caps_peak() -> None:
    sr = 24000
    t = np.arange(sr, dtype=np.float32) / sr
    quiet = 0.01 * np.sin(2 * np.pi * 220 * t)
    leveled = QueueManager._level_turbo_chunk(quiet)
    metrics = QueueManager._audio_metrics(leveled, sr, 3)
    assert metrics["rms_dbfs"] > -21.0
    assert metrics["peak_dbfs"] <= TURBO_CHUNK_PEAK_CEILING_DBFS + 0.2


def test_stability_guard_detects_sudden_loud_fast_chunk() -> None:
    history = [
        {"rms_dbfs": -22.0, "peak_dbfs": -5.0, "seconds_per_word": 0.40},
        {"rms_dbfs": -21.5, "peak_dbfs": -4.5, "seconds_per_word": 0.42},
        {"rms_dbfs": -22.5, "peak_dbfs": -5.5, "seconds_per_word": 0.39},
    ]
    bad = {"rms_dbfs": -14.0, "peak_dbfs": -1.0, "seconds_per_word": 0.22}
    assert QueueManager._turbo_chunk_is_unstable(bad, history)


def test_final_mastering_is_louder_than_v122() -> None:
    source = (ROOT / "queue_manager.py").read_text(encoding="utf-8")
    assert TURBO_FINAL_TARGET_LUFS == -12.5
    assert "measured_I=" in source
    assert "linear=true" in source
    assert "_level_turbo_chunk" in source
    assert "MOTIVATIONAL_TURBO_RETRY_PROFILE" in source


def test_clause_aware_splitter_is_present() -> None:
    source = (ROOT / "utils.py").read_text(encoding="utf-8")
    assert "_CLAUSE_BOUNDARY" in source
