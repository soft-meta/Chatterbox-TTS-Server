from __future__ import annotations

import numpy as np
import sys
import types

stub = types.ModuleType("softmeta_chatterbox")
class _GenerationSettings:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)
class _SoftMetaChatterboxEngine: pass
stub.GenerationSettings = _GenerationSettings
stub.SoftMetaChatterboxEngine = _SoftMetaChatterboxEngine
sys.modules.setdefault("softmeta_chatterbox", stub)

from quality_control import QualityController
from queue_manager import QueueManager


def _tone(seconds: float, sr: int = 24000) -> np.ndarray:
    t = np.arange(max(1, int(sr * seconds)), dtype=np.float32) / sr
    return (0.06 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)


def test_short_slow_tail_is_advisory_not_hard_failure() -> None:
    qc = QualityController()
    # Nine spoken words across 15 seconds can legitimately happen in a final tail
    # containing an expressive breath/event and sentence-ending pauses. The waveform
    # is healthy, so rate alone must never abort the whole narration.
    audio = _tone(15.0)
    report = qc.evaluate_acoustic(
        audio,
        24000,
        "These final nine words close the story very gently.",
    )
    assert report.metrics["wpm"] < 45
    assert report.passed is True
    assert report.hard_failure is False


def test_short_tail_rate_outlier_does_not_trigger_legacy_turbo_retry() -> None:
    metrics = {
        "rms_dbfs": -18.0,
        "peak_dbfs": -3.0,
        "seconds_per_word": 1.10,
        "word_count": 9.0,
    }
    assert QueueManager._turbo_chunk_is_unstable(metrics, [], intentional_emotion=False) is False


def test_long_chunk_extreme_rate_can_still_request_a_retry() -> None:
    metrics = {
        "rms_dbfs": -18.0,
        "peak_dbfs": -3.0,
        "seconds_per_word": 1.10,
        "word_count": 35.0,
    }
    assert QueueManager._turbo_chunk_is_unstable(metrics, [], intentional_emotion=False) is True
