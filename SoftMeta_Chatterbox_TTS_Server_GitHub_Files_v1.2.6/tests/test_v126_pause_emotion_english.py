from __future__ import annotations

import sys
import types

import numpy as np

# QueueManager imports the separately installed engine package. Stub the import
# surface so these server-side tests do not need model weights.
stub = types.ModuleType("softmeta_chatterbox")
class _GenerationSettings:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
class _SoftMetaChatterboxEngine:
    pass
stub.GenerationSettings = _GenerationSettings
stub.SoftMetaChatterboxEngine = _SoftMetaChatterboxEngine
sys.modules.setdefault("softmeta_chatterbox", stub)

from emotion_director import analyze_serious_senior_advisor
from queue_manager import MOTIVATIONAL_TURBO_PROFILE, QueueManager
from utils import prepare_american_english_tts_text


def _filler(words: int) -> str:
    base = (
        "Take your time with this advice and listen carefully because the goal is to understand "
        "the idea without feeling rushed or overwhelmed during a calm senior tutorial. "
    )
    tokens = (base * 20).split()
    return " ".join(tokens[:words])


def test_one_paragraph_250_word_script_can_receive_multiple_sparse_cues() -> None:
    text = " ".join([
        _filler(45),
        "Looking back, I wish I had understood this sooner because it was one of my biggest mistakes.",
        _filler(75),
        "The good news is that this simple change can make daily life easier and safer.",
        _filler(75),
        "I could not believe how much better I felt once I followed the routine consistently.",
        _filler(35),
    ])
    result = analyze_serious_senior_advisor(text)
    assert 230 <= result.total_words <= 290
    assert result.applied_count >= 2
    assert result.applied_count <= 3
    assert "[sigh]" not in result.tagged_text.lower()
    assert "[laugh]" not in result.tagged_text.lower()


def test_excessive_internal_silence_is_compacted_but_normal_breath_is_kept() -> None:
    sr = 24000
    t = np.arange(sr, dtype=np.float32) / sr
    tone = 0.12 * np.sin(2 * np.pi * 220 * t)

    very_long = np.concatenate([tone, np.zeros(sr * 20, dtype=np.float32), tone])
    fixed = QueueManager._compact_excessive_silence(very_long, sr)
    fixed_seconds = fixed.size / sr
    assert 2.25 <= fixed_seconds <= 2.55

    natural = np.concatenate([tone, np.zeros(int(sr * 0.45), dtype=np.float32), tone])
    untouched = QueueManager._compact_excessive_silence(natural, sr)
    assert abs(untouched.size - natural.size) <= int(sr * 0.03)


def test_parentheses_and_brackets_are_normalized_without_losing_words_or_safe_tags() -> None:
    source = (
        "[happy] The good news (especially after 70) is simple; keep moving. "
        "This [small habit] can help: take your time."
    )
    prepared = prepare_american_english_tts_text(source)
    assert "[happy]" in prepared
    assert "especially after 70" in prepared
    assert "small habit" in prepared
    assert "(" not in prepared and ")" not in prepared
    assert "[small habit]" not in prepared
    assert ";" not in prepared and ":" not in prepared


def test_motivational_turbo_is_locked_to_english_and_human_breath_join() -> None:
    assert MOTIVATIONAL_TURBO_PROFILE["language"] == "en"
    assert 180 <= MOTIVATIONAL_TURBO_PROFILE["inter_chunk_pause_ms"] <= 300
