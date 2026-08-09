from pathlib import Path

import numpy as np
import soundfile as sf

from emotion_director import analyze_serious_senior_advisor
from professional_audio import master_professional_voice, shape_professional_pauses
from speech_pipeline import (
    analyze_key_emphasis,
    prepare_senior_clear_speech_text,
)


def test_clear_speech_keeps_words_and_tags_but_simplifies_dense_punctuation():
    source = "[happy] However (and this matters), Dr. Miller said: keep going — slowly; and listen carefully."
    result = prepare_senior_clear_speech_text(source)
    assert "[happy]" in result
    assert "Doctor Miller" in result
    assert "—" not in result
    assert ";" not in result
    assert "(" not in result and ")" not in result
    assert "keep going" in result


def test_key_emphasis_is_sparse_serious_and_never_comedic():
    text = (
        "Many people overlook this habit. "
        "Remember this because it can make a real difference. "
        "The next point is simple. "
        "Do not ignore sudden chest pressure or trouble breathing. "
        "Keep the rest of the advice calm and practical."
    )
    analysis = analyze_key_emphasis(text)
    assert 1 <= len(analysis.placements) <= 2
    assert all(item["kind"] == "serious-emphasis" for item in analysis.placements)
    assert all(item["tag"] in {"narration", "dramatic"} for item in analysis.placements)


def test_auto_emotion_reports_key_emphasis_source_when_applicable():
    text = (
        "Growing older changes many routines, but it does not mean you should stop paying attention to your health. "
        "A calm daily routine can make it easier to notice small changes before they become bigger problems. "
        "Many people focus only on one symptom and forget to look at the pattern across several days. "
        "Remember this because small warning signs can be easy to dismiss when life is busy. "
        "Take your time and notice what your body is telling you, especially when a change is new or persistent. "
        "Keep a short note of what happened, when it happened, and what you were doing at the time. "
        "Do not ignore sudden chest pressure or severe shortness of breath. "
        "Talk with your doctor when something does not feel right, and explain the change in simple clear words."
    )
    result = analyze_serious_senior_advisor(text)
    assert any(item.get("source") in {"key-emphasis", "semantic", "advice-fallback"} for item in result.placements)


def test_pause_shaper_preserves_normal_breath_and_compacts_long_dead_air():
    sr = 24000
    tone = (0.08 * np.sin(2 * np.pi * 220 * np.arange(int(sr * 0.35)) / sr)).astype(np.float32)
    normal = np.zeros(int(sr * 0.42), dtype=np.float32)
    long_gap = np.zeros(int(sr * 2.2), dtype=np.float32)
    audio = np.concatenate([tone, normal, tone, long_gap, tone])
    shaped = shape_professional_pauses(audio, sr)
    # Keep the ordinary breath-sized pause nearly unchanged, while removing >1.5 s of dead air.
    assert len(audio) - len(shaped) > int(sr * 1.4)
    assert len(shaped) > int(sr * (0.35 * 3 + 0.35 + 0.30))


def test_professional_mastering_improves_quiet_speech_presence(tmp_path: Path):
    sr = 24000
    t = np.arange(sr * 3) / sr
    # Speech-like quiet signal with low-frequency rumble and modest upper harmonics.
    audio = (
        0.018 * np.sin(2 * np.pi * 180 * t)
        + 0.006 * np.sin(2 * np.pi * 1200 * t)
        + 0.003 * np.sin(2 * np.pi * 3200 * t)
        + 0.004 * np.sin(2 * np.pi * 35 * t)
    ).astype(np.float32)
    path = tmp_path / "voice.wav"
    sf.write(path, audio, sr, subtype="PCM_16")
    before = sf.read(path, dtype="float32")[0]
    before_rms = float(np.sqrt(np.mean(before**2)))

    master_professional_voice(path)

    after, out_sr = sf.read(path, dtype="float32")
    after_rms = float(np.sqrt(np.mean(after**2)))
    assert out_sr == sr
    assert after_rms > before_rms * 1.8
    assert float(np.max(np.abs(after))) < 0.98
