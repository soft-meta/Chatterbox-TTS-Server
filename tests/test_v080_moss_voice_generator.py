from pathlib import Path

import numpy as np

from voice_designer import VoiceDesigner
from voice_worker import evaluate_acoustic_quality

ROOT = Path(__file__).resolve().parents[1]


def test_v080_uses_moss_as_primary_voice_designer() -> None:
    assert VoiceDesigner.MODEL_ID == "OpenMOSS-Team/MOSS-VoiceGenerator"
    assert VoiceDesigner.MODEL_REVISION == "97521ec"
    source = (ROOT / "voice_worker.py").read_text(encoding="utf-8")
    assert "AutoModel.from_pretrained" in source
    assert "processor.build_user_message" in source
    assert "audio_temperature" in source
    assert "MOSS VoiceGenerator" in source


def test_v080_identity_prompt_keeps_pitch_independent_from_age() -> None:
    profile = VoiceDesigner.build_profile(
        age=84,
        gender="male",
        language="en-US",
        emotion="reflective",
        description="",
        seed=2025,
        candidate_index=3,
    )
    prompt = profile.effective_description
    assert "American English" in prompt
    assert "84-year-old American man" in prompt
    assert "Older age must not force a deep pitch" in prompt
    assert "Do not stretch vowels" in prompt
    assert "pause after every few words" in prompt
    assert "AI assistant" in prompt
    assert "american_background" in profile.identity_traits


def test_v080_acoustic_quality_rejects_dead_air() -> None:
    sample_rate = 24000
    audio = np.zeros(sample_rate * 4, dtype=np.float32)
    result = evaluate_acoustic_quality(audio, sample_rate, 75)
    assert result["status"] == "reject"
    assert "too little active speech" in result["reasons"]


def test_v080_acoustic_quality_accepts_clean_dynamic_audio() -> None:
    sample_rate = 24000
    time = np.arange(sample_rate * 4, dtype=np.float32) / sample_rate
    carrier = 0.16 * np.sin(2 * np.pi * 120 * time)
    envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 1.3 * time) ** 2
    audio = (carrier * envelope).astype(np.float32)
    result = evaluate_acoustic_quality(audio, sample_rate, 70)
    assert result["status"] in {"pass", "review"}
    assert result["score"] > 50


def test_v080_ui_reports_quality_and_duplicate_rejections() -> None:
    script = (ROOT / "ui" / "app.js").read_text(encoding="utf-8")
    html = (ROOT / "ui" / "index.html").read_text(encoding="utf-8")
    assert "quality_rejected_count" in script
    assert "duplicate_rejected_count" in script
    assert "Naturalness" in script
    assert "official MOSS VoiceGenerator" in html
