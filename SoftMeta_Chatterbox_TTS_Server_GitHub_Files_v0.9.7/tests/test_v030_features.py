from pathlib import Path

import pytest
from pydantic import ValidationError

from config import DEFAULT_CONFIG
from models import AudioJobCreate

ROOT = Path(__file__).resolve().parents[1]


def test_generated_voice_job_requires_a_saved_reference() -> None:
    job = AudioJobCreate(
        audio_number=1,
        text="A small test sentence.",
        voice_mode="generated",
        voice_filename="Warm_American_Male_2025.wav",
    )
    assert job.voice_mode == "generated"

    with pytest.raises(ValidationError):
        AudioJobCreate(
            audio_number=1,
            text="A small test sentence.",
            voice_mode="generated",
        )


def test_generated_voice_storage_has_a_default_path() -> None:
    assert DEFAULT_CONFIG["tts_engine"]["generated_voices_path"] == "generated_voices"


def test_v030_ui_contains_requested_controls() -> None:
    html = (ROOT / "ui" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "ui" / "app.js").read_text(encoding="utf-8")

    assert 'id="remove-all"' in html
    assert 'data-voice-mode="generated"' in html
    assert 'data-voice-mode="default"' not in html
    assert "Keep first" not in html
    assert 'class="hidden-audio-engine"' in html
    assert 'data-role="playback-progress"' in html
    assert "function removeTab" in script
    assert "function generateDesignedVoice" in script
