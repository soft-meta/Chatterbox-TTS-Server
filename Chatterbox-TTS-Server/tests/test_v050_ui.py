from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v050_ui_contains_qwen_candidate_workflow() -> None:
    html = (ROOT / "ui" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "ui" / "app.js").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements-voice.txt").read_text(encoding="utf-8")

    assert "Qwen3-TTS VoiceDesign" in html
    assert 'data-field="generated_voice_candidate_count"' in html
    assert 'data-role="voice-candidate-list"' in html
    assert "function renderVoiceCandidates" in script
    assert "function saveVoiceCandidate" in script
    assert "/api/voice-designer/save" in script
    assert "qwen-tts==0.1.1" in requirements
    assert "parler-tts" not in requirements.lower()
