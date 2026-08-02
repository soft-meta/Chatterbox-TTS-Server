from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_generate_voice_ui_uses_moss_candidate_workflow() -> None:
    html = (ROOT / "ui" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "ui" / "app.js").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements-voice.txt").read_text(encoding="utf-8")

    assert "MOSS VoiceGenerator" in html
    assert "Generate Candidates" in html
    assert "Naturalness" in script
    assert "audio_temperature" in (ROOT / "voice_worker.py").read_text(encoding="utf-8")
    assert "transformers==" not in requirements
    assert "speechbrain==1.1.0" in requirements
