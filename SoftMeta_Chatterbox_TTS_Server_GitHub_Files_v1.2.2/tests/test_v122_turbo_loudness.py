from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_turbo_loudness_mastering_is_present() -> None:
    source = (ROOT / "queue_manager.py").read_text(encoding="utf-8")
    assert "def _normalize_turbo_loudness" in source
    assert "loudnorm=I=-16.0:TP=-1.0:LRA=7.0" in source
    assert 'if options.get("model") == "chatterbox-turbo"' in source
    assert "Balancing Turbo loudness" in source


def test_original_is_not_globally_normalised() -> None:
    source = (ROOT / "queue_manager.py").read_text(encoding="utf-8")
    normalise_call = "await asyncio.to_thread(self._normalize_turbo_loudness, output_path)"
    assert normalise_call in source
    turbo_guard = source.index('if options.get("model") == "chatterbox-turbo"')
    call = source.index(normalise_call)
    assert turbo_guard < call
