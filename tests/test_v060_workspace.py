from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v060_defaults_to_one_removable_extra_tab() -> None:
    script = (ROOT / "ui" / "app.js").read_text(encoding="utf-8")
    assert "state.tabs = [defaultTab(1)];" in script
    assert "if (tab.number > 1)" in script
    assert "softMetaChatterboxTabsV6" in script


def test_v060_candidate_players_are_stable_during_polling() -> None:
    script = (ROOT / "ui" / "app.js").read_text(encoding="utf-8")
    assert "container.dataset.renderSignature" in script
    assert "if (!force && container.dataset.renderSignature === signature) return;" in script
    assert "other !== player && !other.paused" in script


def test_v060_api_link_and_professional_branding() -> None:
    html = (ROOT / "ui" / "index.html").read_text(encoding="utf-8")
    css = (ROOT / "ui" / "styles.css").read_text(encoding="utf-8")
    assert 'class="api-sub-link" href="/docs"' in html
    assert "SoftMeta Audio Studio" in html
    assert ".brand-mark" in css
    assert ".voice-candidate-card.is-playing" in css


def test_v060_bundles_five_predefined_voices() -> None:
    voices = sorted((ROOT / "voices").glob("SoftMeta_American_Male_*.wav"))
    metadata = sorted((ROOT / "voices").glob("SoftMeta_American_Male_*.json"))
    assert len(voices) == 5
    assert len(metadata) == 5


def test_v060_voice_profiles_are_more_diverse_and_age_aware() -> None:
    source = (ROOT / "voice_designer.py").read_text(encoding="utf-8")
    assert "_VOICE_FAMILIES" in source
    assert "_age_vocal_character" in source
    assert "not by stretching vowels" in source
    assert "different vocal regions" in source
