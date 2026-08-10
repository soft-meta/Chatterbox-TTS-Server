from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from emotion_director import analyze_serious_senior_advisor, analyze_turbo_avatar_performance, is_heading, strip_turbo_tags
from models import AudioJobCreate, GenerationOptions
from queue_manager import QueueManager
from utils import count_words, prepare_american_english_tts_text, split_text

ROOT = Path(__file__).resolve().parents[1]


SCREENSHOT_STYLE_SCRIPT = """Why Hydration Can Feel Different After 70

Many older adults notice dry mouth, low energy, or mild dizziness during the day. Sometimes they assume those feelings are simply part of getting older, especially when the symptoms come and go.

And sometimes people blame those feelings on age. They say, I guess I am just getting old. But that is not always the whole story. Your body can become less reliable at telling you that it needs water, so thirst may not arrive as early as it once did.

Try keeping water close to the places where you spend the most time. Keep a glass near your chair. Keep a bottle in the kitchen. Take small drinks during the day instead of waiting until you feel very thirsty.

If your doctor has told you to limit fluids because of a health condition, follow that advice. The important thing is to use advice that fits your own health needs instead of copying someone else's routine.

Now pay attention, because the next mistake can happen even when you believe your diet is healthy. Many foods that look harmless can contain more sodium than people expect, and that can make hydration harder to manage.

The good news is that small changes can help. Reading labels, spacing your drinks through the day, and asking your doctor about your own fluid needs can make the routine easier to follow without turning it into a stressful daily task.
"""


def test_screenshot_style_advice_script_does_not_invent_audible_events() -> None:
    result = analyze_turbo_avatar_performance(SCREENSHOT_STYLE_SCRIPT)
    assert 220 <= result.total_words <= 330
    assert result.protected_headings == 1
    # v1.5.6 only auto-inserts audible Turbo events when the wording genuinely
    # supports a laugh/chuckle/sigh/gasp. Ordinary hydration advice must not be
    # turned into a fake sound-effects performance just to satisfy a quota.
    assert result.mode == "Turbo Human Performance Events"
    assert result.applied_count == 0
    assert result.tagged_text.count("[") == 0
    heading_lines = [line for line in result.tagged_text.splitlines() if is_heading(line)]
    assert heading_lines == ["Why Hydration Can Feel Different After 70"]


def test_visible_tagging_is_idempotent_and_does_not_duplicate_tags() -> None:
    first = analyze_serious_senior_advisor(SCREENSHOT_STYLE_SCRIPT)
    second = analyze_serious_senior_advisor(first.tagged_text)
    assert second.tagged_text == first.tagged_text
    assert second.applied_count == first.applied_count
    assert second.protected_headings == first.protected_headings


def test_visible_tags_do_not_inflate_word_count_or_get_lost_before_chunking() -> None:
    result = analyze_serious_senior_advisor(SCREENSHOT_STYLE_SCRIPT)
    assert count_words(result.tagged_text) == count_words(strip_turbo_tags(result.tagged_text))
    prepared = prepare_american_english_tts_text(result.tagged_text)
    chunks = split_text(prepared, 85, prefer_clauses=True)
    joined = " ".join(chunks)
    for tag, count in result.by_tag.items():
        assert joined.lower().count(f"[{tag}]") == count


def test_frontend_keeps_creator_script_raw_and_previews_advanced_events_separately() -> None:
    source = (ROOT / "ui" / "app.js").read_text(encoding="utf-8")
    server = (ROOT / "server.py").read_text(encoding="utf-8")
    assert "applyVisibleEmotionResult" in source
    assert "tab.advanced_tagged_text = result.tagged_text" in source
    assert "preview.textContent = result.tagged_text" in source
    html = (ROOT / "ui" / "index.html").read_text(encoding="utf-8")
    assert "Generate Audio" in html
    assert "Auto Emotion" in html
    assert "Generate Advanced Audio" not in html
    assert "analysis.public_summary(include_text=True)" in server


def test_backend_recomputes_same_canonical_tagged_script_used_by_ui() -> None:
    result = analyze_turbo_avatar_performance(SCREENSHOT_STYLE_SCRIPT)

    class DummyStorage:
        def __init__(self) -> None:
            self.saved = {}
        def load_jobs(self):
            return {}
        def save_jobs(self, jobs):
            self.saved = dict(jobs)

    manager = QueueManager(engine=SimpleNamespace(), storage=DummyStorage())
    request = AudioJobCreate(
        preset="Motivational Speech",
        audio_number=1,
        title="Visible Emotion Test",
        text=result.tagged_text,
        voice_mode="default",
        auto_emotion=True,
        options=GenerationOptions(model="chatterbox-turbo"),
    )
    job = asyncio.run(manager.create(request, enqueue=False))
    raw = manager.get_raw(job["id"])
    from pronunciation_engine import prepare_pronunciation_text
    from speech_pipeline import prepare_senior_clear_speech_text
    expected = prepare_senior_clear_speech_text(prepare_pronunciation_text(result.tagged_text))
    assert raw["generation_text"] == expected
    assert raw["emotion_summary"]["applied_count"] == result.applied_count
    assert raw["emotion_summary"]["protected_headings"] == result.protected_headings
    manager.executor.shutdown(wait=False, cancel_futures=True)
