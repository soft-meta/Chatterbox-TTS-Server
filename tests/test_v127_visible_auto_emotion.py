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


def test_screenshot_style_295_word_script_gets_multiple_visible_tags() -> None:
    result = analyze_turbo_avatar_performance(SCREENSHOT_STYLE_SCRIPT)
    assert 220 <= result.total_words <= 330
    assert result.protected_headings == 1
    assert 2 <= result.applied_count <= 4
    assert result.tagged_text.count("[") == result.applied_count
    assert any(tag in result.tagged_text for tag in ("[narration]", "[surprised]"))
    heading_lines = [line for line in result.tagged_text.splitlines() if is_heading(line)]
    assert heading_lines == ["Why Hydration Can Feel Different After 70"]
    assert all("[" not in line for line in heading_lines)


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


def test_frontend_inserts_backend_tagged_text_and_reports_visible_counts() -> None:
    source = (ROOT / "ui" / "app.js").read_text(encoding="utf-8")
    server = (ROOT / "server.py").read_text(encoding="utf-8")
    assert "applyVisibleEmotionResult" in source
    assert "textarea.value = nextText" in source
    assert "tab.text = nextText" in source
    assert "tags inserted into script" in source or "tag${summary.applied_count" in source
    assert "stripVisibleEmotionTags" in source
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
        options=GenerationOptions(model="chatterbox-turbo"),
    )
    job = asyncio.run(manager.create(request, enqueue=False))
    raw = manager.get_raw(job["id"])
    expected = prepare_american_english_tts_text(result.tagged_text)
    assert raw["generation_text"] == expected
    assert raw["emotion_summary"]["applied_count"] == result.applied_count
    assert raw["emotion_summary"]["protected_headings"] == result.protected_headings
    manager.executor.shutdown(wait=False, cancel_futures=True)
