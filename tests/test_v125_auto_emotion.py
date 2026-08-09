from __future__ import annotations

import json
import re
from pathlib import Path

from emotion_director import (
    AUTO_FORBIDDEN_TAGS,
    analyze_serious_senior_advisor,
    is_heading,
)

ROOT = Path(__file__).resolve().parents[1]


def _long_serious_script() -> str:
    sections = []
    emotional_lines = [
        "Looking back, I wish I had understood this sooner because it was one of my biggest mistakes.",
        "The good news is that a small daily change can make this much easier to manage.",
        "Most people do not realize how quickly this simple habit can affect the way they feel.",
        "By the time I realized what was happening, I had already lost months of good days.",
        "Fortunately, I changed the routine and things began to improve in a steady way.",
        "I could not believe how different I felt after I stayed consistent for several weeks.",
        "I regret waiting so long before I finally took this warning seriously.",
        "The encouraging part is that you can change this without trying to become a different person.",
        "You may be surprised by how much a small adjustment can help over time.",
        "If you think you may be having a heart attack, call 911 or your local emergency number.",
    ]
    filler = (
        "Take your time with this advice and notice how your own body responds. "
        "A calm routine is often easier to follow than a sudden major change. "
        "The goal is to understand the idea, remember it, and use it safely in daily life."
    )
    for index in range(1, 11):
        sections.append(f"Number {index}: Important Senior Habit")
        sections.append(filler)
        sections.append(emotional_lines[index - 1])
        sections.append(filler)
    return "\n\n".join(sections)


def test_headings_are_detected_and_never_tagged() -> None:
    assert is_heading("Number One: Protect Your Sleep")
    assert is_heading("2. Drink Water Regularly")
    assert is_heading("Why Small Habits Matter After 70")
    assert is_heading("IMPORTANT WARNING")

    text = """Why Small Habits Matter After 70

This introduction is calm and gives the listener time to understand the subject before any advice begins. It also explains why the next ideas matter.

Number One: Protect Your Sleep

This section explains the routine in a clear way. Looking back, I wish I had learned this sooner because I lost many good days.

Number Two: Keep Moving

The good news is that a gentle daily walk can make the routine easier to maintain. Keep the pace comfortable and steady.
"""
    result = analyze_serious_senior_advisor(text)
    heading_lines = [line for line in result.tagged_text.splitlines() if is_heading(line)]
    assert result.protected_headings >= 3
    assert heading_lines
    assert all("[" not in line and "]" not in line for line in heading_lines)


def test_auto_emotion_is_sparse_and_non_comedic() -> None:
    text = _long_serious_script()
    result = analyze_serious_senior_advisor(text)
    # About one cue per 125 words, capped at eight, with additional spacing rules.
    expected_cap = min(8, max(1, __import__("math").ceil(result.total_words / 125)))
    assert 1 <= result.applied_count <= expected_cap
    for forbidden in AUTO_FORBIDDEN_TAGS:
        assert f"[{forbidden}]" not in result.tagged_text.lower()
    assert result.protected_headings == 10


def test_pasted_manual_funny_or_angry_tags_are_sanitized() -> None:
    text = (
        "A Calm Senior Advice Video\n\n"
        "[laugh] This is serious advice and it should never become a comedy performance. "
        "[angry] I regret ignoring this lesson for many years, and I wish I had acted sooner. "
        "The good news is that a calm routine can still help you move forward."
    )
    result = analyze_serious_senior_advisor(text)
    assert result.manual_tags == 2
    assert "[laugh]" not in result.tagged_text.lower()
    assert "[angry]" not in result.tagged_text.lower()


def test_backend_and_ui_use_hidden_auto_emotion_pipeline() -> None:
    queue_source = (ROOT / "queue_manager.py").read_text(encoding="utf-8")
    server_source = (ROOT / "server.py").read_text(encoding="utf-8")
    app_source = (ROOT / "ui" / "app.js").read_text(encoding="utf-8")
    html = (ROOT / "ui" / "index.html").read_text(encoding="utf-8")
    assert 'pronounced = prepare_pronunciation_text(emotion_analysis.tagged_text)' in queue_source
    assert 'generation_text = prepare_senior_clear_speech_text(pronounced)' in queue_source
    assert 'data.pop("generation_text", None)' in queue_source
    assert 'intentional_emotion=contains_turbo_tag(chunk)' in queue_source
    assert '@app.post("/api/emotion/analyze")' in server_source
    assert "scheduleEmotionAnalysis" in app_source
    assert "/api/emotion/analyze" in app_source
    assert 'data-role="emotion-status"' in html


def test_colab_requests_l4_and_final_version_is_current() -> None:
    notebook_path = ROOT / "colab" / "SoftMeta_Chatterbox_TTS_Colab_v1.5.3.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["gpuType"] == "L4"
    server_source = (ROOT / "server.py").read_text(encoding="utf-8")
    assert 'APP_VERSION = "1.5.3"' in server_source
