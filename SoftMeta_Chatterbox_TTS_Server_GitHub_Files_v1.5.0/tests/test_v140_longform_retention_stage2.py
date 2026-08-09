import numpy as np

from emotion_director import analyze_serious_senior_advisor, is_heading
from professional_audio import adaptive_tempo_factor, apply_tempo_array
from speech_pipeline import build_long_form_segments


def test_section_planner_protects_heading_and_starts_section_cleanly():
    text = (
        "A calm opening sentence explains what the viewer will learn today. "
        "The next sentence gives one more useful detail before the list begins.\n\n"
        "Number One: Drink Enough Water\n"
        "Many older adults do not notice thirst as quickly, so a simple routine can help. "
        "Keep water nearby and take small drinks during the day unless your doctor gave different advice.\n\n"
        "Number Two: Pay Attention to Fatigue\n"
        "New or persistent fatigue deserves attention because it can have many causes."
    )
    segments = build_long_form_segments(text, max_words=85, heading_detector=is_heading)
    assert segments
    assert segments[0].role == "intro"
    section_starts = [s for s in segments if s.role == "section"]
    assert len(section_starts) == 2
    assert section_starts[0].text.startswith("Number One. Drink Enough Water.")
    assert section_starts[1].text.startswith("Number Two. Pay Attention to Fatigue.")
    assert all(s.pause_before_ms >= 480 for s in section_starts)


def test_intro_profile_is_slightly_more_alert_than_body_and_important_is_slower():
    text = (
        "This opening explains why the topic matters and gives the viewer a reason to keep listening. " * 4
        + "\n\nNumber One: The Important Warning\n"
        + "Remember this because the biggest mistake is ignoring a new warning sign. " * 4
        + "\n\nNumber Two: A Daily Habit\n"
        + "A steady daily routine can make healthy choices easier to remember. " * 8
    )
    segments = build_long_form_segments(text, max_words=75, heading_detector=is_heading)
    intro = segments[0]
    important = next(s for s in segments if s.importance > 0)
    body = next(s for s in segments if s.role == "body" and s.importance == 0)
    assert intro.target_wpm > body.target_wpm
    assert important.target_wpm < body.target_wpm
    assert 140 <= important.target_wpm <= 146
    assert 147 <= body.target_wpm <= 152
    assert 154 <= intro.target_wpm <= 160


def test_adaptive_tempo_slows_fast_generation_but_does_not_over_speed_slow_clone():
    assert 0.84 <= adaptive_tempo_factor(current_wpm=180, target_wpm=150) <= 0.85
    assert adaptive_tempo_factor(current_wpm=148, target_wpm=150) > 1.0
    assert adaptive_tempo_factor(current_wpm=125, target_wpm=150) <= 1.04


def test_apply_tempo_array_changes_duration_pitch_preserving_path():
    sr = 24000
    t = np.arange(sr) / sr
    audio = (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    slowed = apply_tempo_array(audio, sr, 0.90)
    assert len(slowed) > len(audio) * 1.08
    assert len(slowed) < len(audio) * 1.14


def test_long_script_gets_sparse_retention_resets_without_comedy():
    paragraph = (
        "Many older adults build routines slowly, and small changes are easier to keep when the reason is clear. "
        "Take your time, notice what works, and keep the advice practical so it fits normal daily life. "
        "A simple note can help you remember what changed and what you want to discuss with your doctor. "
    )
    text = paragraph * 14  # comfortably above 500 words
    result = analyze_serious_senior_advisor(text)
    reset_items = [p for p in result.placements if p.get("source") == "retention-reset"]
    assert reset_items, result.placements
    assert result.applied_count <= 8
    assert "[laugh]" not in result.tagged_text.lower()
    assert "[chuckle]" not in result.tagged_text.lower()
    assert "[angry]" not in result.tagged_text.lower()
    # Resets remain sparse; they are not every sentence.
    positions = [p["word_position"] for p in result.placements]
    assert all(b - a >= 50 for a, b in zip(positions, positions[1:]))
