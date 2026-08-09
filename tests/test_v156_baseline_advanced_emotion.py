import pytest

from emotion_director import analyze_turbo_avatar_performance
from models import AudioJobCreate, GenerationOptions


def test_current_turbo_planner_should_use_audible_event_tags():
    text = (
        "My grandson looked at me and said I finally seemed happy again. "
        "I laughed because he was completely right. That little moment still makes me smile. "
        "But I also think about the years I lost trying to please everyone. I wish I had known sooner. "
        "Then one morning I discovered something I never expected, and it stopped me in my tracks. "
    ) * 8
    result = analyze_turbo_avatar_performance(text)
    tagged = result.tagged_text.lower()
    assert any(tag in tagged for tag in ("[laugh]", "[chuckle]")), tagged
    assert "[sigh]" in tagged, tagged
    assert "[gasp]" in tagged, tagged


def test_request_should_expose_standard_vs_advanced_generation_mode():
    request = AudioJobCreate(
        preset="Motivational Speech",
        audio_number=1,
        title="compare",
        text="This is a comparison script.",
        voice_mode="clone",
        voice_filename="voice.wav",
        generation_mode="standard",
        options=GenerationOptions(model="chatterbox-turbo"),
    )
    assert request.generation_mode == "standard"
