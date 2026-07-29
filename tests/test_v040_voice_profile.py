from models import VoiceDesignRequest


def test_voice_request_defaults_to_us_english_male_age_50() -> None:
    request = VoiceDesignRequest(
        name="Warm Voice",
        sample_text="This is a clear test sentence.",
    )
    assert request.age == 50
    assert request.gender == "male"
    assert request.language == "en-US"
    assert request.emotion == "warm"
    assert request.candidate_count == 3
    assert request.uniqueness_threshold == 0.68
