from utils import count_words, estimated_audio_seconds, safe_filename, split_text


def test_split_text_keeps_all_words() -> None:
    text = "One two three. Four five six. Seven eight nine."
    chunks = split_text(text, max_words=4)
    assert sum(count_words(chunk) for chunk in chunks) == 9


def test_safe_filename() -> None:
    assert safe_filename("My Video: Part 1") == "My_Video_Part_1"


def test_audio_estimate() -> None:
    assert estimated_audio_seconds(145) == 60.0


def test_clause_aware_fallback_is_opt_in() -> None:
    text = "One two three four, five six seven eight, nine ten eleven twelve."
    default_chunks = split_text(text, max_words=5)
    stable_chunks = split_text(text, max_words=5, prefer_clauses=True)
    assert sum(count_words(chunk) for chunk in default_chunks) == 12
    assert sum(count_words(chunk) for chunk in stable_chunks) == 12
    assert default_chunks != stable_chunks
