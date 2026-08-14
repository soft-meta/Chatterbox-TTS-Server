from queue_manager import split_turbo_long_text, _normalise_space


def test_sentence_splitter_preserves_closing_quotes_and_brackets():
    cases = [
        'He said, "Stay strong." Then he smiled.',
        'This matters.) Next sentence starts here.',
        'First sentence!” Second sentence follows.',
        "It is okay.' Next one.",
        'She asked, “Are you ready?” Then we began.',
    ]
    for text in cases:
        chunks = split_turbo_long_text(text)
        assert _normalise_space(' '.join(chunks)) == _normalise_space(text)


def test_long_quoted_script_preserves_every_character_after_normalized_spacing():
    sentence = 'After 70, he told me, "Protect your peace." Then he added, “Keep your routine simple.” '
    text = (sentence * 40).strip()
    chunks = split_turbo_long_text(text)
    assert len(chunks) > 1
    assert all(len(chunk) <= 300 for chunk in chunks)
    assert _normalise_space(' '.join(chunks)) == _normalise_space(text)
