from queue_manager import split_turbo_long_text, TURBO_MAX_CHARS, _normalise_space


def test_long_script_is_split_at_model_safe_size_without_word_loss():
    paragraph = (
        'Older adults often tell me they want speech that is clear, simple, and easy to follow. '
        'That means every sentence should arrive in order, without hidden rewriting or skipped ideas. '
        'A clean voice matters more than extra effects, and reliable narration matters more than decoration. '
    )
    text = '\n\n'.join(paragraph * 10 for _ in range(7))
    chunks = split_turbo_long_text(text)
    assert len(chunks) > 10
    assert all(1 <= len(c) <= TURBO_MAX_CHARS for c in chunks)
    assert _normalise_space(' '.join(chunks)) == _normalise_space(text)


def test_single_oversize_sentence_still_preserves_all_lexical_content():
    text = 'This is one intentionally long sentence, ' + ', '.join(f'important phrase number {i}' for i in range(1,70)) + '.'
    chunks = split_turbo_long_text(text)
    assert all(len(c) <= TURBO_MAX_CHARS for c in chunks)
    assert _normalise_space(' '.join(chunks)) == _normalise_space(text)
