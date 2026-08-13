from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_generation_options_exposes_sentence_end_pause_control():
    from models import GenerationOptions
    opts = GenerationOptions(sentence_end_pause_ms=360)
    assert opts.sentence_end_pause_ms == 360


def test_ui_has_sentence_end_pause_slider_and_payload():
    html = (ROOT / 'ui' / 'index.html').read_text(encoding='utf-8')
    js = (ROOT / 'ui' / 'app.js').read_text(encoding='utf-8')
    assert '<b>Sentence End Pause</b>' in html
    assert 'data-field="sentence_end_pause_ms"' in html
    assert 'sentence_end_pause_ms:' in js


def test_pause_control_uses_selected_target_without_touching_short_breath():
    from queue_manager import apply_sentence_end_pause
    sr = 24000
    tone = (0.03 * np.sin(2 * np.pi * 180 * np.arange(int(sr * .35)) / sr)).astype(np.float32)
    short_pause = np.zeros(int(sr * .18), dtype=np.float32)
    long_pause = np.zeros(int(sr * .90), dtype=np.float32)
    audio = np.concatenate([tone, short_pause, tone, long_pause, tone])
    out = apply_sentence_end_pause(audio, sr, 'Sentence one with breath. Sentence two.', 360)
    expected_removed = int(sr * (.90 - .36))
    removed = len(audio) - len(out)
    assert abs(removed - expected_removed) < int(sr * .08)


def test_queue_manager_reads_sentence_end_pause_option():
    source = (ROOT / 'queue_manager.py').read_text(encoding='utf-8')
    assert 'sentence_end_pause_ms' in source
    assert 'apply_sentence_end_pause(' in source


def test_version_is_167():
    source = (ROOT / 'server.py').read_text(encoding='utf-8')
    assert 'APP_VERSION = "1.6.8"' in source


def test_sentence_end_pause_reaches_effective_turbo_options():
    from models import AudioJobCreate, GenerationOptions
    from queue_manager import QueueManager
    request = AudioJobCreate(
        text='One sentence. Another sentence.',
        voice_mode='clone',
        voice_filename='voice.wav',
        options=GenerationOptions(sentence_end_pause_ms=420),
    )
    options = QueueManager._effective_options(request)
    assert options['sentence_end_pause_ms'] == 420


def test_pause_control_can_lengthen_sentence_gap_when_creator_requests_it():
    from queue_manager import apply_sentence_end_pause
    sr = 24000
    tone = (0.03 * np.sin(2 * np.pi * 180 * np.arange(int(sr * .3)) / sr)).astype(np.float32)
    pause = np.zeros(int(sr * .22), dtype=np.float32)
    audio = np.concatenate([tone, pause, tone])
    out = apply_sentence_end_pause(audio, sr, 'Sentence one. Sentence two.', 600)
    assert len(out) > len(audio) + int(sr * .30)
