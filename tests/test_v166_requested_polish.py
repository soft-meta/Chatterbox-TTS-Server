from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_ui_only_shows_four_requested_creator_controls_and_formats():
    html = (ROOT / 'ui' / 'index.html').read_text(encoding='utf-8')
    assert '<b>Temperature</b>' in html
    assert '<b>Exaggeration</b>' in html
    assert '<b>Speed Factor</b>' in html
    assert '<b>CFG Weight</b>' in html
    assert 'Top P' not in html
    assert 'Top K' not in html
    assert 'Repetition Penalty' not in html
    assert 'Generation Seed' not in html
    assert '<option value="wav">WAV</option>' in html
    assert '<option value="mp3">MP3</option>' in html
    assert '<option value="m4a">M4A</option>' in html
    assert '<option value="flac">FLAC</option>' in html


def test_option_payload_does_not_send_hidden_sampling_controls_from_ui():
    js = (ROOT / 'ui' / 'app.js').read_text(encoding='utf-8')
    start = js.index('function optionsFor(tab)')
    end = js.index('\n  function jobPayload', start)
    block = js[start:end]
    assert 'temperature:' in block
    assert 'exaggeration:' in block
    assert 'cfg_weight:' in block
    assert 'speed_factor:' in block
    assert 'top_p:' not in block
    assert 'top_k:' not in block
    assert 'repetition_penalty:' not in block
    assert 'seed:' not in block
    assert 'output_format:' in block


def test_sentence_pause_targets_sentence_gap_not_short_breath():
    from queue_manager import apply_sentence_end_pause
    sr = 24000
    tone = (0.03 * np.sin(2 * np.pi * 180 * np.arange(int(sr * .4)) / sr)).astype(np.float32)
    normal_pause = np.zeros(int(sr * .22), dtype=np.float32)
    long_pause = np.zeros(int(sr * .95), dtype=np.float32)
    audio = np.concatenate([tone, normal_pause, tone, long_pause, tone])
    out = apply_sentence_end_pause(audio, sr, 'First clause with breath. Second sentence.', 280)
    # The strongest punctuation-like gap is normalized to 280ms while the shorter breath stays.
    expected_removed = int(sr * (.95 - .28))
    removed = len(audio) - len(out)
    assert abs(removed - expected_removed) < int(sr * .08)


def test_motivational_join_pause_is_shorter_than_previous_140ms():
    from queue_manager import MOTIVATIONAL_TURBO_PROFILE
    assert MOTIVATIONAL_TURBO_PROFILE['inter_chunk_pause_ms'] <= 90


def test_queue_monitor_has_drag_reorder_support():
    js = (ROOT / 'ui' / 'app.js').read_text(encoding='utf-8')
    css = (ROOT / 'ui' / 'styles.css').read_text(encoding='utf-8')
    assert 'draggable = true' in js or "setAttribute('draggable', 'true')" in js
    assert 'dragstart' in js and 'dragover' in js and 'drop' in js
    assert 'queueOrder' in js
    assert '.queue-card.dragging' in css
    assert '.queue-card.drag-over' in css


def test_output_formats_are_accepted_by_models():
    from models import GenerationOptions, CutRequest
    for fmt in ('wav', 'mp3', 'm4a', 'flac'):
        assert GenerationOptions(output_format=fmt).output_format == fmt
        assert CutRequest(start_seconds=0, end_seconds=1, output_format=fmt).output_format == fmt


def test_server_supports_format_aware_full_and_cut_downloads():
    server = (ROOT / 'server.py').read_text(encoding='utf-8')
    assert 'def _convert_audio_format' in server
    assert 'format:' in server or 'output_format' in server
    assert 'libmp3lame' in server
    assert 'aac' in server
    assert 'flac' in server


def test_version_is_166():
    server = (ROOT / 'server.py').read_text(encoding='utf-8')
    assert 'APP_VERSION = "1.6.8"' in server
