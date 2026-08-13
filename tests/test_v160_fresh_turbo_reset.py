from pathlib import Path
import re
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_server_and_ui_are_fresh_turbo_only():
    server = (ROOT/'server.py').read_text()
    html = (ROOT/'ui/index.html').read_text()
    js = (ROOT/'ui/app.js').read_text()
    assert 'APP_VERSION = "1.6.8"' in server
    assert '"id": "chatterbox-turbo"' in server
    # Only Turbo is advertised as a selectable model.
    model_block = server.split('MODELS =',1)[1].split('PRESETS =',1)[0]
    assert '"id": "chatterbox"' not in model_block
    assert 'Chatterbox Turbo generation with voice cloning' not in html
    assert 'No Auto Emotion' not in html
    assert 'Generate Advanced Audio' not in html
    assert "auto_emotion: false" in js
    assert "generation_mode: 'standard'" in js
    assert "quality_gate: false" in js
    assert '/api/emotion/analyze' not in js


def test_config_matches_official_turbo_sampling_defaults_and_no_qc():
    cfg = yaml.safe_load((ROOT/'config.yaml').read_text())
    d = cfg['generation_defaults']
    assert d['preset'] == 'Motivational Speech'
    assert d['temperature'] == 0.72
    assert d['top_p'] == 0.95
    assert d['top_k'] == 1000
    assert d['repetition_penalty'] == 1.2
    assert d['min_p'] == 0.0
    assert d['speed_factor'] == 0.93
    assert d['quality_gate'] is False
    assert d['speaker_consistency'] is False
    assert d['platform_assets'] is False
    req = (ROOT/'requirements-colab.txt').read_text().lower()
    assert 'faster-whisper' not in req
    assert 'speechbrain' not in req


def test_generate_audio_backend_forces_fresh_turbo_profile():
    from models import AudioJobCreate
    from queue_manager import QueueManager
    req = AudioJobCreate(
        text='Hello world.', voice_mode='default',
        options={
            'model':'chatterbox-turbo', 'temperature':1.8, 'top_p':0.2,
            'top_k':5, 'repetition_penalty':2.3, 'min_p':0.8,
            'speed_factor':1.7, 'quality_gate':True, 'speaker_consistency':True,
            'platform_assets':True,
        },
    )
    o = QueueManager._effective_options(req)
    assert o['model'] == 'chatterbox-turbo'
    # Only the four requested creator controls are accepted from UI/API overrides.
    assert o['temperature'] == 1.8
    assert o['top_p'] == 0.95
    assert o['top_k'] == 1000
    assert o['repetition_penalty'] == 1.2
    assert o['min_p'] == 0.0
    assert o['speed_factor'] == 1.7
    assert o['exaggeration'] == 0.5
    assert o['cfg_weight'] == 0.0
    assert o['quality_gate'] is False
    assert o['speaker_consistency'] is False
    assert o['platform_assets'] is False
