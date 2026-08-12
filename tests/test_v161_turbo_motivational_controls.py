from pathlib import Path
import asyncio
import types
import yaml
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_v161_exposes_motivational_preset_and_turbo_controls():
    server = (ROOT/'server.py').read_text(encoding='utf-8')
    html = (ROOT/'ui/index.html').read_text(encoding='utf-8')
    js = (ROOT/'ui/app.js').read_text(encoding='utf-8')
    assert 'APP_VERSION = "1.6.2"' in server
    assert '"name": "Motivational Speech"' in server
    assert '<section class="preset-section">' in html
    assert '<details class="parameters-panel"' in html
    assert 'data-field="temperature"' in html
    assert 'data-field="top_p"' in html
    assert 'data-field="top_k"' in html
    assert 'data-field="repetition_penalty"' in html
    assert 'data-field="speed_factor"' in html
    assert 'Original-only' not in html
    assert 'optionsFor(tab)' in js


def test_motivational_profile_matches_v121_turbo_delivery():
    import server
    p = next(item for item in server.PRESETS if item['name']=='Motivational Speech')
    assert p['temperature'] == 0.72
    assert p['top_p'] == 0.90
    assert p['top_k'] == 1000
    assert p['repetition_penalty'] == 1.2
    assert p['speed_factor'] == 0.93
    assert p['inter_chunk_pause_ms'] == 140
    assert p['exaggeration'] == 0.5
    assert p['cfg_weight'] == 0.0


def test_config_defaults_to_motivational_speech_but_keeps_qc_off():
    cfg = yaml.safe_load((ROOT/'config.yaml').read_text(encoding='utf-8'))
    d = cfg['generation_defaults']
    assert d['preset'] == 'Motivational Speech'
    assert d['temperature'] == 0.72
    assert d['top_p'] == 0.90
    assert d['top_k'] == 1000
    assert d['repetition_penalty'] == 1.2
    assert d['speed_factor'] == 0.93
    assert d['quality_gate'] is False
    assert d['speaker_consistency'] is False
    assert d['platform_assets'] is False


def test_backend_honors_user_turbo_controls_instead_of_forcing_defaults():
    from models import AudioJobCreate
    from queue_manager import QueueManager
    req = AudioJobCreate(
        preset='Motivational Speech', text='Hello world.', voice_mode='default',
        options={
            'model':'chatterbox-turbo', 'temperature':0.61, 'top_p':0.82,
            'top_k':777, 'repetition_penalty':1.31, 'min_p':0.0,
            'speed_factor':0.97, 'exaggeration':0.8, 'cfg_weight':0.7,
            'quality_gate':True, 'speaker_consistency':True, 'platform_assets':True,
            'inter_chunk_pause_ms':120,
        },
    )
    o = QueueManager._effective_options(req)
    assert o['model'] == 'chatterbox-turbo'
    assert o['temperature'] == 0.61
    assert o['top_p'] == 0.82
    assert o['top_k'] == 777
    assert o['repetition_penalty'] == 1.31
    assert o['speed_factor'] == 0.97
    assert o['exaggeration'] == 0.8
    assert o['cfg_weight'] == 0.7
    assert o['min_p'] == 0.0
    # Advanced systems stay disabled in this fresh branch.
    assert o['quality_gate'] is False
    assert o['speaker_consistency'] is False
    assert o['platform_assets'] is False


class FakeStorage:
    def __init__(self, root: Path):
        self.outputs=root/'outputs'; self.outputs.mkdir(parents=True)
    def load_jobs(self): return {}
    def save_jobs(self, jobs): pass
    def output_path(self, filename): return self.outputs/filename
    def voice_path(self, kind, filename): return None


class FakeEngine:
    def __init__(self): self.calls=[]
    def generate(self, text, *, model_name, reference_audio, language, options):
        self.calls.append(dict(options))
        n=max(1200, len(text.split())*500)
        t=torch.linspace(0,1,n)
        return types.SimpleNamespace(waveform=(0.02*torch.sin(2*torch.pi*180*t)).unsqueeze(0), sample_rate=24000)


def test_speed_factor_applies_one_pitch_preserving_final_tempo_pass(tmp_path, monkeypatch):
    from models import AudioJobCreate
    from queue_manager import QueueManager
    engine=FakeEngine(); manager=QueueManager(engine, FakeStorage(tmp_path))
    req=AudioJobCreate(
        preset='Motivational Speech', text='This is a short senior advice narration sentence.',
        voice_mode='default', options={'speed_factor':0.93, 'temperature':0.72, 'top_p':0.90, 'top_k':1000}
    )
    pub=asyncio.run(manager.create(req, enqueue=False)); job=manager.get_raw(pub['id'])
    tempo=[]; loud=[]
    monkeypatch.setattr(manager, '_apply_final_tempo', lambda path, factor: tempo.append((path, factor)))
    monkeypatch.setattr(manager, '_normalise_loudness', lambda path: loud.append(path) or {'profile':'loudness-only'})
    asyncio.run(manager._process(job))
    assert tempo and tempo[0][1] == 0.93
    assert len(tempo) == 1
    assert len(loud) == 1
    assert engine.calls and engine.calls[0]['temperature'] == 0.72
