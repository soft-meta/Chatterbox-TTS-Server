from pathlib import Path
import importlib.util
import sys
import types

ROOT = Path(__file__).resolve().parents[1]

REMOVED_TEXT = [
    'Chatterbox Turbo generation with voice cloning, a senior-friendly Motivational Speech preset, creator controls and up to five queued audio jobs.',
    'Paste your script exactly as you want it spoken. No Auto Emotion, rewriting, QC retries, pacing or professional speech processing is applied.',
    'Use only voices you own or have permission to clone. Clean speech with no music or echo produces the best result.',
    'Motivational Speech is tuned for calm senior-advisor, tutorial, health and life-advice narration. Turbo follows the cloned reference voice for identity, maturity and accent, so use a clean American-English reference when you want an American accent.',
]


def test_v162_removes_requested_explanatory_text_and_enables_controls():
    html = (ROOT / 'ui' / 'index.html').read_text(encoding='utf-8')
    for text in REMOVED_TEXT:
        assert text not in html
    assert 'data-field="exaggeration"' in html
    assert 'data-field="cfg_weight"' in html
    assert 'data-field="exaggeration" type="range"' in html
    assert 'data-field="cfg_weight" type="range"' in html
    # Neither slider may be disabled or labelled Original-only.
    ex_line = next(line for line in html.splitlines() if 'data-field="exaggeration"' in line)
    cfg_line = next(line for line in html.splitlines() if 'data-field="cfg_weight"' in line)
    assert 'disabled' not in ex_line
    assert 'disabled' not in cfg_line
    assert 'Original-only' not in html


def test_v162_ui_sends_creator_exaggeration_and_cfg_values():
    js = (ROOT / 'ui' / 'app.js').read_text(encoding='utf-8')
    assert 'exaggeration: Number(tab.exaggeration)' in js
    assert 'cfg_weight: Number(tab.cfg_weight)' in js


def test_v162_backend_preserves_creator_controls():
    from models import AudioJobCreate
    from queue_manager import QueueManager
    req = AudioJobCreate(
        preset='Motivational Speech', text='Hello world.', voice_mode='default',
        options={
            'model': 'chatterbox-turbo', 'temperature': 0.72, 'top_p': 0.90,
            'top_k': 1000, 'repetition_penalty': 1.2,
            'exaggeration': 1.25, 'cfg_weight': 0.45,
            'speed_factor': 0.93,
        },
    )
    options = QueueManager._effective_options(req)
    assert options['exaggeration'] == 1.25
    assert options['cfg_weight'] == 0.45


def _load_real_engine_module():
    # Isolate engine.py from the repository test conftest's fake `engine` module.
    fake_pkg = types.ModuleType('softmeta_chatterbox')
    class GenerationSettings:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    class SoftMetaChatterboxEngine:
        def __init__(self, device='auto'):
            self.device = 'cpu'
            self.model_name = 'chatterbox-turbo'
        def status_dict(self): return {}
        def load(self, name): self.model_name = name
        def unload(self): pass
        def generate(self, *args, **kwargs):
            self.last_generate = kwargs
            import torch
            return torch.zeros(1, 100), 24000
    fake_pkg.GenerationSettings = GenerationSettings
    fake_pkg.SoftMetaChatterboxEngine = SoftMetaChatterboxEngine
    old = sys.modules.get('softmeta_chatterbox')
    sys.modules['softmeta_chatterbox'] = fake_pkg
    try:
        spec = importlib.util.spec_from_file_location('engine_v162_real', ROOT / 'engine.py')
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if old is None:
            sys.modules.pop('softmeta_chatterbox', None)
        else:
            sys.modules['softmeta_chatterbox'] = old


def test_v162_exaggeration_and_cfg_change_effective_turbo_sampling():
    module = _load_real_engine_module()
    low = module._turbo_creator_sampling(0.72, 0.90, exaggeration=0.0, cfg_weight=0.0)
    neutral = module._turbo_creator_sampling(0.72, 0.90, exaggeration=0.5, cfg_weight=0.0)
    high = module._turbo_creator_sampling(0.72, 0.90, exaggeration=1.5, cfg_weight=0.0)
    guided = module._turbo_creator_sampling(0.72, 0.90, exaggeration=1.5, cfg_weight=1.0)
    assert neutral == (0.72, 0.90)
    assert high[0] > neutral[0] > low[0]
    assert high[1] > neutral[1] > low[1]
    assert guided[0] < high[0]
    assert guided[1] < high[1]


def test_v162_engine_service_applies_creator_bridge_to_generation_settings():
    module = _load_real_engine_module()
    service = module.EngineService(device='cpu')
    options = {
        'temperature': 0.72, 'top_p': 0.90, 'top_k': 1000,
        'repetition_penalty': 1.2, 'min_p': 0.0, 'seed': 123,
        'exaggeration': 1.5, 'cfg_weight': 0.0,
    }
    service.generate(
        'Hello world.', model_name='chatterbox-turbo', reference_audio=None,
        language='en', options=options,
    )
    settings = service.runtime.last_generate['settings']
    assert settings.temperature > 0.72
    assert settings.top_p > 0.90
    assert settings.exaggeration == 1.5
    assert settings.cfg_weight == 0.0
