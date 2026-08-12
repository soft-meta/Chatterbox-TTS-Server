import asyncio
from pathlib import Path
import types

import numpy as np
import torch

from models import AudioJobCreate
from queue_manager import QueueManager, split_turbo_long_text


class FakeStorage:
    def __init__(self, root: Path):
        self.outputs = root / 'outputs'; self.outputs.mkdir(parents=True)
        self.jobs = {}
    def load_jobs(self): return {}
    def save_jobs(self, jobs): self.jobs = jobs
    def output_path(self, filename): return self.outputs / filename
    def voice_path(self, kind, filename): raise AssertionError('default voice should not resolve')


class FakeEngine:
    def __init__(self): self.calls=[]
    def generate(self, text, *, model_name, reference_audio, language, options):
        self.calls.append((text, model_name, reference_audio, language, dict(options)))
        # short deterministic waveform, valid 24 kHz mono
        seconds = max(0.12, len(text.split()) * 0.018)
        n = max(200, int(24000*seconds))
        t = torch.linspace(0, 1, n)
        wave = (0.02*torch.sin(2*torch.pi*220*t)).unsqueeze(0)
        return types.SimpleNamespace(waveform=wave, sample_rate=24000)


def test_each_safe_chunk_is_generated_exactly_once_and_only_loudness_runs(tmp_path, monkeypatch):
    engine=FakeEngine(); storage=FakeStorage(tmp_path)
    manager=QueueManager(engine, storage)
    text = ' '.join(('This is a clear sentence with words that must all be passed to Turbo. ' * 70).split())
    req=AudioJobCreate(text=text, voice_mode='default', title='fresh')
    public=asyncio.run(manager.create(req, enqueue=False))
    job=manager.get_raw(public['id'])
    called=[]
    def fake_loudness(path): called.append(path); return {'profile':'loudness-only','target_lufs':-12.5,'true_peak_dbfs':-0.8}
    monkeypatch.setattr(manager, '_normalise_loudness', fake_loudness)
    asyncio.run(manager._process(job))
    expected=split_turbo_long_text(text)
    assert [c[0] for c in engine.calls] == expected
    assert len(engine.calls) == len(expected)
    assert all(c[1]=='chatterbox-turbo' for c in engine.calls)
    assert all(c[3]=='en' for c in engine.calls)
    assert all(c[4]['temperature']==0.8 and c[4]['top_p']==0.95 and c[4]['top_k']==1000 for c in engine.calls)
    assert called and len(called)==1
    assert job['status']=='completed'
    assert job['quality_summary'] is None
    assert job['prosody_summary'] is None
    assert job['auto_emotion'] is False
